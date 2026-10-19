# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Shared-signature kernel contracts without device or optional runtime imports."""

import json
import struct
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from pypto._kernel_abi import MAX_KERNEL_SCALARS, MAX_KERNEL_TENSORS, KernelABI
from pypto.ir.param_info import MAX_TENSOR_RANK, ParamInfo
from pypto.pypto_core import DataType
from pypto.pypto_core.ir import ParamDirection

SDK = "c1f89ea431f5306f78c0e4ab9573c829756d7f1e"


@pytest.fixture
def params():
    return [
        ParamInfo("x", ParamDirection.In, [8], DataType.FP32),
        ParamInfo("scale", ParamDirection.In, None, DataType.FP32),
        ParamInfo("out", ParamDirection.Out, [8], DataType.FP32),
        ParamInfo("step", ParamDirection.In, None, DataType.INT32),
        ParamInfo("cache", ParamDirection.InOut, [8], DataType.FP32),
    ]


def contract(
    params, *, platform="a2a3", runtime="host_build_graph", sdk_revision=SDK, return_aliases=(2, 4, 2)
):
    return KernelABI(
        params, platform=platform, runtime=runtime, sdk_revision=sdk_revision, return_aliases=return_aliases
    )


def test_interleaved_pools_and_duplicate_external_returns(params):
    abi = contract(params)
    x, out, cache = object(), object(), object()
    for scale, step in ((1.25, 0), (-3.5, 9)):
        tensors, scalars, returns = abi.bind([x, scale, out, step, cache])
        assert tensors == [x, out, cache]
        assert scalars == [scale, step]
        assert returns == [out, cache, out]
        assert returns[0] is returns[2]
    assert abi.tensor_indices == (0, 2, 4)
    assert abi.scalar_indices == (1, 3)
    assert contract(params, return_aliases=(0,)).bind([x, 1.0, out, 0, cache])[2][0] is x
    with pytest.raises(TypeError, match="including all Out/InOut"):
        abi.bind([x, 1.0, out, 0])


def test_round_trip_and_signature_snapshot(params):
    abi = contract(params)
    record = json.loads(json.dumps(abi.record()))
    KernelABI.from_record(record).require_compatible(abi)
    params[0] = replace(params[0], shape=[9])
    abi.record()["params"][0]["shape"][0] = 11
    assert abi.parameters[0].shape == (8,)
    with pytest.raises(ValueError, match="does not match"):
        abi.require_compatible(contract(params))


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", True),
        ("schema", 1.0),
        ("schema", 2),
        ("argument_abi", "TaskArgs"),
        ("sdk_revision", "bad"),
        ("platform", "a2a3sim"),
        ("runtime", "other"),
        ("return_aliases", [1]),
        ("return_aliases", [True]),
        ("return_aliases", [-1]),
        ("return_aliases", [99]),
        ("unknown", 0),
        ("params", None),
    ],
)
def test_reject_noncanonical_descriptor(params, field, value):
    record = contract(params).record()
    record[field] = value
    with pytest.raises(ValueError, match="kernel ABI"):
        KernelABI.from_record(record)


@pytest.mark.parametrize(
    "field,value",
    [
        ("name", ""),
        ("name", "out"),
        ("direction", "mro"),
        ("dtype", "unknown"),
        ("shape", [True]),
        ("shape", [0]),
        ("shape", [2**32]),
        ("shape", [1] * 6),
        ("layout", "unsupported"),
        ("extra", 0),
    ],
)
def test_reject_corrupt_parameter(params, field, value):
    record = contract(params).record()
    record["params"][0][field] = value
    with pytest.raises(ValueError):
        KernelABI.from_record(record)


@pytest.mark.parametrize(
    "changes",
    [
        {"platform": "a5"},
        {"runtime": "tensormap_and_ringbuffer"},
        {"sdk_revision": "a" * 40},
        {"return_aliases": (4, 2)},
    ],
)
def test_artifact_identity_is_checked_separately_from_capabilities(params, changes):
    with pytest.raises(ValueError, match="does not match"):
        contract(params).require_compatible(contract(params, **changes))


def test_scalar_contract_and_dynamic_layout(params):
    for dtype in (DataType.FP16, DataType.BF16):
        contract([replace(params[1], dtype=dtype)], return_aliases=())
    with pytest.raises(ValueError, match="Scalar"):
        contract([replace(params[1], direction=ParamDirection.Out)], return_aliases=())
    abi = contract([replace(params[0], shape=[-1, 16], layout="NZ")], return_aliases=(0,))
    restored = KernelABI.from_record(abi.record())
    assert restored.parameters[0].layout == "NZ"
    assert restored.parameters[0].shape == (-1, 16)


@pytest.mark.parametrize("layout", ["MX_A_ZZ", "MX_B_NN"])
def test_mx_kernel_signature_round_trip(params, layout):
    param = replace(params[0], shape=[16, 32], dtype=DataType.FP8E8M0, layout=layout)
    abi = contract([param], return_aliases=(0,))
    KernelABI.from_record(abi.record()).require_compatible(abi)
    assert abi.parameters[0] is param


@pytest.mark.parametrize("scalar,limit", [(False, MAX_KERNEL_TENSORS), (True, MAX_KERNEL_SCALARS)])
def test_pool_capacity(params, scalar, limit):
    param = params[1 if scalar else 0]
    entries = [replace(param, name=f"p{i}") for i in range(limit)]
    contract(entries, return_aliases=())
    with pytest.raises(ValueError, match="capacity"):
        contract([*entries, replace(param, name="extra")], return_aliases=())


def test_limits_match_actual_pinned_sdk(tmp_path):
    root = Path(__file__).resolve().parents[3]
    source = tmp_path / "abi.cpp"
    executable = tmp_path / "abi"
    source.write_text(
        '#include "task_args.h"\n#include <iostream>\n'
        f"static_assert(CHIP_MAX_TENSOR_ARGS == {MAX_KERNEL_TENSORS});\n"
        f"static_assert(CHIP_MAX_SCALAR_ARGS == {MAX_KERNEL_SCALARS});\n"
        f"static_assert(MAX_TENSOR_DIMS == {MAX_TENSOR_RANK});\n"
        "static_assert(sizeof(((ChipStorageTaskArgs*)nullptr)->scalars_[0]) == 8);\n"
        "int main() { std::cout << to_u64(float(-3.5)) << ' ' << to_u64(int32_t(-7)) "
        "<< ' ' << to_u64(int8_t(-1)); }\n"
    )
    result = subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-o",
            str(executable),
            "-I",
            str(root / "runtime/src/common/task_interface"),
            "-I",
            str(root / "runtime/src/common/platform/include"),
            str(source),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    actual = subprocess.run([str(executable)], check=True, capture_output=True, text=True).stdout.split()
    expected = [
        int.from_bytes(struct.pack("<" + fmt, value), "little")
        for fmt, value in (("f", -3.5), ("i", -7), ("b", -1))
    ]
    assert [int(value) for value in actual] == expected


def test_pure_contract_without_optional_runtime(run_without_optional_runtime):
    result = run_without_optional_runtime("""
from pypto._kernel_abi import KernelABI
from pypto.ir.param_info import ParamInfo
from pypto.pypto_core import DataType
from pypto.pypto_core.ir import ParamDirection
abi = KernelABI([ParamInfo('x', ParamDirection.In, [8], DataType.FP32)],
                platform='a2a3', runtime='host_build_graph', sdk_revision='a'*40, return_aliases=(0,))
KernelABI.from_record(abi.record()).require_compatible(abi)
x = object()
assert abi.bind([x])[2][0] is x
""")
    assert result.returncode == 0, result.stderr


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
