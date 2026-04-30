# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""End-to-end regression for orchestration-level transpose + slice.

Reproduces issue #1209 follow-up: ``pl.transpose(x, 0, 1)`` followed by
``pl.slice(xt, [1, T], [h, 0])`` must access column ``h`` of ``x`` (not the
first contiguous chunk of ``x`` in memory). The runtime ``Tensor::transpose``
is a metadata-only swap, so the IR result must record swapped physical
strides for the codegen to emit a correctly addressed ``make_tensor_view``.

Run via pytest (requires Ascend hardware) or as a script with
``-p {a2a3,a2a3sim,a5,a5sim}``.
"""

import argparse

import pypto.language as pl
import pytest
import torch
from pypto import ir
from pypto.runtime import RunConfig

T = 8
PAD = 16
N = 4


def build_program():
    @pl.program
    class TransposeSliceRepro:
        @pl.function(type=pl.FunctionType.Opaque)
        def main(
            self,
            x: pl.Tensor[[T, PAD], pl.FP32],
            out: pl.Out[pl.Tensor[[T, N], pl.FP32]],
        ):
            xt = pl.transpose(x, 0, 1)
            for h in pl.range(N):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="slice_transposed_row"):
                    col = pl.reshape(pl.slice(xt, [1, T], [h, 0]), [T, 1])
                    out = pl.assemble(out, col, [0, h])
            return out

    return TransposeSliceRepro


def _run(platform: str, device_id: int = 0) -> None:
    x = torch.arange(T * PAD, dtype=torch.float32).reshape(T, PAD)
    out = torch.zeros(T, N)
    p = ir.compile(build_program(), platform=platform)
    p(x, out, config=RunConfig(device_id=device_id))
    expected = x[:, :N]
    torch.testing.assert_close(out, expected)


def test_transpose_slice_assemble_a5sim() -> None:
    """Compile + run the reproducer on a5sim; assert column-h selection is correct.

    a2a3 is intentionally NOT covered: that arch's PTOAS rejects
    ``TLOAD(VecTile_RowMajor, GlobalTensor<DN>)`` (only ``ND2ND``, ``DN2DN``,
    and ``NZ2NZ`` cross-layout pairs are supported on a2a3). a5 lifts this
    restriction, so DN-tagged GlobalTensor with explicit strides flows into
    a Vec tile correctly.
    """
    _run("a5sim")


if __name__ == "__main__":
    # Two run modes:
    #   --pytest       run via pytest discovery (matches the CI entry point)
    #   default        argparse manual runner — useful for picking a specific
    #                  platform/device when iterating locally
    import sys

    if "--pytest" in sys.argv:
        sys.exit(pytest.main([__file__, "-v"]))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--platform",
        type=str,
        default="a5sim",
        choices=["a2a3", "a2a3sim", "a5", "a5sim"],
    )
    parser.add_argument("-d", "--device", type=int, default=0)
    args = parser.parse_args()
    _run(args.platform, device_id=args.device)
    print("OK")
