# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import importlib.util
from pathlib import Path

import pytest
import torch

_PATH = Path(__file__).with_name("reference.py")
_SPEC = importlib.util.spec_from_file_location("indexed_dispatch_reference", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
reference = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(reference)


def test_hand_written_ep4_routes_and_expert_rows():
    x = torch.arange(8, dtype=torch.int32).reshape(4, 2, 1).expand(4, 2, 4).contiguous()
    meta = torch.zeros(4, 2, 16, dtype=torch.int32)
    meta[:, :, :6] = -1
    meta[:, :, :2] = torch.tensor([[[2, 3], [4, 6]], [[0, 1], [2, 4]], [[2, 3], [6, 7]], [[1, 3], [4, 5]]])
    meta[:, :, 6:8] = torch.tensor([0.25, 0.75], dtype=torch.float32).view(torch.int32)
    meta[:, :, 12] = torch.arange(2, dtype=torch.int32)
    result = reference.reference_dispatch(x, meta, local_experts=2, topk=2)

    assert result["send_counts"].tolist() == [[0, 1, 1, 1], [1, 1, 1, 0], [0, 1, 0, 1], [1, 1, 1, 0]]
    assert result["send_index"][0].tolist() == [[-1, -1], [0, -1], [1, -1], [1, -1]]
    assert result["expert_counts"][1].tolist() == [[3, 0, 3], [3, 3, 6]]
    assert result["expert_x"][1, :6, 0].tolist() == [0, 3, 4, 0, 4, 6]
    assert result["expert_metadata"][1, :6, -4:].tolist() == [
        [0, 0, 0, 0],
        [1, 0, 0, 1],
        [2, 0, 0, 0],
        [0, 0, 1, 0],
        [2, 0, 1, 0],
        [3, 0, 1, 0],
    ]
    assert int(result["send_counts"].sum()) == 11
    assert int(result["valid_rows"].sum()) == 16

    padded = reference.reference_dispatch(x, meta, alignment=4, local_experts=2, topk=2)
    assert padded["expert_counts"][1].tolist() == [[3, 0, 3], [3, 4, 7]]
    assert torch.count_nonzero(padded["expert_x"][1, 3]) == 0
    assert torch.equal(padded["expert_x"][1, 4:7], result["expert_x"][1, 3:6])


@pytest.mark.parametrize("dtype", ["bf16", "fp8"])
@pytest.mark.parametrize("routing", ["uniform", "spread", "local", "hotspot", "empty"])
def test_payload_metadata_and_reverse_mapping(dtype, routing):
    x, meta = reference.make_inputs(
        nranks=4, tokens=5, hidden=64, local_experts=3, dtype=dtype, routing=routing
    )
    result = reference.reference_dispatch(x, meta, alignment=4)
    assert torch.equal(result["recv_counts"], result["send_counts"].T)
    assert x.shape == (4, 5, 32 if dtype == "bf16" else 16)
    assert meta.shape == (4, 5, 16 if dtype == "bf16" else 24)
    assert int(result["valid_rows"].sum()) == int(((meta[:, :, :6] >= 0) & (meta[:, :, :6] < 12)).sum())

    restored = torch.zeros(4, 5, 6, dtype=torch.int32)
    for destination in range(4):
        for expert in range(3):
            count, start, end = result["expert_counts"][destination, expert].tolist()
            assert end - start == count
            assert start % 4 == 0
            for row in range(start, end):
                record = result["expert_metadata"][destination, row]
                source, slot, choice, token = record[-4:].tolist()
                assert int(result["send_index"][source, destination, slot]) == token
                assert int(meta[source, token, choice]) == destination * 3 + expert
                assert torch.equal(result["expert_x"][destination, row], x[source, token])
                assert torch.equal(record[:-4], meta[source, token])
                restored[source, token, choice] += 1
            padded_end = start + ((count + 3) // 4) * 4
            assert torch.count_nonzero(result["expert_x"][destination, end:padded_end]) == 0
    assert torch.equal(restored, (meta[:, :, :6] >= 0).to(torch.int32))

    if routing == "local":
        assert int(result["send_counts"].trace()) == 20
        assert int(result["send_counts"].sum()) == 20
    elif routing == "hotspot":
        assert result["send_counts"][:, 0].tolist() == [5, 5, 5, 5]
        assert torch.count_nonzero(result["send_counts"][:, 1:]) == 0
        assert result["expert_counts"][0, :, 0].tolist() == [20, 20, 20]
    elif routing == "empty":
        assert torch.count_nonzero(result["send_counts"]) == 0
        assert torch.count_nonzero(result["expert_counts"]) == 0


def test_report_shape_fp8_scale_bytes_and_reproducibility():
    x, meta = reference.make_inputs(nranks=8, tokens=2, dtype="fp8")
    other_x, other_meta = reference.make_inputs(nranks=8, tokens=2, dtype="fp8")
    assert x.shape == (8, 2, 1280)
    assert meta.shape == (8, 2, 56)
    assert torch.equal(x, other_x)
    assert torch.equal(meta, other_meta)
    assert torch.count_nonzero(meta[:, :, 16:].contiguous().view(torch.uint8)) == 8 * 2 * 160


def test_invalid_expert_ids_are_ignored_and_duplicates_rejected():
    x, meta = reference.make_inputs(nranks=2, tokens=1, hidden=8, routing="empty")
    meta[0, 0, :3] = torch.tensor([-1, 6, -10])
    result = reference.reference_dispatch(x, meta)
    assert torch.count_nonzero(result["send_counts"]) == 0
    meta[0, 0, :2] = 1
    with pytest.raises(ValueError, match="distinct"):
        reference.reference_dispatch(x, meta)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
