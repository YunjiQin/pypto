# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU fixtures and semantic oracle for bit-preserving indexed dispatch."""

import torch

TOPK_CAPACITY = 6
TOKEN_ID_WORD = 12
HEADER_WORDS = 16


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def make_inputs(
    nranks: int = 8,
    tokens: int = 72,
    hidden: int = 5120,
    topk: int = 6,
    local_experts: int = 3,
    dtype: str = "bf16",
    routing: str = "uniform",
    seed: int = 41,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return raw INT32 payload and metadata tensors, both shaped [rank, token, word].

    BF16 payload stores finite, signed BF16 values without conversion during
    dispatch. The FP8 fixture stores arbitrary bytes plus one scale byte per
    32 hidden elements; it tests transport rather than quantization accuracy.
    IDs occupy words 0:6, FP32 weight bits 6:12, token ID word 12, and optional
    scale bytes start at word 16. Invalid Top-K slots have ID -1 and weight 0.
    Local and hotspot fixtures leave excess slots invalid when K exceeds the
    number of experts on their sole destination rank.
    """
    if nranks <= 0 or tokens < 0 or hidden <= 0 or local_experts <= 0:
        raise ValueError("nranks, hidden and local_experts must be positive; tokens must be nonnegative")
    if not 0 < topk <= TOPK_CAPACITY:
        raise ValueError(f"topk must be in [1, {TOPK_CAPACITY}]")
    if dtype not in ("bf16", "fp8"):
        raise ValueError(f"unsupported dtype: {dtype}")
    if routing not in ("uniform", "spread", "local", "hotspot", "empty"):
        raise ValueError(f"unsupported routing: {routing}")
    byte_width = hidden * (2 if dtype == "bf16" else 1)
    if byte_width % 4:
        raise ValueError("payload row size must be a multiple of four bytes")

    generator = torch.Generator().manual_seed(seed)
    if dtype == "bf16":
        payload = torch.randn(nranks, tokens, hidden, generator=generator).to(torch.bfloat16)
    else:
        payload = torch.randint(0, 256, (nranks, tokens, hidden), generator=generator, dtype=torch.uint8)
    x = payload.view(torch.int32)
    scale_bytes = _align_up((hidden + 31) // 32, 32) if dtype == "fp8" else 0
    meta = torch.zeros(nranks, tokens, HEADER_WORDS + scale_bytes // 4, dtype=torch.int32)
    meta[:, :, :TOPK_CAPACITY] = -1
    meta[:, :, TOKEN_ID_WORD] = torch.arange(tokens, dtype=torch.int32)
    if scale_bytes:
        scales = torch.randint(1, 255, (nranks, tokens, scale_bytes), generator=generator, dtype=torch.uint8)
        meta[:, :, HEADER_WORDS:] = scales.view(torch.int32)

    total_experts = nranks * local_experts
    weights = torch.zeros(nranks, tokens, TOPK_CAPACITY, dtype=torch.float32)
    for source in range(nranks):
        for token in range(tokens):
            if routing == "empty":
                experts = []
            elif routing in ("local", "hotspot"):
                destination = source if routing == "local" else 0
                experts = [destination * local_experts + e for e in range(min(topk, local_experts))]
            elif routing == "spread":
                first_rank = (source + token) % nranks
                experts = [
                    ((first_rank + choice) % nranks) * local_experts + choice // nranks
                    for choice in range(min(topk, total_experts))
                ]
            else:
                experts = torch.randperm(total_experts, generator=generator)[:topk].tolist()
            if experts:
                meta[source, token, : len(experts)] = torch.tensor(experts, dtype=torch.int32)
                values = torch.rand(len(experts), generator=generator)
                weights[source, token, : len(experts)] = values / values.sum()
    meta[:, :, TOPK_CAPACITY : 2 * TOPK_CAPACITY] = weights.view(torch.int32)
    return x.contiguous(), meta.contiguous()


def _validate_dispatch_inputs(
    x: torch.Tensor,
    meta: torch.Tensor,
    alignment: int,
    local_experts: int,
    topk: int,
) -> None:
    if x.device.type != "cpu" or meta.device.type != "cpu":
        raise ValueError("reference_dispatch requires CPU tensors")
    if x.dtype != torch.int32 or meta.dtype != torch.int32:
        raise ValueError("payload and metadata must use raw INT32 storage")
    if x.ndim != 3 or meta.ndim != 3 or x.shape[:2] != meta.shape[:2]:
        raise ValueError("payload and metadata must have matching [rank, token, word] shapes")
    if alignment <= 0 or local_experts <= 0 or not 0 < topk <= TOPK_CAPACITY:
        raise ValueError("alignment and local_experts must be positive, and topk must be in [1, 6]")
    if meta.shape[2] < HEADER_WORDS:
        raise ValueError("metadata must contain the 16-word route header")


def reference_dispatch(
    x: torch.Tensor,
    meta: torch.Tensor,
    alignment: int = 1,
    local_experts: int = 3,
    topk: int = 6,
) -> dict[str, torch.Tensor]:
    """Build expert-grouped outputs and the two-level reverse mapping on CPU.

    Each destination receives one record per (source rank, input token),
    irrespective of its number of local expert matches. Expert rows are
    ordered by source rank, input token and Top-K position. Counts store
    [valid count, aligned start, valid end]. Only valid rows and expert padding
    have defined payload; this oracle initializes the unused tail to zero.
    """
    _validate_dispatch_inputs(x, meta, alignment, local_experts, topk)
    nranks, tokens, row_words = x.shape
    total_experts = nranks * local_experts
    send_lists = [[[] for _ in range(nranks)] for _ in range(nranks)]
    choices = [[[] for _ in range(local_experts)] for _ in range(nranks)]
    for source in range(nranks):
        for token in range(tokens):
            ids = meta[source, token, :topk].tolist()
            valid_ids = [expert for expert in ids if 0 <= expert < total_experts]
            if len(valid_ids) != len(set(valid_ids)):
                raise ValueError("valid expert IDs within one token must be distinct")
            destinations = sorted({expert // local_experts for expert in valid_ids})
            slots = {}
            for destination in destinations:
                slots[destination] = len(send_lists[source][destination])
                send_lists[source][destination].append(token)
            for choice, expert in enumerate(ids):
                if 0 <= expert < total_experts:
                    destination, local_expert = divmod(expert, local_experts)
                    choices[destination][local_expert].append((source, token, slots[destination], choice))

    capacity = _align_up(
        nranks * tokens * min(topk, local_experts) + (alignment - 1) * local_experts, alignment
    )
    expert_x = torch.zeros(nranks, capacity, row_words, dtype=torch.int32)
    expert_metadata = torch.zeros(nranks, capacity, meta.shape[2] + 4, dtype=torch.int32)
    expert_metadata[:, :, -4:] = -1
    expert_counts = torch.zeros(nranks, local_experts, 3, dtype=torch.int32)
    valid_rows = torch.zeros(nranks, capacity, dtype=torch.bool)
    for destination in range(nranks):
        start = 0
        for expert, rows in enumerate(choices[destination]):
            end = start + len(rows)
            expert_counts[destination, expert] = torch.tensor([len(rows), start, end], dtype=torch.int32)
            valid_rows[destination, start:end] = True
            for row, (source, token, slot, choice) in enumerate(rows, start):
                expert_x[destination, row] = x[source, token]
                expert_metadata[destination, row, : meta.shape[2]] = meta[source, token]
                expert_metadata[destination, row, -4:] = torch.tensor(
                    [source, slot, choice, token], dtype=torch.int32
                )
            start += _align_up(len(rows), alignment)

    send_index = torch.full((nranks, nranks, tokens), -1, dtype=torch.int32)
    send_counts = torch.zeros(nranks, nranks, dtype=torch.int32)
    for source in range(nranks):
        for destination, token_ids in enumerate(send_lists[source]):
            send_counts[source, destination] = len(token_ids)
            send_index[source, destination, : len(token_ids)] = torch.tensor(token_ids, dtype=torch.int32)
    return {
        "expert_x": expert_x,
        "expert_metadata": expert_metadata,
        "expert_counts": expert_counts,
        "valid_rows": valid_rows,
        "send_index": send_index,
        "send_counts": send_counts,
        "recv_counts": send_counts.T.contiguous(),
    }
