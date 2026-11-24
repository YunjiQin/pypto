/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */
#pragma once


// Indexed transport follows the PyPTO indexed_all_to_all_v template.

#include <cstdint>
#include <pto/pto-inst.hpp>

#include "platform_comm/comm_context.h"
#include "pto/comm/comm_types.hpp"
#include "pto/comm/pto_comm_inst.hpp"
#include "tensor.h"

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

// Keep this after the qualifier fallbacks so hardware builds retain the
// [aicore] entry annotation while using simpler's logical SPMD context.
#include "intrinsic.h"

#if !defined(PTO_URMA_SUPPORTED) || !defined(PTO_NPU_ARCH_A5)
#error "indexed_all_to_all_v requires A5 URMA"
#endif

namespace {

// Peer counts are read after Barrier A. Barrier B follows payload completion.
// Credits add +1/+1, wait GE(1)/GE(2), then subtract 2: never reset a signal
// that may already contain the next invocation's credits. All ranks use the
// same block count and collective order; receive-window reuse follows its consumer.
// Callers must order tasks sharing URMA queues (e.g. with the same InOut signal).
// One writer per peer here does not provide exclusion between independent tasks.

template <typename T>
AICORE inline __gm__ T *CommRemotePtr(__gm__ CommContext *ctx, __gm__ T *local_ptr, int pe) {
    uint64_t local_base = ctx->windowsIn[ctx->rankId];
    uint64_t offset = reinterpret_cast<uint64_t>(local_ptr) - local_base;
    return reinterpret_cast<__gm__ T *>(ctx->windowsIn[pe] + offset);
}

static constexpr uint32_t kBarrierThreads = 256;

__simt_vf__ __aicore__ __launch_bounds__(kBarrierThreads) void ExchangeCredits(
    __gm__ CommContext *ctx, __gm__ int32_t *signal, uint32_t rank, uint32_t peers, uint64_t stride, uint32_t block,
    int32_t threshold, uint32_t drop
) {
    if (drop == 0) {
        const uint64_t offset = reinterpret_cast<uint64_t>(signal + rank * stride + block) - ctx->windowsIn[rank];
        for (uint32_t peer = threadIdx.x; peer < peers; peer += kBarrierThreads) {
            if (peer != rank) {
                auto *remote = reinterpret_cast<__gm__ int32_t *>(ctx->windowsIn[peer] + offset);
                atomicAdd(remote, int32_t{1});
            }
        }
    }
    for (uint32_t peer = threadIdx.x; peer < peers; peer += kBarrierThreads) {
        if (peer == rank) continue;
        auto *local = signal + static_cast<uint64_t>(peer) * stride + block;
        if (drop != 0) {
            atomicAdd(local, int32_t{-2});
        } else {
            uint32_t spins = 0;
            while (atomicAdd(local, int32_t{0}) < threshold) {
                if (++spins == 100000000U) *reinterpret_cast<__gm__ volatile uint8_t *>(~uint64_t{0}) = 0;
// Match DeepEP's backoff: do not flood one remote atomic address.
#pragma unroll 1
                for (uint32_t delay = 0; delay < 20; ++delay)
                    __nop();
            }
        }
    }
}

AICORE inline void RunCreditRound(
    __gm__ CommContext *ctx, __gm__ int32_t *signal, int rank, int peers, int64_t stride, int32_t block,
    int32_t threshold, uint32_t drop
) {
    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
    cce::async_invoke<ExchangeCredits>(
        cce::dim3{kBarrierThreads, 1, 1}, ctx, signal, static_cast<uint32_t>(rank), static_cast<uint32_t>(peers),
        static_cast<uint64_t>(stride), static_cast<uint32_t>(block), threshold, drop
    );
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    dsb(DSB_DDR);
}

AICORE inline void ExchangeBarrier(
    __gm__ CommContext *ctx, __gm__ int32_t *signal, int rank, int peers, int64_t stride, int32_t block,
    int32_t threshold
) {
    RunCreditRound(ctx, signal, rank, peers, stride, block, threshold, 0);
}

AICORE inline void DropBarrierCredits(__gm__ int32_t *signal, int rank, int peers, int64_t stride, int32_t block) {
    RunCreditRound(nullptr, signal, rank, peers, stride, block, 0, 1);
}

// Fixed 4-WQEBB slots hold a 48-byte WRITE header and up to 12 16-byte
// SGEs. Like DeepEP dispatch, unused trailing WQEBBs contain valid NOPs.
static constexpr uint32_t kRecordsPerSqe = 6;
static constexpr uint32_t kBbsPerSlot = 4;
static constexpr uint32_t kSimtThreads = 256;
static constexpr uint32_t kSqesPerDoorbell = 64;

__simt_vf__ __aicore__ __launch_bounds__(kSimtThreads) void BuildIndexedSqes(
    __gm__ uint64_t *sq, __gm__ const int32_t *index, uint64_t input, uint64_t metadata, uint64_t remote,
    uint32_t row_bytes, uint32_t meta_bytes, uint32_t local_token, uint32_t head, uint32_t depth, uint32_t first,
    uint32_t rows, uint32_t num_sqes, uint64_t header0, uint64_t header1, uint64_t eid0, uint64_t eid1, uint64_t token,
    uint32_t input_rows
) {
    const uint32_t lane = threadIdx.x % 32;
    const uint32_t warp = threadIdx.x / 32;
    for (uint32_t q = warp; q < num_sqes; q += kSimtThreads / 32) {
        const uint32_t slot = first + q * kRecordsPerSqe;
        const uint32_t n = rows - slot < kRecordsPerSqe ? rows - slot : kRecordsPerSqe;
        const uint32_t slot_head = head + q * kBbsPerSlot;
        const uint32_t bb_head = slot_head + lane / 8;
        const uint64_t owner = static_cast<uint64_t>((bb_head & depth) == 0) << 31;
        const uint32_t used_bbs = (48 + 32 * n + 63) / 64;
        uint64_t word = 0;
        if (lane == 0) {
            // Normal odr=1; final WRITE odr=5, fence=1, CQE=1 (DeepEP).
            const uint64_t flags = slot + n == rows ? 0x2dULL : 0x01ULL;
            word = header0 | owner | (slot_head % depth) | (flags << 16);
        } else if (lane == 1) {
            word = header1 | (static_cast<uint64_t>(2 * n) << 24);
        } else if (lane == 2) {
            word = eid0;
        } else if (lane == 3) {
            word = eid1;
        } else if (lane == 4) {
            word = token;
        } else if (lane == 5) {
            word = remote + static_cast<uint64_t>(slot) * (row_bytes + meta_bytes);
        } else if (lane < 6 + n * 4) {
            const uint32_t j = (lane - 6) / 4;
            const bool meta = ((lane - 6) % 4) >= 2;
            const uint32_t bytes = meta ? meta_bytes : row_bytes;
            if ((lane & 1) == 0) {
                word = bytes | (static_cast<uint64_t>(local_token) << 32);
            } else {
                const int32_t t = index[slot + j];
                if (t < 0 || static_cast<uint32_t>(t) >= input_rows)
                    *reinterpret_cast<__gm__ volatile uint8_t *>(~uint64_t{0}) = 0;
                word = (meta ? metadata : input) + static_cast<uint64_t>(t) * bytes;
            }
        } else if (lane / 8 >= used_bbs && lane % 8 == 0) {
            word = (0x11ULL << 40) | owner | (bb_head % depth);
        }
        __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(
            sq + static_cast<uint64_t>(bb_head % depth) * 8 + lane % 8, word
        );
    }
}

AICORE inline void FailIndexed(__gm__ const char *message) {
    PTO_ASSERT(false, message);
    trap();
}

AICORE inline void SubmitIndexed(
    __gm__ CommContext *ctx, int peer, __gm__ uint8_t *input, __gm__ uint8_t *metadata, __gm__ uint8_t *target,
    __gm__ const int32_t *index, uint32_t rows, uint32_t row_bytes, uint32_t meta_bytes, uint32_t input_rows
) {
    if (rows == 0) return;
    using namespace pto::comm;
    using namespace pto::comm::urma;
    AsyncSession session;
    if (!BuildAsyncSession<DmaEngine::URMA>(reinterpret_cast<__gm__ uint8_t *>(ctx->workSpace), peer, session)) {
        FailIndexed("indexed_all_to_all_v: invalid URMA session");
        return;
    }
    auto *info = reinterpret_cast<__gm__ UrmaInfo *>(session.contextGm);
    auto *wq = urma::detail::GetWqContextAt(session, peer, 0);
    auto *cq = urma::detail::GetCqContextAt(session, peer, 0);
    auto *mem = urma::detail::GetRemoteMemInfo(info, peer, 0);
    if (wq->wqeShiftSize != 6 || wq->depth < kBbsPerSlot || wq->depth > 65536 || (wq->depth & (wq->depth - 1)) != 0 ||
        cq->depth == 0) {
        FailIndexed("indexed_all_to_all_v: unsupported queue geometry");
        return;
    }
    const uint64_t offset = reinterpret_cast<uint64_t>(target) - ctx->windowsIn[ctx->rankId];
    const uint64_t bytes = static_cast<uint64_t>(rows) * (row_bytes + meta_bytes);
    if (offset > mem->len || bytes > mem->len - offset) {
        FailIndexed("indexed_all_to_all_v: target exceeds registered peer window");
        return;
    }
    const uint64_t remote = mem->addr + offset;
    const uint64_t header0 = (static_cast<uint64_t>(mem->tokenValueValid) << 28) |
                             (static_cast<uint64_t>(mem->rmtJettyType) << 29) |
                             (static_cast<uint64_t>(mem->targetHint) << 32) | (0x03ULL << 40);
    const uint64_t header1 = (mem->tpn & 0xffffffULL) | (static_cast<uint64_t>(mem->tid & 0xfffffU) << 32);
    auto *eid = reinterpret_cast<__gm__ uint64_t *>(mem->eidAddr);
    const uint64_t eid0 = eid[0], eid1 = eid[1];
    uint32_t head = ld_dev(reinterpret_cast<__gm__ uint32_t *>(wq->headAddr), 0);
    uint32_t completions = wq->submittedWqeCount;
    const uint32_t total_sqes = (rows + kRecordsPerSqe - 1) / kRecordsPerSqe;
    // Like DeepEP, the entire peer transfer must fit before publishing any SQE.
    if (total_sqes > wq->depth / kBbsPerSlot) {
        FailIndexed("indexed_all_to_all_v: peer transfer exceeds SQ capacity");
        return;
    }
    const uint32_t required_bbs = total_sqes * kBbsPerSlot;
    if (!urma::detail::UrmaRingHasRoom(wq, cq, head, completions, required_bbs, 1)) {
        bool done = false;
        if (!urma::detail::UrmaPollCqAt(session, peer, 0, head, completions, true, done) || !done ||
            !urma::detail::UrmaRingHasRoom(wq, cq, head, completions, required_bbs, 1)) {
            FailIndexed("indexed_all_to_all_v: cannot reclaim queue space");
            return;
        }
    }
    for (uint32_t first = 0; first < rows;) {
        const uint32_t left = (rows - first + kRecordsPerSqe - 1) / kRecordsPerSqe;
        const uint32_t n = left < kSqesPerDoorbell ? left : kSqesPerDoorbell;
        // DeepEP local source SGEs use tokenId=0. mem->localTokenId describes
        // the symmetric window registration, which does not cover a local input Tensor.
        set_flag(PIPE_S, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
        cce::async_invoke<BuildIndexedSqes>(
            cce::dim3{kSimtThreads, 1, 1}, reinterpret_cast<__gm__ uint64_t *>(wq->bufAddr), index,
            reinterpret_cast<uint64_t>(input), reinterpret_cast<uint64_t>(metadata), remote, row_bytes, meta_bytes,
            uint32_t{0}, head, wq->depth, first, rows, n, header0, header1, eid0, eid1,
            static_cast<uint64_t>(mem->rmtTokenValue), input_rows
        );
        // Match DeepEP's V->S handoff before publishing the fully constructed SQ.
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        dsb(DSB_DDR);
        head += n * kBbsPerSlot;
        first += n * kRecordsPerSqe;
        // Intermediate doorbells start transport while the next SQEs are built.
        // Only the final WRITE requests a CQE for the entire peer transfer.
        if (first >= rows) ++completions;
        urma::detail::CommitPostedWqes(wq, head, completions);
    }
}

AICORE inline void WaitIndexed(__gm__ CommContext *ctx, int peer) {
    using namespace pto::comm;
    using namespace pto::comm::urma;
    AsyncSession session;
    if (!BuildAsyncSession<DmaEngine::URMA>(reinterpret_cast<__gm__ uint8_t *>(ctx->workSpace), peer, session)) {
        FailIndexed("indexed_all_to_all_v: invalid URMA completion session");
        return;
    }
    auto *wq = urma::detail::GetWqContextAt(session, peer, 0);
    const uint32_t head = ld_dev(reinterpret_cast<__gm__ uint32_t *>(wq->headAddr), 0);
    const uint32_t completions = wq->submittedWqeCount;
    bool done = false;
    if (!urma::detail::UrmaPollCqAt(session, peer, 0, head, completions, true, done) || !done) {
        FailIndexed("indexed_all_to_all_v: payload completion failed");
        return;
    }
}

}  // namespace
