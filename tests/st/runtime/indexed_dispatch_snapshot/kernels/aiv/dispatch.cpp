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
#include "indexed_transport.h"

namespace {
constexpr uint32_t kExperts = 3;
constexpr uint32_t kTopK = 6;
constexpr uint32_t kControlStride = 512;
constexpr uint32_t kIndexOffset = 16;
constexpr uint32_t kMapOffset = 128;
constexpr uint32_t kHistogramReady = 7;
constexpr uint32_t kRouteReady = 8;
constexpr uint32_t kDataReady = 9;
constexpr uint32_t kCountReady = 10;
constexpr uint32_t kTileWords = 2560;
using DispatchShape = pto::Shape<1, 1, 1, 1, pto::DYNAMIC>;
using DispatchStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
using DispatchGlobal = pto::GlobalTensor<uint32_t, DispatchShape, DispatchStride>;
using DispatchTile = pto::Tile<pto::TileType::Vec, uint32_t, 1, kTileWords, pto::BLayout::RowMajor, -1, -1>;

template <typename T>
AICORE inline __gm__ T *Data(__gm__ Tensor *t) {
    return reinterpret_cast<__gm__ T *>(t->buffer.addr) + t->start_offset;
}
AICORE inline uint64_t Stamp() {
    uint64_t value;
    asm volatile("MOV %0, SYS_CNT" : "=l"(value));
    return value;
}
AICORE inline void LocalBarrier(__gm__ int32_t *control, uint32_t workers, uint32_t block, uint32_t epoch) {
    dsb(DSB_DDR);
    st_dev(epoch, control + block * kControlStride, 0);
    dsb(DSB_DDR);
    using FlagShape = pto::Shape<1, 1, 1, pto::DYNAMIC, 8>;
    using FlagGlobal = pto::GlobalTensor<uint32_t, FlagShape, DispatchStride>;
    using FlagTile = pto::Tile<pto::TileType::Vec, uint32_t, 32, 8, pto::BLayout::RowMajor, -1, 8>;
    FlagTile flags(workers);
    // Flag staging is disjoint from the hidden ring and the routing tile.
    pto::TASSIGN(flags, 8 * kTileWords * sizeof(uint32_t) + 72 * 8 * sizeof(uint32_t));
    FlagShape shape(1, 1, 1, workers, 8);
    DispatchStride stride(
        workers * kControlStride, workers * kControlStride, workers * kControlStride, kControlStride, 1
    );
    FlagGlobal source(reinterpret_cast<__gm__ uint32_t *>(control), shape, stride);
    bool ready = false;
    while (!ready) {
        pto::TLOAD(flags, source);
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        ready = true;
        for (uint32_t worker = 0; worker < workers; ++worker)
            ready &= flags.GetValue(worker * 8) >= epoch;
        set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
    }
    dsb(DSB_DDR);
}

AICORE inline void PublishReady(__gm__ int32_t *flag, uint32_t epoch) {
    dsb(DSB_DDR);
    st_dev(epoch, flag, 0);
    dsb(DSB_DDR);
}
AICORE inline void WaitReady(__gm__ int32_t *flag, uint32_t epoch) {
    while (ld_dev(flag, 0) < epoch) {
#pragma unroll 1
        for (uint32_t delay = 0; delay < 20; ++delay)
            __asm__ volatile("nop");
    }
    dsb(DSB_DDR);
}
AICORE inline void ComputeOwnPrefixes(
    __gm__ int32_t *control, __gm__ int32_t *counts, uint32_t workers, uint32_t block, uint32_t alignment,
    uint32_t epoch, uint32_t *cursor, uint32_t *starts, uint32_t *ends
) {
    using CountShape = pto::Shape<1, 1, 1, pto::DYNAMIC, 8>;
    using CountGlobal = pto::GlobalTensor<uint32_t, CountShape, DispatchStride>;
    using CountTile = pto::Tile<pto::TileType::Vec, uint32_t, 32, 8, pto::BLayout::RowMajor, -1, 8>;
    CountTile histograms(workers);
    pto::TASSIGN(histograms, 8 * kTileWords * sizeof(uint32_t) + 72 * 8 * sizeof(uint32_t));
    CountShape shape(1, 1, 1, workers, 8);
    DispatchStride stride(
        workers * kControlStride, workers * kControlStride, workers * kControlStride, kControlStride, 1
    );
    CountGlobal source(reinterpret_cast<__gm__ uint32_t *>(control), shape, stride);
    bool ready = false;
    while (!ready) {
        pto::TLOAD(histograms, source);
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        ready = true;
        for (uint32_t worker = 0; worker < workers; ++worker)
            ready &= histograms.GetValue(worker * 8 + kHistogramReady) >= epoch;
        set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
    }
    // Count loads follow acquisition of all producer epochs.
    dsb(DSB_DDR);
    pto::TLOAD(histograms, source);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    uint32_t start = 0;
    for (uint32_t expert = 0; expert < kExperts; ++expert) {
        uint32_t total = 0, before = 0;
        for (uint32_t worker = 0; worker < workers; ++worker) {
            const uint32_t count = histograms.GetValue(worker * 8 + 4 + expert);
            total += count;
            if (worker < block) before += count;
        }
        cursor[expert] = start + before;
        starts[expert] = start;
        ends[expert] = start + total;
        if (block == 0) {
            st_dev(total, counts + expert * 3, 0);
            st_dev(start, counts + expert * 3 + 1, 0);
            st_dev(start + total, counts + expert * 3 + 2, 0);
        }
        start += (total + alignment - 1) / alignment * alignment;
    }
}

AICORE inline void Copy(__gm__ uint32_t *dst, __gm__ uint32_t *src, uint32_t words, DispatchTile &tile) {
    tile.ColMaskInternal = words;
    DispatchShape shape(1, 1, 1, 1, words);
    DispatchStride stride(words, words, words, words, 1);
    DispatchGlobal target(dst, shape, stride), source(src, shape, stride);
    pto::comm::TPUT(target, source, tile);
}
AICORE inline void StoreDone() {
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    dsb(DSB_DDR);
}

// Each worker owns a disjoint interval; slots are reused only after MTE3 drains.
AICORE inline void CopySelf(
    __gm__ uint32_t *x, __gm__ uint32_t *meta, __gm__ uint32_t *target, __gm__ int32_t *index, uint32_t count,
    uint32_t row_words, uint32_t meta_words, uint32_t worker, uint32_t workers
) {
    constexpr uint32_t stages = 8;
    const uint32_t first = count * worker / workers, end = count * (worker + 1) / workers;
    if (first == end) return;
    for (uint32_t stage = 0; stage < stages; ++stage)
        set_flag(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(stage));
    uint32_t stage = 0;
    for (uint32_t slot = first; slot < end; ++slot) {
        const uint32_t token = ld_dev(index + slot, 0);
        for (uint32_t part = 0; part < 2; ++part) {
            const uint32_t words = part == 0 ? row_words : meta_words;
            auto *src = part == 0 ? x + token * row_words : meta + token * meta_words;
            auto *dst = target + slot * (row_words + meta_words) + (part == 0 ? 0 : row_words);
            const auto event = static_cast<event_t>(stage);
            wait_flag(PIPE_MTE3, PIPE_MTE2, event);
            DispatchTile staging(1, words);
            pto::TASSIGN(staging, stage * kTileWords * sizeof(uint32_t));
            DispatchShape shape(1, 1, 1, 1, words);
            DispatchStride stride(words, words, words, words, 1);
            DispatchGlobal source(src, shape, stride), destination(dst, shape, stride);
            pto::TLOAD(staging, source);
            set_flag(PIPE_MTE2, PIPE_MTE3, event);
            wait_flag(PIPE_MTE2, PIPE_MTE3, event);
            pto::TSTORE<pto::TStoreL2Hint::NotAllocClean>(destination, staging);
            set_flag(PIPE_MTE3, PIPE_MTE2, event);
            stage = (stage + 1) % stages;
        }
    }
    for (uint32_t stage = 0; stage < stages; ++stage)
        wait_flag(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(stage));
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    dsb(DSB_DDR);
}

__simt_vf__ __aicore__ __launch_bounds__(256) void ExpandMetadata(
    __gm__ uint32_t *records, __gm__ int32_t *control, __gm__ uint32_t *output, uint32_t source, uint32_t count,
    uint32_t row_words, uint32_t meta_words, uint32_t rank, uint32_t first
) {
    const uint32_t record_words = row_words + meta_words;
    const uint32_t lane = threadIdx.x % 32;
    for (uint32_t item = threadIdx.x / 32; item < count * kTopK; item += 8) {
        const uint32_t slot = item / kTopK, choice = item % kTopK;
        auto *meta = records + slot * record_words + row_words;
        const int32_t expert = static_cast<int32_t>(meta[choice]) - rank * kExperts;
        if (expert < 0 || expert >= kExperts) continue;
        const uint32_t row = control[kMapOffset + slot * kExperts + expert];
        auto *dst = output + row * (meta_words + 4);
        for (uint32_t word = lane; word < meta_words + 4; word += 32) {
            uint32_t value;
            if (word < meta_words) value = meta[word];
            else if (word == meta_words) value = source;
            else if (word == meta_words + 1) value = first + slot;
            else if (word == meta_words + 2) value = choice;
            else value = meta[12];
            dst[word] = value;
        }
    }
}
}  // namespace

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    auto *xt = reinterpret_cast<__gm__ Tensor *>(args[0]);
    auto *mt = reinterpret_cast<__gm__ Tensor *>(args[1]);
    auto *x = Data<uint32_t>(xt);
    auto *meta = Data<uint32_t>(mt);
    auto *records = Data<uint32_t>(reinterpret_cast<__gm__ Tensor *>(args[2]));
    auto *control = Data<int32_t>(reinterpret_cast<__gm__ Tensor *>(args[3]));
    auto *expert_x = Data<uint32_t>(reinterpret_cast<__gm__ Tensor *>(args[4]));
    auto *expert_meta = Data<uint32_t>(reinterpret_cast<__gm__ Tensor *>(args[5]));
    auto *counts = Data<int32_t>(reinterpret_cast<__gm__ Tensor *>(args[6]));
    auto *stats = Data<uint64_t>(reinterpret_cast<__gm__ Tensor *>(args[7]));
    auto *ctx = reinterpret_cast<__gm__ CommContext *>(args[8]);
    const uint32_t rounds = args[9], alignment = args[10], transport = args[11];
    const uint32_t peers = ctx->rankNum, rank = ctx->rankId, block = get_block_idx(args);
    const uint32_t workers = get_block_num(args);
    const uint32_t tokens = xt->shapes[0], row_words = xt->shapes[1], meta_words = mt->shapes[1];
    const uint32_t record_words = row_words + meta_words;
    auto *own = control + block * kControlStride;
    auto *index = own + kIndexOffset;
    auto *signal = control + workers * kControlStride;
    if (workers < peers || workers % peers != 0 || tokens > 72 || row_words > kTileWords || meta_words < 16 ||
        alignment == 0 || (transport != 0 && ctx->workSpace == 0)) {
        FailIndexed("indexed_dispatch: invalid geometry or missing URMA workspace");
        return;
    }
    const uint32_t chunks = workers / peers;
    const uint32_t source_rank = block / chunks, chunk = block % chunks;
    DispatchTile tile(1, row_words);
    pto::TASSIGN(tile, 0);
    for (uint32_t round = 0; round < rounds; ++round) {
        // Every source partition is immutable until all local consumers leave the round.
        LocalBarrier(control, workers, block, round + 1);
        const uint32_t epoch = round + 1;
        const uint64_t begin = Stamp();
        uint32_t sent = 0;
        for (uint32_t t = 0; block < peers && t < tokens; ++t) {
            bool selected = false;
            for (uint32_t k = 0; k < kTopK; ++k) {
                const int32_t expert = static_cast<int32_t>(meta[t * meta_words + k]);
                selected |= expert >= 0 && expert < peers * kExperts && expert / kExperts == block;
            }
            if (selected) st_dev(t, index + sent++, 0);
        }
        if (block < peers) {
            st_dev(sent, own + 1, 0);
            PublishReady(own + kRouteReady, epoch);
        }
        const uint64_t routed = Stamp();
        if (block < peers) {
            for (uint32_t peer = 0; peer < peers; ++peer)
                WaitReady(control + peer * kControlStride + kRouteReady, epoch);
            ExchangeBarrier(ctx, signal, rank, peers, peers, block, 1);
            const uint32_t count = ld_dev(CommRemotePtr(ctx, control + rank * kControlStride + 1, block), 0);
            st_dev(count, own + 2, 0);
            PublishReady(own + kCountReady, epoch);
        }
        const uint64_t counted = Stamp();
        auto *target = records + rank * tokens * record_words;
        if (transport != 0 && block < peers && block != rank) {
            SubmitIndexed(
                ctx, block, reinterpret_cast<__gm__ uint8_t *>(x), reinterpret_cast<__gm__ uint8_t *>(meta),
                reinterpret_cast<__gm__ uint8_t *>(target), index, sent, row_words * 4, meta_words * 4, tokens
            );
        } else if (block < peers && block != rank) {
            auto *remote = CommRemotePtr(ctx, target, block);
            for (uint32_t slot = 0; slot < sent; ++slot) {
                const uint32_t token = ld_dev(index + slot, 0);
                Copy(remote + slot * record_words, x + token * row_words, row_words, tile);
                Copy(remote + slot * record_words + row_words, meta + token * meta_words, meta_words, tile);
            }
            if (sent != 0) StoreDone();
        }
        const uint64_t submitted = Stamp();
        if (source_rank == rank) {
            auto *self_control = control + rank * kControlStride;
            WaitReady(self_control + kRouteReady, epoch);
            CopySelf(
                x, meta, target, self_control + kIndexOffset, ld_dev(self_control + 1, 0), row_words, meta_words, chunk,
                chunks
            );
        }
        const uint64_t self_done = Stamp();
        if (transport != 0 && block < peers && block != rank && sent != 0) WaitIndexed(ctx, block);
        if (block < peers && block != rank)
            PublishReady(CommRemotePtr(ctx, control + rank * kControlStride + kDataReady, block), epoch);
        const uint64_t transferred = Stamp();
        if (block < peers) {
            ExchangeBarrier(ctx, signal, rank, peers, peers, block, 2);
            DropBarrierCredits(signal, rank, peers, peers, block);
        }
        if (source_rank != rank) {
            WaitReady(control + source_rank * kControlStride + kCountReady, epoch);
            WaitReady(control + source_rank * kControlStride + kDataReady, epoch);
        }
        const uint64_t ready = Stamp();
        const uint32_t count = ld_dev(control + source_rank * kControlStride + (source_rank == rank ? 1 : 2), 0);
        const uint32_t first = count * chunk / chunks, end = count * (chunk + 1) / chunks;
        const uint32_t received = end - first;
        auto *input = records + (source_rank * tokens + first) * record_words;
        using RouteShape = pto::Shape<1, 1, 1, pto::DYNAMIC, 8>;
        using RouteGlobal = pto::GlobalTensor<uint32_t, RouteShape, DispatchStride>;
        using RouteTile = pto::Tile<pto::TileType::Vec, uint32_t, 72, 8, pto::BLayout::RowMajor, -1, 8>;
        RouteTile routes(received);
        pto::TASSIGN(routes, 8 * kTileWords * sizeof(uint32_t));
        if (received != 0) {
            RouteShape shape(1, 1, 1, received, 8);
            DispatchStride stride(tokens * record_words, tokens * record_words, tokens * record_words, record_words, 1);
            RouteGlobal source(input + row_words, shape, stride);
            pto::TLOAD(routes, source);
            set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        }
        uint32_t histogram[kExperts] = {};
        for (uint32_t slot = 0; slot < received; ++slot) {
            for (uint32_t k = 0; k < kTopK; ++k) {
                const int32_t e = static_cast<int32_t>(routes.GetValue(slot * 8 + k)) - rank * kExperts;
                if (e >= 0 && e < kExperts) ++histogram[e];
            }
        }
        for (uint32_t e = 0; e < kExperts; ++e)
            st_dev(histogram[e], own + 4 + e, 0);
        PublishReady(own + kHistogramReady, epoch);
        uint32_t cursor[kExperts], starts[kExperts], ends[kExperts];
        ComputeOwnPrefixes(control, counts, workers, block, alignment, epoch, cursor, starts, ends);
        const uint64_t prefixed = Stamp();
        for (uint32_t stage = 0; stage < 8; ++stage)
            set_flag(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(stage));
        for (uint32_t slot = 0; slot < received; ++slot) {
            const auto event = static_cast<event_t>(slot % 8);
            wait_flag(PIPE_MTE3, PIPE_MTE2, event);
            pto::TASSIGN(tile, (slot % 8) * kTileWords * sizeof(uint32_t));
            tile.ColMaskInternal = row_words;
            DispatchShape shape(1, 1, 1, 1, row_words);
            DispatchStride stride(row_words, row_words, row_words, row_words, 1);
            DispatchGlobal source(input + slot * record_words, shape, stride);
            pto::TLOAD(tile, source);
            set_flag(PIPE_MTE2, PIPE_MTE3, event);
            wait_flag(PIPE_MTE2, PIPE_MTE3, event);
            for (uint32_t k = 0; k < kTopK; ++k) {
                const int32_t e = static_cast<int32_t>(routes.GetValue(slot * 8 + k)) - rank * kExperts;
                if (e >= 0 && e < kExperts) {
                    const uint32_t row = cursor[e]++;
                    st_dev(row, own + kMapOffset + slot * kExperts + e, 0);
                    DispatchGlobal target(expert_x + row * row_words, shape, stride);
                    pto::TSTORE<pto::TStoreL2Hint::NotAllocClean>(target, tile);
                }
            }
            set_flag(PIPE_MTE3, PIPE_MTE2, event);
        }
        for (uint32_t stage = 0; stage < 8; ++stage)
            wait_flag(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(stage));
        StoreDone();
        set_flag(PIPE_S, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
        cce::async_invoke<ExpandMetadata>(
            cce::dim3{256, 1, 1}, input, own, expert_meta, source_rank, received, row_words, meta_words, rank, first
        );
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        for (uint32_t expert = block; expert < kExperts && alignment > 1; expert += workers) {
            tile.ColMaskInternal = row_words;
            pto::TEXPANDS(tile, uint32_t{0});
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            const uint32_t padding_end =
                starts[expert] + (ends[expert] - starts[expert] + alignment - 1) / alignment * alignment;
            DispatchShape shape(1, 1, 1, 1, row_words);
            DispatchStride stride(row_words, row_words, row_words, row_words, 1);
            for (uint32_t row = ends[expert]; row < padding_end; ++row) {
                DispatchGlobal target(expert_x + row * row_words, shape, stride);
                pto::TSTORE<pto::TStoreL2Hint::NotAllocClean>(target, tile);
            }
            StoreDone();
        }
        const uint64_t finished = Stamp();
        auto *timing = stats + (round * workers + block) * 16;
        timing[0] = begin;
        timing[1] = routed;
        timing[2] = transferred;
        timing[3] = ready;
        timing[4] = prefixed;
        timing[5] = finished;
        timing[6] = sent;
        timing[7] = received;
        timing[8] = transport != 0 && block < peers && block != rank ? (sent + 5) / 6 : 0;
        timing[9] = 2;
        timing[10] = counted;
        timing[11] = submitted;
        timing[12] = self_done;
        pto::comm::urma::DcciCachelines(reinterpret_cast<__gm__ uint8_t *>(timing), 128);
        dsb(DSB_DDR);
    }
    pipe_barrier(PIPE_ALL);
}
