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

// CPU contract test for the generated AIV wrapper, not a hardware CCU simulator.
#include <array>
#include <cassert>
#include <cstdint>
#include <vector>

#define __gm__
#define __aicore__
struct Tensor {
  struct {
    uint64_t addr;
  } buffer;
  uint64_t start_offset;
  uint64_t shapes[2];
};
struct CommContext {
  uint32_t rankId;
  uint32_t rankNum;
  uint64_t windowsIn[TEST_RANKS];
};
struct AsyncCtx {
  int64_t* original;
};
inline AsyncCtx get_async_ctx(int64_t* args) { return {args}; }
constexpr int SIMPLER_ERROR_ASYNC_COMPLETION_INVALID = 1;
namespace pto2::detail {
inline void defer_error(AsyncCtx&, int) { assert(false); }
}  // namespace pto2::detail
namespace simpler::ccu {
struct CcuCallArg {
  enum class Kind : uint32_t { Scalar, Window, PeerRank };
  Kind kind;
  uint32_t rank;
  uint64_t value;
  uint64_t size;
};
}  // namespace simpler::ccu
using Arg = simpler::ccu::CcuCallArg;
std::array<Arg, 3 + 2 * TEST_RANKS> captured;
int64_t* original_args;
inline bool submit_ccu_mission(AsyncCtx& ctx, const CommContext* comm, uint32_t id,
                               const Arg (&args)[3 + 2 * TEST_RANKS]) {
  assert(ctx.original == original_args);
  assert(comm == reinterpret_cast<const CommContext*>(original_args[4]));
  assert(id == 7);
  for (std::size_t i = 0; i < captured.size(); ++i) captured[i] = args[i];
  return true;
}

#include "generated_kernel.inc"

int main() {
  for (uint32_t rank = 0; rank < TEST_RANKS; ++rank) {
    for (uint64_t count : {7U, 64U}) {
      std::vector<uint16_t> window(512 + TEST_RANKS * count);
      const auto base = reinterpret_cast<uint64_t>(window.data());
      // Nonzero buffer displacement and start_offset catch byte/element confusion.
      Tensor input{{base + 64}, 3, {1, count}};
      Tensor target{{base + 512}, 5, {TEST_RANKS, count}};
      auto* src = reinterpret_cast<uint16_t*>(input.buffer.addr) + input.start_offset;
      auto* dst = reinterpret_cast<uint16_t*>(target.buffer.addr) + target.start_offset;
      for (uint64_t i = 0; i < count; ++i) src[i] = static_cast<uint16_t>(0x7f80 + i);
      for (uint64_t i = 0; i < TEST_RANKS * count; ++i) dst[i] = 0xdead;
      CommContext comm{};
      comm.rankId = rank;
      comm.rankNum = TEST_RANKS;
      for (auto& address : comm.windowsIn) address = base;
      int64_t args[]{reinterpret_cast<int64_t>(&input), reinterpret_cast<int64_t>(&target), 0, 0,
                     reinterpret_cast<int64_t>(&comm)};
      original_args = args;
      kernel_entry(args);
      for (uint64_t i = 0; i < TEST_RANKS * count; ++i) assert(dst[i] == 0xdead);
      assert(captured[0].kind == Arg::Kind::Window && captured[0].rank == rank);
      assert(captured[0].value == 70 && captured[0].size == 2 * count);
      assert(captured[1].kind == Arg::Kind::Scalar && captured[1].value == 2 * count);
      assert(captured[2].kind == Arg::Kind::Scalar && captured[2].value == rank * 2 * count);
      for (uint32_t peer = 0; peer < TEST_RANKS; ++peer) {
        const auto& window_arg = captured[3 + 2 * peer];
        const auto& rank_arg = captured[4 + 2 * peer];
        assert(window_arg.kind == Arg::Kind::Window && window_arg.rank == peer);
        assert(window_arg.value == 522 && window_arg.size == TEST_RANKS * 2 * count);
        assert(rank_arg.kind == Arg::Kind::PeerRank && rank_arg.rank == peer);
      }
    }
  }
}
