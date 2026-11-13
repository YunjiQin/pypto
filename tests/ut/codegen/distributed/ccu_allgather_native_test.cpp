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

// Execute the emitted native definition with CPU primitive stubs. This checks
// address/token wiring, completion ordering and BF16 bits, not HCOMM synthesis.
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

constexpr uint32_t SIMPLER_CCU_MAX_RANKS = 64;
constexpr uint32_t SIMPLER_CCU_TASK_ARGUMENT_COUNT = 2;
constexpr uint32_t SIMPLER_CCU_REGISTRATION_ARGUMENT_COUNT = 1;
constexpr uint32_t SIMPLER_CCU_KERNEL_CAP_THREE_EVENTS = 1;
// The source-only manifest contract from simpler #2986.
struct SimplerCcuModuleManifest {
  uint32_t struct_size, manifest_size, kernel_record_size, kernel_count;
  uint32_t kernel_records_offset, string_table_offset, string_table_size, capability_flags;
  uint8_t sdk_build_id[32], pto_isa_build_id[32];
};
struct SimplerCcuKernelRecord {
  uint32_t struct_size, kernel_id;
  uint64_t parameter_size;
  uint32_t parameter_alignment, capability_flags, entry_symbol_offset, entry_symbol_size;
  uint32_t registration_binding_size, registration_argument_count, task_argument_count;
};
struct SimplerCcuRegistrationBinding {
  uint32_t struct_size, ccu_id, rank_id, rank_count;
  uint64_t channels[64], ready_event_handle, trigger_event_handle, done_event_handle;
};
static_assert(sizeof(SimplerCcuModuleManifest) == 96);
static_assert(sizeof(SimplerCcuKernelRecord) == 48);
static_assert(sizeof(SimplerCcuRegistrationBinding) == 552);
extern const uint8_t pypto_ccu_sdk_identity[32] = {0x12};
extern const uint8_t pypto_ccu_isa_identity[32] = {0x34};

enum CcuResult { CCU_SUCCESS, CCU_E_INTERNAL };
namespace model {
std::array<uint64_t, 4 + 3 * TEST_RANKS> words;
std::array<uint32_t, TEST_RANKS> physical_ranks;
std::array<bool, TEST_RANKS> written;
uint16_t transfer_bits;
uint64_t peers, records, waits;
uint32_t source_rank, next_slot;
int phase, done_count;
bool fail_local_copy;

uint64_t ChannelBit(uint64_t channel) {
  assert(channel >= 1000 && channel < 1064);
  const auto bit = uint64_t{1} << (channel - 1000);
  assert(peers & bit);
  return bit;
}
}  // namespace model

namespace AscendC::ccu {
struct Variable {
  uint64_t value = 0;
  void operator=(uint64_t v) { value = v; }
  Variable operator+(const Variable& rhs) const { return {value + rhs.value}; }
  bool operator!=(uint64_t rhs) const { return value != rhs; }
  bool operator==(uint64_t rhs) const { return value == rhs; }
};
struct Address {
  uint64_t value = 0;
  void operator=(const Variable& rhs) { value = rhs.value; }
  void operator+=(const Variable& rhs) { value += rhs.value; }
};
struct LocalAddr {
  Address addr;
  Variable token;
};
struct RemoteAddr {
  Address addr;
  Variable token;
};
struct Event {
  uint64_t handle;
  explicit Event(uint64_t h = 100) : handle(h) {}
};
inline CcuResult LoadArg(Variable& dst, uint32_t index) {
  assert(index < SIMPLER_CCU_TASK_ARGUMENT_COUNT);
  dst = index == 0 ? reinterpret_cast<uint64_t>(model::words.data()) : 999;
  return CCU_SUCCESS;
}
inline CcuResult Load(const Variable& address, Variable& dst) {
  dst = *reinterpret_cast<const uint64_t*>(address.value);
  return CCU_SUCCESS;
}
inline CcuResult EventWait(const Event& event, uint16_t mask) {
  if (event.handle == 11) {
    assert(mask == 1 && model::phase == 0);
    model::phase = 1;
  } else {
    assert(event.handle == 100 && model::phase == 3);
    assert(model::next_slot % 16 == 0 || model::next_slot == TEST_RANKS);
    const auto batch = (model::next_slot - 1) % 16 + 1;
    assert(mask == (1u << batch) - 1 && model::transfer_bits == mask);
    model::transfer_bits = 0;
  }
  return CCU_SUCCESS;
}
inline CcuResult EventRecord(const Event& event, uint16_t mask) {
  assert(event.handle == 12 && mask == 1 && model::phase == 5);
  assert(model::waits == model::peers);
  ++model::done_count;
  model::phase = 0;
  model::next_slot = 0;
  model::records = model::waits = 0;
  return CCU_SUCCESS;
}
inline CcuResult NotifyRecord(uint64_t channel, uint32_t index, uint16_t mask) {
  assert(index == 7);
  if (mask == 1) {
    assert(model::phase == 1);
  } else {
    assert(mask == 2 && model::next_slot == TEST_RANKS && model::transfer_bits == 0);
    if (model::phase == 3) model::records = model::waits = 0;
    assert(model::phase == 3 || model::phase == 4);
    model::phase = 4;
  }
  const auto bit = model::ChannelBit(channel);
  assert(!(model::records & bit));
  model::records |= bit;
  return CCU_SUCCESS;
}
inline CcuResult NotifyWait(uint64_t channel, uint32_t index, uint16_t mask) {
  assert(index == 7 && model::records == model::peers);
  assert((mask == 1 && (model::phase == 1 || model::phase == 2)) ||
         (mask == 2 && (model::phase == 4 || model::phase == 5)));
  const auto bit = model::ChannelBit(channel);
  assert(!(model::waits & bit));
  model::waits |= bit;
  model::phase = mask == 1 ? 2 : 5;
  return CCU_SUCCESS;
}
inline void Copy(const Address& dst, const LocalAddr& src, const Variable& bytes, const Event& event,
                 uint16_t mask) {
  assert(model::phase == 2 || model::phase == 3);
  assert(model::waits == model::peers && src.token.value == 101);
  assert(event.handle == 100 && mask == (1u << (model::next_slot % 16)));
  assert(!(model::transfer_bits & mask));
  std::memcpy(reinterpret_cast<void*>(dst.value), reinterpret_cast<const void*>(src.addr.value), bytes.value);
  model::written[model::next_slot++] = true;
  model::transfer_bits |= mask;
  model::phase = 3;
}
inline CcuResult Write(uint64_t channel, const RemoteAddr& dst, const LocalAddr& src, const Variable& bytes,
                       const Event& event, uint16_t mask) {
  assert(model::next_slot != model::source_rank);
  assert(channel == 1000 + model::physical_ranks[model::next_slot]);
  assert(dst.token.value == 200 + model::next_slot);
  Copy(dst.addr, src, bytes, event, mask);
  return CCU_SUCCESS;
}
inline CcuResult LocalCopy(const LocalAddr& dst, const LocalAddr& src, const Variable& bytes,
                           const Event& event, uint16_t mask) {
  assert(model::next_slot == model::source_rank && dst.token.value == 200 + model::source_rank);
  if (model::fail_local_copy) return CCU_E_INTERNAL;
  Copy(dst.addr, src, bytes, event, mask);
  return CCU_SUCCESS;
}
}  // namespace AscendC::ccu
#define CCU_WHILE(condition) while (condition)
#define CCU_IF(condition) if (condition)
#include "generated_module.inc"

int main() {
  const auto* manifest = simpler_ccu_module_manifest();
  const auto* base = reinterpret_cast<const uint8_t*>(manifest);
  const auto* kernel =
      reinterpret_cast<const SimplerCcuKernelRecord*>(base + manifest->kernel_records_offset);
  assert(manifest->struct_size == 96 && manifest->kernel_count == 1 && manifest->kernel_record_size == 48);
  assert(manifest->sdk_build_id[0] == 0x12 && manifest->pto_isa_build_id[0] == 0x34);
  assert(kernel->parameter_size == (4 + 3 * TEST_RANKS) * 8 && kernel->parameter_alignment == 64);
  assert(kernel->registration_argument_count == 1 && kernel->task_argument_count == 2);
  assert(kernel->capability_flags == SIMPLER_CCU_KERNEL_CAP_THREE_EVENTS);
  assert(std::memcmp(base + kernel->entry_symbol_offset, "pypto_ccu_allgather_test",
                     kernel->entry_symbol_size) == 0);
  for (uint32_t rank = 0; rank < TEST_RANKS; ++rank) {
    for (uint64_t count : {7U, 64U}) {
      for (bool failure : {false, true}) {
        std::vector<uint16_t> input(count + 2);
        std::vector<std::vector<uint16_t>> outputs(TEST_RANKS,
                                                   std::vector<uint16_t>(TEST_RANKS * count + 6, 0xdead));
        for (uint64_t i = 0; i < count; ++i) input[2 + i] = static_cast<uint16_t>(0x7f80 + i);
        model::words[0] = reinterpret_cast<uint64_t>(input.data() + 2);
        model::words[1] = 101;
        model::words[2] = count * 2;
        model::words[3] = rank * count * 2;
        model::phase = model::done_count = model::transfer_bits = 0;
        model::next_slot = 0;
        model::source_rank = rank;
        model::peers = model::records = model::waits = 0;
        model::written.fill(false);
        model::fail_local_copy = failure;
        SimplerCcuRegistrationBinding binding{};
        binding.rank_count = 64;
        binding.ccu_id = 7;
        binding.trigger_event_handle = 11;
        binding.done_event_handle = 12;
        for (uint32_t slot = 0; slot < TEST_RANKS; ++slot) {
          // Permuted, noncontiguous subsets catch domain/physical rank confusion.
          const uint32_t physical = (63 - 5 * slot) % 64;
          model::physical_ranks[slot] = physical;
          model::words[4 + 3 * slot] = reinterpret_cast<uint64_t>(outputs[slot].data() + 3);
          model::words[5 + 3 * slot] = 200 + slot;
          model::words[6 + 3 * slot] = physical;
          binding.channels[physical] = 1000 + physical;
          if (slot == rank)
            binding.rank_id = physical;
          else
            model::peers |= uint64_t{1} << physical;
        }
        const auto result = pypto_ccu_allgather_test(&binding);
        assert(result == (failure ? CCU_E_INTERNAL : CCU_SUCCESS));
        assert(model::done_count == (failure ? 0 : 2));
        // Every destination gets exactly this source rank's row. Nonzero base
        // offsets and sentinels catch double slot offsets and neighbor corruption.
        for (uint32_t slot = 0; slot < TEST_RANKS; ++slot) {
          assert(model::written[slot] == (!failure || slot < rank));
          for (uint64_t i = 0; i < outputs[slot].size(); ++i) {
            const bool copied = model::written[slot] && i >= 3 + rank * count && i < 3 + (rank + 1) * count;
            const auto expected = copied ? input[2 + i - 3 - rank * count] : uint16_t{0xdead};
            assert(outputs[slot][i] == expected);
          }
        }
      }
    }
  }
}
