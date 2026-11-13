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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_CCU_COLLECTIVE_UTILS_H_
#define PYPTO_IR_TRANSFORMS_UTILS_CCU_COLLECTIVE_UTILS_H_

#include <cstddef>
#include <string>

#include "pypto/ir/function.h"

namespace pypto::ir {
inline constexpr const char* kCcuBroadcastTemplate = ":pypto.runtime.builtins.ccu.broadcast";

inline constexpr const char* kCcuAllGatherTemplate = ":pypto.runtime.builtins.ccu.allgather";

inline bool IsCcuAllGatherBuiltin(const FunctionPtr& func) {
  return func && func->GetAttr<std::string>(kAttrBuiltinTemplateDir) == kCcuAllGatherTemplate;
}

inline bool IsCcuBroadcastBuiltin(const FunctionPtr& func) {
  return func && func->GetAttr<std::string>(kAttrBuiltinTemplateDir) == kCcuBroadcastTemplate;
}

// CCU signal/order are reference handles with conservative task effects, not
// returned SSA data values. The payload target still obeys ordinary rebind rules.
inline bool IsCcuControlParameter(const FunctionPtr& func, size_t index) {
  return (IsCcuBroadcastBuiltin(func) && (index == 1 || index == 2)) ||
         (IsCcuAllGatherBuiltin(func) && (index == 2 || index == 3));
}
}  // namespace pypto::ir

#endif  // PYPTO_IR_TRANSFORMS_UTILS_CCU_COLLECTIVE_UTILS_H_
