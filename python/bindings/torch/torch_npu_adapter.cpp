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

// Optional lifecycle bridge. No Simpler SDK implementation or submission ABI is linked.
#include <acl/acl.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.h>

#include <cstdint>

#include "pypto/core/error.h"

namespace nb = nanobind;
namespace {
void Require(bool condition, const char* message) {
  if (!condition) throw pypto::ValueError(message);
}
bool FrameworkAlive() { return c10_npu::NpuSysCtrl::GetInstance().GetInitFlag(); }

uintptr_t BorrowContext(int32_t device_id) {
  Require(FrameworkAlive(), "Cannot borrow a context after torch_npu teardown");
  int32_t current = -1;
  aclrtContext context = nullptr;
  Require(aclrtGetDevice(&current) == ACL_SUCCESS && current == device_id,
          "Kernel lifecycle must borrow the current torch NPU device");
  Require(aclrtGetCurrentContext(&context) == ACL_SUCCESS && context != nullptr,
          "Cannot borrow the current torch NPU context");
  return reinterpret_cast<uintptr_t>(context);
}

void BindContext(uintptr_t context) {
  Require(FrameworkAlive() && context != 0, "Kernel lifecycle cannot use a torn-down framework context");
  Require(aclrtSetCurrentContext(reinterpret_cast<aclrtContext>(context)) == ACL_SUCCESS,
          "Cannot bind the borrowed framework context to the kernel lifecycle thread");
}

c10_npu::NPUStream CallStream(int64_t stream_id, int32_t device_id) {
  auto stream = c10_npu::getCurrentNPUStream(device_id);
  Require(stream.id() == stream_id, "Kernel frame is not on the current NPU stream");
  return stream;
}

}  // namespace

NB_MODULE(_torch_npu, m) {
  m.def("framework_alive", &FrameworkAlive);
  m.def("borrow_context", &BorrowContext);
  m.def("bind_context", &BindContext);
  // Intentionally survive interpreter finalization if safe cleanup failed.
  m.def("retain_until_exit", [](nb::object owner) { Py_INCREF(owner.ptr()); });
  m.def("check_call", [](int64_t stream_id, int32_t device_id) {
    auto stream = CallStream(stream_id, device_id);
    aclmdlRICaptureStatus status{};
    aclmdlRI model = nullptr;
    Require(aclmdlRICaptureGetInfo(stream.stream(false), &status, &model) == ACL_SUCCESS,
            "Cannot query kernel initialization capture state");
    Require(status == ACL_MODEL_RI_CAPTURE_STATUS_NONE || status == ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE,
            "Kernel initialization encountered an invalidated graph capture");
    return status == ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE;
  });
  m.def("storage_shape", [](nb::handle object) {
    Require(THPVariable_Check(object.ptr()), "Storage shape requires a torch tensor");
    const at::Tensor& tensor = THPVariable_Unpack(object.ptr());
    Require(tensor.device().type() == c10::DeviceType::PrivateUse1, "Storage shape requires an NPU tensor");
    return at_npu::native::get_npu_storage_sizes(tensor);
  });
}
