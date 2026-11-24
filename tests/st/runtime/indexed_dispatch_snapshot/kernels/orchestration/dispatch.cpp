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
#include "orchestration_api.h"
#include "platform_comm/comm_context.h"
extern "C" {
__attribute__((visibility("default"))) OrchestrationConfig indexed_dispatch_config(const ChipTaskArgs &args) {
    (void)args;
    return OrchestrationConfig{.expected_arg_count = 12};
}
__attribute__((visibility("default"))) void indexed_dispatch(const ChipTaskArgs &args) {
    CoreTaskArgs task;
    task.add_input(args.tensor(0).ref());
    task.add_input(args.tensor(1).ref());
    task.add_inout(args.tensor(2).ref());
    task.add_inout(args.tensor(3).ref());
    for (int i = 4; i < 8; ++i)
        task.add_output(args.tensor(i).ref());
    for (int i = 0; i < 4; ++i)
        task.add_scalar(args.scalar<uint64_t>(i));
    const auto *ctx = reinterpret_cast<const CommContext *>(args.scalar<uint64_t>(0));
    task.launch_spec.set_block_num((32 / ctx->rankNum) * ctx->rankNum);
    rt_submit_aiv_task(0, task);
}
}
