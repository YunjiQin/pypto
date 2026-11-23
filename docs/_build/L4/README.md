# PyPTO L4（跨机）支持方案

参考 simpler 的 `compute_then_tload_mixed_l3` 层级模型与
`examples/workers/l4/global_tload_mpirun_l3` 启动路径，为 PyPTO 增加
CLUSTER_0（L4）层级支持，使一份 PyPTO 程序可以跨多台机器、多张卡运行，且与现有
L3 分布式方案最大程度兼容。

> 通信域与 MPI 启动接口按 Simpler `d2b37652d` 核对。PyPTO `ca08d06bd` 锁定的 Simpler `b7cb9e73` 已提供这两组接口。
> 本文的 L4 生成代码和端到端流程是 PyPTO 的目标设计。见 [Simpler 通信域契约](https://github.com/hw-native-sys/simpler/blob/d2b37652d/docs/comm-domain.md)。
> `worker=`、`num_workers()`、`@pl.jit.pod`、`ClusterConfig` 均为本方案新增能力。
>
> **范围约束：L4 不直接调用 collective。** L4 创建全局域、组织控制流并下发 L3；
> collective 在 L3/L2 表达并由设备任务执行。L3 的本地域与上层传入域可以并存。

## 文档索引

| 文档 | 内容 |
| ---- | ---- |
| [00-design.md](00-design.md) | 设计：层级模型、`worker=` 统一词汇、通信域模型、DSL 形态、硬约束 |
| [01-frontend.md](01-frontend.md) | 前端实现：层级放行、`worker=`、`num_workers()`、`@pl.jit.pod` |
| [02-passes.md](02-passes.md) | Pass 实现：层级泛化、按调用点解析窗口实参、ctx 查询层级放行与 verifier 同步 |
| [03-codegen.md](03-codegen.md) | Codegen 实现：`LevelHandler` 抽象与 `DistributedCodegen` 改造 |
| [04-runtime.md](04-runtime.md) | 运行时：`ClusterConfig`、`cluster_runner`、MPI 启动、L3 bundle 分发与直接注册下发 |
| [05-plan.md](05-plan.md) | 分阶段实施、测试策略、风险与待决问题 |
| [06-example-lowering.md](06-example-lowering.md) | 例子的目标降级链路：关键 IR、各层产物示意、参数布局与运行时时序 |

## 一页纸摘要

**目标**：一份程序，四层函数，跨机运行。

```text
cluster_orch  @pl.jit.pod     L4  跨机编排（新增装饰器）
node_orch     @pl.jit.host    L3  节点内编排
chip_orch     @pl.jit         L2  单卡编排
kernel        @pl.jit.incore  L0  核内计算
```

**核心结论**：

1. **IR 零新节点**。`ir::Level::CLUSTER_0` 已存在（`LevelToLinquLevel` = 4 =
   simpler `Worker(level=4)`），`ir.compile` 的分布式判定 `linqu_level >= 3` 已覆盖
   L4。复用 `CommDomainScopeStmt` / `WindowBuffer` / `DistributedTensorType`。

2. **层级词汇统一**。下发 kwarg 在每一层都是 `worker=`（下一级的哪个子节点），
   循环上界都是 `pld.num_workers()`。不区分 cluster/host 拼写——层级已由 caller 的
   `level_` 唯一确定。

3. **域跨层传递复用 CommCtx 形参**。`CommDomainScopeStmt` 只出现在建域的那一层；
   下层通过 `MaterializeDistTensorCtx`（pass 48）追加的 `CommCtx` 形参拿到域——与
   L3 传给 L2 使用同一传播机制。需将 pass 48 的运行时 ctx 查询及其 verifier
   从 HOST 放宽到 HOST/CLUSTER_0。

4. **每一层都可以自建域**。`node_orch` 可以既接收 L4 建的全局域，又自建只跨本节点
   若干卡的本地域——两级 collective 分解正是分层的主要动机。

5. **不新增 TaskArgs**。L4→L3 与 L3→L2 用同一个 `simpler.task_interface.TaskArgs`，
   前端 L3 形参仍为 `DistributedTensor`，IR 保留张量及配对的 `CommCtx`。L4 每节点提交一次：窗口占 0 个 tensor 槽，每个窗口形参在业务参数顺序中发射一个 `buffer_index` 标量，配对 ctx 在 IR 后缀位置发射完整 `CommDomainRef` 三槽。L3 按本地 chip 构造窗口 Tensor，向 L2 下传该 Tensor 和 1 个 `device_ctx` 标量。

6. **Codegen 差异收进 `LevelHandler`**。共享 IR visitor 与下发骨架，handler
   决定成员表达式、ctx 编解码、完整张量打包语句与入口签名。L3→L2 保留
   `add_chip_tensor_arg`，L4→L3 使用节点路由。建域能力与 collective 能力分开判断。

7. **直接注册和下发生成的 orch**。L4/L3 编排都生成普通 Python；分发 L3 模块后，
   通过 `register(RemoteCallable(generated_target), workers=[node])` 注册生成的 L3
   函数。L4 orch 直接 `orch.submit_next_level(handle, args, cfg, worker=node)`，
   无统一 trampoline 或按 artifact 标识二次分派。

8. **L4 统一编译、自动分发**。通过现有 MPI group 启动 L3，无需远端预启动 daemon。
   L2 CHIP 二进制复用已有 manifest 分发；生成的 L3 Python 与 metadata 打成 bundle，
   通过 `MpiL3GroupSpec.startup_artifacts` 分发，`startup_hooks_by_rank` 在 L2 fork 前绑定节点状态。远端只预装兼容运行环境，
   不手工部署本次生成物、不运行 PyPTO/PTOAS 编译。详见 [04-runtime.md](04-runtime.md) § 3、§ 7。

**不做的事**：L4 直接调用 collective；L4 直发 L2 的语法糖；`copy_to/from_global_domain` 前端接口（填/取窗口
数据一律经 L2 task，与 L3 一致）；program 内多相位切分（program 边界 = 一次
`worker.run`）；下层 IR 里的"继承 scope"；新的 TaskArgs 类型。
启动阶段校验产物版本和摘要，运行阶段直接使用已注册 callable handle。
