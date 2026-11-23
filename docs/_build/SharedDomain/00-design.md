# 设计与边界

## 1. 要解决的问题

现有持久化路径按 `(program_id, generated_domain_name)` 隔离域。同一个 program 的多次 submit 可以复用；Prefill 与 Decode 分别编译后，即使初始化来源相同，也会创建两份 Host 表。

```text
Before:
  Prefill -> p0:comm_d0 -> allocation A
  Decode  -> p1:comm_d0 -> allocation B

Target:
  worker owns domain D -> buffers A, B
  Prefill(table_a=A, table_b=B)
  Decode (table_a=A, table_b=B)
```

不能删除 program 前缀来合并：生成域名是编译器内部名字，独立程序的同名域可能有不同成员、布局和用途。不能按相同 shape、初始化 key 或文件对象自动共享。

采用显式资源共享：program 外创建域，program 声明输入窗口，调用者传同一个资源对象。域身份跨 submit 存活，编译程序只描述一次执行如何消费它。

实施顺序固定为 SharedDomain → L4：先让 #3497 在独立 L3 上完成并验收，再由 #3531 复用其公共编译/runtime 基础。对齐 L4 的 ABI 设计不意味着等待 L4 代码落地。

## 2. 基线与证据范围

本地检查基线：PyPTO HEAD `fe651f2961a3f63c0eb2dcc9b0893c6ecbe13bdc`，runtime checkout `6e383fc57c12a8bd4dce045289b8b4f5d55a15f1`，并存在用户未提交修改。以下位置是当前工作区的实现入口，不表示它们与最新集成分支完全相同。

| 位置 | 当前约束 | 目标改动 |
| --- | --- | --- |
| `python/pypto/runtime/distributed_runner.py::_submit_persistent` | `program_domains` + `pN:generated_name` | 保留内部域；外部域另有 worker 资源 registry |
| 同文件 `_submit_compiled` | 实参数量等于 `param_infos` 数量 | 只校验业务参数；从窗口实参自动补 ctx |
| `python/pypto/ir/compiled_program.py::_extract_func_param_infos` | shaped/scalar 类型 | distributed 入口分离用户 metadata 和 ctx ABI |
| `src/codegen/distributed/distributed_codegen.cpp::ScopeForWindowBuffer` | 依赖当前打开的 owning scope | 传入形参从入口绑定获取域与槽 |
| 同文件 `ResolveCommCtxArg` | 本层窗口查设备 ctx；显式变量转发 | 在 HOST→CHIP 边把逻辑域绑定转成目标设备 ctx |
| `materialize_dist_tensor_ctx_pass.cpp` | 已向各函数追加 ctx | 复用配对，修正 HOST 参数查询与 verifier |
| `distributed_compiled_program.py::_META_SCHEMA` | 本地为 3 | 为新的分布式入口契约提供明确版本迁移 |

L4 文档按更近的 Simpler `d2b37652d` 契约设计，提及 PyPTO `ca08d06bd` 锁定 `b7cb9e73` 已具有统一域接口。实施前须在目标分支核对这些依赖，不能直接在旧 checkout 假定已有 `CommDomainRef/get_domain(ref)`。

本方案要求的 Simpler 能力：

- `allocate_domain(members=..., retain_after_run=True, ...)`；
- 完整 `CommDomainRef` 和当前 run 的 `get_domain(ref)`；
- 读出的 ordered buffer definitions、成员本地 context 和 Buffer tensor；
- 受控初始化、借用授权、release/run/descendant fence；
- descriptor-backed POSIX shared Buffer，可在 init 后让子进程按身份映射。

这里只据源码/契约设计，未声称复现 #3497 或测得节省内存。

## 3. 三种身份

| 身份 | 示例 | 职责 |
| --- | --- | --- |
| 形参身份 | `prefill` 的参数 1 | 类型、方向、ctx 配对、ABI 位置 |
| 域身份 | `(owner_instance_id, domain_id, generation)` | 分配、注册、成员、生命周期 |
| buffer 身份 | 该域 ordered buffers 中的 index 1 | 选择实际命名窗口 |

`table_a`、`weights_a` 和 `comm_d0` 不需要同名。用户传参建立第一种到后两种的绑定。
同一个编译产物可以在不同调用接收域 D 的槽 1、域 E 的槽 0，只要满足它的类型和执行要求。

## 4. 层级与范围

| 调用边 | 窗口实参 | 配对 ctx |
| --- | --- | --- |
| Python submit→L3 | DomainTensor 解析出的 buffer_index | 完整 ref；本地 frame 可直接保存 ref 对象 |
| L4→L3 | 一个 buffer_index scalar | 三个 uint64 scalar |
| L3→L2 | 当前 chip 的 Buffer.tensor | 一个 device_ctx scalar |
| L2→kernel | 现有 tensor ABI | 原样转发 device_ctx |

L3 worker 创建域时，members 为逻辑 chip worker 下标，例如 `[2, 0]`。其中 worker 2 的 domain rank 为 0，worker 0 的 domain rank 为 1。实际物理卡由 DistributedConfig 映射。
L4 创建域时成员为 `(node_worker_id, local_chip_worker_id)`；同一 L3 接口消费该域的本地可访问上下文。ref 不是授权，不允许未获 grant 的节点仅凭 ref 导入域。

本地 `world_size()` 仍是本地 worker 数，不是外部域大小。派发覆盖必须包含于域的本地成员。某张卡同时消费两个域时，它必须属于两者。

## 5. 两条资源路径并存

```text
Internal allocation:
  alloc_window_buffer -> inferred CommDomainScopeStmt
                      -> local handle + static slot expression

External parameter:
  DistributedTensor + paired CommCtx
                      -> incoming buffer_index + ref
                      -> borrowed/owned handle + dynamic slot expression

Common dispatch:
  (domain_expr, slot_expr) -> domain[w].buffers[slot].tensor(...)
                          -> domain[w].device_ctx
```

只在分配所在函数创建 scope。外部形参不加入内部 scope，不要求其 WindowBuffer 在当前独立 program 内有 owner。替代检查是“有合法的形参绑定来源与配对 ctx”。
复用现有 IR 类型，不新增 inherited scope 或 runtime-pointer IR 节点。

## 6. 与 #3531 的精确对齐

DistributedTensor 自身贡献 buffer_index，ctx 后缀贡献 ref。普通 scalar 和窗口下标按业务形参相对顺序排列，ctx 后缀在最后。先发全部 tensor，再发 scalar。

L4 文档的 call-site window resolution 已支持不同域/槽传给同一 L3。SharedDomain 提供它的本地调用方：外部域创建 + Python submit 自动填参。
两者共用参数计划与 binder，不能各维护一套按名字猜测的配对。

需要在实现阶段同步的 L4 补充：

- 独立 L3 根入口可以有外部 DistributedTensor 形参；
- 内部 provider 的跨 program 共享 key 不再是 #3497 的必经路径；仍可保留用于其他内部域复用；
- fork 前准备适用于 fork-inherited 存储；新增外部域 token 若使用 POSIX descriptor 路径，应明确另一条受控准备/映射路线，不能把未继承的 torch 指针交给 child。

本轮不修改 L4 文档；上述为后续实现的同步项。

## 7. 首期边界

- 实现同一 L3 worker 中跨 program 的外部 Host 域共享；Device-source 域复用相同框架，按后端能力测试。
- 共享表可以读；Out/InOut 窗口必须显式传入已有分配，不能自动创建不知成员的外部域。
- 初版跨入口窗口为静态形状、连续 ND、从 named buffer 起始处开始的 view。没有额外 offset/stride wire 字段。
- 可变批量普通 tensor/scalar 沿用现有 ABI。仅由外部窗口提供的动态维度不能在缺少传输契约时隐式支持；见 ABI 文档。
- program 内部受支持的 slice/reshape 仍可基于已绑定窗口降低，但必须保留 base/ctx 和实际视图范围。
- 不增加 L4 Host backend 能力，不改变 URMA 支持矩阵，不解决多卡共享同一物理整表的 #3482。
- 不承诺新的跨 run 并行调度或跨节点同步；保持运行时 fence 契约并验收。

## 8. 未采用的替代方案

| 方案 | 未采用原因 |
| --- | --- |
| 去掉 `p0/p1` 域名前缀 | 相同内部编号不代表相同资源 |
| 用 `init="engram"` 作为共享身份 | 初始化源不是域，也不是唯一 buffer 身份 |
| 给 submit 额外传 `domains={"engram": ...}` | 缺少隐式域与业务参数的可靠对应；参数绑定已经能表达 |
| 把 buffer_index 塞进 ctx 四元组 | 混淆域与窗口身份，偏离 #3531 的三字段 ctx |
| ctx 后另加 WindowDescriptor 尾部 | 与更新的 L4 参数规则重复 |
| 把调用者 WindowBuffer 固化到 callee | 无法独立编译和复用同一 L3 |
| 只传每张卡裸地址 | 缺失注册、域成员、授权、生命周期及远端寻址信息 |
