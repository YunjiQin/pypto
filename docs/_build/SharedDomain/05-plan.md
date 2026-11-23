# 实施顺序、文件清单与验收

本文件描述后续代码实现。此次只编写设计文档，没有运行编译、pytest 或 NPU 作业。

**交付顺序：SharedDomain 先独立落地，L4 后接入。** P0–P4 构成 #3497 的完整本地 L3 实现与验收；后续 L4 工作属于 #3531，不作为 #3497 合入、发布或设备验收的前置条件。

```text
P0 runtime contract
  -> P1 entry ABI
  -> P2 standalone L3 compiler/codegen
  -> P3 external domain + submit
  -> P4 local L3 device acceptance / deliver #3497
  -> later L4 integration / #3531
```

P0 只依赖 Simpler 已有的本地域能力；不等待 PyPTO 的 `@pl.jit.pod`、ClusterConfig、LevelHandler 重构、MPI bundle/startup 或远端入口实现。

## 1. P0：确认 runtime 基线与冻结 ABI

前置依赖：目标分支使用支持统一域 ref/lookup/retention 的 Simpler；当前旧 checkout 不能直接视作满足。

交付：

- 核对 owner grant、`get_domain(ref)` 当前 run 登记、失败撤销、release fence。
- 核对 buffer ordered specs、初始化回调和后 init descriptor-backed Buffer。
- 验证相邻 run 的 FIFO/完成顺序，确定异步共享域提交是否可直接复用；否则补显式串行准入策略。
- 固定 #3531 的 index 一槽、ctx 三槽与 tensor-before-scalar 规则。
- 固定静态 ND、buffer 起点 view 的首期范围及诊断。

验收：形成能执行的低层小例子/测试，而不是仅凭 hasattr 放行。版本不满足则升级正式依赖并重新构建，不能复制其他 checkout 的二进制或通过 private retention hook 绕过。

## 2. P1：ABI builder 与 metadata

新增 C++ layout builder，Python bindings/pyi，同步版本化 loader。保存用户参数与隐藏参数的区别。

| 文件 | 操作 |
| --- | --- |
| `include/pypto/codegen/distributed/entry_abi.h` | 新增内部布局结构和 builder 声明 |
| `src/codegen/distributed/entry_abi.cpp` | 实现配对、pool 分配及校验 |
| 对应 CMake 源文件列表 | 注册新 `.cpp` |
| `python/bindings/modules/codegen.cpp` | 导出 builder 与只读 metadata |
| `python/pypto/pypto_core/codegen.pyi` | 同步类型签名 |
| `python/pypto/ir/distributed_entry_abi.py` | 新增 metadata 解析、序列化、验证叶子模块 |
| `python/pypto/ir/distributed_compiled_program.py` | public metadata、l3_entry、版本与冷加载 |
| `python/pypto/ir/compiled_program.py` | 若复用参数提取 helper，增加显式索引路径；保持 L2 默认行为 |

验收：两窗口夹用户 scalar 的精确布局、同域不压缩 ABI、错误配对拒绝、schema 3 普通 L3 兼容、无 IR 冷加载。

## 3. P2：JIT、pass 与 codegen

依赖 P1 的共享布局定义。

| 文件 | 操作 |
| --- | --- |
| `python/pypto/jit/decorator.py` | DomainTensor metadata adapter、注解与实参校验 |
| `python/pypto/jit/specializer.py` | 保留 DistributedTensor 与静态 window view 约束 |
| `src/ir/transforms/materialize_comm_domain_scopes_pass.cpp` | 只为内部 alloc 建域，外部来源合法性 |
| `src/ir/transforms/materialize_dist_tensor_ctx_pass.cpp` | HOST 形参 ctx 优先转发、Call/Submit、verifier |
| `include/pypto/codegen/distributed/distributed_codegen.h` | 参数 window/domain binding 状态与 helper |
| `src/codegen/distributed/distributed_codegen.cpp` | common binder、slot 动态解析、per-chip ctx、ordering 去重 |
| `python/pypto/backend/pto_backend.py` | distributed 生成与 ABI 产物接线 |
| `src/codegen/distributed/` 中 builtin 实际注册文件 | 审核内部 scope 假设，支持或明确拒绝外部窗口 |

验收：纯外部参数 HOST、内部 scratch 与外部表混用、不同域/槽调用同一 L3、原有内部域 golden 与 host-control 行为回归。
不在这里引入 #3531 的全部 MPI 功能；共享的 helper 应可由后续 L4 handler 复用。

## 4. P3：外部 owner 与 submit

依赖 P0 的 runtime 能力与 P1/P2 的 ABI。

| 文件 | 操作 |
| --- | --- |
| `python/pypto/runtime/comm_domain.py` | 新增 DomainBufferSpec、CommDomain、DomainTensor、逐 buffer 64 字节向上对齐的纯布局 helper、事务与 lease |
| `python/pypto/runtime/distributed_args.py` | 新增 entry binder/ref codec/参数验证 |
| `python/pypto/runtime/distributed_runner.py` | API、frame 扩展、自动补 ctx、登记、release/close |
| `python/pypto/runtime/__init__.py` | 导出公共 Python 符号 |
| 目标分支 `python/pypto/runtime/window_sources.py` | 复用/抽出初始化逻辑，避免第二套文件复制实现 |
| `python/pypto/runtime/debug/` 参数重放消费者 | 明确外部资源必须由调用者重建/注入，不能保存裸指针 |

验收：一次分配初始化、跨 program reuse、不同 worker 拒绝、资源关闭 race、native accepted 后异常、reset 隔离、token 身份一致。

新调试/重放工具不能声称仅靠 artifact 即能恢复外部表内容。记录资源规格，明确需要再提供 sources，禁止将设备地址作为重放输入。

## 5. P4：独立 L3 设备验收与交付

依赖 P0–P3，不依赖任何 L4 实现：

- 运行两个独立编译的 L3 program，共用一个外部域及多个 buffer。
- 验证非 64 字节倍数的逻辑 buffer 自动补齐物理布局，源文件和输出保持逻辑大小。
- 验证 Host 本地读取、支持平台上的远端 GET、重复提交、初始化一次、成员重排和释放。
- 完成 cold reload、连续异步提交、错误清理与内部 persistent 窗口回归。
- 更新正式中英文文档并交付 #3497；编译和设备阶段分开，详见第 7 节。

### 后续 L4 接入（#3531，不阻塞本次交付）

- L4 使用已经落地的 ABI builder、ref codec 和 common window binder，不另建同义实现。
- L4 从实际窗口发 index；本地 submit 从 DomainTensor 发 index，L3 的逻辑接口和编排体相同。
- 节点启动/域准备层提供 ordering 状态，不把 parent owner 对象跨机序列化。
- 单独验收 L4 owner/child 双侧 run 登记、grant、跨节点通信和释放。
- 同步 L4 文档的共享约定和 token backing 条款，保持 index/ctx ABI 一致。

L4 backend、平台和 Host memory 能力与接口复用是两个验收维度。不能以本地 A5 Host 通过推导 L4 Host 域也支持。

## 6. 测试矩阵

| 层 | 文件（新建项标明） | 关键断言 |
| --- | --- | --- |
| ABI | `tests/ut/codegen/distributed/test_entry_abi.py`（新） | exact offsets、uint64 高位、非法配对、ctx 后缀 |
| Pass | `tests/ut/ir/transforms/test_materialize_dist_tensor_ctx.py` | HOST 参数查询消除、别名、返回 lineage、Call/Submit |
| 域 pass | `tests/ut/ir/transforms/test_materialize_comm_domain_scopes.py` | 外部参数无 scope、内部 scope 保留 |
| Verifier | `tests/ut/ir/transforms/test_verify_dist_tensor_ctx_materialized.py` | 无 owner 且无参数来源拒绝、错误 ctx 拒绝 |
| Codegen | `tests/ut/codegen/distributed/test_host_orch_distributed.py` | 运行生成入口，捕获实际 TaskArgs，不仅字符串断言 |
| JIT | `tests/ut/jit/test_decorator.py`、`test_specializer.py` | annotation-only 编译、sample adapter、cache 不含 ref/index |
| Metadata | `tests/ut/ir/test_distributed_compiled_program.py` | 无 live IR 冷加载、版本、错布局、旧产物 |
| Domain | `tests/ut/runtime/test_comm_domain.py`（新） | spec、逐槽向上 64 字节对齐、offset/window_size、逻辑长度、源短读、事务回滚、generation |
| Binder | `tests/ut/runtime/test_distributed_args.py`（新） | 普通 scalar 与两窗口、不同域/槽、same-ref 去重 |
| Worker | `tests/ut/runtime/test_distributed_worker.py` | 双 program、frame 隔离、并发 admission/release、accepted failure |
| Hardware | `tests/st/distributed/test_shared_domain.py`（新） | Host 本地读/远端 GET、跨程序写读与清理 |
| L4（后续，不阻塞 SharedDomain） | #3531 的节点测试 | 双节点各有 chip 0、成员重排、同一入口不同 ref/index |

### 64 字节自动对齐的专项覆盖

- UINT8 logical bytes 为 1、63、64、65、127、128、129 时，physical bytes 分别为 64、64、64、128、128、128、192。
- FP32 17 元素为 68 logical bytes、128 physical bytes；view 仍为 17 个 FP32。
- 多 buffer 使用逻辑字节 `[1, 64, 65]`，实际 offset 为 `[0, 64, 128]`、window_size 为 256；捕获传给 Simpler 的 native specs，不能只测试 helper 返回值。
- initializer 对上述三片只读取/写入 `[1, 64, 65]` 字节，第二/第三片定位正确，相邻槽和 padding 不被当成数据。
- 已对齐的 buffer 不多补 64 字节；padding 不影响 buffer_index、public shape/dtype 或两程序间绑定。
- 乘积、对齐后大小和累计 window_size 超限，均在 native allocation 前失败；同时拒绝零/负/非法维度。
- 设备小例子包含不对齐逻辑长度，逐字节核对末尾数据和下一 buffer 开头，不能全部使用天然 64 字节对齐的表来代替。

### 必须有的反例

- buffer_index 为 -1、超界、bool 或错误标量宽度。
- same shape 但不同域/数据，不能因形状相同合并。
- 把 a/b 参数交换，输出必须随实际参数改变，不能绑定固定 slot 名。
- 目标 chip 不在域中；外部域没有 program 操作要求的 workspace。
- release 后仍保留 DomainTensor Python 对象；同名重建不能使旧 ref 重新有效。
- 只有一个窗口提供动态 shape，但 ABI 无处传维度；必须拒绝。
- 参数是非起点/非连续 view，不能忽略 offset/stride。
- native 已接受后 submit 抛错，不能提前释放 owner lease。
- 用户不调用 handle.result() 就 release/close，仍必须排空和回收。
- program reset=True/False 交替不应影响外部只读表；内部窗口原行为不变。

### 必须有的正例

- 同域两窗口、不同域两窗口、同一窗口作为两个参数。
- 一个 domain 的 members 为 `[2, 0]`，对目标 worker 解析的 domain rank 正确。
- 同一冷加载程序先接域 D slot 1，再接域 E slot 0，不重新编译。
- 第一个 program 写外部窗口，第二个读取；改变输入、多次提交暴露陈旧缓存。
- 每次调用都执行 run 登记，initializer 只执行一次。
- 输入表是 UINT8 字节，逐字节对比 CPU reference，覆盖首末行和相邻 buffer 边界。

## 7. 执行方式

实现后按 repository testing workflow，在当前 worktree 构建，不复用其他 checkout 的 `.so`：

```bash
source .claude/skills/testing/load-env.sh
cmake --build build --parallel "$PYPTO_BUILD_JOBS"
export PYTHONPATH="$PWD/python:${PYTHONPATH:-}"
python -m pytest tests/ut/ir/transforms/test_materialize_dist_tensor_ctx.py tests/ut/ir/transforms/test_materialize_comm_domain_scopes.py tests/ut/ir/transforms/test_verify_dist_tensor_ctx_materialized.py -n "$PYPTO_TEST_JOBS" -v
python -m pytest tests/ut/codegen/distributed/test_host_orch_distributed.py tests/ut/ir/test_distributed_compiled_program.py tests/ut/runtime/test_distributed_worker.py -n "$PYPTO_TEST_JOBS" -v
```

如当前 worktree 尚未配置 build，先按 testing skill 执行 CMake 配置。新增测试文件创建后加入同样的定向检查；按 repository 要求运行受影响代码 lint/type checks。此处不是本轮已执行日志。

NPU 验收严格分离 CPU 准备和 on-device 阶段：

1. CPU 上完成 PyPTO/PTOAS 编译、artifact 固化、bin fixtures 和 goldens。
2. 通过 pto TaskQueue 申请 2 卡，再按需要申请 4 卡；队列内仅加载预编译 artifact、分配域和执行。
3. 测试 driver 的 run-only 路径断言没有触发编译，使用队列分配的 device 列表。
4. 输出 artifact/运行时版本、创建与初始化次数、每 rank correctness、连续 submit、释放结果。
5. 不运行全量 ST；8 卡单独验收，不能由 4 卡通过推导。

队列命令按安装的 `pto-task-operations` skill 执行，不在设计中虚构当前 pytest 尚不存在的 run-only 参数。

## 8. 内存与正确性验收分开

小表测试证明绑定、初始化、次序及释放；不能单独证明 183 GiB 表可注册。
大表验收记录每域/per-rank 的实际分配大小和创建次数，并以共享页不重复计数的统计核对。两个 program 合计应是一组 Host shard 分配，而非两组。

同一个 worker 创建一次域的 Python 计数不能替代底层分配证据；CPU file mappings、page cache、goldens、padding 与 runtime heap 分开报告。
性能变化不是此功能的正确性门槛；不把首次初始化时间混入 steady-state lookup。

## 9. 文档与交付

正式实现后更新中英文：

- `docs/{en,zh}/dev/06-persistent-l3.md`：内部 persistent 与外部 owner 的差异。
- `docs/{en,zh}/dev/passes/46-materialize_comm_domain_scopes.md`。
- `docs/{en,zh}/dev/passes/48-materialize_dist_tensor_ctx.md`。
- codegen、运行时参数/产物文档：隐藏 ctx、index ABI、cold reload。
- 用户分布式文档：双 program 外部表示例，所有新 API 必须已验证。

本设计从独立分支发布，未包含原工作区的 pass/codegen 修改。后续实现应先核对目标分支及运行时基线。本分支通过窄范围 `.gitignore` 例外跟踪 SharedDomain 和 L4 的 Markdown，其他构建输出继续忽略。

## 10. 实现前必须关闭的验证项

| 项 | 决策/门槛 |
| --- | --- |
| Simpler 统一域能力 | 核对并锁定支持 ref/授权/初始化/fence 的版本 |
| 连续异步共享域提交 | 证明已有 run 顺序，或实现有界的前序 fence 准入 |
| external-domain token | 本地后 init POSIX 路径需在 SharedDomain 跑通；L4 节点准备路线留给后续 #3531 |
| metadata 版本 | 实施时协调新 schema 号，旧版普通 L3 显式兼容 |
| HOST builtin 外部域 | 逐项支持或诊断，不靠放宽 scope 检查默许 |
| runtime-only offset/stride | v1 拒绝；未来扩展须与 #3531 同步 |

没有把这些验证项解释为当前已经支持的能力，也不以默认 fallback 掩盖不兼容。
