# L4 实施计划

通信域与启动接口按 Simpler `d2b37652d` 核对；PyPTO `ca08d06bd` 锁定的 `b7cb9e73` 已含这些接口。
保留四层架构、共享 codegen、CommCtx 跨层传递和递归产物。
**L4 不直接调用 collective**；L4 建域、下发 L3，通信由 L3/L2 表达及设备 kernel 执行。

## 1. 分阶段

### P0 — 抽出 `LevelHandler`，保持当前 L3 行为

| 项 | 位置/要求 |
| -- | --------- |
| 接口与注册 | `include/pypto/hierarchy/level_handler.h`、`src/hierarchy/` |
| HostLevelHandler | 承接当前 Buffer API、完整参数添加语句、ctx 和入口发射 |
| 共享骨架 | `src/codegen/distributed/distributed_codegen.cpp` 及头文件 |
| 构建 | CMake 注册新源文件 |

**验收**：当前 upstream 的 L3 golden 逐字节不变，包含 `add_chip_tensor_arg`、
ordering token、标量后置、Submit 依赖和返回值。不是复刻旧稿的裸指针模板。

### P0.5 — dispatch 命名统一

| 项 | 范围 |
| -- | ---- |
| `device=` → `worker=` | parser、IR attr、域分析、codegen；保留弃用别名，双写报错 |
| 下发发射 | L4 直接 submit_next_level；L3 保留 _submit_chip，无运行时统一包装 |
| golden | 仅刷新命名差异 |

`world_size` 不做全局更名：HOST 保留其本地设备数语义及入口绑定。
`num_workers` 是新增 op，在 P1/P2/P3 联动适配，L4 不开放 world_size。

### P1 — 前端与配置入口

- `alloc_window_buffer` 放行 HOST/CLUSTER_0；新增 `num_workers()` 两层查询。
- `@pl.jit.pod`、HOST 依赖发现、NextLowerLevel、outlining 映射。
- L4 collective 显式拒绝，保持相邻层调用规则。
- `cluster_config` 接入 RunConfig、CompileOptions、JIT 和 ir.compile，并纳入缓存身份。
- 用户侧采用 backend_type/execution_mode，内部仍转换成 Simpler platform。

**验收**：四层程序解析/打印往返；L4 collective 和非法层级诊断；JIT compile 配置透传。

### P2 — Pass

| 项 | 文件 |
| -- | ---- |
| 按层配对、num_workers 覆盖分析 | `materialize_comm_domain_scopes_pass.cpp` |
| 窗口实参来源解析 | 同文件；按 Call/Submit 关联实参与形参，保留可复用的 callee 类型 |
| ctx 查询放行、verifier 同步 | `materialize_dist_tensor_ctx_pass.cpp` |
| HOST collective 限制保留 | `lower_host_tensor_collectives_pass.cpp`、`synthesize_allreduce_signals_pass.cpp` |
| 非设备闸门与 L4 拒绝路径 | `lower_composite_ops_pass.cpp`、`legalize_tile_cast_pass.cpp` 及 IR 检查 |
| 文档 | `docs/{en,zh}/dev/passes/46-materialize_comm_domain_scopes.md`、`48-materialize_dist_tensor_ctx.md` |

不新增 pass。域唯一 owner 的 verifier 可保留；ctx 查询的 verifier 必须修改。

### P3 — Codegen

- 新增 ClusterLevelHandler，复用 shared visitors 与任务组装。
- 嵌入 L4 的 L3 直接生成 Simpler 调用签名；入口内绑定参数和节点状态，独立 L3 ABI 保持兼容。
- L4 使用节点路由；L3 保留 add_chip_tensor_arg 和 Buffer.tensor。
- 新增 num_workers 发射；HOST world_size 兼容。
- L4→L3 的 DistributedTensor 占 0 个 tensor 槽，窗口实参发射单槽 buffer_index，ctx 后缀发射完整 ref 三槽；L3→L2 按 chip 物化窗口并附加 device_ctx。
- 自建域/传入域的 ordering token 统一身份和生命周期。
- 递归输出 `orchestration/cluster_orch.py` 与
  `next_levels/node_orch/orchestration/node_orch.py`。

**验收**：目标生成代码可加载，嵌入 L4 的 L3 函数可直接注册和下发；独立 L3 golden
仅因统一建域 `members=` 拼写而调整相应快照。

### P4.0 — 接入 Simpler MPI 启动接口

PyPTO 打包生成的 L3 模块与 metadata，配置 `MpiL3GroupSpec.startup_artifacts` 与 `startup_hooks_by_rank`，由准备函数绑定本地 CHIP 句柄与共享状态。接口见 [运行时设计](04-runtime.md) § 7。

**验收**：parent 本地载荷经 rank 0 广播，各 rank 校验/安装/hook 后才初始化 L2；
任一 rank 安装或准备失败时整组有界失败并清理；无需远端 daemon 或共享文件系统。

### P4 — PyPTO 运行时集成

| 项 | 范围 |
| -- | ---- |
| ClusterConfig / ClusterNodeSpec | 配置校验、JIT 身份、保存/冷加载 |
| cluster_runner / distributed_runner | MPI group 组装、生成 L3 orch 直接注册、启动状态绑定 |
| L3 bundle | L4 统一编译、确定性打包、摘要缓存、入口与 CHIP 依赖绑定 |
| 节点 bootstrap | 自动安装 bundle，L2 fork 前准备共享 intermediates/token |
| metadata | 以 schema 3 为基线新增版本化节点参数描述及 loader |
| TaskArgs 布局 | tensor/scalar 顺序、window 跳过和 ctx 绑定；无加载器前缀 |
| host tensor 路由 | MPI 节点统一 remote-like binding、方向、view、回填、keepalive |
| callable 元数据 | 远端和冷加载后仍保留 CHIP host-control 信息 |
| persistent | 显式共享约定、槽映射、ordering 状态、释放 |

**验收**：单 rank MPI L4 在 sim 上跑通；空缓存自动安装、冷加载及共享存储启动
时序通过；远端不调用编译器，CHIP host_tensor_indices 在加载后仍正确。

### P5 — 单机模拟多 L3 节点

测试配置使用两个 MPI rank，而不是把一个 L3 的两张卡当成两个节点：

```python
ClusterConfig(
    nodes=[
        ClusterNodeSpec(host="localhost", devices=[0, 1]),
        ClusterNodeSpec(host="localhost", devices=[2, 3]),
    ],
    python_executable=python_launcher,
    comm_profile="sim",
)
```

`python_launcher` 由测试环境提供；验证 mpirun 实际放置与声明顺序一致。
单节点覆盖只保留一个 nodes 条目。两种测试均走生产 MPI/bundle 启动路径，
不以手工 fork 两个本地 Worker 替代。MPI rank 0 即使同机也测试 remote buffer 路由。

首个多节点测试验证：L4 建四卡全局域；两个 L3 各自建两卡本地域；L3 下发 L2
执行节点内、跨节点通信。**L4 源码中没有 collective**。
位置：`tests/st/distributed/test_l4_cluster.py`，需运行环境具备对应 Simpler sim 能力。

### P6 — 真实跨机 ST

两台 A3 机器，`comm_profile="a3-fabric-v1"`，对照
`compute_then_tload_mixed_l3` 两 round 流程。验证传输、窗口保留、冷加载及清理，
不能以单机 sim 通过替代跨机验证。

必需验收场景：远端仅有兼容运行环境，**无本次生成物、缓存为空、
无预启动 L3 daemon**；L4 统一编译后直接运行成功。观测并断言远端未调用 PyPTO/
PTOAS 编译链。再覆盖缓存命中、产物更新、损坏 bundle、缺失 CHIP 依赖、单 rank
准备失败/超时及清理；本地与远端的 artifact 目录可以不同。

## 2. 测试矩阵

| 层 | 必需覆盖 |
| -- | -------- |
| 前端 | jit.pod、worker 别名、num_workers、L4 collective/world_size 拒绝 |
| Pass | 查询层级、窗口传播、乱序函数、别名、Call/Submit、冲突与 verifier |
| Codegen | 同一 L3 前端和 IR 窗口形参；L4 每节点一次，0 个窗口 tensor 槽；普通 tensor、窗口形参和用户 scalar 交错时仍先发射全部 tensor；窗口下标按 scalar pool 顺序排列，两个 ctx 在后缀各占三槽；L3 每 chip 各取窗口和一个 device_ctx |
| Metadata | schema 3 兼容、节点扩展往返、未知版本/缺字段/错配拒绝；同一域的两个同形窗口按相反声明顺序传入并更名，用同一个冷加载入口再接收另一域不同位置的窗口；越界下标拒绝 |
| Runtime | MPI 节点索引映射、bundle 安装后导入注册、版本隔离、fork 前 bootstrap、直接下发、tensor 方向/view、冷加载 |
| 启动协议 | 空缓存安装后导入、同名 orch 多版本隔离、重复安装、版本/摘要错误、缺依赖、单 rank 失败/超时、READY 与清理 |
| ST | 双节点各有 chip 0，一节点多 chip；重排成员或使用子集；同一 L3 同时用自建域与借入域；跨 program 第二轮读到首轮数据；同名重建后拒绝旧 ref |

按 P0、P0.5/P1、P3 分别建立/刷新/回归 golden，明确比较基线，不能同时承诺
“文本不变”又在同阶段更名或更换接口。

## 3. 风险与待验证

### 3.1 运行时基线已确定

统一入口为 `allocate_domain(members=...)`、`get_or_allocate_domain` 和 `get_domain(ref)`。L3 根据本地 chip worker 下标解析窗口；父 L4 或无 grant 的兄弟 L3 不能仅凭 ref 借用 L3 自建域。

### 3.2 全局域与本地域并存

必须验证同一节点同时持有两种域、设备 ctx 不混淆、ordering 状态正确及释放顺序。
跨机参考例先走 L3→L2→kernel 的显式 ctx 路径。L3 HOST builtin 对传入全局域的
支持单独验证，不因 L4 能建域就自动宣称支持。

### 3.3 PyPTO 启动产物集成

当前 MPI 能自动拉起 L3，CHIP callable 也已随 manifest 下发；但 `RemoteCallable`
只导入远端符号，不携带生成的 Python 编排。必须先安装生成模块，才能完成远端注册。
必须完成 P4.0 的 PyPTO bundle 打包和 hook 绑定，并由 PyPTO 适配器在 L2 fork 前分配共享
状态。init 后 remote_copy 或首次任务中加载都不能替代这段启动流程。

### 3.4 远端数据传输

按方向决定 copy_to/copy_from，明确 view/偏移/非连续张量及缓存失效策略。
大张量优先放窗口跨 round 复用；控制面传输开销在 ST 中记录。

## 4. 待决问题

| 问题 | 约束与当前方向 |
| ---- | -------------- |
| device 迁移窗口 | 保留 DeprecationWarning 别名，后续再决定删除版本 |
| 跨 program 共享域 | 需要显式共享意图、成员/布局校验及槽名映射；不能仅按尺寸自动合并 |
| 节点扩展 schema | 优先保留公共 schema 3、新增版本化 l3_entry 块；不兼容公共结构时升级 loader/schema |
| bundle/hook 集成 | 定义 PyPTO manifest、hook 参数、共享状态持有和清理 |

## 5. Simpler 接口集成验收

1. MPI 启动从 parent 安装 bundle，hook 在 L2 fork 前获得本地 CHIP digest 到 handle 的映射。
2. 生成的 L3 入口可直接注册和执行；验证冷缓存、启动失败及节点共享状态的清理。
3. 验证 retained 域在仅传 ref 的 run 中由 owner 登记使用，借用者不能释放，未参与节点无法查询。

## 6. 工作量

旧稿“约 830 行 C++ / 750 行 Python”的估算建立在 pass/verifier 零改动、metadata
零扩展和运行时仅两处改动的假设上，撤销该估算。
按上述阶段实际变更面估算，重点是 P2 的实参来源解析、P3 的参数/ordering 状态，以及
P4.0 的打包与启动接入与 P4 的 metadata、bootstrap 和远端生命周期。先完成可验证的接口样例，再估具体工作量。
