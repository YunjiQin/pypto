# L4 Pass 实现方案

复用现有 pass 流水线，不新增 pass。编号以 `upstream/main@d202cede9` 为基线：
`MaterializeCommDomainScopes` 为 46，`MaterializeDistTensorCtx` 为 48。
改动包含域分析的层级泛化、调用点实参的窗口解析，以及 ctx 查询和 verifier 的层级放行。

**L4 不直接调用 collective。** `LowerHostTensorCollectives` 与
`SynthesizeAllReduceSignals` 的 HOST 限制保留，不能把“可以建域”等同于
“可以执行 HOST collective lowering”。

## 1. 域跨层传递：复用 CommCtx 形参

现有 `MaterializeDistTensorCtx` 为每个 `DistributedTensorType` 形参在签名尾部
追加 `CommCtxType` 形参，并为 `Call`/`Submit` 补实参。它已经追踪形参、返回值、
别名及循环携带值的 ctx，无需另建一套 L4 参数传播机制。

但当前 `materialize_dist_tensor_ctx_pass.cpp` 的三个位置只允许 HOST 查询运行时 ctx：

| 位置 | 当前行为 | L4 适配 |
| ---- | -------- | ------- |
| `GetCtxForArg` | 找不到已有 ctx 时要求 `IsHostOrch(current_func_)` | 允许 HOST/CLUSTER_0 为本层窗口合成查询 |
| `VisitExpr_(CallPtr)` | 仅 HOST 保留显式 `get_comm_ctx` | HOST/CLUSTER_0 保留；设备层仍必须消除 |
| `DistTensorCtxMaterializedPropertyVerifierImpl` | 仅豁免 HOST | 与上述合法查询层级同步 |

示意：

```python
# L4 source
buf = pld.alloc_window_buffer(1024)
data = pld.window(buf, [1, 256], dtype=pl.FP32)
node_orch(data, worker=w)

# Target IR after MaterializeDistTensorCtx
ctx = pld.get_comm_ctx(data)
node_orch(data, ctx, worker=w)
```

`data` 是 L4 本地创建的窗口，没有可直接转发的 ctx 形参。因此必须修改上述查询
闸门；“pass 48 与 verifier 零改动”不再成立。L3 接收窗口后，已有的形参 ctx
转发机制仍可复用。动态合并不同 ctx 等现有限制保持不变。

| 调用边 | ctx 的运行时值 |
| ------ | -------------- |
| L4→L3 | `CommDomainRef`（3 个 uint64），另随配对窗口携带本次实参的 buffer_index |
| L3→L2 | 目标设备的 `device_ctx` |
| L2→kernel | 原样转发 `device_ctx` |

IR 签名、函数体和调用点保留 `DistributedTensor` 与配对的 `CommCtx`。L4→L3 的窗口占 0 个 wire tensor 槽，属于 codegen 的参数布局规则；pass 仍须传播窗口身份及其视图信息。

IR 中不引入“继承 scope”：`CommDomainScopeStmt` 只出现在创建域的函数中。

## 2. `MaterializeCommDomainScopes` 的改动

文件：`src/ir/transforms/materialize_comm_domain_scopes_pass.cpp`。

### 2.1 建域宿主与下一级函数配对

将 `IsHostOrch` / `IsChipOrch` 泛化为按层级查询的能力判定：

```cpp
bool IsDomainOwnerOrch(const FunctionPtr& func);
bool IsOrchAtLevel(const FunctionPtr& func, Level level);
Level NextLowerLevel(Level level);  // HOST -> CHIP, CLUSTER_0 -> HOST
```

建域能力可查询 `hierarchy::GetLevelHandler(level)`；collective 能力另行判定。
按层级建立 callee 索引，并按从高到低的调用依赖处理，不能依赖函数名的排序。
保留现有 CHIP Graph 排除规则及 Call/Submit 区别，不能用泛化覆盖掉已有条件。

### 2.2 循环覆盖范围与 `worker=`

`ProcessHostOrch` 泛化为 `ProcessOrchFunction`，复用 alloc/window 收集、下发分析、
view 绑定和 scope 构造，但不是内部零改动：

- `DispatchAnalyzer` 读取统一的 `worker` attr。
- 动态全覆盖循环识别 `pld.num_workers()`，包括 SSA 临时变量包装后的形式。
- 保留 HOST 的 `world_size()` 兼容路径；L4 不开放 `world_size()`。
- 常量子集循环保留原有非负、单位步长等检查。
- L4 `devices_` 表示下一级节点索引；L3 `devices_` 表示本地设备索引。

```python
# L4: all nodes; L3: all local devices
for w in pl.range(pld.num_workers()):
    child(data, worker=w)
```

这是原定域分析改造的一部分，不需要新增 pass。

### 2.3 在调用点解析实参窗口

建域函数的 `pld.window` 结果通过 `window_buffer_` 指向本地 owning scope 的槽。在每个 L4→L3 Call/Submit 上，按位置将实参与 callee 的逻辑形参配对，记录这次调用使用的域和该实参在域的有序 buffer 布局中的下标。别名须追溯到实际来源；已绑定的转发形参沿用本次调用的绑定。持久域若配置了显式槽映射，调用方使用映射后的实际物理下标。

L3 形参保留通用的 `DistributedTensorType`、形状、dtype 和配对 ctx。具体 `WindowBuffer` 只属于创建它的 scope；同一 L3 函数的不同调用边可以提供不同域、任意兼容槽及顺序。调用点的校验覆盖实参类型、视图和来源可解析性。L3→L2 由 L3 在下发时物化窗口。

### 2.4 L3 自建域与传入域并存

```text
node_orch(x, inter, out, inter_ctx)
└── CommDomainScopeStmt slots=[intra_buf]
    └── dispatch chip_orch(... intra ..., inter, ...)
```

`inter` 通过形参 ctx 绑定全局域，本地 `intra_buf` 由当前 scope 拥有。
窗口归属与 ctx 绑定分别维护；不把传入窗口重复加入本地 scope。

## 3. 其余层级闸门

| 文件/职责 | 改动 |
| --------- | ---- |
| `lower_host_tensor_collectives_pass.cpp` | 保留 HOST 限制；不在 L4 生成按设备调用的 builtin |
| `synthesize_allreduce_signals_pass.cpp` | 保留 HOST 限制；不在 L4 自动合成 collective 信号 |
| `lower_composite_ops_pass.cpp` | L4 collective 显式拒绝，不能落入设备 composite lowering；HOST 现有延后路径保留 |
| `legalize_tile_cast_pass.cpp` | 非设备编排层跳过设备 cast 合法化，可扩展到 Linqu level >= 3 |
| `scope_outline_utils.h` | 核对并补齐 CLUSTER_0 的 outlining 后缀 |
| 前端/IR 合法性检查 | L4 中所有 collective 调用给出明确诊断，包括直接构造 IR 的路径 |

本地域 collective 继续使用已有 HOST 路径。上层传入的全局域若要在 L3 直接调用
collective，需要另外验证 builtin 生成、窗口绑定、域大小和信号容量是否支持该来源；
不能假设单机 HOST builtin 自动支持全局域。首期跨机参考例通过 L3 下发 L2，由
设备 kernel 使用显式 `ctx` 执行通信，不依赖这一扩展。

## 4. Property Verifier

两类 verifier 必须区分：

- **`CommDomainScopesMaterialized`**：保留 slots 非空、slot 有效、每个 WindowBuffer
  全程序至多一个 owning scope 的规则；跨层引用不会新增 owning scope。
- **`DistTensorCtxMaterialized`**：必须修改，允许 HOST/CLUSTER_0 的运行时 ctx 查询；
  L2 及设备函数仍不得残留查询。

逐调用点检查实际窗口有 owning scope 或有效的传入绑定，并校验实参与形参的视图类型兼容。可复用形参按配对 ctx 和调用参数解析，类型检查允许不同调用传入不同来源。

## 5. Pass 顺序与文档

不新增 pass；实现时按 `pass_manager.py` 的实际依赖运行，文档编号不是 dump 序号。

- `docs/{en,zh}/dev/passes/46-materialize_comm_domain_scopes.md`：补多层域、循环识别与逐调用点窗口解析。
- `docs/{en,zh}/dev/passes/48-materialize_dist_tensor_ctx.md`：更新允许查询的层级和属性定义。
- HOST collective 文档明确 L4 不支持直接调用；已有 HOST/L2 功能不变。

## 6. Pass 测试

| 测试 | 断言 |
| ---- | ---- |
| L3 回归 | 当前 L3 IR 与 codegen 行为保留 |
| L4 自建全局域 | 仅 L4 存在该域 owning scope |
| L3 接收窗口 | 同一签名在不同调用点接收不同域和槽，body 通过形参绑定解析窗口 |
| L4 ctx 查询 | pass 48 正常补 ctx，verifier 接受 L4 查询 |
| 设备 ctx 查询 | 仍消除或报错，不因放宽 L4 而放宽设备层 |
| 动态循环 | num_workers 直接/SSA 形式识别全覆盖；HOST world_size 兼容 |
| 三层传播与乱序定义 | 与函数名、定义顺序无关 |
| Call/Submit | 保留依赖、TASK_ID 返回、Out 缺口和 ctx 后缀 |
| 两级域并存 | 本地域与全局域身份、ctx 不混淆 |
| 无效窗口 | 来源无法解析或视图不兼容时在对应调用点报错 |
| L4 collective | 前端与直接构造 IR 两条路径均拒绝 |
