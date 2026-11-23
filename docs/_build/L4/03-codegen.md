# L4 Codegen 实现方案

## 0. 共享骨架，按调用边适配运行时

L3 与 L4 继续共享 `DistributedCodegen` 的 IR visitor、控制流和任务组装骨架。
`LevelHandler` 封装各层的成员表达式、ctx 编解码、参数打包与入口差异。
L4 不生成 collective builtin：它只建全局域并下发 L3。

通信域接口按 Simpler `d2b37652d` 核对。以下是目标接口草案，
并非当前已实现的类或实际运行产物。

| 调用边 | 普通张量 | 窗口张量 | CommCtx |
| ------ | -------- | -------- | ------- |
| L4→L3 | MPI 节点统一 `_io_arg` 路由 | 一个 `buffer_index` 标量，L3 按本次实参物化 | `CommDomainRef` 三个标量 |
| L3→L2 | `add_chip_tensor_arg` | `context.buffers[slot].tensor(...)` | 目标设备 ctx |

`add_chip_tensor_arg` 只属于 L3→L2。它依据 callee 的 `host_tensor_indices`，
为 host 构图控制张量设置 `TensorTransfer.NONE`，其余使用默认传输；内部调用
`make_tensor_arg(worker, tensor)`。L4 本层不使用这个 CHIP helper。

## 1. `LevelHandler` 接口

新建 `include/pypto/hierarchy/level_handler.h`，让 pass 可以查询层级能力，
不依赖 codegen。接口返回字符串/数据，由骨架负责发射。

原方案 `FormatHostTensorArg` 仅返回 `.add_tensor(...)` 的参数表达式，无法直接
复用现在负责整个添加动作的 CHIP helper。因此改成返回完整的添加语句：

```cpp
struct HostArgSpec {
  std::string task_args;
  std::string tensor_expr;
  std::string tag;
  std::string worker_expr;
  std::string callee_name;
  size_t tensor_index;  // Index in the tensor pool, not the IR parameter list.
};

class LevelHandler {
 public:
  virtual ~LevelHandler() = default;
  virtual ir::Level GetLevel() const = 0;
  virtual std::string FormatDomainMembership(const std::vector<int64_t>& children) const = 0;
  virtual std::vector<std::string> FormatCommCtxAppend(const std::string& task_args,
                                                       const std::string& domain,
                                                       const std::string& child) const = 0;
  virtual std::string FormatCommCtxBind(const std::string& task_args, size_t scalar_offset) const = 0;
  virtual std::string FormatHostTensorAppend(const HostArgSpec& spec) const = 0;
  virtual bool MaterializesWindowTensor() const = 0;
  virtual std::vector<std::string> GetEntryKeywordParams() const = 0;
  virtual std::vector<std::string> GetTopologyBindings() const = 0;
  virtual std::vector<std::string> GetExtraImports() const = 0;
};

const LevelHandler* GetLevelHandler(ir::Level level);
```

这个草案只展示职责；实际头文件需补 includes、属性和项目风格。
`MaterializesWindowTensor` 指从 Buffer 创建 wire Tensor，不是暴露裸指针。
`GetLevelHandler` 注册 HOST/CLUSTER_0；“存在 handler”不代表允许 collective。

## 2. 两层的返回值

| 方法 | HostLevelHandler | ClusterLevelHandler |
| ---- | ---------------- | ------------------- |
| 建域方法（共享骨架） | `allocate_domain`，持久路径用 provider | 同左 |
| 成员 | `members=list(range(num_workers))` 或设备子集 | `members=[(nodes[n], d) for n in node_subset for d in range(node_sizes[n])]` |
| ctx 下传 | `table[w].device_ctx` | `table.ref` 三个 uint64 |
| 接收 ctx 绑定 | 解码 ref 后 `orch.get_domain(ref)` | 顶层无传入 ctx |
| 窗口 Tensor 物化 | 是 | 否 |
| 拓扑绑定 | `num_workers=world_size`，两者为本地设备数 | `num_workers=len(nodes)`，不绑定 L4 world_size |

`nodes` 是 DSL 节点索引到实际 L3 worker id 的表；`node_sizes` 与它同序。
`node_bindings` 保存节点路由和数据传输状态，与 worker id 区分；L3 callable handle
由启动注册返回，按 callee 和节点索引保存在 callables 中。

普通张量的目标发射形态：

```python
# HostLevelHandler: L3 -> L2
add_chip_tensor_arg(
    _ta_0, orch._worker, tensors["x"], TensorArgType.INPUT,
    callables, "chip_orch", 0,
)

# ClusterLevelHandler: L4 -> L3; helper and binding are proposed runtime additions
_ta_0.add_tensor(
    _io_arg(node_bindings[w], tensors["x"], TensorArgType.INPUT),
    TensorArgType.INPUT,
)
```

`_io_arg` 接收方向，才能记录 OUT/INOUT 回填；不能只在外层 `add_tensor` 传方向，
却假设 helper 自动知道它。MPI 节点（包括同机 rank）使用 remote buffer 路径，
不按主机名推断与 L4 parent 共享内存。

## 3. 域、窗口与 ctx

### 3.1 入口绑定传入域

codegen 为每个分布式张量形参及其配对 ctx 形参建立绑定：

```python
from simpler.task_interface import CommDomainRef

# The entry layout gives independent offsets for the tensor argument and its ctx.
buffer_index = int(args.scalar(window_offset))
ref = CommDomainRef(
    int(args.scalar(ctx_offset)).to_bytes(8, "little"),
    int(args.scalar(ctx_offset + 1)),
    int(args.scalar(ctx_offset + 2)),
)
__dom_data_ctx = orch.get_domain(ref)
if not 0 <= buffer_index < len(__dom_data_ctx.buffers):
    raise ValueError("window buffer index out of range")
slot = __dom_data_ctx.buffers[buffer_index].name
```

L3 codegen 按 IR 值的来源建立 `(domain_expr, slot_name_expr)` 绑定。两项均为生成代码里的表达式。

| 来源 | 建立绑定的时机 | 域表达式 | buffer 名表达式 |
| ---- | -------------- | -------- | --------------- |
| 当前 L3 scope 创建的 WindowBuffer | 发射建域语句时 | 本地建域返回的 handle | 对应 `CommBufferSpec.name`，例如字符串字面量 `"scratch"`；存在显式物理槽映射时使用映射后的键 |
| L3 的 DistributedTensor 形参 | 生成的入口每次被调用时 | `orch.get_domain(ref)` 的结果 | 对本次传入的 buffer 下标校验范围，再取 `domain.buffers[buffer_index].name` 存入局部变量 |

生成器为别名沿用其来源绑定。多个形参可共享域；每个形参保留自己的 slot 名与视图。下发点只消费这对表达式，无需在逐 chip 循环中判断域的创建层级。

绑定依据 pass 48 的形参配对和新增节点 metadata，不能从任意标量位置猜测。
入口 metadata 固定每个窗口形参与 ctx 的配对、读取位置及视图约束。本次域身份和 buffer_index 均从 TaskArgs 解码；同一 ref 可共用 handle，各形参分别选择自己的 buffer。

### 3.2 自建域发射

骨架继续负责 `CommBufferSpec`、逻辑字节数、物理槽对齐、window_size、scope 入栈
和 body 递归。handler 提供各层成员表达式；共享骨架发射临时域创建或持久域 provider 调用。
保留当前对齐策略，不能用早期示例的未对齐总和覆盖实际 codegen。

L3 本地域依旧由 `CommDomainScopeStmt` 发出；L4 全局域的成员将节点子集展开为
`(l3_worker_id, local_l2_worker_id)`。

### 3.3 窗口张量物化

`EmitCallToWorker` 在 L3→L2 组装窗口实参时按 IR 值取出绑定，发射 `domain_expr[w].buffers[slot_name_expr].tensor(...)`。例如同一个 L2 任务同时用节点内 scratch 和传入窗口 data，生成代码可写成：

```python
# local_domain owns the buffer declared as "scratch".
local_ctx = local_domain[w]
_ta_0.add_tensor(local_ctx.buffers["scratch"].tensor((1, 256), DataType.FLOAT32), TensorArgType.INOUT)

# slot was decoded once for the incoming data argument at node_orch entry.
data_ctx = __dom_data_ctx[w]
_ta_0.add_tensor(data_ctx.buffers[slot].tensor((1, 256), DataType.FLOAT32), TensorArgType.INOUT)
```

两个窗口对应的设备 ctx 分别取自 `local_ctx.device_ctx`、`data_ctx.device_ctx`，按 L2 签名顺序放在 tensor 参数之后。

移除旧 `.buffer_ptrs` / `Tensor.make(data=..., child_memory=True)` 发射模板。
L4 把这个窗口实参归入 scalar pool 并发射 `add_scalar(buffer_index)`。节点入口从其 scalar 偏移读取下标。
这意味着两层共享组装算法，但 wire 参数布局并非逐项相同。

### 3.4 ctx 下传

```python
# L4 -> L3: the DistributedTensor argument contributes one scalar.
_ta_0.add_scalar(buffer_index_for_actual)
# The appended CommCtx parameter contributes three scalars at its own position.
ref = __comm_d0.ref
_ta_0.add_scalar(int.from_bytes(ref.owner_instance_id, "little"))
_ta_0.add_scalar(ref.domain_id)
_ta_0.add_scalar(ref.generation)

# L3 -> L2, including an explicit ctx parameter received from L4
_ta_0.add_scalar(__dom_data_ctx[w].device_ctx)
```

L4→L3 的窗口实参与普通业务 scalar 在 scalar pool 中保持原形参的相对顺序；随后按 IR 的 ctx 后缀顺序，每个 ctx 展开为三个连续 scalar。handle 和 token 可按完整 ref 去重。L3→L2 的 `w` 是本地 chip worker 下标，`domain[w].domain_rank` 才是域内 rank。
普通同层标量转发保持现有语义。

### 3.5 通信 ordering token

当前 L3 codegen 为通信域添加按 rank 的 ordering token，放在最后一个 tensor 槽，
然后才添加 scalars。自建域继续复用现有实现；传入域需要补域身份到 token 的映射。

- 同一物理域、同一设备的派发共享 token，多个窗口不能创建互不关联的 token。
- token 和其他共享 intermediates 必须在本地 CHIP children fork 前准备。
- 节点共享存储不能在第一次 L3 task 中才临时创建。
- L4→L3 的依赖和跨 round 排空单独验证；不能机械复制 CHIP token 到 L4 参数表。

自建域/传入域并存及跨 program 复用时的 token 生命周期属于 P3/P4 验收内容。

## 4. 任务组装与下发

### 4.1 参数顺序

复用当前 `EmitCallToWorker` 的分组规则：先添加全部 tensor，再添加全部 scalar，
两组内部各保持形参相对顺序。保留 caller-supplied Out、runtime-allocated Out、
ctx 后缀、Submit deps 和 TASK_ID 返回值的现有处理。
Simpler 在第一次 `add_scalar` 之后拒绝 `add_tensor`。生成器先按调用边把每个逻辑实参归入 wire tensor 或 scalar，再分两遍发射。L4→L3 的窗口实参归入 scalar，遇到它时记录下标表达式，等本次调用的全部 tensor 添加完毕再发射。[TaskArgs 实现](https://github.com/hw-native-sys/simpler/blob/ce02bb2640fbd27fcc6d656cfb1423d55b7bdf46/src/common/task_interface/task_args_wire.h#L93)

例如 L4→L3 的 IR 参数为 `(x_dist, lhs, n, y_dist, out, x_ctx, y_ctx)`。第一遍按顺序为 `lhs, out` 调用 `add_tensor`；第二遍按顺序为 `x_idx, n, y_idx, *x_ref, *y_ref` 调用 `add_scalar`。两个 ref 在 scalar pool 的起始槽分别为 3 和 6。节点 metadata 独立记录各参数所在 pool 及其偏移。L3→L2 的窗口按 tensor 处理，通信 ordering token 是最后一个 tensor，随后才添加设备 ctx 等标量。

### 4.2 共享组装，按层发射下发语句

```python
# L4 -> L3: directly submit the registered generated orchestration.
orch.submit_next_level(
    callables["node_orch"][w], _ta_0, config, worker=nodes[w],
)

# L3 -> L2: preserve the existing CHIP helper.
_submit_chip(orch, callables["chip_orch"], _ta_0, config, w)
```

- L4：handle 由 `register(RemoteCallable(generated_target), workers=[nodes[w]])`
  返回，直接标识生成的 L3 orch；不增加统一入口或 artifact 任务封装。
- L3：复用当前 `_submit_chip` 的 worker 解析、DFX 和返回值，不强制更名。
- 共享 visitor 与参数组装，codegen 根据调用边选择下发语句。若需要 helper 抽象，
  放在 codegen 发射接口内，不新增运行时转调层。
- 未显式指定 worker 时两层都需要合法目标选择，不把 L3 的设备数直接用于 L4。
- 依赖等待、Submit 的 TaskId 返回及 DFX 节点/设备层级均需验证。

### 4.3 入口与系统 op

| 场景 | 入口与状态绑定 |
| ---- | -------------- |
| 独立 L3 | 保留现有 keyword 参数与 runner 绑定方式 |
| 嵌入 L4 的 L3 | 直接生成 `(orch, args, config)`；函数体从 TaskArgs 绑定业务参数，使用启动时绑定的本地 callables/拓扑/共享状态 |
| L4 | 保留本地 runner 注入的 tensors、callables、nodes、node_sizes、node_bindings、keepalive、可选 domain_provider |

`src/codegen/distributed/distributed_codegen.cpp` 当前生成 L3 keyword 参数入口；
需增加嵌入 L4 的直接调用约定。只是入口绑定不同，L3 编排 body 仍复用原发射逻辑。
函数本身就是注册目标，不额外生成转发到另一个 orch 的包装函数。
示例见 [06-example-lowering.md](06-example-lowering.md) § 5。

`num_workers()` 需新增 codegen 分支；HOST `world_size()` 的本地设备数语义保留。
用户入口名称与当前 `_pypto_distributed_entry` 标记发现机制继续兼容。

## 5. 产物递归

```text
<out>/
  distributed_meta.json
  orchestration/cluster_orch.py
  next_levels/node_orch/
    distributed_meta.json
    orchestration/node_orch.py
    next_levels/chip_orch/
      kernel_config.py
      orchestration/chip_orch.cpp
      kernels/...
```

每个下一级编排函数生成独立子树。L3 子树复用已有产物加载机制，但 metadata 要新增
窗口/ctx 绑定信息，节点启动还要准备共享 intermediates；不能称“只换域获取一行”。
L4 机器统一完成所有子树编译；生成的 L3 Python 与必要 metadata 打包后在 MPI
启动阶段自动分发，L2 二进制沿用 CHIP manifest 路径。节点按 digest 绑定已安装
CHIP 句柄，不在远端重新编译。bundle 和 fork 前准备使用 MPI group 的 `startup_artifacts` 和 `startup_hooks_by_rank`。
当前 metadata schema 与扩展策略见 [04-runtime.md](04-runtime.md) § 6。

## 6. 回归与测试

P0 从当前 upstream 的生成结果抽 handler，不回退到旧接口，L3 golden 逐字节不变。
命名/新 op 的可见变动集中在 P0.5/P1 更新 golden；P3 若将 L3 建域拼写由 `workers=` 迁为 `members=`，仅更新该调用的 golden，并对其余 L3 生成结果回归。

| 测试 | 断言 |
| ---- | ---- |
| L3 golden | Buffer、CHIP host-control 打包、ordering token、Submit 等全部保留 |
| L4 参数 | 使用节点路由，不调用 add_chip_tensor_arg，不发窗口设备 Tensor |
| L3 参数 | 保留 add_chip_tensor_arg 及 host_tensor_indices，Buffer 窗口正常物化 |
| ctx | L4 每个窗口与 ctx 配对传 ref 三槽和 buffer_index 一槽；多个 ctx 与用户 scalar 偏移正确，L3 每 chip 传一个 device_ctx |
| 拓扑 | L4 节点数、L3 设备数分离；节点索引正确映射实际 worker id |
| ordering | 同域多窗口去重；共享 token 在 fork 前存在；相邻派发保持顺序 |
| 两级域 | 自建和传入域不混淆，传入域无需重复 scope |
| L4 collective | codegen 前明确拒绝，不落入 HOST builtin 路径 |
| 产物 | 两层入口和 metadata 正确嵌套；嵌入 L4 的 L3 可直接注册；冷加载可恢复绑定 |
| 无支持层级 | CLUSTER_1 等明确报错 |
