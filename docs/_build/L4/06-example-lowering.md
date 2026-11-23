# 例子的完整降级链路

把 [00-design.md](00-design.md) § 8 的 compute round 展开为目标 IR 与生成代码。
**以下均为设计示意，不是实际编译 dump 或通过验证的生成物。** 用户源码使用 JIT；
这里用 `@pl.program` 展示特化后的 IR，类名统一写成 `ComputeRound` 便于阅读。
仅保留一个窗口，NN/ND 为动态节点数/每节点设备数；源码不在 L4 调用 collective。
文档编号 46/48 不等于运行时 passes_dump 文件序号。

## 0. 产物目录

产物目录按层级递归嵌套——每一层是 `orchestration/`（本层）+ `next_levels/<callee>/`
（下一级），与现有 L3→L2 的 `next_levels/chip_orch/` 约定一致：

```text
build_output/ComputeRound_<ts>/                     ← L4
├── distributed_meta.json                             L4 形参 + cluster_config
├── orchestration/cluster_orch.py                     L4 entry（在 L4 进程执行）
└── next_levels/node_orch/                          ← L3
    ├── distributed_meta.json                         L3 形参（含 window / comm_ctx 标注）
    ├── orchestration/node_orch.py                    L3 entry（在每个节点执行）
    └── next_levels/chip_orch/                      ← L2
        ├── kernel_config.py
        ├── orchestration/chip_orch.{cpp,so}
        ├── kernels/aiv/add_kernel.{cpp,o}          ← L0
        └── ptoas/add_kernel.{cpp,pto}
```

`next_levels/node_orch/` 复用 L3 的目录与模块加载机制，但 metadata 要增加节点参数
绑定信息，bootstrap 要在 CHIP fork 前准备共享状态。纯 L3 产物保持兼容；不能把
整个节点适配描述成“只换一行”。

## 1. 前端 IR（`00_frontend.py`）

```python
# pypto.program: ComputeRound
import pypto.language as pl
import pypto.language.distributed as pld

NN = pl.dynamic("NN")
ND = pl.dynamic("ND")

@pl.program
class ComputeRound:
    @pl.function(type=pl.FunctionType.InCore)
    def add_kernel(self, lhs: pl.Tensor[[1, 256], pl.FP32], rhs: pl.Tensor[[1, 256], pl.FP32], data: pl.InOut[pld.DistributedTensor[[1, 256], pl.FP32]], echo: pl.Out[pl.Tensor[[1, 256], pl.FP32]]) -> pl.Tensor[[1, 256], pl.FP32]:
        a: pl.Tile[[1, 256], pl.FP32] = pl.tile.load(lhs, [0, 0], [1, 256])
        b: pl.Tile[[1, 256], pl.FP32] = pl.tile.load(rhs, [0, 0], [1, 256])
        s: pl.Tile[[1, 256], pl.FP32] = pl.tile.add(a, b)
        data: pld.DistributedTensor[[1, 256], pl.FP32] = pl.tile.store(s, [0, 0], data)
        return pl.tile.store(s, [0, 0], echo)

    @pl.function(type=pl.FunctionType.Orchestration)
    def chip_orch(self, lhs: pl.Tensor[[1, 256], pl.FP32], rhs: pl.Tensor[[1, 256], pl.FP32], data: pl.InOut[pld.DistributedTensor[[1, 256], pl.FP32]], echo: pl.Out[pl.Tensor[[1, 256], pl.FP32]]) -> pl.Tensor[[1, 256], pl.FP32]:
        return self.add_kernel(lhs, rhs, data, echo)

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def node_orch(self, lhs: pl.Tensor[[ND, 1, 256], pl.FP32], rhs: pl.Tensor[[ND, 1, 256], pl.FP32], data: pl.InOut[pld.DistributedTensor[[1, 256], pl.FP32]], echo: pl.Out[pl.Tensor[[ND, 1, 256], pl.FP32]]) -> pl.Tensor[[ND, 1, 256], pl.FP32]:
        for w in pl.range(pld.system.num_workers()):
            self.chip_orch(lhs[w], rhs[w], data, echo[w], worker=w)
        return echo

    @pl.function(level=pl.Level.CLUSTER_0, role=pl.Role.Orchestrator)
    def cluster_orch(self, lhs: pl.Tensor[[NN, ND, 1, 256], pl.FP32], rhs: pl.Tensor[[NN, ND, 1, 256], pl.FP32], echo: pl.Out[pl.Tensor[[NN, ND, 1, 256], pl.FP32]]) -> pl.Tensor[[NN, ND, 1, 256], pl.FP32]:
        data_buf: pl.Ptr = pld.tensor.alloc_window_buffer(1024)
        for w in pl.range(pld.system.num_workers()):
            data: pld.DistributedTensor[[1, 256], pl.FP32] = pld.tensor.window(data_buf, [1, 256], dtype=pl.FP32)
            self.node_orch(lhs[w], rhs[w], data, echo[w], worker=w)
        return echo
```

四个函数的 `level_` / `role_` 此时已解析：`InCore` → `AIV/SubWorker`，
`Orchestration` → `CHIP/Orchestrator`，另两个由装饰器显式给出。

## 2. MaterializeCommDomainScopes 后（文档编号 46）

`cluster_orch` 是本程序唯一的建域宿主（它体内有 `alloc_window_buffer`）。pass 在其
body 外包一层 `CommDomainScopeStmt`——打印为一行 `# pld.comm_domain:` 注释——并把
`window_buffer_` 填进 `pld.tensor.window` 结果的类型里。

```python
    @pl.function(level=pl.Level.CLUSTER_0, role=pl.Role.Orchestrator)
    def cluster_orch(self, lhs__ssa_v0: pl.Tensor[[NN, ND, 1, 256], pl.FP32], rhs__ssa_v0: pl.Tensor[[NN, ND, 1, 256], pl.FP32], echo__ssa_v0: pl.Out[pl.Tensor[[NN, ND, 1, 256], pl.FP32]]) -> pl.Tensor[[NN, ND, 1, 256], pl.FP32]:
        # pld.comm_domain: devices=all, slots=[data_buf__ssa_v0]
        data_buf__ssa_v0: pl.Ptr = pld.tensor.alloc_window_buffer(1024)
        t__tmp_v0: pl.Scalar[pl.INT64] = pld.system.num_workers()
        for w__idx_v0 in pl.range(t__tmp_v0):
            data__ssa_v0: pld.DistributedTensor[[1, 256], pl.FP32] = pld.tensor.window(data_buf__ssa_v0, [1, 256], dtype=pl.FP32)
            t__tmp_v1: pl.Tensor[[ND, 1, 256], pl.FP32] = pl.tensor.slice(lhs__ssa_v0, [1, ND, 1, 256], [w__idx_v0, 0, 0, 0], [], [0])
            t__tmp_v2: pl.Tensor[[ND, 1, 256], pl.FP32] = pl.tensor.slice(rhs__ssa_v0, [1, ND, 1, 256], [w__idx_v0, 0, 0, 0], [], [0])
            t__tmp_v3: pl.Tensor[[ND, 1, 256], pl.FP32] = pl.tensor.slice(echo__ssa_v0, [1, ND, 1, 256], [w__idx_v0, 0, 0, 0], [], [0])
            self.node_orch(t__tmp_v1, t__tmp_v2, data__ssa_v0, t__tmp_v3, worker=w__idx_v0, attrs={"arg_directions": [pl.adir.input, pl.adir.input, pl.adir.inout, pl.adir.inout]})
        return echo__ssa_v0
```

本例的 `node_orch` 接收一个通用的 `DistributedTensor` 形参。具体窗口属于 L4 调用点，pass 保留形参的视图类型和 ctx 配对；L3 的函数体可用于不同域和槽。下例省略存储规划细节。

```python
    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def node_orch(self, lhs__ssa_v0: pl.Tensor[[ND, 1, 256], pl.FP32], rhs__ssa_v0: pl.Tensor[[ND, 1, 256], pl.FP32], data__ssa_v0: pl.InOut[pld.DistributedTensor[[1, 256], pl.FP32]], echo__ssa_v0: pl.Out[pl.Tensor[[ND, 1, 256], pl.FP32]]) -> pl.Tensor[[ND, 1, 256], pl.FP32]:
        t__tmp_v0: pl.Scalar[pl.INT64] = pld.system.num_workers()
        for w__idx_v0 in pl.range(t__tmp_v0):
            t__tmp_v1: pl.Tensor[[1, 256], pl.FP32] = pl.tensor.slice(lhs__ssa_v0, [1, 1, 256], [w__idx_v0, 0, 0], [], [0])
            t__tmp_v2: pl.Tensor[[1, 256], pl.FP32] = pl.tensor.slice(rhs__ssa_v0, [1, 1, 256], [w__idx_v0, 0, 0], [], [0])
            t__tmp_v3: pl.Tensor[[1, 256], pl.FP32] = pl.tensor.slice(echo__ssa_v0, [1, 1, 256], [w__idx_v0, 0, 0], [], [0])
            self.chip_orch(t__tmp_v1, t__tmp_v2, data__ssa_v0, t__tmp_v3, worker=w__idx_v0, attrs={"arg_directions": [pl.adir.input, pl.adir.input, pl.adir.inout, pl.adir.inout]})
        return echo__ssa_v0
```

对照点：这段与一个**纯 L3 程序的 host_orch** 结构上完全一样，唯一区别是窗口来自形参
而不是本地 `alloc_window_buffer`。

## 3. MaterializeDistTensorCtx 后（文档编号 48）

pass 48 对每个带 `DistributedTensorType` 形参的函数在签名尾部追加 `CommCtxType`
形参，并在每个调用点补上匹配实参。**先完成 ctx 查询及 verifier 的 L4 放行后**，
所以四个函数里有三个被改写（`add_kernel` / `chip_orch` / `node_orch` 各追加
`data_ctx`），`cluster_orch` 自身没有分布式张量形参，只是在调用点绑定实参。

```python
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def add_kernel(self, lhs__ssa_v0: ..., rhs__ssa_v0: ..., data__ssa_v0: pl.InOut[pld.DistributedTensor[[1, 256], pl.FP32, ...]], echo__ssa_v0: ..., data_ctx: pld.CommCtxType) -> pl.Tensor[[1, 256], pl.FP32]:
        ...

    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def chip_orch(self, lhs__ssa_v0: ..., rhs__ssa_v0: ..., data__ssa_v0: pl.InOut[pld.DistributedTensor[[1, 256], pl.FP32, ...]], echo__ssa_v0: ..., data_ctx: pld.CommCtxType) -> pl.Tensor[[1, 256], pl.FP32]:
        t__tmp_v0: ... = self.add_kernel(lhs__ssa_v0, rhs__ssa_v0, data__ssa_v0, echo__ssa_v0, data_ctx, attrs={"arg_directions": [pl.adir.input, pl.adir.input, pl.adir.inout, pl.adir.inout, pl.adir.scalar]})
        return t__tmp_v0

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def node_orch(self, lhs__ssa_v0: ..., rhs__ssa_v0: ..., data__ssa_v0: pl.InOut[pld.DistributedTensor[[1, 256], pl.FP32]], echo__ssa_v0: ..., data_ctx: pld.CommCtxType) -> pl.Tensor[[ND, 1, 256], pl.FP32]:
        t__tmp_v0: pl.Scalar[pl.INT64] = pld.system.num_workers()
        for w__idx_v0 in pl.range(t__tmp_v0):
            ...
            # data 是本函数的形参 => 转发本函数已物化的 ctx 形参（pass 48 规则 2）
            self.chip_orch(t__tmp_v1, t__tmp_v2, data__ssa_v0, t__tmp_v3, data_ctx, worker=w__idx_v0, attrs={"arg_directions": [pl.adir.input, pl.adir.input, pl.adir.inout, pl.adir.inout, pl.adir.scalar]})
        return echo__ssa_v0

    @pl.function(level=pl.Level.CLUSTER_0, role=pl.Role.Orchestrator)
    def cluster_orch(self, lhs__ssa_v0: ..., rhs__ssa_v0: ..., echo__ssa_v0: ...) -> pl.Tensor[[NN, ND, 1, 256], pl.FP32]:
        # pld.comm_domain: devices=all, slots=[data_buf__ssa_v0]
        data_buf__ssa_v0: pl.Ptr = pld.tensor.alloc_window_buffer(1024)
        t__tmp_v0: pl.Scalar[pl.INT64] = pld.system.num_workers()
        for w__idx_v0 in pl.range(t__tmp_v0):
            data__ssa_v0: pld.DistributedTensor[[1, 256], pl.FP32] = pld.tensor.window(data_buf__ssa_v0, [1, 256], dtype=pl.FP32)
            ...
            # data 不是本函数形参 => 就地绑定（pass 48 规则 2 的另一分支）
            data_ctx: pld.CommCtxType = pld.system.get_comm_ctx(data__ssa_v0)
            self.node_orch(t__tmp_v1, t__tmp_v2, data__ssa_v0, t__tmp_v3, data_ctx, worker=w__idx_v0, attrs={"arg_directions": [pl.adir.input, pl.adir.input, pl.adir.inout, pl.adir.inout, pl.adir.scalar]})
        return echo__ssa_v0
```

**这就是全部的跨层域传递机制**——IR 里没有出现任何 L4 专属节点、属性或作用域。
`data_ctx` 在三条边上是同一个概念、不同的物理取值，取值差异全部落在 codegen：

| 边 | ctx 的物理值 |
| -- | ------------ |
| `cluster_orch` → `node_orch` | ctx 后缀为 `__comm_d0.ref` 的三个 uint64；业务实参 `data` 自身编码为 buffer_index |
| `node_orch` → `chip_orch` | 设备 ctx 指针（`__dom_data_ctx[w].device_ctx`） |
| `chip_orch` → `add_kernel` | 同上，按标量原样转发（L2 C++ 侧 `orch_args.scalar(0)`） |

## 4. L4 入口的目标代码

下例省略导入和常规元数据加载。`_io_arg` 是待实现的数据路由 helper；
callables 中已保存各目标节点上生成 L3 orch 的注册 handle，直接传给 Simpler 下发。

```python
def cluster_orch(
    orch, _args, config, *, tensors, callables, nodes, node_sizes,
    node_bindings, _keep, _domain_provider=None,
):
    num_workers = len(nodes)
    with (_domain_provider or orch.allocate_domain)(
        name="comm_d0",
        members=[(nodes[n], d) for n in range(num_workers) for d in range(node_sizes[n])],
        window_size=1024,
        buffers=[CommBufferSpec(name="data_buf__ssa_v0", dtype="opaque", count=1024, nbytes=1024)],
    ) as __comm_d0:
        ref = __comm_d0.ref
        for w in range(num_workers):
            _ta_0 = TaskArgs()
            _ta_0.add_tensor(_io_arg(node_bindings[w], tensors["lhs"][w], TensorArgType.INPUT), TensorArgType.INPUT)
            _ta_0.add_tensor(_io_arg(node_bindings[w], tensors["rhs"][w], TensorArgType.INPUT), TensorArgType.INPUT)
            _ta_0.add_tensor(_io_arg(node_bindings[w], tensors["echo"][w], TensorArgType.INOUT), TensorArgType.INOUT)
            _ta_0.add_scalar(0)  # data: index of the actual window buffer
            _ta_0.add_scalar(int.from_bytes(ref.owner_instance_id, "little"))
            _ta_0.add_scalar(ref.domain_id)
            _ta_0.add_scalar(ref.generation)
            _keep.append(_ta_0)
            orch.submit_next_level(callables["node_orch"][w], _ta_0, config, worker=nodes[w])

cluster_orch._pypto_distributed_entry = True
```

持久路径的 `_domain_provider` 在当前 run 调用 `get_or_allocate_domain(..., retain_after_run=True)`，并返回退出时保留域的 lease。普通路径使用临时域。这里只展示已对齐的 1024 字节单槽情况，实际 codegen 保留现有物理槽对齐规则。
窗口 `data` 不占 L4 wire tensor 槽；L4 不调用 add_chip_tensor_arg，也不生成 collective。
方向传给 `_io_arg`，供它决定远端上传与回填。

## 5. L3 入口的目标代码

此函数本身就是 `RemoteCallable` 的注册目标，直接包含生成的编排 body。
`_bind_task_tensors` 在本例只绑定 tensor 槽 0、1、2 的 lhs、rhs、echo。`data` 留在 IR，由本次调用的 ref、buffer_index 及入口视图布局绑定，在 L3→L2 调用点才生成 Tensor；
参数读取/视图恢复的具体 API 需在实现时对齐。`_prepared_state` 表示启动阶段为本模块
绑定的 session 状态；其中共享存储在 CHIP children fork 前准备，并按真实域身份复用。
以下是目标生成代码草案，独立 L3 的现有入口 ABI 不随之改变。

```python
def node_orch(orch, args, config):
    tensors = _bind_task_tensors(args)
    buffer_index = int(args.scalar(0))
    ref = CommDomainRef(
        int(args.scalar(1)).to_bytes(8, "little"),
        int(args.scalar(2)),
        int(args.scalar(3)),
    )
    state = _prepared_state
    callables = state.callables
    _keep = state.keepalive
    num_workers = state.world_size  # Local device count, not global domain size.
    __dom_data_ctx = orch.get_domain(ref)
    if not 0 <= buffer_index < len(__dom_data_ctx.buffers):
        raise ValueError("window buffer index out of range")
    slot = __dom_data_ctx.buffers[buffer_index].name
    ordering_token = state.tokens_for_domain(ref)
    for w in range(num_workers):
        ctx = __dom_data_ctx[w]  # w is the local chip worker index.
        _ta_0 = TaskArgs()
        add_chip_tensor_arg(_ta_0, orch._worker, tensors["lhs"][w], TensorArgType.INPUT, callables, "chip_orch", 0)
        add_chip_tensor_arg(_ta_0, orch._worker, tensors["rhs"][w], TensorArgType.INPUT, callables, "chip_orch", 1)
        _ta_0.add_tensor(
            ctx.buffers[slot].tensor(shapes=(1, 256), dtype=DataType.FLOAT32),
            TensorArgType.INOUT,
        )
        add_chip_tensor_arg(_ta_0, orch._worker, tensors["echo"][w], TensorArgType.INOUT, callables, "chip_orch", 3)
        # Last tensor; all scalar arguments follow it.
        _ta_0.add_tensor(make_tensor_arg(orch._worker, ordering_token[w]), TensorArgType.INOUT)
        _ta_0.add_scalar(ctx.device_ctx)
        _keep.append(_ta_0)
        _submit_chip(orch, callables["chip_orch"], _ta_0, config, w)

node_orch._pypto_distributed_entry = True
```

`tokens_for_domain` 是示意的状态查询，只关联/复用启动时已准备的共享存储，
不能在首个任务中才分配需 fork 继承的存储；域身份去重和清理仍需实现。
普通 host 张量经过当前 CHIP helper；窗口使用 Buffer.tensor；排序 token 保留。
`callables` 必须携带 host_tensor_indices，冷加载/远端注册后也不能退化为丢失元数据
的普通字典。传入全局域与自建本地域复用这些打包规则，但绑定和启动准备不同。

## 6. L2 与 kernel

L2 不需要知道上层是 standalone L3 还是嵌套 L3。它接收四个业务张量和一个 ctx
标量；附加 ordering tensor 由 runtime 建依赖，不改变业务张量的已有索引。
目标 L2 接口关键部分如下（省略普通配置、导出和任务 scope）：

```cpp
const Tensor& ext_lhs = orch_args.tensor(0).ref();
const Tensor& ext_rhs = orch_args.tensor(1).ref();
const Tensor& ext_data = orch_args.tensor(2).ref();
const Tensor& ext_echo = orch_args.tensor(3).ref();
uint64_t data_ctx = orch_args.scalar(0);

L0TaskArgs params;
params.add_input(ext_lhs);
params.add_input(ext_rhs);
params.add_inout(ext_data);
params.add_inout(ext_echo);
params.add_scalar(data_ctx);
rt_submit_aiv_task(0, params);
```

沿用当前后端生成完整 L2/kernel 产物，不再把旧模板的完整 C++ 声称为当前 golden。
设备 kernel 中的 `rank(ctx)`、`nranks(ctx)` 和 remote_load 继续操作该 ctx 指定的域。

## 7. Metadata 与任务布局

当前分布式 artifact 的公共字段以 schema 3 为基线，例如以下**字段摘录**：

```json
{
  "schema": 3,
  "supported_execution_modes": ["program"],
  "params": [
    {"name": "lhs", "direction": "In", "shape": [-1, -1, 1, 256], "dtype": "fp32", "layout": "ND"}
  ],
  "backend_type": "Ascend910B"
}
```

这不是完整可加载文件。cluster_config 和 L3 节点参数扩展仍需实现，不能把新字段
描述成当前 schema 已有能力。节点扩展的目标语义示例如下，具体编码及版本策略见
[04-runtime.md](04-runtime.md) § 6：

| 逻辑形参序号 | 参数 | 来源 |
| ------------ | ---- | ---- |
| 0 | lhs | tensor pool 0 |
| 1 | rhs | tensor pool 1 |
| 2 | data | 无 wire tensor；与逻辑参数 4 配对，从本次调用的 scalar pool 0 读取 buffer_index |
| 3 | echo | tensor pool 2 |
| 4 | data_ctx | scalar pool，起始槽 1，宽度 3：owner_instance_id、domain_id、generation |

本例的 L4 调用点选择 `data_buf__ssa_v0`，在其域内下标为 0。另一个调用点可以向同一 `node_orch` 传入不同 ref 和下标 1，只须目标节点可以访问该域且窗口满足形参视图约束。下标由每次调用提供，节点入口只固定读取位置。

L4→L3 最终任务：

```text
tensors: [lhs, rhs, echo]
scalars: [buffer_index, owner_instance_id_le64, domain_id, generation]
```

生成的 L3 orch 直接按布局绑定参数；不能按逻辑参数序号索引 tensor pool。
该布局与节点内 chip 数无关。例如成员表 `[(A, 0), (B, 0), (B, 1)]` 中，B 只收到一次调用，`w=0` 和 `w=1` 各自取到 domain rank 1 和 2 的窗口。目标函数已由 callable handle 确定，TaskArgs 中不携带 artifact 标识。L3→L2 的任务为：

```text
tensors: [lhs, rhs, data, echo, ordering_token]
scalars: [device_ctx]
```

两条边复用 TaskArgs，均无加载器前缀；窗口 tensor 是否占槽及 ctx 的物理值不同。

## 8. 运行时时序

```text
L4 parent                                  MPI L3 ranks (rank 0 on parent host)
  compile all L4/L3/L2/kernel artifacts
  pack L3 Python + metadata bundle
  register CHIP callables / generated L3 orch handles
  configure startup_artifacts and startup_hooks_by_rank
  init: mpirun                 ----------> rank 0 reads and broadcasts payload
                                           validate / install importable modules
                                           install generated L3 + CHIP registries
                                           bind local CHIP handles to L3 modules
                                           prepare shared intermediates/token
                                           inner_worker.init() / L2 fork
  init returns after READY     <---------- all ranks prepared and L2 ready
  run(cluster_orch)
    allocate_domain            <---------> export/import/commit device windows
    route host tensors         ----------> remote buffers
    submit_next_level(handle)  ----------> registered node_orch binds TaskArgs
                                               get_domain(decoded_ref)
                                               add_chip_tensor_arg / Buffer.tensor
                                               append ordering token and device_ctx
                                               submit L2 -> kernel
  wait for DAG drain           <---------- completion
  copy remote OUT/INOUT        <---------- outputs
  next round reuses explicitly shared domain
  release buffers/domains and close workers
```

bundle 自动分发、fork 前 bootstrap、跨 program 域共享和两级域并存均需实现及验证。
PyPTO 的 bundle 与 hook 接入现有 MPI 启动接口，见 [04-runtime.md](04-runtime.md)
§ 7。同机与远端 MPI 节点使用相同的产物加载协议，节点按任务选择 L3 入口；每个 rank
收到相同集合的已注册 L2 CHIP 二进制。远端不提前部署本次生成物，也不重新编译。
