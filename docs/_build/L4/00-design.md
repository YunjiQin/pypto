# L4 设计

## 1. 层级对齐

simpler 的 Worker 树与 PyPTO 的 `ir::Level` 已经对齐，无需引入新概念：

| simpler | PyPTO `ir::Level` | `LevelToLinquLevel` | DSL 声明 | 现状 |
| ------- | ----------------- | ------------------- | -------- | ---- |
| L2（一张 NPU） | `CHIP` | 2 | `@pl.jit` | 已有 |
| core | `AIV` / `AIC` | 0 | `@pl.jit.incore` | 已有 |
| L3（一台机器） | `HOST` | 3 | `@pl.jit.host` | 已有 |
| **L4（跨机）** | **`CLUSTER_0`**（别名 `POD`） | **4** | `@pl.jit.pod`（新增） | **本方案** |

依据：`include/pypto/ir/function.h`（Level 枚举与 `LevelToLinquLevel`）、`python/pypto/language/parser/enum_utils.py`
（`"CLUSTER_0"` 已映射）、`python/pypto/ir/compile.py`
（`linqu_level >= 3` 判定为分布式程序）。

**结论**：CLUSTER_0 已经能被前端解析、被 compile 识别为分布式程序。缺的是
pass 闸门、codegen 分派与运行时驱动。

## 2. 与 L3 参考用例的逐层对应

以 simpler `examples/workers/l4/compute_then_tload_mixed_l3` 为基准：

```text
PyPTO function                        对应用例文件 / 函数
──────────────────────────────────────────────────────────────────────
cluster_orch   (CLUSTER_0)            main.py: compute_phase / communication_phase
node_orch      (HOST)                 main.py: local_compute_orch / remote_compute_orch
chip_orch      (CHIP)                 kernels/orchestration/local_add_orch.cpp
                                                            global_tload_orch.cpp
kernel         (InCore)               kernels/aiv/local_add_kernel.cpp
                                                 global_tload_kernel.cpp
```

`node_orch` 复用 L3 编排语言与 L3→L2 下发路径。已有程序接入 L4 时，可以继续
自建本地域；需要参与全局通信的窗口改由 L4 创建并作为形参传入。

**L4 不允许直接调用 collective**（包括 tensor/tile collective）。L4 只创建域、
组织节点级控制流并下发 L3；L3/L2 表达通信，设备 kernel 执行数据交换。
L4 对 collective 必须报错，不能通过放宽 HOST collective lowering 的闸门来支持。

## 3. 统一的层级词汇：`worker=`

下发 kwarg 在**每一层都叫 `worker=`**，含义恒定为"下一级的哪一个子节点"——与
simpler 的 `orch.submit_next_level(..., worker=<id>)` 一致：

| 调用边 | `worker=` 的含义 |
| ------ | ---------------- |
| CLUSTER_0 → HOST | L3 节点索引 |
| HOST → CHIP | 本节点内的 L2 索引 |

不区分 `node=` / `device=`——层级已经由 caller 的 `level_` 唯一确定，再造一个拼写
只会制造两套等价语义。循环上界同理统一为 `pld.num_workers()`（下一级子节点数）：

| 内建 | 层 | 含义 |
| ---- | -- | ---- |
| `pld.num_workers()` | 任一编排层 | 下一级子节点数（L4：节点数；L3：本节点卡数） |
| `pld.world_size()` | HOST | 保留当前本节点设备数语义；不作为全局域大小查询，L4 不开放 |
| `pld.nranks(ctx)` | 设备侧 | 指定通信域的 rank 数，本地域与全局域可不同 |

纯 L3 程序里 `num_workers() == world_size()`。在嵌套 L3 中两者仍表示本节点设备数，
不改写成全局 rank 数。全局通信窗口/信号的形状由 L4 配置、张量形参或显式标量
传入 L3/L2；设备端用 `nranks(ctx)` 获取实际域大小。
例如两节点各四卡：L4 `num_workers()=2`，L3 `num_workers()=4`，全局域 `nranks(ctx)=8`。

> **迁移**：`device=` 是既有拼写，仓库内有大量使用。方案采用
> `worker=` 为正名、`device=` 保留为带 `DeprecationWarning` 的别名一个迁移窗口。
> 别名删除时机见 [05-plan.md](05-plan.md) § 待决问题。

## 4. 运行时约束与自动分发

从 simpler 源码与协议文档读出，直接决定设计形态：

### (a) 远端 Python callable 只能 `PYTHON_IMPORT`

`docs/remote-l3-worker-design/protocol.md`：`REMOTE_TASK_DISPATCHER` /
`INNER_L3_WORKER` 的必备基线是 `PYTHON_IMPORT`（`"module:qualname"` 字符串），
`PYTHON_SERIALIZED`（cloudpickle）标注为 "optional negotiated extension"。

→ 远端需要**可导入的生成模块**，不要求固定的统一入口。先将生成的 L3 Python
分发并安装，再由 manifest 注册其导入目标；parent 通过
`worker.register(RemoteCallable(generated_target), workers=[node])` 获得 handle。
L4 orch 直接 `orch.submit_next_level(handle, args, cfg, worker=node)`。

`RemoteCallable` 本身不传源码，由 PyPTO 将生成模块打包为 MPI startup artifact；模块按产物版本命名，
保证不同版本的 callable 身份不冲突。生成的 L3 函数直接遵守 Simpler 调用约定，
在函数体入口绑定业务参数和本地状态。无需统一 trampoline 或任务中的 artifact 标识。

### (b) chip 二进制随会话自动下发

同文档："Pre-init `ChipCallable` registrations on an L4 worker are serialized
into each remote session manifest and installed on that L3's L2 children."

→ PyPTO 编出的 kernel `.o` 与 orchestration `.so` 无需手工同步到对端，但
本方案在 `Worker.init()` 前统一 register，确保进入启动 manifest。
各 L3 rank 收到 parent 已注册的同一组 L2 CHIP callable；各自编排选择实际执行的
callee。它们与 L3 Python dispatcher 的注册表不同，也不要求本地句柄相同。

### (c) 跨机没有共享内存

L3→L2 的普通张量走 `add_chip_tensor_arg(...)`，内部调用
`make_tensor_arg(orch._worker, tensor)` 并按 callee 元数据保留 host control 参数的
`TensorTransfer.NONE`。共享 host 存储须在 fork 前准备。L4→L3 不调用这个 CHIP
helper：MPI 启动的所有 L3 节点统一走 Simpler 的对偶通路：
`Worker.remote_malloc(worker=, nbytes=)` → `remote_copy_to` →
`RemoteTensorRef(handle, shape, dtype)` → `TaskArgs.add_tensor(...)`
（`python/simpler/worker.py`、`task_interface.py`）。

→ **DSL 零差异**：普通 `pl.Tensor` 参数照常传给下一级，节点 binding 负责传输。
同机 MPI rank 也不是 parent fork 出的 L3，不能仅凭 hostname 推断共享内存可用。

### (d) MPI 启动期间分发 L3 产物

现有 `MpiL3GroupSpec` / `add_mpirun_worker_group` 在 init 时自动启动 MPI L3，无需
提前启动远端 daemon。当前 rank 0 必须位于 L4 机器，用本机 mailbox 与 parent
通信，再通过 MPI 分发任务；运行时 MPI collective 不改变 L4 DSL 的禁用约束。

L4 机器编译完整程序。PyPTO 将 L3 Python 和 metadata 打包，通过 `startup_artifacts` 提交给 MPI group；各 rank 安装产物，执行 `startup_hooks_by_rank` 配置的准备函数，再初始化本地 L2。见 [运行时设计](04-runtime.md) § 7。

## 5. 通信域模型

### 5.1 域只在建它的那一层有 scope

`CommDomainScopeStmt` **只出现在建域的那个函数里**。下层拿到域的方式与 L3 向 L2
传递的方式完全一致：**通过 `CommCtx` 参数**。

这正是 `MaterializeDistTensorCtx`（pass 48）已经在做的事——为每个
`DistributedTensorType` 形参在签名尾部追加一个 `CommCtxType` 形参，并在调用点
转发调用方的 ctx 形参、或就地 `pld.system.get_comm_ctx(dist)` 绑定。L4→L3 这条边
复用同一个 pass 的参数追加与别名追踪，但需把 `IsHostOrch` 限定的运行时查询
放宽到 HOST/CLUSTER_0，并同步 `DistTensorCtxMaterialized` verifier。
当前源码只允许 HOST 为本层创建的窗口查询 ctx；不改这处，L4→L3 传窗口会被拒绝。

L3 和 L4 都调用 `orch.allocate_domain(name=..., members=..., window_size=..., buffers=...)`。L3 的成员是本地 chip worker 下标，L4 的成员是 `(node_worker_id, local_worker_id)`。两层的 handle 都提供 `domain.ref`。成员的声明顺序决定 domain rank。

| 调用边 | IR 形参 | TaskArgs 与接收端 |
| ------ | ------- | ---------------- |
| L4→L3（每节点一次） | 窗口 `DistributedTensor` 和配对的 `CommCtx` | 窗口占 0 个 tensor 槽；`DistributedTensor` 实参自身编码为一个 `buffer_index` 标量，尾部配对 `CommCtx` 独立编码为 owner_instance_id、domain_id、generation 三个 uint64。参与节点按各自的 scalar 偏移解码，再用 `domain = orch.get_domain(ref)` 和 `slot = domain.buffers[buffer_index].name` 选取窗口。 |
| L3→L2（每 chip 一次） | 同一对逻辑形参 | 以本地 chip 下标 `w` 取 `ctx = domain[w]`，用 `ctx.buffers[slot].tensor(...)` 构造窗口视图，传该 Tensor 和一个 `ctx.device_ctx` 标量。 |

例如成员为 `[(A, 0), (B, 0), (B, 1)]` 时，B 的两颗 chip 下标为 0、1，对应 domain rank 1、2。L4 给 B 的一次调用为每个窗口形参传一份 `(ref, buffer_index)`，L3 在循环中逐 chip 选窗口。需要 domain rank 时读 `ctx.domain_rank`。自建域和借用域均按此方式下发，只有创建者负责释放；仅收到 ref 的其他节点无法据此导入域。

### 5.2 每一层都可以自建域

L4 程序里的 `node_orch` 可以既接收 L4 建的全局域（ctx 形参），又自建一个只跨本节点
若干卡的本地域——两级 collective 分解正是分层的主要动机：

```python
@pl.jit.host
def node_orch(
    x: pl.Tensor[[NW, 1, COUNT], pl.FP32],
    inter: pl.InOut[pld.DistributedTensor[[1, COUNT], pl.FP32]],   # 上层传入（ctx 形参随之追加）
    out: pl.Out[pl.Tensor[[NW, 1, COUNT], pl.FP32]],
) -> pl.Tensor[[NW, 1, COUNT], pl.FP32]:
    intra_buf = pld.alloc_window_buffer(COUNT * pl.FP32.get_byte())   # 自建：本节点域
    for w in pl.range(pld.num_workers()):
        chip_orch(
            x[w],
            pld.window(intra_buf, [1, COUNT], dtype=pl.FP32),   # 节点内窗口
            inter,                                              # 跨节点窗口
            out[w],
            worker=w,
        )
    return out
```

自建域走 `CommDomainScopeStmt`（本函数内），传入域走 ctx 形参——两者互不干扰，
IR 里也不需要"继承 scope"这类新概念。

### 5.3 跨 run 存活

沿用 L3 persistent 机制（`docs/en/dev/06-persistent-l3.md`）：
`prepare(persistent=True)` 注入 `_domain_provider`，生成入口原样接受该 kwarg；
两层的 provider 每次 run 都以同一配置调用 `get_or_allocate_domain(..., retain_after_run=True)`，并返回退出时保留物理域的租约。后续 run 若只传 ref 标量，创建者须在当前回调中复用或查找该域，以登记本次使用。
`reset_persistent_windows=False` 对应"下一次 run 直读上一次写入窗口的内容"。

**待定**：租约索引键需从 `(compiled program, 生成域名)` 放宽，见
[05-plan.md](05-plan.md) § 待决问题。

## 6. 程序边界与相位

**一个 compiled program = 一次 dispatch = 一次 `worker.run`。** 不在 program 内做
相位切分。参考用例的两个 round 对应两个 program，在同一个持久 worker 上顺序
dispatch：`rt.run(...)` 返回即该 round 的 DAG 已排空，天然是相位边界。

## 7. 数据进出窗口

Simpler 的设备内存域在两层均提供 `copy_to_domain` / `copy_from_domain`。本方案的 DSL 通过 L2 task 填写和读取窗口，与 L3 quickstart
（`docs/en/user/distributed/00-model.md` § Quickstart 第 1 步 "Stage-in"）一致。

## 8. 目标示例：改写 `compute_then_tload_mixed_l3`

以下为待实现 DSL 的目标源码；`jit.pod`、`worker`、`num_workers` 和 cluster_config
接入尚未实现，不能当作当前可运行示例。`06-example-lowering.md` 展示对应 IR/产物。

```python
import pypto.language as pl
import pypto.language.distributed as pld
from pypto.backend import BackendType
from pypto.runtime import ExecutionMode, RunConfig

COUNT = 256
NN = pl.dynamic("NN")
ND = pl.dynamic("ND")

@pl.jit.incore
def add_kernel(
    lhs: pl.Tensor[[1, COUNT], pl.FP32],
    rhs: pl.Tensor[[1, COUNT], pl.FP32],
    data: pl.InOut[pld.DistributedTensor[[1, COUNT], pl.FP32]],
    echo: pl.Out[pl.Tensor[[1, COUNT], pl.FP32]],
) -> pl.Tensor[[1, COUNT], pl.FP32]:
    a = pl.load(lhs, [0, 0], [1, COUNT])
    b = pl.load(rhs, [0, 0], [1, COUNT])
    s = pl.add(a, b)
    data = pl.store(s, [0, 0], data)
    return pl.store(s, [0, 0], echo)

@pl.jit
def chip_orch(
    lhs: pl.Tensor[[1, COUNT], pl.FP32],
    rhs: pl.Tensor[[1, COUNT], pl.FP32],
    data: pl.InOut[pld.DistributedTensor[[1, COUNT], pl.FP32]],
    echo: pl.Out[pl.Tensor[[1, COUNT], pl.FP32]],
) -> pl.Tensor[[1, COUNT], pl.FP32]:
    return add_kernel(lhs, rhs, data, echo)

@pl.jit.host
def node_orch(
    lhs: pl.Tensor[[ND, 1, COUNT], pl.FP32],
    rhs: pl.Tensor[[ND, 1, COUNT], pl.FP32],
    data: pl.InOut[pld.DistributedTensor[[1, COUNT], pl.FP32]],
    echo: pl.Out[pl.Tensor[[ND, 1, COUNT], pl.FP32]],
) -> pl.Tensor[[ND, 1, COUNT], pl.FP32]:
    for w in pl.range(pld.num_workers()):
        chip_orch(lhs[w], rhs[w], data, echo[w], worker=w)
    return echo

@pl.jit.pod
def cluster_orch(
    lhs: pl.Tensor[[NN, ND, 1, COUNT], pl.FP32],
    rhs: pl.Tensor[[NN, ND, 1, COUNT], pl.FP32],
    echo: pl.Out[pl.Tensor[[NN, ND, 1, COUNT], pl.FP32]],
) -> pl.Tensor[[NN, ND, 1, COUNT], pl.FP32]:
    data_buf = pld.alloc_window_buffer(COUNT * pl.FP32.get_byte())
    for w in pl.range(pld.num_workers()):
        node_orch(
            lhs[w], rhs[w], pld.window(data_buf, [1, COUNT], dtype=pl.FP32),
            echo[w], worker=w,
        )
    return echo
```

第二个 round 由 L2/kernel 读取全局域中的各 rank 数据。前一个 run 已排空，
所以本例不需要在同一 run 内额外协调 stage-in 与 remote_load：

```python
@pl.jit.incore
def tload_kernel(
    data: pl.InOut[pld.DistributedTensor[[1, COUNT], pl.FP32]],
    out: pl.Out[pl.Tensor[[1, COUNT], pl.FP32]],
) -> pl.Tensor[[1, COUNT], pl.FP32]:
    ctx = pld.get_comm_ctx(data)
    me = pld.rank(ctx)
    n = pld.nranks(ctx)
    acc = pl.load(data, [0, 0], [1, COUNT])
    for peer in pl.range(n):
        if peer != me:
            acc = pl.add(
                acc,
                pld.tile.remote_load(data, peer=peer, offsets=[0, 0], shape=[1, COUNT]),
            )
    return pl.store(acc, [0, 0], out)

@pl.jit
def comm_chip_orch(
    data: pl.InOut[pld.DistributedTensor[[1, COUNT], pl.FP32]],
    out: pl.Out[pl.Tensor[[1, COUNT], pl.FP32]],
) -> pl.Tensor[[1, COUNT], pl.FP32]:
    return tload_kernel(data, out)

@pl.jit.host
def comm_node_orch(
    data: pl.InOut[pld.DistributedTensor[[1, COUNT], pl.FP32]],
    out: pl.Out[pl.Tensor[[ND, 1, COUNT], pl.FP32]],
) -> pl.Tensor[[ND, 1, COUNT], pl.FP32]:
    for w in pl.range(pld.num_workers()):
        comm_chip_orch(data, out[w], worker=w)
    return out

@pl.jit.pod
def comm_cluster_orch(
    out: pl.Out[pl.Tensor[[NN, ND, 1, COUNT], pl.FP32]],
) -> pl.Tensor[[NN, ND, 1, COUNT], pl.FP32]:
    data_buf = pld.alloc_window_buffer(COUNT * pl.FP32.get_byte())
    for w in pl.range(pld.num_workers()):
        comm_node_orch(pld.window(data_buf, [1, COUNT], dtype=pl.FP32), out[w], worker=w)
    return out
```

驱动目标：

```python
from pypto.ir.distributed_compiled_program import ClusterConfig, ClusterNodeSpec

cc = ClusterConfig(
    nodes=[
        ClusterNodeSpec(host="localhost", devices=[0]),
        ClusterNodeSpec(host="192.0.2.20", devices=[0]),
    ],
    python_executable=python_launcher,
    comm_profile="a3-fabric-v1",
)
config = RunConfig(
    backend_type=BackendType.Ascend910B,
    execution_mode=ExecutionMode.ONBOARD,
    cluster_config=cc,
)
compute = cluster_orch.compile(config=config)
comm = comm_cluster_orch.compile(config=config)

# A proposed explicit cross-program domain-sharing contract is also required.
with compute.prepare(
    extra_compiled=[comm], persistent=True, reset_persistent_windows=False,
) as rt:
    rt.run(compute, lhs, rhs, echo)
    rt.run(comm, res)
```

`python_launcher` 由运行环境提供，须在各机可执行；MPI rank 0 位于 L4 机器。
`lhs/rhs/echo/res` 由调用者按拓扑准备；节点内的共享 intermediates/token 则由启动
hook 在 L2 fork 前准备。上面 prepare 形态沿用当前 API；**它目前不保证两个不同 program 自动共享域**。
要实现该两 round 目标，还必须完成 [04-runtime.md](04-runtime.md) § 8 的显式共享
约定与槽映射，再把最终共享参数补入此例。不能靠相同 nbytes 隐式合并所有域。

L4 两个入口都只有建域、循环和 L3 下发，没有 collective。

## 9. 全局 rank 编号

全局设备 rank 按 `nodes` 声明序、各条目的 devices 顺序分配。每个通信域独立按其 `members` 顺序分配 domain rank；MPI rank 0 在本机。
`ClusterConfig` 据此计算每个节点的 `global_device_ranks`（对应用例中手写的
`(0,)` / `(1,)`）。

## 10. 明确不做

| 不做 | 原因 |
| ---- | ---- |
| L4 直接调用 collective | 仅负责建域与节点下发，通信留在 L3/L2 |
| L4 直发 L2 的语法糖 | 破坏层级链，节点级语义无处表达 |
| `copy_to/from_global_domain` 前端接口 | DSL 统一经 L2 task 填写和读取窗口 |
| program 内多相位切分 | program 边界 = 一次 `worker.run` |
| 下层 IR 里的"继承 scope" | 域通过 ctx 形参传递，与 L3→L2 完全一致 |
| 新 TaskArgs 类型或加载器任务封装 | 复用运行时容器，callable handle 已标识目标 L3 orch |
| 远端节点本地 SubWorker 回调 | 需要对端存在用户 Python 代码，与约束 (a) 冲突 |
