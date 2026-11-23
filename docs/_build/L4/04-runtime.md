# L4 运行时实现方案

新增 `python/pypto/runtime/cluster_runner.py`，负责 L4 驱动、产物分发接入和注册。
节点启动准备复用并扩展 `distributed_runner.py`；直接注册编译生成的 L3 orch。
`distributed_runner.py` 复用现有 CHIP 注册、host-control 元数据、域租约与共享
intermediates 逻辑，并适配逐节点下发和传入域状态。不能再按“两处小改”估算。

**复用同一个 `simpler.task_interface.TaskArgs` 容器**。张量和标量分别按形参
相对顺序排列，窗口参数在 L4→L3 不占 wire tensor 槽。callable handle 确定执行入口，
业务 TaskArgs 不增加 artifact digest 或加载器 ABI 前缀。

L4 不直接调用 collective。运行时只把全局域、数据与控制流交给 L3/L2。

## 1. 配置

### 1.1 `ClusterConfig` / `ClusterNodeSpec`

加在 `python/pypto/ir/distributed_compiled_program.py`，与 `DistributedConfig`
并列。**主路径采用 MPI 拉起 L3**，替换旧稿的 local_devices / remote_nodes + TCP
守护进程配置。以下是 PyPTO 待实现配置草案；Simpler 的 `MpiL3GroupSpec` 已存在。

```python
@dataclass
class ClusterNodeSpec:
    """One MPI L3 rank and the devices it owns on its host."""

    host: str
    devices: list[int] = field(default_factory=lambda: [0])


@dataclass
class ClusterConfig:
    """MPI ranks in declaration order; rank zero runs on the L4 parent host."""

    nodes: list[ClusterNodeSpec]
    python_executable: str
    comm_profile: str = "a3-fabric-v1"
    runtime: str = "tensormap_and_ringbuffer"
    num_sub_workers: int = 0
    startup_timeout_s: float = 120.0
    mpirun_path: str = "mpirun"
    mpirun_args: tuple[str, ...] = ()
    aicpu_thread_num: int = 0

    def node_device_counts(self) -> list[int]:
        """Device counts in MPI rank / DSL node-index order."""
        return [len(n.devices) for n in self.nodes]

    def global_device_ranks(self) -> list[tuple[int, ...]]:
        """Per-node global device ranks, distinct from MPI ranks."""
        ranks, base = [], 0
        for count in self.node_device_counts():
            ranks.append(tuple(range(base, base + count)))
            base += count
        return ranks

    def world_size(self) -> int:
        """Total devices in the config, not the HOST DSL world_size op."""
        return sum(self.node_device_counts())
```

`nodes` 和每组 devices 必须非空，同一主机的设备归属不能重叠。每个条目对应一个
L3 MPI rank；同一主机可有多个条目。MPI rank、DSL 节点索引、全局设备 rank 和
实际 worker id 是不同概念。启动参数必须保证 rank 放置与声明顺序一致。

当前 Simpler 通过本机 named mailbox 连接 L4 parent 与 MPI rank 0，**rank 0 必须
位于 L4 所在机器**。`python_executable` 是每台主机均可执行的 Python 或 launcher
路径；各机预装兼容的 Python、MPI/mpi4py、PyPTO/Simpler 运行环境。
不要求预装本次程序生成物、提前运行远端 daemon 或在远端执行 PyPTO/PTOAS 编译。
配置中的超时和 AICPU 线程数仍须接入已有运行时配置解析。

### 1.2 接入点

| 文件 | 改动 |
| ---- | ---- |
| `python/pypto/ir/compile.py` | 增 cluster_config；CLUSTER_0 entry 要求集群配置 |
| `python/pypto/runtime/runner.py` | RunConfig、CompileOptions、compile_options/as_compile_kwargs 同步增加 cluster_config；JIT compile 路径透传 |
| `distributed_compiled_program.py` | 按 `cluster_config` 是否存在，把执行分派给 `cluster_runner` 而非 `distributed_runner` |

`ir.compile` 的分布式判定（`compile.py`，`linqu_level >= 3`）本就覆盖
CLUSTER_0(=4)，无需改变这一判定。

用户侧采用 `backend_type` + `execution_mode`，不再新增 `platform=` 示例。
底层 Simpler Worker 仍接收由统一 target 解析得到的 platform 字符串；两者不混淆。
ClusterConfig 的配置校验、序列化、JIT 缓存身份和冷加载也要接入，不能仅修改 dataclass。

## 2. Worker 组装

当前锁定 Simpler 已提供 `Worker.add_mpirun_worker_group`。注册 worker group 后，
`Worker.init()` 内部启动 `mpirun ... -m simpler.mpi_l3_session`，无需用户先去远端
启动 L3。rank 0 读取 parent 本地 group manifest，再通过 MPI 广播给各 rank；
后续任务由 parent mailbox 经 rank 0 转发给目标 rank。
参考 `examples/workers/l4/global_tload_mpirun_l3`。

```python
# Proposed PyPTO wrapper around existing Simpler APIs.
def _construct_cluster_worker(cc, platform: str, artifacts, hooks_by_rank) -> tuple[Any, list[int]]:
    from simpler.worker import MpiL3GroupSpec, Worker

    w = Worker(level=4, num_sub_workers=0, remote_session_timeout_s=cc.startup_timeout_s)
    node_ids = w.add_mpirun_worker_group(
        MpiL3GroupSpec(
            hosts=tuple(n.host for n in cc.nodes),
            platform=platform,
            runtime=cc.runtime,
            device_ids_by_rank=tuple(tuple(n.devices) for n in cc.nodes),
            num_sub_workers_by_rank=tuple(cc.num_sub_workers for _ in cc.nodes),
            comm_profile=cc.comm_profile,
            global_device_ranks_by_rank=tuple(cc.global_device_ranks()),
            mpirun_path=cc.mpirun_path,
            mpirun_args=cc.mpirun_args,
            python_executable=cc.python_executable,
            startup_artifacts=tuple(artifacts),
            startup_hooks_by_rank=tuple(hooks_by_rank),
        )
    )
    return w, list(node_ids)
```

PyPTO 在组装 group 前准备 artifact 描述与逐 rank 的 hook 配置，见 § 7。
MPI 内部的广播/收集属于运行时启动协议，不放开 L4 DSL collective。

运行时**不组装 members 列表**——`allocate_domain` 的 `members=` 展开式由
codegen 从 `CommDomainScopeStmt::devices_` 发出（[03-codegen.md](03-codegen.md)
§ 2），与 L3 的 `members=` 同源。运行时只提供两个普通拓扑 kwarg，性质等同 L3 现有的
`world_size`：

```python
entry_kwargs = {
    "nodes": node_ids,                      # Actual L3 worker IDs in topology order.
    "node_sizes": cc.node_device_counts(),  # Local device counts in the same order.
}
```

生成代码里的 `members=[(nodes[n], d) for n in range(num_workers) for d in
range(node_sizes[n])]` 因此展开成 `(l3_worker_id, local_l2_id)` 对，顺序决定该域的 domain rank。子集域（`devices_=[0, 2]`）自然发成 `for n in (0, 2)`，无需运行时参与。

## 3. 注册与节点启动顺序

目标是 **仅 L4 机器统一编译，MPI 启动期间自动分发 L3 Python 编排产物**。
远端只加载和执行产物，不运行 PyPTO 编译器。当前基础能力与新增工作如下：

| 对象 | 当前 Simpler 行为 | 本方案 |
| ---- | ---------------- | ------ |
| L3 进程 | MPI 自动拉起 | 复用，无需远端常驻 daemon |
| L2 CHIP callable | init 前注册的二进制随 manifest 下发并安装 | 复用现有路径 |
| L3 Python callable | `PYTHON_IMPORT` 导入目标机器可见的符号，不传源码 | bundle 安装后直接导入生成的 L3 orch |
| 生成的 L3 Python / metadata | MPI group 支持 startup artifacts 与逐 rank 的 hook | 打包生成物并实现 PyPTO 节点准备函数 |

启动顺序（包含待实现部分）：

1. L4 机器编译完整 L4/L3/L2/kernel 子树，打包 L3 bundle，并将全部依赖 CHIP
   callable 注册到 parent。多 round 使用的产物在本次 prepare/init 前一并准备。
2. 创建 MPI group，逐节点注册生成的 L3 orch 的 `RemoteCallable(module:qualname)`，
   保存返回的 handle；group spec 携带 bundle 描述和准备参数。
   此时 parent 记录注册，远端实际 import/install 在 init 阶段进行。
3. `Worker.init()` 拉起 MPI ranks。rank 0 读取本机 bundle，通过 MPI 分发字节；
   各 rank 校验摘要和版本、原子安装并建立可导入的模块路径，然后才能安装 Python
   dispatcher 注册表。生成模块按产物版本隔离，不能覆盖另一版本的同名模块。
4. 各 rank 创建 `inner_worker`，安装 **L2 CHIP callable** 与生成的 **L3 orch**
   注册表；PyPTO 启动准备逻辑按 CHIP digest 绑定本地句柄、恢复 host_tensor_indices，
   为生成模块绑定节点状态，并在 L2 children fork 前准备共享 intermediates/token。
5. 调用 `inner_worker.init()` 启动本地 L2；所有 rank 的安装、准备及 L2 初始化
   均成功后才发布 READY，parent 的 init 才返回。失败处理见 § 7。
6. 执行本地生成的 L4 orch；其中直接调用
   `orch.submit_next_level(l3_handle, task_args, config, worker=node_id)`。
   Simpler 按已注册 handle 执行对应生成的 L3 orch，没有统一转调入口。

注册与下发示意（`register` / `submit_next_level` 为现有 API；生成模块及分发待实现）：

```python
from simpler.worker import RemoteCallable

# Produced from the bundle's versioned module namespace and entry metadata.
remote_compute_handle = worker.register(
    RemoteCallable(REMOTE_COMPUTE_TARGET), workers=[remote_node],
)
# init must install the bundle before importing REMOTE_COMPUTE_TARGET.
worker.init()

# Inside the generated L4 orchestration, after its task arguments are built:
orch.submit_next_level(remote_compute_handle, remote_task, cfg, worker=remote_node)
```

`REMOTE_COMPUTE_TARGET` 直接指向生成模块中的 `node_orch`，例如
`pypto_artifacts.p_<bundle_digest>.node_orch:node_orch`（命名示意，前缀保证合法标识符）。
产物版本纳入模块名，因为现有 PYTHON_IMPORT callable 身份由 module/qualname 决定，
不对远端 Python 源码做内容散列。bundle 内容校验仍由启动传输协议完成。
对每个目标节点重复注册并保存 handle；各节点的 L3 函数可以相同，也可以不同。

步骤 4 安装的有两类注册表，不能混淆：Python dispatcher 执行 L3 入口，
inner registry 保存给 L2 执行的 CHIP callable。当前 `_inner_registry_entries_for_spec`
会把 parent 在 init 前注册的**全部 LOCAL_CHIP callable**放进每个 rank 的 manifest，
因此各 L3 收到同一组 L2 二进制；本地句柄可以不同，实际执行由各自 L3 编排决定。
Python 入口则可通过 `workers=` 限定节点。按节点裁剪 CHIP 集合不在首版范围。

当前 Python 远端注册也支持 init 后的 prepare/commit；这里选择启动前统一注册和
准备，是为了满足共享存储生命周期，不是宣称所有 callable 都只能在 init 前注册。
`PYTHON_SERIALIZED` 仍被 remote protocol v1 拒绝；本方案不依赖它。

## 4. 直接下发与任务参数

L4 的 callable 表按 L3 callee 和节点索引保存注册返回的 handle，`nodes[w]` 是
该索引对应的实际 L3 worker id。生成代码直接下发：

```python
orch.submit_next_level(
    callables["node_orch"][w], task_args, config, worker=nodes[w],
)
```

L3→L2 继续复用当前 `_submit_chip`，保留 worker 解析、DFX、TaskId 返回和
`_ChipCallables.host_tensor_indices`。共享 codegen 的任务组装逻辑不要求额外引入
运行时 `_submit_next` 包装；下发语句按调用层级发射，依赖等待和返回值仍需保留。

| 区域 | 内容 |
| ---- | ---- |
| tensor pool | 普通张量，按逻辑形参顺序，跳过 window 形参 |
| scalar pool | `DistributedTensor` 发射 1 个 `buffer_index`，用户标量各占 1 槽，按业务形参相对顺序排列；末尾的每个 `CommCtx` 发射 3 个连续 uint64 |

保持 TaskArgs 的 tensor-before-scalar 要求。生成的 L3 orch 按编译时确定的参数
布局直接绑定 TaskArgs；窗口参数按 slot/ctx 信息恢复。入口由 callable handle
确定，不增加二次选择入口的 artifact digest 或版本前缀。元数据、启动协议版本
在安装阶段检查，参数数量/类型在执行前检查。

## 5. host tensor 路由

`_io_arg` 属于 L4→L3；它需要节点 binding、tensor 和方向。以下为目标伪代码：

```python
def _io_arg(node, tensor, tag):
    handle = node.remote_buffer_for(tensor)
    if is_input_or_inout(tag):
        node.parent_worker.remote_copy_to(handle, tensor.data_ptr(), tensor.nbytes)
    if is_output_or_inout(tag):
        node.record_pending_output(handle, tensor)
    return RemoteTensorRef(handle=handle, shape=tuple(tensor.shape), dtype=_dtype_of(tensor))
```

`is_input_or_inout` 等是待实现的方向判断；区分 OUTPUT/OUTPUT_EXISTING/INOUT。
MPI group 中的所有 L3（包括同机 rank 0）均走 remote-like binding。MPI 启动的进程
不是 parent fork 出的共享内存 L3，不能按主机名相同就走 `make_tensor_arg(parent_worker, ...)`。
未来若增加共享存储优化，必须显式验证该能力。现有 `remote_malloc/copy` 支持 MPI
worker id，但只能在 init 后使用，不能替代 fork 前的 bundle/bootstrap 通路。
每个 binding 持有实际 worker id 与 parent worker，`remote_buffer_for` 调用
`remote_malloc(worker=node.worker_id, nbytes=...)`。回填在对应 run 排空后进行。
缓存需要区分 tensor view 的范围、偏移及生命周期，不得仅按临时 Python 对象 id 复用。

L3→L2 则保留当前 `add_chip_tensor_arg`，它在 `make_tensor_arg(worker, tensor)`
之上按 callee 元数据设置 host control 参数的 `TensorTransfer.NONE`。L4 不调用它。

大张量尽量留在窗口中跨 round 复用。OUT 无需预先 copy_to；IN 无需 copy_from。
远端输入/输出拷贝和 keepalive 也需覆盖非连续 view 的处理或明确拒绝策略。

## 6. 生成的 L3 入口与 metadata

嵌入 L4 的 L3 orch 直接生成符合 Simpler 调用约定的
`def node_orch(orch, args, config)`，函数体包含原有编排逻辑。其入口处绑定业务
TaskArgs 和启动时准备的本地状态，然后执行循环、建域和 L2 下发。
不生成统一 `node_entry` 或调用其他目标函数的 trampoline。

当前独立 L3 的生成入口还有 `tensors`、`callables`、`world_size` 等 keyword 参数，
因此嵌入 L4 的入口需要 codegen 适配，不能声称现有入口无需改动就能直接注册。
目标绑定方式见 [06-example-lowering.md](06-example-lowering.md) § 5：

- 每次调用的张量、标量和 ctx 从 TaskArgs 绑定；窗口不占 wire tensor 槽。
- CHIP handles、host-control 元数据、拓扑和共享存储由启动准备逻辑绑定到生成模块；
  状态按 session 管理，不能把 parent 的 Python 对象或指针直接传给远端。
- TaskArgs/张量及节点状态保持到 DAG 排空，session 结束时清理；不得无限积累 keepalive。
- 独立 L3 的既有入口 ABI 保持兼容，嵌入 L4 的入口选择记录在 metadata 中。

当前 `DistributedCompiledProgram` 的 `distributed_meta.json` 使用 **schema 3**，
而非旧稿的 schema 1。现有参数记录包含 name/direction/shape/dtype/layout；
顶层还包含 supported_execution_modes（产物消费方式，如 program，并非 SIM/ONBOARD）
等字段。`from_dir` 严格检查版本。

**当前并不存在** window/comm_ctx 的 `kind`、`slot`、配对索引字段。方案需新增
节点参数描述及绑定器：

| 逻辑形参 | wire 来源 | 额外元数据 |
| -------- | --------- | ---------- |
| 普通 Tensor | tensor pool 对应位置 | shape/dtype/layout/direction |
| 用户 Scalar | scalar pool | dtype、位置 |
| DistributedTensor | scalar pool 中的单个 `buffer_index` | 记录该 scalar 偏移、配对 ctx 的逻辑形参序号及视图布局 |
| CommCtx | scalar pool 中连续 3 个 uint64 | 配对形参、ctx 起始槽和宽度 3。依次编码 `owner_instance_id`（8 字节 little-endian）、`domain_id`、`generation` |

同一已注册入口可以先接收域 A 的第 1 个下标，再接收域 B 的第 0 个下标。对每个窗口形参，L3 读取该次调用的 ref 和 buffer_index，查 `domain = orch.get_domain(ref)`，验证 `0 <= buffer_index < len(domain.buffers)`，取 `slot = domain.buffers[buffer_index].name`。目标 chip 的 Tensor 从 `domain[w].buffers[slot]` 创建；必须检查实际视图类型约束和切片范围。一次调用可用同一 ref 的不同槽，或使用多个域，入口的绑定均为本次调用局部状态。

节点入口 ABI 对布局进行版本化。打包、加载时校验窗口的独立 scalar 偏移、ref 的起始槽和宽度，拒绝不匹配的布局。跨 program 的稳定物理槽映射由调用方应用，L3 入口与具体域布局解耦。

优先将节点扩展放在显式版本化的 `l3_entry` 块中，保留当前 schema 3 的公共字段；
若实际改动公共参数结构的兼容语义，则同步升级 schema 和 loader。具体版本号随实现
确定，不能只把示例中的 1 改成 3 就认为完成适配。扩展块读写及老 L3 产物兼容要测试。

CHIP callable 的 host_tensor_indices 来自 L2 artifact/runtime config，必须一路保留到
节点的 `_ChipCallables`；它与节点 window/ctx 元数据是不同职责。

## 7. L3 bundle 自动分发与节点准备

### 7.1 产物及职责边界

PyPTO 在 L4 机器生成内容寻址的 bundle，包含所需 L3 入口子树的
`orchestration/*.py`、节点 metadata，以及版本化 manifest。manifest 描述入口、
相对路径、参数/window/ctx 映射、依赖 CHIP digest、host-control 元数据和共享状态
准备信息；动态形状或调用相关的准备参数随启动描述传入，不能假设全是编译期常量。
bundle 对规范化文件顺序及内容计算 SHA-256，摘要由外层启动描述携带，避免自引用。
多入口可打包在同一载荷中。bundle digest 用于安装校验、缓存及模块命名空间隔离；
manifest 为每个入口记录相对模块名、qualname 和参数布局。打包后根据 bundle
digest 确定外层模块命名空间，再组成最终导入目标，避免摘要参与自身计算。parent
用该导入目标注册 callable 并保存 handle，任务按 handle 直接执行。

只打包运行 L3 所需的生成模块与元数据；L2 二进制沿用 CHIP manifest 路径，不在
bundle 中重复运输，也不发送编译环境或序列化用户闭包。当前 loader 若依赖完整
L2 目录，必须改为从 digest/已安装句柄及必要元数据恢复，不能触发远端重新编译。

| 层 | 责任 |
| -- | ---- |
| PyPTO parent | 统一编译、收集依赖、确定性打包、注册 CHIP 和生成的 L3 orch、提交启动描述 |
| Simpler | 通用 artifact 载荷传输、安装协议、启动 hook 生命周期、就绪及失败聚合 |
| PyPTO 启动准备逻辑 | 解释 metadata、绑定生成模块的 CHIP 句柄和共享状态 |
| 生成的 L3 orch | 直接接收 TaskArgs、绑定业务参数、执行编排并下发 L2 |

Simpler 无需理解 PyPTO IR 或 window/ctx 绑定；PyPTO 的启动准备逻辑和生成代码负责这些语义。
节点执行 Python 模块加载不等于重新进行 PyPTO/PTOAS 编译。

### 7.2 MPI 启动接口

用 `StartupArtifactSpec(archive_path, sha256, module_prefix)` 描述 parent 本地 bundle，在 `MpiL3GroupSpec.startup_artifacts` 中提交。通过 `startup_hooks_by_rank` 为每个 rank 配置 `NodeStartupHook(import_path, parameters)`。入口为可导入的 `module:qualname`，参数为 JSON 对象。

Simpler 依次安装启动产物、建立注册表、执行 hook、初始化本地 worker。PyPTO 的准备函数从 `NodeStartupContext` 获取 `rank`、`artifact_roots`、`inner_worker`、按 digest 索引的 `chip_handles` 及 `parameters`，绑定本 session 的模块状态，并在 L2 fork 前准备共享存储。集成测试覆盖空缓存、命中缓存、摘要错误、hook 失败和启动清理。[MPI 启动顺序](https://github.com/hw-native-sys/simpler/blob/d2b37652d/python/simpler/mpi_l3_session.py#L574-L601)

## 8. persistent 域租约

PyPTO 主线的 provider 按 `(program_id, generated_name)` 隔离域，每次 run 均调用 `orch.get_or_allocate_domain(..., retain_after_run=True)`，再用 `_RetainedDomainLease` 包装 handle。L4 适配要直接接收 `members`；仍使用 `workers` 的 L3 调用在 provider 入口规范化一次。临时域可直接使用 `with orch.allocate_domain(...)`；持久路径使用 `with provider(...)`，由 lease 在退出时保留底层域。直接对原始 handle 使用 `with` 会释放域。[L3 provider 实现](https://github.com/hw-native-sys/pypto/blob/ca08d06bd/python/pypto/runtime/distributed_runner.py#L2472-L2491)

跨 program 共享须显式选择同一个 owner、域名和稳定的物理槽名。调用方将各 program 的 SSA 槽名映射到物理槽，使用一致的有序成员、窗口大小、buffer 完整配置和保留策略请求复用。未声明共享时继续按 program 隔离。共享键的用户侧拼写待定；同名配置冲突应报错。复用命中保留数据；`reset_persistent_windows` 的显式清零由 PyPTO 控制。

L4 只下传 ref 标量的 run 中，创建者通过当次 provider 复用或 `orch.get_domain(ref)` 将使用登记到该 run；子节点的 lookup 不能代替登记。借用 handle 仅能用于已获授权的节点，释放由创建者执行。排序状态按完整 ref 和参与成员复用；同名重建后旧 ref 失效。

## 9. 生命周期与清理

```python
try:
    ...
    w.run(entry_fn, args=None, config=call_config)
    _drain_remote_outputs(node_bindings)  # Copy back only after successful drain.
finally:
    _release_remote_buffers(node_bindings)             # remote_free
    if domain is not None and not domain.freed:
        domain.release()                               # retain_after_run 的显式释放
    w.close()                                          # 兜底：释放所有残留域与会话
```

对应用例 `main.py` 的 `finally` 块。`Worker.close()` 是最后一道防线，但不能
依赖它做常规释放——`retain_after_run=True` 的域必须显式 `release()`。
