# Runtime：创建、提交、排序与释放

本文件的 Python helper 名称均为拟议实现。复用 Simpler 的 domain/ref/fence 能力；不访问旧版 `_live_domains` 私有字典来伪造新的域生命周期。

## 1. 状态分层

```text
DistributedWorker
  internal persistent domains     # existing per-program ownership
  external domain registry        # worker-owned CommDomain records
  prepared program states         # immutable entry ABI and callable handles
  dispatch frames / run handles   # per-call args, leases, cleanup
```

registry 按完整 `CommDomainRef` 索引。可另有创建事务 ID，但不能以 Python id 或域标签替代物理身份。
DomainTensor 的 owner 引用不等于正在执行的 lease；每个已接受 dispatch 必须持有独立使用记录。

建议的内部对象：

```python
@dataclass
class _ExternalDomainRecord:
    native_handle: object
    spec: object                 # Immutable normalized creation spec.
    state: DomainState
    ordering_buffers: dict[int, object]
    admitted_uses: dict[int, object]
    cleanup_error: BaseException | None = None


@dataclass
class _EntryBindings:
    tensor_values: tuple[object, ...]
    scalar_values: tuple[object, ...]
    domains: tuple[object, ...]   # Distinct owner leases, not wire values.
```

只读 ABI 放 program state，实际 bindings 放 `_DispatchFrame`。当前 frame 已有 `keepalive`、`cleanup` 和 native handle 绑定，不新增一个无界请求列表。

## 2. 外部域创建事务

`create_comm_domain()` 是同步控制面 API。初版采取保守时序：关闭该 worker 的新提交准入，排空已接受运行，再创建并初始化域。不能把 create 偷插到正在使用设备控制面的 run 中。

具体步骤：

1. 在 Python 上验证并规范化 spec，调用纯布局 helper 对每个 buffer 自动向上 64 字节对齐，再累计 offset/window_size。native specs 使用 `dtype="opaque", count=logical_bytes, nbytes=physical_bytes`；用户 source/view 不补齐。
2. 进入 worker 的控制操作准入；取得所有需排空的 run handles。
3. 释放 Python 锁后等待完成；不要拿着 completion cleanup 会用到的锁等待。
4. 通过 `worker.create_buffer` 准备每个参与本地 chip 的 ordering token backing。
5. 调用受控的 Simpler run callback 创建域：

```python
# Proposed wrapper around the target Simpler API.
def create_domain_callback(orch, _args, _config):
    created = orch.allocate_domain(
        name=transaction.unique_runtime_name,
        members=spec.members,
        window_size=spec.window_size,
        buffers=spec.native_buffer_specs,
        memory_source=spec.memory_source,
        workspace_engines=spec.workspace_engines,
        retain_after_run=True,
        initializer=transaction.initializer,
    )
    transaction.native_handle = created
```

1. callback 的初始化完成，整个控制 run 成功后，再按 `handle.ref` 发布 registry 记录。
2. 返回 CommDomain，恢复新提交准入。

`memory_source` 的公共字符串由 wrapper 转成目标 runtime enum；不能把不同版本的 spelling 当成无检查透传。

失败时：未发布资源由事务回滚；已创建的 native handle 经正式 release/failure cleanup 回收。token backing 只有在安全回收后关闭。若回收无法确认，保留 cleanup debt 并报告，不能丢掉句柄只返回初始异常。
创建和初始化不得在预编译阶段执行。

## 3. Host initializer

复用目标分支 `window_sources` 的分块读取/CPU tensor 原始字节复制逻辑，并从显式 specs 构造初始化绑定。旧 checkout 未含该模块时，应使用已集成分支的实现，而不是复制另一个版本。

```text
for buffer in ordered_specs:
    offset, allocated_bytes = domain.buffer_range(buffer.name)
    assert buffer.logical_bytes <= allocated_bytes
    for domain_rank in range(len(members)):
        storage = domain.member(domain_rank).buffers[buffer.name].shm
        target = storage.buf[offset : offset + buffer.logical_bytes]
        copy_source(rank_sources[buffer.name][domain_rank], target)
```

以上是算法示意：实际 memoryview 必须用 context manager 及时释放，异常 traceback 不能使 rollback 的 shm.close 永久失败。遵循 runtime 的 `.shm.buf` 覆盖整个 rank allocation 的契约，不能重复加 buffer offset。

读完每片后才发布可用域，不为各 program 再次初始化。支持 restore source cursor、短读循环、短文件 rollback。创建返回后 sources 可以关闭；owner 不应无期限保留大 CPU 源副本。

## 4. submit 的前半段：验证与占用资源

在 `_submit_compiled` 中，先按 user ABI 校验，再获取 frame 并填参数。不把隐藏 ctx 放进用户参数计数。

```python
# Proposed control flow; exceptions must use existing frame cleanup paths.
user_values = bind_user_args(state.entry_abi, args)
validated = validate_all_arguments(user_values, state.entry_abi, self)
frame = self._acquire_dispatch_frame()
try:
    leases = admit_distinct_domain_uses(validated.domains, frame.dispatch_id)
    frame.keepalive.extend(leases)
    frame.cleanup.extend(lease.finish for lease in leases)
    frame.entry_bindings = encode_entry_args(state.entry_abi, validated)
    self._submit_native_dispatch(handle, frame, state, tensors, call_config)
except BaseException:
    finish_or_recover_submission_using_existing_handle_logic(frame)
    raise
```

这里不是直接可复制的代码：当前 `_acquire_dispatch_frame` 返回 frame/dispatch_id tuple，应按实际接口接入。关键不变量是 lease 在 native submit 前获得，接受与 release 的状态变化在同一准入规则下原子化。

验证与 pin 不能存在 TOCTOU：一次普通属性检查之后、真正 admission 时还要检查 owner 未进入 CLOSING。多个域的锁按稳定身份顺序获取，失败则撤销此前尚未提交的 pin。

每个 DomainTensor 参数只写自己的 buffer index；根据 ABI 配对把其 ref 编码到对应 ctx 后缀。普通 tensor/scalar 仍使用现有校验、动态维度和 host-control 逻辑。

错误诊断应包含 program、user 参数名/index、期望类型及具体字段。错误示例：

```text
prefill parameter table_a: expected DomainTensor, received torch.Tensor
prefill parameter table_b: domain is closing; new submissions are rejected
prefill parameter table_a: target local worker 2 is not a domain member
```

## 5. 当前 callback 内登记并恢复绑定

独立 L3 runner 给生成入口 `_entry_bindings`；L4 entry 从 TaskArgs 读取同布局数据。common binder 必须在当前 orchestration callback 运行。

```python
# Proposed common binder, local state for this invocation only.
run_domains = {}
for window_param in entry_abi.windows:
    index = bindings.scalar(window_param.window_offset)
    ref = decode_domain_ref(bindings, window_param.ctx_offset)
    if ref not in run_domains:
        run_domains[ref] = orch.get_domain(ref)
    domain = run_domains[ref]
    validate_index_view_members_and_engine(domain, index, window_param)
    slot = domain.buffers[index].name
    bound_windows[window_param.ir_index] = (domain, slot)
```

`get_domain` 同时登记本次 run 的使用，不能只用旧 handle 的 cached `.device_ctx` 绕过登记。先绑定和验证所有外部参数，再执行 L3 body；不在第一个 chip 派发后才验证第二个 table。

L4 owner 转发 ref 的 run 也要登记 owner 使用；接收节点 lookup 不能替代 owner 登记。授权由 Simpler grant 验证，不由 PyPTO ref 相等性替代。

## 6. 提交失败的三种情况

| 时机 | 处理 |
| --- | --- |
| Python 校验失败或 native 未接受 | 释放本次 pin、归还未发布 frame，不释放共享域 |
| native 已接受，但 submit 抛异常/句柄回传失败 | 复用 `_recover_accepted_native_handle` 路径恢复；资源跟随真实 native 完成 |
| native 执行或域发生故障 | 等待安全终态/后代 cleanup；若底层撤销域，则 registry 标记失效，不再复用 |

不能把所有异常都当作“没有提交”。当前 `_submit_native_dispatch` 已区分 native acceptance；新 lease 必须接入这条机制。无确认完成时，`keepalive.clear()` 不是回收许可。

swimlane 两次执行同样要两次登记域使用，外部表仍不自动 reset。首轮失败不能直接复用受污染的域进行第二轮。

## 7. ordering token

key 为 `(full_ref, local_chip_worker_id)`，一个域多个窗口与多个 program 共用。一次 L2 派发对它触及的 distinct domains 各追加一个 token tensor；token 在所有业务 tensor 之后、所有 scalar 之前。

外部域在 worker init 后创建，推荐以 Simpler 的 descriptor-backed POSIX `create_buffer` 承载 token，而非新建 `torch.zeros(...).share_memory_()` 后假定 child 继承了地址。

- buffer 在受控准备阶段分配一次，子进程按 canonical identity 映射。
- 不按 dispatch/frame 创建新 token identity；否则无法串起同域使用。
- `tokens_for_domain(ref)` 只查询，不在生成代码的逐 chip 循环中触发建域/分配控制操作。
- token 生命周期至少覆盖域的全部 admitted uses；release 时在安全 fence 后关闭。
- L4 节点也需要域导入/运行准备阶段建立相同 registry；这是 L4 runtime 集成工作，不能假定远端拥有 Python 的本地 owner 对象。
- 当前自建域的旧 token 路径保持回归；只有明确共享同一物理域时才统一身份。

L4 文档现有 pre-fork 条款继续约束 FORK_SHM；POSIX descriptor 路径需补单独契约和测试，不能无验证修改所有内部 intermediate 的创建时机。

## 8. 跨 submit 顺序

本方案继承 Simpler 的 run FIFO、子任务完成和域 release fence，不在 PyPTO 上层凭空生成一个跨 run TaskId。

必须验证：

```python
h1 = worker.submit(write_program, table, ...)
h2 = worker.submit(read_program, table, ...)
h2.result()
h1.result()
```

若目标 runtime 不保证这类共享域使用的必要完成顺序，则首期必须在提交准入/激活处基于前序 run fence 串行化，或明确拒绝该异步模式，不能让数据竞争进入发布版本。此能力核对是 P0 的准入项；优先复用已提供的运行时顺序，避免无依据给所有无关域增加全局 fence。
只读表本身无写 hazard，但 URMA workspace/队列仍有协议并发约束，不能仅根据 tensor direction=In 就宣布任意并发安全。

本地程序仅使用域成员子集时，也不能自动推断其 collective 协议正确。全域参与、信号协议和远端数据准备是算法契约，需要专门验收。

## 9. release 与 close 状态机

```text
CREATING -> LIVE -> CLOSING -> RELEASED
    |          |       |
    v          v       v
ROLLBACK    INVALID  CLEANUP_FAILED
```

- LIVE：可以获取/提交新使用。
- CLOSING：拒绝新的 DomainTensor 使用，但已被 Python admission 接受的 frame 必须仍能登记并完成。
- RELEASED：确认 backend freed，关闭 tokens，删除可用 registry 记录；保留必要终态诊断。
- INVALID：底层域已失败/撤销，拒绝复用；走 failure cleanup。
- CLEANUP_FAILED：未确认回收，句柄不可再用，worker 保留清理责任；不要把 Python 引用数归零当作 freed。

显式 release 顺序：

1. 在 owner admission 锁下从 LIVE 转 CLOSING，快照所有已接受使用。
2. 放开锁，等待这些 frame 的提交结果和真实 native completion；不持有 `_dispatch_submit_mu` 或 cleanup 所需锁等待。
3. 使用 Simpler 正式 release，等待后代任务和注册/映射释放完成。
4. 确认 freed 后关闭 tokens，记录 RELEASED。重复 release 幂等。

关键 race：frame 已取得 lease、尚未拿到 native handle 时，release 不能先把低层域撤销，导致该已接受 callback 的 get_domain 失败。使用记录需要包含 submission-pending 状态与完成事件。

`worker.close()` 先全局停止 admission，再排空 frames，释放内部 retained 域和外部 owner 域，最后关闭 Simpler。不要因一个域释放失败跳过其他独立清理；汇总并保留原始错误。
程序移除/关闭只释放 program 自己的 callable/state，不释放外部 owner。GC 不作为正确性的释放机制，用户使用 context manager 或显式 release。

## 10. reset、sources 与数据更新

内部 `reset_persistent_windows` 仅遍历内部 registry。外部域不会因 Prefill/Decode 的不同 reset 设置而被清零。
初版没有异步 `reinitialize` API。需要更新表时，先等待全部使用，随后通过明确且受控的数据写入路径，或 release 后创建新域。新 generation 使旧 DomainTensor 失效。
`window_sources` 与公开的直接共享物理整表绑定不是一回事；本设计复用已建立的 per-rank 分片域，不宣称实现 #3482。
