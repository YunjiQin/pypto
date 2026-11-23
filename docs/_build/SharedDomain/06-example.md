# 两个独立 L3 入口的目标示例与降低链

以下使用拟议 API，**尚不可在当前 checkout 直接运行，也不是实际编译 dump**。这是共享资源和参数绑定的最小验收形态，不是完整 Engram 模型或性能测试。使用四卡只是示例拓扑，实际设备测试应先从两卡开始。

## 1. 用户源码

两个 HOST 入口都消费两张 UINT8 表的某一行，写到本地输出。它们独立编译，窗口来自参数；业务 kernel 不涉及远端 GET，便于先验证基础共享链路。

```python
import io
from contextlib import ExitStack

import torch
import pypto.language as pl
import pypto.language.distributed as pld
from pypto import DataType
from pypto.backend import BackendType
from pypto.ir import DistributedConfig
from pypto.runtime import (
    DistributedWorker,
    DomainBufferSpec,  # Proposed API.
    ExecutionMode,
    RunConfig,
)

N = 4
ROWS_A = 8
ROWS_B = 16
DIM = 256


@pl.jit.incore
def read_pair(
    ids: pl.Tensor[[1], pl.INT32],
    table_a: pld.DistributedTensor[[ROWS_A, DIM], pl.UINT8],
    table_b: pld.DistributedTensor[[ROWS_B, DIM], pl.UINT8],
    out: pl.Out[pl.Tensor[[2, DIM], pl.UINT8]],
    step: pl.Scalar[pl.INT32],
) -> pl.Tensor[[2, DIM], pl.UINT8]:
    index = pl.read(ids, [0]) + step
    a = pl.load(table_a, [index % ROWS_A, 0], [1, DIM])
    b = pl.load(table_b, [index % ROWS_B, 0], [1, DIM])
    out = pl.store(a, [0, 0], out)
    return pl.store(b, [1, 0], out)


@pl.jit
def chip_read(
    ids: pl.Tensor[[1], pl.INT32],
    table_a: pld.DistributedTensor[[ROWS_A, DIM], pl.UINT8],
    table_b: pld.DistributedTensor[[ROWS_B, DIM], pl.UINT8],
    out: pl.Out[pl.Tensor[[2, DIM], pl.UINT8]],
    step: pl.Scalar[pl.INT32],
) -> pl.Tensor[[2, DIM], pl.UINT8]:
    return read_pair(ids, table_a, table_b, out, step)


@pl.jit.host
def prefill(
    ids: pl.Tensor[[N, 1], pl.INT32],
    table_a: pld.DistributedTensor[[ROWS_A, DIM], pl.UINT8],
    table_b: pld.DistributedTensor[[ROWS_B, DIM], pl.UINT8],
    out: pl.Out[pl.Tensor[[N, 2, DIM], pl.UINT8]],
    step: pl.Scalar[pl.INT32],
) -> pl.Tensor[[N, 2, DIM], pl.UINT8]:
    for w in pl.range(pld.world_size()):
        chip_read(ids[w], table_a, table_b, out[w], step, device=w)
    return out


@pl.jit.host
def decode(
    ids: pl.Tensor[[N, 1], pl.INT32],
    table_a: pld.DistributedTensor[[ROWS_A, DIM], pl.UINT8],
    table_b: pld.DistributedTensor[[ROWS_B, DIM], pl.UINT8],
    out: pl.Out[pl.Tensor[[N, 2, DIM], pl.UINT8]],
    step: pl.Scalar[pl.INT32],
) -> pl.Tensor[[N, 2, DIM], pl.UINT8]:
    for w in pl.range(pld.world_size()):
        chip_read(ids[w], table_a, table_b, out[w], step, device=w)
    return out


class CountingSource(io.BytesIO):
    def __init__(self, data):
        super().__init__(data)
        self.bytes_read = 0

    def readinto(self, target):
        count = super().readinto(target)
        self.bytes_read += count
        return count


def make_tables(rows, tag):
    row = torch.arange(rows, dtype=torch.int64).reshape(rows, 1)
    col = torch.arange(DIM, dtype=torch.int64).reshape(1, DIM)
    base = row * 29 + col * 7
    return [(base + 17 * rank + tag).remainder(256).to(torch.uint8)
            for rank in range(N)]


def run_example(prefill_compiled, decode_compiled):
    # Host IO must be prepared before the worker forks its chip children.
    ids = torch.arange(N, dtype=torch.int32).reshape(N, 1).share_memory_()
    out_p = torch.zeros((N, 2, DIM), dtype=torch.uint8).share_memory_()
    out_d = torch.zeros_like(out_p).share_memory_()
    tables_a = make_tables(ROWS_A, 3)
    tables_b = make_tables(ROWS_B, 101)

    with ExitStack() as stack:
        a_files = [stack.enter_context(CountingSource(t.numpy().tobytes()))
                   for t in tables_a]
        b_files = [stack.enter_context(CountingSource(t.numpy().tobytes()))
                   for t in tables_b]
        with DistributedWorker([prefill_compiled, decode_compiled]) as worker:
            with worker.create_comm_domain(
                members=list(range(N)),
                memory_source="host",
                workspace_engines=(),
                buffers=[
                    DomainBufferSpec("weights_a", (ROWS_A, DIM), DataType.UINT8),
                    DomainBufferSpec("weights_b", (ROWS_B, DIM), DataType.UINT8),
                ],
                window_sources={"weights_a": a_files, "weights_b": b_files},
            ) as domain:
                a = domain.tensor("weights_a")
                b = domain.tensor("weights_b")
                initialized_bytes = sum(f.bytes_read for f in a_files + b_files)
                assert initialized_bytes == N * (ROWS_A + ROWS_B) * DIM

                for program, output, step in [
                    (prefill_compiled, out_p, 0),
                    (decode_compiled, out_d, 1),
                    (prefill_compiled, out_p, 2),
                ]:
                    output.fill_(255)
                    worker.submit(program, ids, a, b, output, step).result()
                    for rank in range(N):
                        index = int(ids[rank, 0]) + step
                        expected = torch.stack([
                            tables_a[rank][index % ROWS_A],
                            tables_b[rank][index % ROWS_B],
                        ])
                        assert torch.equal(output[rank], expected)
                    assert sum(f.bytes_read for f in a_files + b_files) == initialized_bytes
```

`CountingSource` 验证后续执行没有再次读取初始化源；它不能单独证明底层没有重复分配。底层分配次数、domain ref/context 与实际内存占用由 runtime/ST 额外采样。

编译与设备执行分开：

```python
# CPU preparation: no domain creation here.
config = RunConfig(
    backend_type=BackendType.Ascend950,
    execution_mode=ExecutionMode.ONBOARD,
    distributed_config=DistributedConfig(device_ids=[0, 1, 2, 3], num_sub_workers=0),
)
prefill_compiled = prefill.compile(config=config)
decode_compiled = decode.compile(config=config)
# Persist both artifact directories and the fixture before a queued device run.
```

设备阶段从预编译目录使用 `DistributedCompiledProgram.from_dir` 加载，并按队列分配的 device_ids 构造执行配置，再调用 `run_example`。不在占用 NPU 的作业中重新执行上述 compile。

## 2. 对应的 IR

下面为目标 IR 签名示意，省略 SSA 后缀和 tensor 详细类型：

```text
read_pair(ids, table_a, table_b, out, step, table_a_ctx, table_b_ctx)
chip_read(ids, table_a, table_b, out, step, table_a_ctx, table_b_ctx)
prefill(ids, table_a, table_b, out, step, table_a_ctx, table_b_ctx)
decode (ids, table_a, table_b, out, step, table_a_ctx, table_b_ctx)
```

四个函数都没有共享表的 owning scope。HOST 调用 `chip_read` 时转发两个 ctx，L2 调用 kernel 时继续转发。同一 runtime domain 绑定到两个参数，不使 IR 签名缩短。

如果其中一个 HOST 函数另有 `scratch_buf = pld.alloc_window_buffer(...)`，只有 scratch 进入它的 owning scope。

## 3. Python submit 绑定

`a`、`b` 属于同一 ref `R`，在 ordered specs 中的下标分别为 0、1。

```text
User call:
  worker.submit(prefill_compiled, ids, a, b, out_p, 0)

Logical internal values:
  ids, a_index=0, b_index=1, out_p, step=0, a_ctx=R, b_ctx=R

Entry pools:
  tensors = [ids, out_p]
  scalars = [0, 1, 0, R.owner_le64, R.id, R.generation,
                      R.owner_le64, R.id, R.generation]
```

本地 runner 可以用 `_EntryBindings` 的 scalar accessor 表示相同槽位，普通 tensor 保留现有 kwargs 路径。L4 sender 使用同一布局写 TaskArgs。

## 4. L3 入口绑定与下发

以下为局部片段，省略普通 tensor 的绑定和现有 callable/config 参数。common binder 只在入口执行一次，不在逐卡循环中反复判断域来自哪里。

```python
# Proposed generated entry fragment for the canonical five-business-arg ABI.
a_index = int(entry_args.scalar(0))
b_index = int(entry_args.scalar(1))
step = entry_args.scalar(2)
a_ref = decode_domain_ref(entry_args, 3)
b_ref = decode_domain_ref(entry_args, 6)

run_domains = {}
domain_a = resolve_domain_for_run(orch, a_ref, run_domains)
domain_b = resolve_domain_for_run(orch, b_ref, run_domains)
validate_window_argument(domain_a, a_index, a_constraints)
validate_window_argument(domain_b, b_index, b_constraints)
slot_a = domain_a.buffers[a_index].name
slot_b = domain_b.buffers[b_index].name

for w in range(world_size):
    ctx_a = domain_a[w]
    ctx_b = domain_b[w]
    task_args = TaskArgs()
    append_existing_chip_tensor(task_args, ids[w], ids_direction)
    task_args.add_tensor(
        ctx_a.buffers[slot_a].tensor((ROWS_A, DIM), DataType.UINT8),
        TensorArgType.INPUT,
    )
    task_args.add_tensor(
        ctx_b.buffers[slot_b].tensor((ROWS_B, DIM), DataType.UINT8),
        TensorArgType.INPUT,
    )
    append_existing_chip_tensor(task_args, out[w], out_direction)
    append_distinct_domain_tokens(task_args, w, (domain_a, domain_b))
    task_args.add_scalar(step)
    task_args.add_scalar(ctx_a.device_ctx)
    task_args.add_scalar(ctx_b.device_ctx)
    _submit_chip(orch, chip_handle, task_args, config, w)
```

`append_existing_chip_tensor` 表示复用当前 CHIP helper 及 host_tensor_indices，不是建议新增绕过传输语义的 wrapper。`append_distinct_domain_tokens` 对同一 ref 只发一个该 chip 的 token；两个 ctx 标量仍分别存在。

L2 所见：

```text
tensors = [ids_local, table_a_local, table_b_local, out_local, domain_token]
scalars = [step, device_ctx_a, device_ctx_b]
```

L2/kernel 既不接收 buffer_index，也不接收三字段 ref；它们继续使用既有设备参数 ABI。

## 5. 独立编译与跨来源复用

同一份 L3 可以绑定另一域：

```text
Invocation 1: a = D.slot[0], b = D.slot[1]
Invocation 2: a = E.slot[2], b = F.slot[0]
```

类型、view、成员和 engine 要求符合即可；不按 buffer 名或创建 program 匹配。
L4 调用方在自己的调用点解析这组 index/ref，接收端执行上面完全相同的绑定和 L3→L2 body。

## 6. 生命周期时间线

```text
compile P, D on CPU
    |
prepare worker / register both programs / init
    |
create shared domain -> allocate -> initialize -> publish ref R
    |
submit P -> acquire lease -> get_domain(R) enroll run -> execute -> complete
    |
submit D -> acquire lease -> get_domain(R) enroll run -> execute -> complete
    |
release domain -> reject new uses -> drain admitted uses -> native release
    |
close worker
```

P 与 D 每次使用共享映射和原有内容。源文件内容相同但新建另一个 domain 时，仍然是另一份分配；只有显式传同一资源才共享。
