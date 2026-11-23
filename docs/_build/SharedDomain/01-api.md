# 公共 API 与资源对象

本文件所有新增符号均为拟议 API，目标位置为 `python/pypto/runtime/comm_domain.py`，由 `pypto.runtime` 导出。DSL 的 `pld.DistributedTensor` 保持现有拼写。

## 1. 域创建签名

```python
@dataclass(frozen=True)
class DomainBufferSpec:
    name: str
    shape: tuple[int, ...]
    dtype: DataType
    layout: str = "ND"


class DistributedWorker:
    def create_comm_domain(
        self,
        *,
        members: Sequence[int],
        buffers: Sequence[DomainBufferSpec],
        memory_source: str = "device",
        workspace_engines: Sequence[str] = (),
        window_sources: Mapping[str, Sequence[BinaryIO | torch.Tensor]] | None = None,
    ) -> CommDomain:
        ...


class CommDomain:
    @property
    def ref(self) -> CommDomainRef: ...

    @property
    def members(self) -> tuple[int, ...]: ...

    @property
    def buffers(self) -> tuple[DomainBufferSpec, ...]: ...

    def tensor(self, name: str) -> DomainTensor: ...
    def release(self) -> None: ...
    def __enter__(self) -> CommDomain: ...
    def __exit__(self, exc_type, exc, traceback) -> None: ...
```

`release` 为显式释放入口；context manager 在退出时调用它。PyPTO owner wrapper 可以使用 `with`，不能把 Simpler 借用 handle 的 context manager 当作释放途径。

外部域天然跨 submit 保留，不受 `persistent=False` 影响。`persistent` 与 `reset_persistent_windows` 继续只控制 program 内部生成的域。
第一版没有根据用户 name 自动命中另一分配的行为。runtime 为新域生成 worker 内唯一标签；复用由传同一个 CommDomain/DomainTensor 表达。

## 2. 创建前验证

验证顺序应在设备分配之前完成：

1. worker 尚未关闭，所需 Simpler 能力与 backend 支持存在。
2. members 非空、无重复，元素为合法本地 worker 下标，不接受 bool 代替整数。
3. buffers 非空，name 非空且唯一，声明顺序固定。
4. shape 为具体正整数，dtype 是已支持的可物化字节类型，初版 layout 必须 ND。
5. 乘法、对齐和总和不溢出，符合后端窗口容量限制。
6. workspace 配置规范化；不静默补不支持的 engine，也不因 capability 失败自动 bypass。
7. Host sources 的 key 覆盖声明的初始化需求，每个列表长度等于成员数。

第一版可限制所有 Host buffer 都提供初始化源，以保证可重复的发布语义。Device 域不接收 Host initializer 语法：通过后续 L2 task 填充，或单独接入已有 Device copy helper；不要用 Host shm 逻辑初始化 Device window。

初始化文件按 source 的既有读取契约读取，不接管调用者的 close；文件要保持打开直到同步 create 返回。支持短读循环，短文件报错并回滚。CPU tensor 仅接受连续、可按原始字节读取的存储。

## 3. 自动向上 64 字节对齐

这是 `create_comm_domain` 的强制职责，在交给 Simpler 分配之前完成。用户的 `DomainBufferSpec` 只表达逻辑 shape/dtype；不能因逻辑大小不是 64 的倍数而要求用户修改 shape、扩充 tensor 或给 bin 文件补 padding。

```text
logical_bytes[i] = product(shape[i]) * element_bytes(dtype[i])
physical_bytes[i] = ((logical_bytes[i] + 63) // 64) * 64
offset[i] = sum(physical_bytes[:i])
window_size = sum(physical_bytes)
```

**逐 buffer 对齐，再求和，不能只对最终 window_size 对齐。** 底层保证 window 基址满足其对齐要求；PyPTO 保证各槽相对偏移和物理长度都是 64 字节的倍数。

| buffer | 用户 shape/dtype | logical bytes | physical bytes | offset |
| --- | --- | ---: | ---: | ---: |
| a | `(1,) / UINT8` | 1 | 64 | 0 |
| b | `(16,) / FP32` | 64 | 64 | 64 |
| c | `(65,) / UINT8` | 65 | 128 | 128 |

该例 `window_size=256`，不是 `align_up(1 + 64 + 65, 64)=192`。三个 `buffer_index` 仍是 0、1、2，padding 不产生新槽。

实现时新增一个纯布局 helper，供创建前校验、native specs 构造和测试共用。示意：

```python
def build_domain_layout(buffer_specs, max_size):
    offset = 0
    entries = []
    for spec in buffer_specs:
        logical = checked_logical_nbytes(spec, max_size)
        physical = ((logical + 63) // 64) * 64
        if physical > max_size or offset > max_size - physical:
            raise ValueError("aligned communication-domain layout exceeds size limit")
        entries.append((spec, logical, physical, offset))
        offset += physical
    return entries, offset
```

`checked_logical_nbytes` 验证正整数维度、受支持 dtype，并按目标 native 大小字段和后端上限检查乘积；Python 整数不溢出不代表 C++ 参数可以接收该数。对齐后的单槽大小和累计大小仍要检查，所有错误在设备分配之前报告。

下发采用与内部窗口一致的 opaque 字节布局：

```python
CommBufferSpec(
    name=spec.name,
    dtype="opaque",
    count=logical_bytes,
    nbytes=physical_bytes,
)
```

这里 native `count` 是 opaque 字节数，不是原 dtype 的元素数；公开 spec/DomainTensor 保留原 shape/dtype，实际 `.tensor(shape, dtype)` 仍按逻辑视图物化。显式保存 logical/physical/offset，不能在后续初始化中仅凭 padded nbytes 反推内容长度。

初始化只消费和写入 logical_bytes。padding 不属于表内容，不要求源文件包含它，也不在公开 view 中暴露；域的 backend 分配大小或映射大小可以更大，但不能因此扩大允许的逻辑访问范围。

packed/sub-byte dtype 首期拒绝，除非实现者同时接通已有的存储形状转换并增加测试。64 字节 padding 与后端页对齐是两层不同约束，页对齐不能替代逐 buffer 的 64 字节对齐。

## 4. DomainTensor

```python
@dataclass(frozen=True)
class DomainTensor:
    # Conceptual fields; construction is controlled by CommDomain.tensor().
    _owner: CommDomain
    buffer_index: int
    shape: tuple[int, ...]
    dtype: DataType
    layout: str

    @property
    def domain_ref(self) -> CommDomainRef:
        return self._owner.ref
```

不开放用户拼装裸 ref/任意 owner 的 constructor，不提供可修改的 slot 属性。只读 metadata 不等于底层数据只读；访问方向由函数声明与所有者权限共同约束。

- `domain.tensor("weights_a")` 根据 ordered specs 找 index，返回形状/类型固定的 view。
- 同一 domain.tensor 多次调用可以返回不同 Python 对象，但必须保持相同物理域、slot、buffer canonical identity。
- 不暴露 `.data_ptr()` 作为分布式资源入口，不能伪装成 torch.Tensor。
- `_owner` 的 Python 引用只是保活的一部分；显式 release 后旧 DomainTensor 仍须拒绝新使用。
- ref 的 generation 变化意味着新分配，不能按域名重定向旧 tensor。

## 5. 业务调用

```python
# Proposed API; a_files/b_files remain open through creation.
with DistributedWorker([prefill_compiled, decode_compiled]) as worker:
    with worker.create_comm_domain(
        members=[0, 1, 2, 3],
        memory_source="host",
        workspace_engines=("urma",),
        buffers=[
            DomainBufferSpec("weights_a", (ROWS_A, DIM), DataType.UINT8),
            DomainBufferSpec("weights_b", (ROWS_B, DIM), DataType.UINT8),
        ],
        window_sources={"weights_a": a_files, "weights_b": b_files},
    ) as domain:
        a = domain.tensor("weights_a")
        b = domain.tensor("weights_b")
        worker.submit(prefill_compiled, ids_p, a, b, out_p, 0).result()
        worker.submit(decode_compiled, ids_d, a, b, out_d, 1).result()
```

调用参数名与物理 buffer 名无关。两个参数可传同一个 tensor；两张表也可来自两个不同域，前提是 callee 使用它们的目标 chip 属于各自域，且相关操作支持该组合。

初始化一次和共享一次分配不意味着对所有模型都最优；本方案不引入性能结论。

## 6. 方向与返回

| 形参 | 接受的资源 | 行为 |
| --- | --- | --- |
| `pld.DistributedTensor` / In | 有效且满足读取要求的 DomainTensor | 不复制 payload |
| `pl.InOut[DistributedTensor]` | 有效、可写的 DomainTensor | 原地更新并保持域归属 |
| `pl.Out[DistributedTensor]` | 调用方显式传入的可写 DomainTensor | 不自动建立新域 |
| 普通 `pl.Tensor` | 现有支持的 tensor 参数 | 不接受 DomainTensor 隐式降级 |
| 编译器追加 CommCtx | 不接受用户实参 | 自动从配对 DomainTensor 提取 |

第一阶段外部域入口采用已有 prepared worker 的 in-place 调用方式。未提供的外部 Out 必须在提交前报错；对普通 Out 的既有支持不能回退。
返回窗口的合法形式是现有参数的别名；第一版不支持从 L3 内部临时域导出一个新 DomainTensor 给 Python。返回 lineage 在 pass 和 codegen 中保持一致。

## 7. Shape、view 与成员契约

窗口参数初版使用静态 shape/ND contiguous、buffer 起点 view；runtime shape 必须与 compiled formal 一致。可以为同一物理槽显式准备多个静态编译入口，但不能在没有 runtime shape 信息的情况下伪装成任意动态窗口。

动态窗口后续可利用同一入口已有的业务 scalar/普通 tensor 维度表达形状；必须定义双方都可恢复的表达式与范围校验。若窗口是某动态维度的唯一来源，当前一 scalar 的窗口 ABI 不足，首期拒绝。不能在本地偷偷用 DomainTensor.shape 而让 L4 无法恢复。

program 内部 slice/reshape 的语义与跨入口传入一个切片不同。前者由已编译 IR 表达 offset/stride；后者需要可传输的参数契约。本功能不新增 offset wire 字段。

对静态或拓扑可求值的派发集合，入口 metadata 记录每个外部参数需要的本地 worker 覆盖。提交前求值校验；无法分析的合法控制流仍在首次实际访问前检查，并且不能在已部分派发后才发现可预检的错误。

## 8. 与 L4 的 API 边界

此 API 首期在单个 DistributedWorker 上创建本地域。L4 全局域的创建/授权走 #3531，不能把本机 owner Python 对象传到远端。

L4 codegen 从 DSL 实参解析 ref/index，Python submit 从 DomainTensor 解析相同二元绑定；接收端只见 ref/index。L3 使用同一个参数声明和同一套窗口/ctx 解析，无需 `source="l4"` 等标记。
