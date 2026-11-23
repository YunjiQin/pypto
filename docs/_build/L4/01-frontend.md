# L4 前端实现方案

前端改动包括建域层级放行、`worker=`、`num_workers()`、jit 子装饰器，以及 L4 collective 拒绝。
**没有新的 DSL 语法结构**——通信域仍由 pass 推断，用户在 L3 orch 中照常声明 `pld.DistributedTensor`（按需标记 `pl.InOut`）。pass 为它追加逻辑 `CommCtx`；调用点自动传递 `domain.ref` 和实参在该域的 buffer 序号，同一 L3 入口可以在不同调用中绑定不同域和窗口。

## 1. 层级放行

### 1.1 现状

`@pl.function(level=pl.Level.CLUSTER_0, role=pl.Role.Orchestrator)` **已经能解析**
（`python/pypto/language/parser/enum_utils.py` 已映射 `"CLUSTER_0"`）。
建域操作需要放开 HOST 限制；拓扑操作按各自语义校验。L4 不开放 collective 和 world_size。

### 1.2 引入 `_is_orchestration_level`

```python
# python/pypto/language/parser/ast_parser.py（模块级私有常量 + 方法）

#: 可承载通信域 / 分布式控制面 op 的层级（Linqu L3 及以上）。
_ORCHESTRATION_LEVELS = frozenset({ir.Level.HOST, ir.Level.CLUSTER_0})

def _is_orchestration_level(self) -> bool:
    """True when the enclosing function may host distributed control-plane ops.

    ``pld.alloc_window_buffer`` / ``pld.num_workers`` are
    legal in any orchestration level (HOST, CLUSTER_0, ...), not only HOST.
    Nested device-side scopes (InCore / SPMD) are excluded by the caller.
    """
    return self._func_level in _ORCHESTRATION_LEVELS
```

### 1.3 `alloc_window_buffer` 闸门（`ast_parser.py`）

```python
# Before
if not is_printed and (self._func_level != ir.Level.HOST or in_device_scope):
    raise ParserSyntaxError(
        "pld.tensor.alloc_window_buffer() can only be called in HOST orchestration "
        "context (not inside InCore / SPMD scopes); "
        f"current function level: {self._func_level}",
        span=span,
        hint="Use '@pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)' "
        "on the enclosing function and call outside any nested device-side scope",
    )

# After
if not is_printed and (not self._is_orchestration_level() or in_device_scope):
    raise ParserSyntaxError(
        "pld.tensor.alloc_window_buffer() can only be called in an orchestration "
        "context (HOST or CLUSTER_0, not inside InCore / SPMD scopes); "
        f"current function level: {self._func_level}",
        span=span,
        hint="Use '@pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)' "
        "(node-local window) or 'level=pl.Level.CLUSTER_0' (cluster-global window) "
        "on the enclosing function, and call outside any nested device-side scope",
    )
```

保留现有 `is_printed` 的 IR dump 往返处理；源码合法性和 IR 合法性分别检查。

### 1.4 保留 `world_size` 的 HOST 限制

`world_size()` 保留当前 HOST 本节点设备数语义，不扩展为全局域大小查询。
新 `num_workers()` 在 HOST/CLUSTER_0 合法，分别返回本地设备数/节点数。

### 1.5 L4 collective 拒绝

在解析 collective 操作时，根据实际操作注册身份/语义判断；若当前层是 CLUSTER_0，
报 `ParserSyntaxError`，提示将 collective 放入 L3/L2，再由 L4 下发。
检查 tensor/tile collective 及别名；不能只匹配用户书写的函数名。
直接构造 IR 的路径也需要相同合法性检查，见 [02-passes.md](02-passes.md)。

### 1.6 层级链校验不变

`TryEmitHierarchyCall`（`src/codegen/distributed/distributed_codegen.cpp`）
的规则 `callee_level == current_level - 1` **保持不变**：CLUSTER_0(6) - 1 =
HOST(5)。不做跳级语法糖。

## 2. `worker=` 统一命名

### 2.1 语义

下发 kwarg 在每一层都叫 `worker=`，含义恒定为"下一级的哪一个子节点"，与
`orch.submit_next_level(..., worker=<id>)` 一致。层级由 caller 的 `level_` 唯一
确定，无需第二种拼写。

| 调用边 | `worker=` 的含义 |
| ------ | ---------------- |
| CLUSTER_0 → HOST | L3 节点索引 |
| HOST → CHIP | 本节点内的 L2 索引 |

### 2.2 `allowed_kwargs`（`ast_parser.py`）

```python
# Before
if func_obj is not None and func_obj.role == ir.Role.Orchestrator:
    allowed_kwargs.add("device")

# After
if func_obj is not None and func_obj.role == ir.Role.Orchestrator:
    allowed_kwargs.add("worker")
    allowed_kwargs.add("device")   # deprecated alias, see §2.4
```

### 2.3 kwarg 解析（`ast_parser.py`）

```python
def _parse_dispatch_worker_kwarg(self, keywords: list[ast.keyword]) -> ir.Expr | None:
    """Extract the optional ``worker=`` kwarg on a dispatch call.

    ``worker=`` names which child of the *next* level receives the dispatch —
    an L3 node when the caller is CLUSTER_0, one of that node's L2 devices when
    the caller is HOST. The spelling is level-independent by design: the level
    is already fixed by the caller, so a second spelling would only create two
    equivalent vocabularies. Callee-role validation happens in
    :meth:`parse_call` via ``allowed_kwargs`` before this method runs.

    The parser stays permissive about the value form: any IR expression works.
    Downstream passes inspect the resolved expression and decide:

    * ``ConstInt`` (literal or after simplify-time folding) → fixed target.
    * IR Var that is a ``for``-loop induction variable → target subset / kAll.
    * Anything else → rejected by the comm-collection pass.
    """
    dispatch_kws = [k for k in keywords if k.arg in ("worker", "device")]
    if len(dispatch_kws) > 1:
        raise ParserSyntaxError(
            "Specify only one of worker= and device=",
            span=self.span_tracker.get_span(dispatch_kws[1].value),
        )
    kw = next(iter(dispatch_kws), None)
    if kw is None:
        return None
    if kw.arg == "device":
        self._warn_deprecated_device_kwarg(kw)
    return self.parse_expression(kw.value)
```

### 2.4 `device=` 的迁移

`device=` 是既有拼写，仓库内仍广泛使用。迁移时区分 dispatch kwarg 与 torch/device 配置等同名参数，不再依赖旧版出现次数。方案：

1. `worker=` 为正名，新代码一律用它。
2. `device=` 作为别名保留一个迁移窗口，命中时发 `DeprecationWarning`，提示
   `"device= is deprecated on a dispatch call; use worker= (the next-level child index)"`。
3. IR attr 键同步改名 `attrs["device"]` → `attrs["worker"]`（内部键，不对外暴露），
   打印/解析往返与 pass dump 一并更新。
4. 迁移窗口结束后删除别名。

别名删除时机见 [05-plan.md](05-plan.md) § 待决问题。

### 2.5 受影响的下游

| 位置 | 改动 |
| ---- | ---- |
| `distributed_codegen.h` `ResolveRankExpr` | 读 `attrs["worker"]`；文档串改写 |
| `materialize_comm_domain_scopes_pass.cpp` `DispatchAnalyzer` | 读 `attrs["worker"]` |
| `distributed_codegen.cpp` 的 `INTERNAL_CHECK_SPAN` 文案 | `device= attr` → `worker= attr` |
| 现有 L3 测试 / 示例 / 用户文档 | 逐步迁移到 `worker=` |

## 3. `pld.num_workers()`

`worker=` 统一之后，循环上界也统一：`for w in pl.range(pld.num_workers())`——L4 得到
节点数，L3 得到本节点卡数。不再需要 `num_nodes()` / `local_size()` 两个层级特化的
拼写。

### 3.1 C++ 注册

新建 `src/ir/op/distributed/num_workers.cpp`，逐字镜像
`src/ir/op/distributed/world_size.cpp` 的形状：

```cpp
namespace {

TypePtr DeduceNumWorkersType(const std::vector<ExprPtr>& args,
                             const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CHECK(args.empty()) << "pld.system.num_workers takes no positional arguments, but got " << args.size();
  CHECK(kwargs.empty()) << "pld.system.num_workers takes no kwargs, but got " << kwargs.size();
  return std::make_shared<ScalarType>(DataType::INT64);
}

}  // namespace

REGISTER_OP("pld.system.num_workers")
    .set_description(
        "Return the number of children this orchestration level dispatches to, as a scalar "
        "INT64 — L3 nodes for a CLUSTER_0 orchestrator, that node's L2 devices for a HOST "
        "orchestrator. Orchestration-only. In a standalone L3 program this equals "
        "``pld.system.world_size``. Codegen lowers each call site to the ``num_workers`` "
        "binding in the emitted entry.")
    .set_op_category("DistributedOp")
    .no_argument()
    .no_memory_spec()
    .f_deduce_type(DeduceNumWorkersType);
```

### 3.2 Python 包装

`python/pypto/ir/op/distributed/system_ops.py` 增 sentinel，
`python/pypto/language/distributed/op/system_ops.py` 增 DSL 包装：

```python
def num_workers() -> Scalar:
    """Return the next-level child count as an ``INT64`` :class:`Scalar`.

    L3 nodes when called from a CLUSTER_0 orchestrator, that node's L2 devices
    when called from a HOST orchestrator. In a standalone L3 program this equals
    :func:`world_size`.

    .. warning::

       Callable **only** inside an orchestration function (HOST or CLUSTER_0),
       outside any nested device-side scope.
    """
    return Scalar(_ir_system.num_workers())
```

导出：`python/pypto/language/distributed/op/__init__.py` 与
`python/pypto/language/distributed/__init__.py` 的 `__all__` 各加一项。

### 3.3 Parser 层级校验

```python
if op_name == "num_workers":
    allowed_level = self._is_orchestration_level()
elif op_name == "world_size":
    allowed_level = self._func_level == ir.Level.HOST
```

两者均禁止出现在嵌套 InCore/SPMD scope 内。保持 `world_size()` 的兼容语义，
不要将它与 `num_workers()` 一起无条件放开到 L4。

### 3.4 配套消费者

操作注册和 parser 放行后，还需同步：

- 域分析识别 `num_workers()` 循环上界（含 SSA 临时变量），见 [02-passes.md](02-passes.md)。
- codegen 将它转换为入口的 `num_workers` 绑定；HOST 的 `world_size` 绑定保留。
- runner 分别绑定 L4 节点数、L3 本地设备数；全局域大小不能从本地 worker 数推导。

## 4. `@pl.jit.pod` 子装饰器

类形式（`@pl.program` + `@pl.function`）无需改动。`@pl.jit` 家族补一个子装饰器：

```python
# python/pypto/jit/decorator.py
self.host = _SubFunctionDecorator("host", allow_level=False, allow_auto_scope=True)
self.pod  = _SubFunctionDecorator("pod",  allow_level=False, allow_auto_scope=True)
```

`python/pypto/jit/specializer.py`：

```python
if kind == "pod":
    return f"@pl.function(level=pl.Level.CLUSTER_0, role=pl.Role.Orchestrator{auto_scope_suffix})"
```

`_discover_deps` 的调用许可（`decorator.py` 附近）：新增
`caller_func_type == "pod"` 承认 `host` 依赖（且**不**承认 `orchestration`——层级链
严格）。

## 5. 跨层同步

按 `cross-layer-sync.md`：

| 层 | 文件 | 改动 |
| -- | ---- | ---- |
| C++ | `src/ir/op/distributed/num_workers.cpp` | 一个 `REGISTER_OP` |
| Python IR | `python/pypto/ir/op/distributed/system_ops.py` | 一个 sentinel |
| Python DSL | `python/pypto/language/distributed/op/system_ops.py` | 一个包装 + `__all__` |
| 类型桩 | `python/pypto/language/distributed/*.pyi`（若存在） | 一个签名 |

`ClusterConfig` / `ClusterNodeSpec` 的桩见 [04-runtime.md](04-runtime.md)。

## 6. 前端测试

| 测试 | 位置 | 断言 |
| ---- | ---- | ---- |
| CLUSTER_0 函数解析 | `tests/ut/ir/parser/` | `func.level == ir.Level.CLUSTER_0`、`role == Orchestrator` |
| `@pl.jit.pod` 特化 | `tests/ut/jit/` | 展开为 `level=CLUSTER_0, role=Orchestrator` |
| `worker=` 落 attr | `tests/ut/ir/parser/` | 调用点 `attrs["worker"]` 存在且值为循环变量 |
| `device=` 别名仍可用 | 同上 | 解析通过 + `DeprecationWarning` |
| `worker=` 在两层同形 | 同上 | L4 与 L3 的下发点 attr 结构一致 |
| `alloc_window_buffer` 在 L4 合法 | 同上 | 解析通过 |
| L4 collective / world_size 拒绝 | 同上 | ParserSyntaxError，指出合法层级 |
| worker/device 同时指定 | 同上 | 明确报错，不静默选一个 |
| `num_workers` 在 InCore 报错 | 同上 | `ParserSyntaxError`，提示含 "orchestration context" |
| 打印-解析往返 | `tests/ut/ir/printing/` | 四层程序 `assert_structural_equal` 往返一致 |
