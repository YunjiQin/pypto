# JIT、Pass 与 Codegen 实施方案

## 1. 改动基线

复用已存在的 [MaterializeDistTensorCtx](../../en/dev/passes/52-materialize_dist_tensor_ctx.md)，接口设计与更新后的 [L4 pass 方案](../L4/02-passes.md)一致。不新增 IR 类型，不复制一套 SharedDomain 专属 ctx pass。

**编译侧基础先在 SharedDomain 中实现。** 首期只打通独立 HOST 入口、HOST→CHIP 和既有下层 ctx 转发；不依赖尚未落地的 L4 LevelHandler、CLUSTER_0 查询放行、worker 命名迁移或递归 L4 产物。ABI builder 与 window binder 在这里建立，L4 后续调用它们。

本地入口位置：

| 文件/函数 | 职责 |
| --- | --- |
| `python/pypto/jit/decorator.py::_is_tensor/_extract_tensor_meta/_bind_args` | 从真实实参获得特化信息 |
| `python/pypto/jit/specializer.py` | 保留 DistributedTensor 形参类型 |
| `src/ir/transforms/materialize_comm_domain_scopes_pass.cpp::ProcessHostOrch` | 收集内部 alloc、推导成员、构造 scope |
| `materialize_dist_tensor_ctx_pass.cpp::BuildFunctionCtxPlan/ExtendFunctionSignature` | 一窗口形参一 ctx 后缀 |
| 同文件 `GetCtxForArg/VisitExpr_(CallPtr)` | 追踪参数/别名并改写 ctx 查询 |
| `src/codegen/distributed/distributed_codegen.cpp::EmitFunction` | 生成 L3 入口与局部绑定 |
| 同文件 `EmitCallToWorker/ResolveCommCtxArg` | 物化窗口 tensor，选择每 chip ctx |

## 2. JIT 与用户签名

当前 `_is_tensor` 只识别 torch.Tensor。不要直接把 DomainTensor 冒充普通 tensor，也不要让 scalar fallback 接受它。

新增有明确类型的 runtime metadata 适配：

```python
# Proposed classification, before ordinary tensor/scalar fallback.
if is_domain_tensor(value):
    require_distributed_annotation(parameter)
    tensor_meta[name] = tensor_meta_from_domain_tensor(value)
elif is_torch_tensor(value):
    require_ordinary_tensor_or_compile_sample_contract(parameter)
    tensor_meta[name] = existing_tensor_meta(value)
else:
    bind_existing_scalar_or_constexpr(parameter, value)
```

`tensor_meta_from_domain_tensor` 只读 shape/dtype/layout；不包含 ref/index/owner/source。避免 JIT 导入 runtime 重型模块形成循环，可把不可变类型协议/判别放在叶子模块，实际 owner 由 runtime 实现。

- `prefill.compile(config=...)` 的 annotation-only 编译不需要活域，不应启动设备或初始化表。
- 用 sample args 编译时只读 DomainTensor 元信息，不隐式注册内存。
- L3 用户签名只包含业务参数；pass 后 ctx 是实现参数，不回写 Python inspect.signature。
- DomainTensor 传给普通 Tensor 形参、torch.Tensor 直接传给外部 DistributedTensor 执行形参，均给出有参数位置的错误。
- dtype/ND 静态 shape 等首期限制由前端和 runtime 双方一致检查。

## 3. 域物化 pass

`MaterializeCommDomainScopes` 继续以内部 alloc 为域创建来源。没有 alloc 的 HOST 函数可以原样保留，但不能因此跳过后续入口 ABI/codegen。

```text
host(table_a, table_b, out)
  external inputs: table_a, table_b
  internal alloc: scratch
  owning scope: [scratch] only
```

不为外部参数制造 synthetic alloc/WindowBuffer，不把传入参数和同 members 的内部 buffer 聚类。
`DistributedTensorType.window_buffer_` 对外部形式参数可以为空；它不是缺失物理域的错误，前提是有有效参数绑定来源。

verifier 的来源规则：

- 本地窗口：WindowBuffer 必须有且只有一个 owning scope。
- 传入窗口：可追溯到 HOST/下层合法形参，并有匹配 ctx/入口绑定。
- 既非本地窗口也非有效参数/别名：明确报错。

不再要求 caller 的 WindowBuffer 被复制到通用 L3 callee。每个调用点检查实际窗口来源和 view 兼容性，允许不同调用传不同域/槽。

## 4. ctx pass

### 4.1 保留签名策略

```text
Before: host(x, a, b, y, n)
After:  host(x, a, b, y, n, a_ctx, b_ctx)
```

每个 DistributedTensor 形参对应一个 `CommCtxType`；保持当前方向和显式数据流。名称可用于打印，但配对依据参数位置/类型契约。
同域实参不在编译期合并 ctx 参数。

### 4.2 修正 HOST 显式查询

本地代码当前在 HOST 中遇到 `get_comm_ctx` 先直接保留。目标是优先寻找已有形参 ctx，只有本层创建的窗口才需要运行时查询。

```cpp
// Current shape:
if (IsHostOrch(current_func_)) return call;
auto ctx = LookupExistingCtx(call->args_[0]);

// Proposed shape:
if (auto ctx = LookupExistingCtx(call->args_[0])) return ctx;
if (IsDomainOwnerOrch(current_func_) && IsLocallyBoundWindow(call->args_[0])) {
  return call;
}
// Otherwise diagnose unresolved provenance.
```

这是行为示意，`IsLocallyBoundWindow` 要复用实际来源分析。L4 接入时 `IsDomainOwnerOrch` 同时支持 HOST/CLUSTER_0，不能顺带放开设备侧 runtime 查询。

### 4.3 别名与 Call/Submit

复用现有返回 lineage、tuple、循环携带值和输出复用语义。ctx lineage 与窗口 view lineage 分开：域相同不代表 slot 或偏移相同。

同时检查：

- `Call` 与 `Submit` 都追加正确 ctx 实参，保留 deps、返回 TaskId 和 source span。
- 有 omitted Out 的 Submit 按 gap 映射 ctx 后缀，不能按 arg_index 直接索引全部 callee 参数。
- 同一参数的 view/reshape 沿用域绑定，但保留自己的 view 表达式。
- 不把不同 ctx 的动态 if/loop 合并当成此功能顺带支持；保留现有诊断。
- HOST 中纯形参查询应该消失，合法内部窗口查询可以保留。

## 5. 统一 ABI builder

新增 `include/pypto/codegen/distributed/entry_abi.h` 和 `src/codegen/distributed/entry_abi.cpp`，实现 [ABI 文档](02-abi.md) 的唯一布局算法。codegen 与 metadata writer 共享它。

绑定新增位置：

```python
# Proposed stub in python/pypto/pypto_core/codegen.pyi

def build_distributed_entry_abi(func: Function) -> DistributedEntryAbi:
    """Build the validated public/window/context argument layout."""
    ...
```

通过 `python/bindings/modules/codegen.cpp` 导出只读字段。不得只更新 C++ 或只更新 Python stub。
`python/pypto/backend/pto_backend.py` 的 distributed codegen 调用点与 `DistributedCompiledProgram` metadata 路径需要拿到同一版本计划。可重复调用纯 builder，但不能各自重写计算。

## 6. Codegen 的统一 binding

建议内部增加：

```cpp
// Proposed internal codegen state, not IR fields.
struct WindowBindingExpr {
  std::string domain_expr;
  std::string slot_name_expr;
  WindowViewExpr view;
};

WindowBindingExpr ResolveWindowBinding(const ir::ExprPtr& value);
std::string ResolveDomainExpr(const ir::ExprPtr& ctx);
```

`WindowViewExpr` 表示当前 codegen 已支持的 shape/layout/offset 表达式，不是新 wire descriptor。第一次实现只处理首期可表达的 view，不悄悄丢失信息。

### 6.1 本地域

进入 owning scope 时记录静态 binding：

```text
window_buffer -> (__comm_d0, "scratch")
```

### 6.2 外部参数

入口对每个参数按 ABI 解码，查域和槽一次：

```python
# Proposed generated entry fragment.
buffer_index = int(entry_args.scalar(abi_a_window_offset))
ref = decode_domain_ref(entry_args, abi_a_ctx_offset)
__dom_a = resolve_domain_for_run(orch, ref, run_domains)
if not 0 <= buffer_index < len(__dom_a.buffers):
    raise ValueError("window buffer index out of range")
__slot_a = __dom_a.buffers[buffer_index].name
```

`run_domains` 是每次调用局部的按完整 ref 去重表。入口校验在首个 chip 派发前完成。

```text
a parameter/aliases -> (__dom_a, __slot_a)
a_ctx               -> __dom_a
```

外部参数不再调用 `ScopeForWindowBuffer`。本地窗口继续走 scope 规则；不能为了放行外部参数把内部无 owner 错误也忽略。

### 6.3 下发

```python
ctx_a = __dom_a[w]
ta.add_tensor(ctx_a.buffers[__slot_a].tensor(shape_a, dtype_a), direction_a)
# Append other tensors and per-domain ordering tensors first.
ta.add_scalar(ctx_a.device_ctx)
```

当前 `ResolveCommCtxArg` 对显式 ctx Var 返回标量变量名的做法，不适合 L3 的 ref/handle。HOST→CHIP 必须从相应域表达式选 `domain[w].device_ctx`；L2→kernel 的 ctx 仍是标量原样转发。

## 7. 普通参数与入口兼容

独立 L3 保留当前 `_args/config/tensors/callables/sub_ids/_keep/world_size` 调用机制，增加仅新 ABI 入口使用的 `_entry_bindings` keyword。运行时按 artifact capability 决定是否传入，不能用 TypeError 重试掩盖真实错误。

嵌入 L4 的 L3 保留 #3531 的 `(orch, args, config)` 入口；由 TaskArgs adapter 构造等价绑定。两者 common binder 输出一致的 domain/slot 局部状态，编排 body 不按来源分支。
普通 tensor 的 host-control 传输标记、动态维度恢复与 CHIP callable metadata 必须保留。

## 8. collective 与依赖

共享 Host 表首期通过 L3→L2→kernel 读取/GET，不因支持外部参数就宣称所有 HOST builtin collective 已适配。
检查 builtin 中所有 `GetCommDomainHandleVar/ScopeForWindowBuffer` 使用点：

- 能以统一 binding 正确生成者接入；
- 要求同域、signal 容量或全域参与的操作做静态/运行时检查；
- 尚未适配的外部域 builtin 在编译期给出明确诊断，保留内部域原有支持。

显式 Out/InOut 的效果、alias 和排序 token 不能被新句柄包装隐藏。共享物理 Buffer 应保留同一 canonical identity；不要按 program 重新包成不相关身份。

## 9. 复杂度与验证

按函数建立一次参数/ctx 配对索引和一次窗口来源表；按调用边查表。避免对每个参数反复扫描整个 program 或所有 scope。目标 O(IR 节点数 + 调用边数 + 参数数)，局部有序映射允许 O(log N)。

需要新增/扩展 verifier：来源可解析、ctx 配对完整、ABI 后缀与 metadata 一致、别名 view 不丢失。直接构造 IR 的测试必须覆盖，不能只依赖 parser 校验。

正式文档更新位置：`docs/{en,zh}/dev/passes/46-materialize_comm_domain_scopes.md`、`48-materialize_dist_tensor_ctx.md`、codegen 和参数方向文档。L4 的入口、builder 与 codec 复用这些契约。
