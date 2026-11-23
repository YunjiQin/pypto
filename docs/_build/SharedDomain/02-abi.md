# 参数 ABI、metadata 与冷加载

本文件固定 SharedDomain 与 #3531 共用的规则。实现中不得同时存在“本地 submit 一种排列、L4 sender 另一种排列”的算法。

实现先后为 SharedDomain → L4：本地 L3 先交付 ABI builder、metadata 与 binder。L4 的发送/接收适配随后复用这些实现，不能把本方案挂在尚未落地的 L4 loader 或 handler 上。

## 1. 三套索引

| 名称 | 含义 |
| --- | --- |
| user_index | 用户 `submit` 的业务参数位置，不含 ctx 后缀 |
| ir_index | lowering 后函数签名位置，包含追加 ctx |
| pool_offset | tensor/scalar pool 内的起始位置，与 ir_index 独立 |

举例：

```text
Source:
  host(ids, table_a, table_b, out, step)

IR:
  host(ids, table_a, table_b, out, step, table_a_ctx, table_b_ctx)
```

| ir_index | kind | user_index | pool | offset | width | 配对 |
| ---: | --- | ---: | --- | ---: | ---: | --- |
| 0 | tensor | 0 | tensor | 0 | 1 | — |
| 1 | distributed_tensor | 1 | scalar | 0 | 1 | ctx ir_index=5 |
| 2 | distributed_tensor | 2 | scalar | 1 | 1 | ctx ir_index=6 |
| 3 | tensor | 3 | tensor | 1 | 1 | — |
| 4 | scalar | 4 | scalar | 2 | 1 | — |
| 5 | comm_ctx | 无 | scalar | 3 | 3 | window ir_index=1 |
| 6 | comm_ctx | 无 | scalar | 6 | 3 | window ir_index=2 |

```text
tensors = [ids, out]
scalars = [a_index, b_index, step,
           a_owner, a_domain_id, a_generation,
           b_owner, b_domain_id, b_generation]
```

同域 a/b 的后三字段数值相同，但两组 ABI 槽都存在。不能按 runtime 相等性压缩。

## 2. 唯一布局算法

布局 builder 在 `MaterializeDistTensorCtx` 之后、最终签名已确定时运行：

1. 识别类型为 CommCtx 的追加后缀；按照 pass 的契约与业务 DistributedTensor 顺序配对。
2. 检查后缀数量、类型、方向及完整配对；不按变量名或 `_ctx` 后缀推断。
3. 枚举业务形参。普通 tensor 分配 tensor offset；DistributedTensor 分配一个 scalar offset；用户 scalar 分配一个 scalar offset。
4. 按 ctx 后缀顺序，每个 ctx 分配三个 scalar offset。
5. 发射调用时分两遍：先所有 tensor，再按 scalar offset 发所有 scalar。
6. 记录固定 public arity、每个参数的 view/direction，以及派发成员与 engine 要求。

初版不开放用户在 HOST 源签名手写 CommCtx；发现非编译器契约的显式 ctx 参数应诊断，不能误剔除为隐藏参数。公开参数映射以实际 lowering 签名为准，保留当前 return-lowered Out 的调用规则。

用于回归的混排样例：

```text
IR: (x_dist, lhs, n, y_dist, out, x_ctx, y_ctx)
tensors: [lhs, out]
scalars: [x_index, n, y_index, *x_ref, *y_ref]
ctx offsets: x=3, y=6
```

这与 [L4 参数排序](../L4/03-codegen.md)相同。

## 3. builder 与持久化只有一个定义来源

建议新增内部 C++ helper：

```cpp
// Proposed: include/pypto/codegen/distributed/entry_abi.h
struct EntryParamBinding {
  size_t ir_index;
  std::optional<size_t> user_index;
  EntryParamKind kind;
  ArgumentPool pool;
  size_t offset;
  size_t width;
  std::optional<size_t> paired_ir_index;
};

struct DistributedEntryAbi {
  int version;
  size_t user_param_count;
  size_t tensor_count;
  size_t scalar_count;
  std::vector<EntryParamBinding> bindings;
};

DistributedEntryAbi BuildDistributedEntryAbi(const ir::FunctionPtr& function);
```

此处是职责草图，实际补齐 includes、enum、约束字段和项目风格。它是 codegen metadata，不是新 IR 节点或 runtime 数据。

- `DistributedCodegen` 调该 helper 决定入口绑定与发送偏移。
- `python/bindings/modules/codegen.cpp` 暴露只读结果和同一 builder。
- `python/pypto/pypto_core/codegen.pyi` 同步声明。
- `DistributedCompiledProgram` 调同一个 builder 提取/保存布局，Python 不重写一遍 offset 算法。
- loader 只解析并严格验证持久化计划，不能在缺少 IR 时按名称猜布局。

Python binder 可从这份计划派生只读的 `WindowParamBinding` 便捷记录，包含
`user_index`、`tensor_ir_index`、`ctx_ir_index`、`window_offset`、`ctx_offset` 和 view 约束。
`entry_abi.windows` 是这些记录的列表；offset 直接取已有 binding，通过配对索引连接，
不在 Python 中重新分配槽位。这也是 runtime 文档中各 helper 的输入。

## 4. metadata 目标形态

以下为第一节样例的绑定记录，省略普通 params 已有的完整 shape/dtype/direction 字段：

```json
{
  "schema": 4,
  "l3_entry": {
    "abi_version": 1,
    "user_param_count": 5,
    "tensor_count": 2,
    "scalar_count": 9,
    "bindings": [
      {"ir_index": 0, "user_index": 0, "kind": "tensor", "pool": "tensor", "offset": 0, "width": 1},
      {"ir_index": 1, "user_index": 1, "kind": "distributed_tensor", "pool": "scalar", "offset": 0, "width": 1, "paired_ir_index": 5},
      {"ir_index": 2, "user_index": 2, "kind": "distributed_tensor", "pool": "scalar", "offset": 1, "width": 1, "paired_ir_index": 6},
      {"ir_index": 3, "user_index": 3, "kind": "tensor", "pool": "tensor", "offset": 1, "width": 1},
      {"ir_index": 4, "user_index": 4, "kind": "scalar", "pool": "scalar", "offset": 2, "width": 1},
      {"ir_index": 5, "kind": "comm_ctx", "pool": "scalar", "offset": 3, "width": 3, "paired_ir_index": 1},
      {"ir_index": 6, "kind": "comm_ctx", "pool": "scalar", "offset": 6, "width": 3, "paired_ir_index": 2}
    ]
  }
}
```

`schema=4` 是基于本地 schema 3 的目标下一版；实施时若已占用，协调使用新的真实版本号。`abi_version=1` 只标识这里定义的入口规则。

顶层 `params` 继续保存用户可见的业务参数 metadata；其窗口/普通 tensor 身份由 `l3_entry.bindings.kind` 明确区分。ctx 不伪装成 dtype=uint64 的普通业务参数。

每个 window 的静态形状、dtype、layout、方向及 view 范围约束必须可从 metadata 恢复。不能把本次 `domain_ref`、buffer_index、设备指针、owner 对象或 sources 序列化到产物。

## 5. 版本与旧产物

采用以下明确策略，避免旧 loader 忽略扩展块后将 DomainTensor 当 torch.Tensor：

| 产物 | 新 writer | 新 loader | 旧 loader |
| --- | --- | --- | --- |
| 没有外部窗口的普通 L3 | 可继续写现有 schema 3 | 走 legacy adapter，保留行为 | 按既有方式工作 |
| 带外部窗口的新 L3 | 写新 schema + 必需 l3_entry | 校验后生成新版 binder | 版本不匹配，提示升级/重编译 |
| 未知 ABI 版本 | 不生成 | 明确拒绝 | 不保证支持 |

不要全局放宽 `_load_meta` 的严格检查。distributed loader 按明确支持的 schema 分支；单卡 `CompiledProgram` 的 schema 和 tensor 参数消费者不应因本功能无关变更。

将 public metadata 提取做成分布式专用 helper：根据 ABI 的 user_index/ir_index 映射提取，不让通用 `_extract_func_param_infos` 遍历隐藏 CommCtx 后报错。现有 L2 参数处理独立回归。

生成模块还应包含对应 ABI version/布局指纹，加载时与 metadata 核对，避免把另一个产物的 sidecar 配到错误 Python 入口。指纹不进入每次业务 TaskArgs。

## 6. 本地 submit 与 L4 sender

### 本地 prepared submit

保留既有 kwargs 入口和普通 tensor 绑定。新增 dispatch-local `EntryBindings`，存放业务参数及自动产生的标量槽；传给支持新 ABI 的生成入口。

```text
DomainTensor object
    -> validate type/owner/member/view
    -> scalar slot = buffer_index
    -> paired ctx slots = encode(domain_ref)
    -> frame-local bindings
```

生成入口的 common binder 通过布局取 ref/index，构造 `(domain_expr, slot_expr)`。不要通过全局字典或修改 compiled.state 保存本次域。

### L4 sender

调用点 codegen 从实际窗口解析 ref/index，按同一计划构造 TaskArgs。借用/转发形参沿用其当前绑定。不要让 L3 sender 针对“来自 L4 的窗口”另选函数体。

远端入口从 TaskArgs 得到与本地 EntryBindings 等价的值；普通 tensor 的节点路由遵守 L4 既有设计。本地与远端可以有入口适配差异，但窗口 body 只消费统一绑定。

## 7. ref 与 index 校验

- buffer_index 是非负整数，并满足 `index < len(domain.buffers)`；不能接受 Python 负下标语义。
- owner_instance_id 是八字节，按 little-endian 转为 uint64；domain_id/generation 也必须符合完整 uint64 表示。
- 不通过有符号溢出或截断接受无效值；若底层 scalar API 是有符号载体，codec 要按契约保留 64 位模式并双向测试最高位。
- 先从 `orch.get_domain(ref)` 获得有效且授权的 handle，再解引用 index 与本地 context；ref 不能转成跨进程指针。
- 验证 ordered specs、声明 view 和 named buffer 的逻辑/物理范围。padding 不是可读表数据。
- metadata 的索引、配对、pool 区间必须完整无重叠，ctx width 必须为 3。未知 kind、遗漏配对、错序后缀在加载期拒绝。

## 8. shape 与 view 的边界

本版不增加 offset、shape、stride 专用 wire 槽。静态 ND view 由形式参数和 buffer 定义恢复，运行时 shape 要一致。
已有业务 scalar 能表示的动态 shape，只有当其表达式、计算顺序和边界校验同时出现在双方的入口计划中时才能支持。初版可以明确只开放静态窗口形参；普通 tensor 的动态维度不能因此退化。

## 9. 同域去重与 L2 ABI

ctx ABI 不去重；只有以下 runtime 状态去重：

```text
key = (owner_instance_id, domain_id, generation)
lookup/run lease -> one per distinct key per dispatch
ordering token   -> one per distinct key and local member
window binding   -> one per formal, including its own buffer_index/view
```

在 HOST→CHIP 上，窗口成为真正的 tensor；ctx 只占一个 scalar，值为对应 chip 的 device_ctx。ordering token 为最后的 tensor，然后才是业务 scalar 和 ctx。
IR `Submit` 有未提供 Out 时，ctx 在实参与形参中的索引存在 gap；使用 call-like 参数映射，不把 L3 入口的 in-place arity 规则错误套到所有 IR Submit 上。
