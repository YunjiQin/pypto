# LowerL2TensorCollectives Pass

## 概述

`LowerL2TensorCollectives` 是托管集合通信（managed collective）的 CHIP/L2 通路。
它把写在 **CHIP orchestration 函数体**里的 `pld.tensor.*` 集合通信改写成对编译器
合成的 AIV kernel 的调用，使该集合通信成为调用方自身 pipeline 里的一个普通 task。

目前支持 `pld.tensor.all_to_all_v`，且要求 `core_num` 为编译期常量。

HOST 通路（[`LowerHostTensorCollectives`](50-lower_host_tensor_collectives.md)）
在上一层解决同一问题，做法不同：它把集合通信按设备扇出成**每个设备一次**
`builtin.tensor.*` chip dispatch。每次这样的 dispatch 都是一个完整的 L2
orchestration task，而它唯一的工作就是提交一个 AIV kernel，因此
`compute -> collective -> consume` 序列每张卡要花三次 L3 -> L2 往返。在本通路上
只需一次。

```text
HOST 通路                                 CHIP 通路（本 pass）
─────────                                ─────────────────────
L3 -> L2  stage task                     L3 -> L2  chip_pipeline
L3 -> L2  builtin collective dispatch              ├── stage         (AIV task)
          └── rt_submit_aiv_task                   ├── collective    (AIV task)
L3 -> L2  consume task                             └── consume       (AIV task)
```

## 在流水线中的位置

```text
... -> FuseCreateAssembleToSlice -> LowerL2TensorCollectives -> LegalizeSpmdLaunches -> DeriveCallDirections -> AutoDeriveTaskDependencies -> ...
```

这个位置是必要条件而非偏好。改写后的调用必须像任何其他 kernel 调用一样经过
[`DeriveCallDirections`](45-derive_call_directions.md) 和
[`AutoDeriveTaskDependencies`](46-auto_derive_task_dependencies.md)：正是这两个
pass 把合成 kernel 的参数方向转换成排序 `compute -> collective -> consume` 的
TensorMap 依赖边。放在它们之后改写，会让该 collective task 失去顺序约束。

它同样运行在 [`MaterializeDistTensorCtx`](51-materialize_dist_tensor_ctx.md)
之前 —— 后者会补上 kernel 需要的 `CommCtx` 实参（见下文 *ABI*）。

## 行为

对于一个 CHIP orchestration 函数体：

```python
@pl.function(type=pl.FunctionType.Orchestration)
def chip_pipeline(self, inp, out, stage, data, signal, counts, recv):
    stage, counts = self.stage_step(inp, stage, counts)
    data = pld.tensor.all_to_all_v(stage, data, signal, counts, recv, core_num=1)
    return self.consume_step(data, recv, out)
```

该集合通信被改写为：

```python
data = self.__builtin_all_to_all_v__fp32(stage, data, signal, counts, recv)
# INT8（RFC #2521 A1 规范负载）合成为 __builtin_all_to_all_v__int8，
# builtin_template_vars = "dtype_cpp=int8_t"
```

其中 `__builtin_all_to_all_v__{fp32,int8}` 是新增到 program 中的合成
`FunctionType.AIV` 函数：

| 方面 | 取值 |
| ---- | ---- |
| 参数 | `input, target, signal, send_counts, recv_counts` —— 规范化类型，**不是**调用点的类型：`input` / `send_counts` 声明为普通 `Tensor`，其余三个为 `DistributedTensor` |
| 方向 | `In, InOut, InOut, In, InOut` |
| 函数体 | `return target` —— 单条 `ReturnStmt`，永不参与代码生成 |
| Attrs | `builtin_template_dir`、`builtin_template_vars`、`require_contiguous_operands` |

每个 variant 只合成一个函数，被该 variant 的所有调用点共享。

### 为什么函数体是 `ReturnStmt` 而不是空声明

该 kernel 的实现是手写 builtin 源码，函数体不会被代码生成 —— 对 backend 来说空
声明就够了。之所以写成返回 `target` 参数的真实 `ReturnStmt`，是因为后续仍会读取
该函数的 pass 需要它：`ReturnParamsExplicit` 成立，且 `MaterializeDistTensorCtx`
能把返回的 `DistributedTensor` 解析回它写入的那个参数。返回 `target` 也与 public
op 的 window-as-result 语义一致，使调用点保持为一次普通 rebind。

## Kernel 源码：一份实现，两条通路

合成函数并不像
[external kernel](../language/04-external-kernels.md) 那样直接给出 `.cpp` 路径，
而是指向 builtin **模板包** —— 与 `builtin.tensor.all_to_all_v` 通过
`set_template_dir` 声明的是同一个 —— 外加渲染所需的替换项：

```text
builtin_template_dir  = ":pypto.runtime.builtins.collectives.all_to_all_v"
builtin_template_vars = "dtype_cpp=float"
```

PTO backend 把 `templates/kernel.cpp.in` 渲染到 chip 子构建的
`kernels/aiv/<name>.cpp`，并写入生成的 `kernel_config.py` —— 与 PyPTO 生成的
kernel 走同一条路径，区别只在于文本来自模板而非 ptoas。

`dtype_cpp` 是两条通路**唯一**的替换项，且取值相同，因此两边渲染出的 kernel
**逐字节相同**。ST 通过 diff 两条通路的渲染产物端到端断言这一点。

## ABI

两条通路以**相同**的实参布局到达 kernel：

| 槽位 | HOST 通路 | CHIP 通路（本 pass） |
| ---- | --------- | -------------------- |
| `args[0..4]` | `input, target, signal, send_counts, recv_counts` | 相同 |
| `args[5]` | `CommContext*` | `CommContext*` |
| `args[6..7]` | — | `args[5]` 的未读副本 |

两条通路都不传 rank 数标量。kernel 读 `CommContext::rankNum`，它与 HOST dispatch
过去传的 `domain_size` 是同一个数：`comm_derive_context` **按 comm domain** 建
context，其 `rankNum` 就是该域的 rank 数。去掉这个标量的代价是入口处多一次 GM
读取，换来的是一份共享源码。这在 CHIP 通路上也是唯一可行方案 —— 它根本无法计算
rank 数，`pld.system.nranks` 只有 InCore 代码生成，没有 orchestration 代码生成。

`args[6..7]` 的副本是 `MaterializeDistTensorCtx` **为每个 `DistributedTensor`
参数各追加一个 `CommCtx` 参数**的产物。合成签名中恰好声明三个这样的参数 ——
`target`、`signal`、`recv_counts` —— 因为无论调用点传入哪种类型，`input` 与
`send_counts` 都被规范化为普通 `Tensor`。因此该尾部恒为三项，且第一个 ctx 总是
落在 `args[5]`，因为尾部跟在全部五个 tensor 参数之后。三者全部解析到同一个
`device_ctx`，因为一次调用的五个操作数必属同一个 comm domain。

## 启动宽度

`core_num` 是请求的 block 上限 `L`。在本通路上它必须是编译期常量，因此 pass 自己折叠出准入的
block 数 `B = CalAllToAllVBlocks(P, L)`：当 `L < P` 时取 `L`，否则取不超过 `L` 的最大 `P` 的倍数。
`P` 是 `[NR, S]` signal 的静态 `NR`。

- `B == 1` 时 kernel 调用保持不变。
- `B > 1` 时在 kernel 调用上附加 `core_num=B` 与 `sync_start=True`。感知 block 的 barrier 要求所有
  准入的 block 同时驻留，而 runtime 会拒绝超出容量的 sync-start 启动，而不是只启动一部分。

启动规格放在调用的 attrs 上，而 attrs 不是 kernel 参数，所以上文的五操作数 ABI 不变。被校验的是 `B`
而不是 `L`：`signal` 必须满足 `S >= B`，因此 `L=10, P=8`（实际启动 8 个 block）接受 `S=8`。HOST
通路在其生成的 entry 中计算同一个 `B`；两处实现都对照 RFC #2521 的算例表固定。`NR` 必须等于运行期
通信域大小，因为 kernel 从 `CommContext` 读取 rank 数。

## 约束与诊断

| 条件 | 诊断 |
| ---- | ---- |
| `core_num` 不是编译期常量 | 拒绝 —— 本通路暂不支持运行期 `core_num` |
| `core_num` 不在 `[1, INT32_MAX]` 内 | 拒绝 —— 与 HOST 通路 entry 在运行期强制的上界相同 |
| `signal` 步长 `S < B` | 拒绝 —— `B` 是折叠后的 block 数（见 *启动宽度*），不是原始 `core_num` |
| `dtype != FP32 && dtype != INT8` | 拒绝 —— 支持 FP32 与 INT8（与 HOST 通路的 allowlist 一致） |
| 集合通信残留在非 HOST 的 orchestration 函数体中 | 被本 pass 自身的后置条件检查拒绝 |

残留检查覆盖除 HOST orchestrator 之外的每个 orchestration 函数体（HOST 交由自己的
通路处理，在五个 pass 之后）。InCore 函数体不检查：它们归 composite 通路
（[`LowerCompositeOps`](14-lower_composite_ops.md)）所有，且该 pass 早在 26 个
pass 之前就已运行，在这里重复报告会指向错误的 pass。

## 本 pass 不做的事

- 不创建 `CommDomain`，也不分配任何集合通信 staging buffer。L3 仍负责创建域、交换
  window 地址并绑定 window；L2 只消费本地视图和已经建好的 context。
- 不做按设备扇出。`device=` 派发仍留在 HOST orchestrator，每张卡一次
  `chip_pipeline`。
- 不产生嵌套的 L2 -> L2 dispatch。该集合通信是调用方 pipeline 的一个 AIV task，
  不是另一个 chip callable。

## 当前限制

- **运行期 `core_num`**。仅接受编译期 `core_num`。运行期值需要在生成的 orchestration 中计算 `B`，并做
  运行期 `S >= B` 检查，而 orchestration 目前没有抛出该致命错误的途径。HOST 通路因其 entry 运行在
  host 上而接受运行期值。
- **除连续性外，操作数校验是静态的，且别名只覆盖了一部分**。`pld.tensor.all_to_all_v` 的类型
  推导只拒绝仅凭操作数类型即可证明的违规：非 ND 布局、与紧凑步长不符的 stride
  向量、比 shape 更窄的 `valid_shape`，以及 `input` 与 `target` 是**同一个表达式**。
  它**不会**拒绝同一块 allocation 上的两个不同 `pld.window()` 视图 —— 类型推导在
  构造 Call 时就已运行，而 `DistributedTensorType::window_buffer_` 要到
  [`MaterializeCommDomainScopes`](49-materialize_comm_domain_scopes.md)（pass 46）
  才被绑定。整块 allocation 层面的互不相同是 **HOST 通路**的保证：
  `LowerHostTensorCollectives` 能在同一个 `host_orch` 函数体内把每个操作数溯源回
  其 `WindowBuffer`，并对五个操作数运行 `CheckPairwiseDistinctWindows`。本通路
  —— 与 InCore composite 通路一样 —— 看到的操作数是外层 pipeline 的参数，没有这样
  的溯源能力，因此在这里「窗口两两互不相同」是调用方的义务。**连续性在本 pass 中于编译期判定，
  并保留运行期兜底**。类型推导看不到切片的步长（切片类型不携带 stride），而运行期判定是按
  rank 独立做出的：若只抑制某一个 rank 的提交而其对端仍然启动，对端会阻塞在 kernel 的
  `ExchangeBarrier` 上（表现为挂死，而非拒绝）。因此本 pass 沿着所有可能到达的值（赋值、
  循环携带值以及 `if`/`for`/`while` 的结果；循环结果还会追溯其初值，因为零次迭代的循环
  返回的就是初值）追溯调用中按原样书写的每个操作数，并采用
  **白名单**：参数、窗口、`tensor.create` 以及 `pl.submit` 交由运行期分配的 Out 是行主序的根；
  `reshape`、`view`、`set_validshape`、`assemble`（其目标）、collective 的结果（其目标）、
  kernel 或 `pl.submit` 的结果或其元组元素（其返回的实参）以及仅收窄静态形状 tensor 首维的切片
  会被继续追溯到其来源；其余一律拒绝，包括 `tensor.transpose` 及其他任何切片；链上任何非 ND
  布局、或显式 stride 不等于紧凑 stride 的值也会被拒绝（上述各步都假定来源是紧凑行主序的，而切片
  会丢弃其父 tensor 的 stride：DN 或带填充 stride 的 tensor 的首维切片并不连续，而 DN tensor 的
  ND `view` 会降级为运行期转置）。一些安全的程序也会被拒绝 ——
  `tensor.write` 的结果、`[1, W]` tensor 的 `[1, N]` 切片、偏移不是字面元组的切片、被调函数
  ReturnStmt 未直接写为参数的结果 —— 因为漏掉一次接受是安全的，错误的接受则不是。合成的
  kernel 仍带有 `require_contiguous_operands`，orchestration codegen 会在提交之前为每个
  tensor 操作数生成一条 `is_contiguous` 检查，并在第一个违规处锁存
  `SIMPLER_ERROR_INVALID_ARGS`，作为调用方提供的操作数的兜底：参数打包会拒绝非连续的 host
  torch tensor，但预先构建的运行期 `Tensor` 只在这里检查。另一项需要运行期形式的检查
  （signal stride `>= B`）仍由 lowering 阶段静态校验：`S` 与折叠出的 `B` 都是编译期常量。
- **rank 数**。kernel 读取 `CommContext::rankNum`，因此当显式设备子集小于 context
  的 rank 数时，其行为与传入 comm domain `domain_size` 的 HOST 通路不同。
- **「同属一个通信域」是未经检查的前置条件**。kernel 通过单个 `CommContext`
  （`args[5]`）解析所有对端地址，因此绑定到不同域的操作数会寻址到错误的远端窗口。
  HOST 通路用 `FindScopeForBuffers` 强制了等价约束——它能直接看到 window buffer；
  本通路做不到：comm domain 在 `MaterializeCommDomainScopes`（pass 46）与
  `MaterializeDistTensorCtx`（pass 48）之前没有 IR 表示，而这两者都在本 pass
  之后运行；到那时该集合通信的操作数已是外层 pipeline 的参数，把它们追溯回绑定它们
  的 host window 需要目前不存在的跨函数分析。改为比较追加的 `CommCtx` 实参也不成立：
  每个 `DistributedTensor` 参数各自生成一个，单域调用本就携带多个互不相同的 SSA 值。

## 测试

- `tests/ut/ir/transforms/test_lower_l2_tensor_collectives.py` —— 改写后的形态、
  合成签名与方向、模板 attrs、variant 共享、InCore 透传、启动规格附加与 block 数折叠、
  容量检查、运行期 `core_num` 拒绝、INT8 variant。
- `tests/ut/codegen/distributed/test_builtin_collective_kernel_source.py` ——
  HOST 与 CHIP 通路对 FP32 / INT8 渲染逐字节相同的 kernel。
- `tests/ut/ir/transforms/test_lower_composite_ops.py` —— composite 通路把
  CHIP orchestration 中的集合通信交给本 pass，并在 InCore 函数体中拒绝
  `core_num != 1`。
- `tests/ut/ir/transforms/test_lower_host_tensor_collectives.py` —— 把窗口别名
  的拒绝固定为 **HOST 通路**的保证（`CheckPairwiseDistinctWindows`），这正是上文
  *当前限制* 一节所对照的另一面。
- `tests/st/distributed/collectives/test_l2_tensor_all_to_all_v.py` —— P=2/4 的
  硬件正确性；InCore 与 HOST 通路同样会跑的 0 / 1 / 容量 / 超容量 / 负数计数矩阵，
  它把三条通路约束到同一份链路 golden；外加三条结构断言：不产生 builtin chip
  dispatch、builtin kernel 被渲染进 pipeline 自己的子构建、HOST 通路渲染出逐字节
  相同的 kernel 源码。

## A5 URMA 索引交换

`pld.tensor.indexed_all_to_all_v(input, target, signal, send_counts,
recv_counts, index, core_num=L)` 使用托管 L2 任务流程及独立的内核模板。
六个 Tensor 操作数之后，第一个 CommContext 位于槽位 6。input、index 和
recv_counts 可以是本地 Tensor；send_counts 必须是窗口。
包括 `index` 在内的全部六个操作数，与普通算子一样接受连续性检查与兜底（见[当前限制](#当前限制)）。
`L` 为编译期正整数，表示精确启动核数。全部 AIV 在远端提交与 CQ 等待之间
分担本地搬运；超过 P 的 AIV 只参与本地搬运。仅支持 A5 URMA PER_PEER 传输，
每个目标队列只有一个写入者。调用者必须为共享队列的交换建立顺序；即使
payload 窗口不同，复用同一个 InOut signal 也能建立此依赖。
不支持 HOST 和 InCore 调用。参见[用户约定](../../user/distributed/20-all_to_all.md#a5-urma-索引-alltoallv)。

## CCU broadcast

此 pass 同时支持 Ascend950 上显式的 `pld.tensor.broadcast(..., engine="ccu")`。
它生成 AIV builtin，模板身份为 `:pypto.runtime.builtins.ccu.broadcast`，按 root/shape
区分变体，三个参数 `target`、`signal`、`order` 均为 `InOut`，函数体为 `return target`，
并显式设置单块 launch。支持 FP32、静态正尺寸、完整连续 ND target，INT32 `[2, 1]`
signal，root 0/1 和 CHIP 直线调用。默认 broadcast 保持原有限制。

在方向和依赖推导前，在 CHIP 函数体内为每个 CCU function 分配一个普通 UINT8 `[1]`
tensor。function key 由 CCU 模板及 dtype/rank/root 特化条件组成；shape 只区分 AIV wrapper
签名，因为 element count 是 CCU 调用参数。共享同一入口的调用复用 `InOut` token，形成
写后写（WAW）依赖链。不同入口使用独立 token，除非 payload 或 signal 数据依赖要求排序，
否则允许并行。token 属于单次 CHIP 调用，不涉及远端访问或完成语义；不需要 Host 窗口、
隐藏 CHIP 参数或 function attr。Host Call/Submit 的签名、输出空缺、返回映射和依赖保持不变。
支持直接窗口/SSA 别名、常量设备或单个静态 rank 分发循环；动态、嵌套及条件绑定明确报错。

`MaterializeDistTensorCtx` 仅为 builtin 追加 target/signal 的 context，形成
`target, signal, order, ctx_target, ctx_signal`。本地 order tensor 不需要 context。
target/signal 同域与实际 rank 数匹配仍为调用前置条件，不新增算法 scratch。
源码渲染识别该模板身份并明确报错，等待 PR2 接通；入口去重和编号必须与 token 共享使用相同
CCU function key。Engine 收集现已识别两种 CCU collective；broadcast 源码与运行时接入尚未交付。

CCU builtin 的 signal/order 在 `InOutUseDiscipline` 中属于可复用控制句柄：保留 `InOut`
任务效果，但不要求通过 SSA 返回值重新绑定。target 继续遵循普通的重新绑定检查。
token 在 CHIP 函数体内分配，因此不需要对 CHIP 参数添加豁免。

## CCU allgather

`pld.tensor.allgather(input, target, signal, engine="ccu")` 支持 Ascend950 CHIP
编排、BF16 输入 `[1, N]`、目标 `[NR, N]`、INT32 signal `[NR, 1]`、2–64 个 rank，
以及完整连续 ND 静态正尺寸视图。三个操作数必须属于同一域；输入和目标不得别名。
按每个 callable 的 rank 数检查 Host 绑定，复用连续性来源分析和本地 UINT8 order token；
拒绝 HOST/InCore、动态尺寸、循环、条件调用和嵌套 CHIP dispatch。
同域关系及跨 rank 调用顺序一致仍是调用前置条件；所有 rank 的目标视图必须具有
相同的窗口内偏移。64 rank 上限来自 simpler 注册 ABI，并非 AllGather 算法限制。
Host dispatch 支持常量设备 ID 或最多 64 个设备、步长为 1 的静态 rank 循环。

AIV builtin 的模板身份为 `:pypto.runtime.builtins.ccu.allgather`，ABI 是
`input, target, signal, order, ctx_target`，方向依次为
`In, InOut, InOut, InOut, Scalar`，返回 target 别名。公共输入仍为
DistributedTensor；builtin 的 input/signal 使用普通 Tensor 参数类型，以便
context 物化只追加一个域 context，满足 simpler 的任务准入契约。
不同 rank 数特化为不同入口；相同入口的不同 shape 调用共享 order，形成完整异步完成依赖；signal/order
作为控制句柄免于返回值 rebind 检查。单个 AIV block 用原始
`get_async_ctx(args)` 提交 typed `CcuCallArg` 元数据；CCU 完成本地和远端
数据搬运，保留 BF16 位模式。

### Native source contract

对齐 [simpler #2986](https://github.com/hw-native-sys/simpler/pull/2986) 的
`a6566e3197302b2ac3c7a015adac6861c9d41d5e`。请求展开为 `4 + 3 * NR` 个 uint64 word，
共 `8 * (4 + 3 * NR)` 字节，参数对齐为 64：

| Words | Meaning |
| ----- | ------- |
| 0–1 | 本地输入 W 地址和 token |
| 2 | BF16 分块字节数 |
| 3 | 输出槽位字节偏移：源域 rank × 分块字节数 |
| 4 + 3r, 5 + 3r | 域内目标 r 的输出基址 W 地址和 token |
| 6 + 3r | 从域 W 表解析出的目标 r 的物理 L2 rank |

AIV 根据 tensor start offset 与本地 windowsIn 算字节偏移，由 runtime 解析
CCU 地址/token；不把 AIV 映射当作 CCU 地址。域 rank 确定输出行顺序，物理
L2 rank 选择 registration channel，支持重排子集的映射。

Native 定义等待 trigger、完成就绪握手后，对每个 peer 发起远端 `Write`，
对自身发起 `LocalCopy`。每批最多 16 个目标，各占一个传输事件位；整批完成后
复用 16 位事件，处理后续批次。所有批次和 peer 均完成后才发布 done。
就绪和完成均先向所有 peer 发布通知，再等待各 peer，避免依赖 peer 遍历顺序。
就绪和完成使用不同通知位。有限 CCU 循环处理静态统计的调用次数，
最后一次业务请求发布 done 后正常退出；不增加退出任务、scratch 或 signal 协议。
Runtime 先回收异步 done 再释放任务依赖。

HBM 数据通路参考 [PTO-ISA CCU 指南](https://gitcode.com/cyz00/pto-isa/blob/ccu_test/docs/zcy/ccu_collective_usage_zh.md)
提交 `6a60d0d3525aeecd76b0138018cc39fabf708794` 中的 `CcuAllGatherOp`。
A5 CCU translator 缺少乘法，因此由 AIV 计算槽位偏移，CCU 对各输出基址只加一次
该偏移。实现保留 simpler 的 ready/trigger/done 契约，不依赖参考分支私有头文件；
未引入 `CcuFusionSession`、其描述符表生命周期或 LoopGroup/MS 性能路径。
因此，参考 LoopGroup 调度器的 32 KiB 分块限制不适用于本次源码生成契约。

Codegen 按模板身份及特化参数去重，每个提取后的 CHIP callable 独立排序编号，
输出 `kernels/aiv/*.cpp` 和 `ccu/module_<id>.cpp`。`kernel_config.py` 中
`CCU_MODULES` 按 ID 顺序记录源码、入口、参数布局及调用次数；每个模块通过
`simpler_ccu_module_manifest` 导出一个 kernel record。后续二进制构建需根据
`CCU_BUILD_IDENTITY` 指定的 SDK/ISA 头文件生成 SHA256，链接 32 字节的
`pypto_ccu_sdk_identity` 和 `pypto_ccu_isa_identity`，保留模块顺序并显式注册 CCU。
`CCU_RUNTIME_SUPPORTED = False` 标明仅交付源码；本次不构建模块、不装配
ChipCallable、不宣称硬件验证。Broadcast 保留源码尚未交付的明确诊断。

`MaterializeCommDomainScopes` 在全 Program 范围检测原始 CCU Call/Submit 和
lowering 后的 builtin 身份，包括未调用函数；合并 ccu 与 urma 等已有需求，
重复物化保留窗口绑定。CPU 测试覆盖单 CHIP 与 Host→CHIP 源码输出。

源码生成边界也支持多个独立 CHIP 入口；现有公共 `ir.compile` 多入口执行元数据
封装在写出源码后拒绝 `CommCtxType`。该封装支持隐藏 context 前，公共多 callable
编译应使用 Host→CHIP 通路。
