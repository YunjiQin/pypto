# Kernel mode 集成基础

内部 torch 适配器为后续 kernel executor 描述借用的 NPU 参数，不新增公开 kernel
执行入口。已有 JIT、编译 program 和 Worker 调用保持当前行为。

## 公共签名与 L2 ABI

`pypto._kernel_abi.KernelABI` 对现有 `ParamInfo` 建立快照。program 和 kernel
共用参数 JSON 序列化，不新增 kernel 参数类型或 native dtype tag 表。Tensor/Scalar
池索引从签名顺序派生。`bind()` 要求全部 Out/InOut 实参，并返回两个参数池及外部
Tensor 返回别名，支持重复别名和只读输入返回。

descriptor 记录目标配置和构建上下文传入的**编译 SDK revision**。
`from_record()` 拒绝非规范记录，`require_compatible()` 核对完整签名、返回映射、目标和
SDK；这些操作不加载可选 runtime 或初始化设备。kernel Scalar 声明目前覆盖 FP32、
各宽度整数、bool 和 index，不改变 program 更宽的 Scalar 映射。descriptor 的维度为正
u32 或 `-1`，rank 为 1–5，参数池容量与固定的 L2 SDK 一致。

program 提交构造无地址的 `TaskArgs`，由 Simpler 解析为 `ChipStorageTaskArgs`。
后续 Torch adapter 将从借用的设备存储构造同一种边界容器。两者均进入 runtime 的
`EntryArgsStorage`，随后转成 orchestration 使用的 `ChipTaskArgs`。
`ScalarArgument.encode()` 把正确类型的 ctypes 值交给现有 `scalar_to_uint64`，
与 program 提交共用 Simpler 编码器。

`runtime.kernel._compat.inspect_kernel_capabilities(sdk_revision)` 先执行既有 runtime
pin 检查，再要求 native build 身份与编译 SDK 一致。当前审定 revision 为 `c1f89ea4`。
其 Python 封装支持 `handle = worker.register_callable(callable_)` 和
`unregister_callable(handle)`，PyPTO 不分配 native callable slot。borrowed-context
初始化、caller-stream 提交和 run 退休在 native 接口单独检查。这些检查仅描述 runtime
前提，不表示 Torch executor 已接通。其他 revision 需复核契约后启用。

旧 supported bit 不关闭 native eager 能力。graph capture 独立标记为不支持；保留诊断
要求逐 run 配置和 flush。缺失能力抛 `NotImplementedError`；SDK/native 不匹配、可选依赖
错误和 native 异常保留原错误类型。本阶段不新增 Worker 初始化、注册或 launch。

## 调用元数据与所有权

[`CallSignature`](../../../../python/pypto/torch/interop.py) 一次性复制共享的
`ir.param_info.ParamInfo` 签名。`describe_call(args)` 复用 `bind_complete_args`，
所有 Out/InOut 参数必须在查询框架上下文前完整传入。返回别名（return alias）使用
完整参数列表中的 Tensor 索引，不能指向 Scalar 或不存在的参数。

每次调用创建新的不可变 `CallFrame`：

| 字段 | 含义 |
| ---- | ---- |
| `tensors` | 按签名顺序保存 Tensor，包含原始 `param_index`、方向、dtype、shape、stride、format、逻辑地址和 storage 边界。 |
| `scalars` | 按签名顺序保存本次类型化基本值及原始 `param_index`；可变 ctypes 输入被复制。 |
| `device_index` | 本次 NPU 当前设备，与全部 Tensor 核对。 |
| `stream` | 本次 torch_npu 当前 stream 对象，由 frame 保持引用。 |
| `return_tensors` | 经校验的返回别名选择的原始调用方对象。 |

Tensor/Scalar 的排列不定义 native ABI 布局。后续 native 参数编码需同时消费这些值
和经过校验的 kernel descriptor。地址、Scalar 实际值和 stream 属于每次调用状态，
不进入编译 key 或持久 metadata。

frame 持有每个 Tensor 及其 storage 对象。`alias_result()` 返回 `None`、单个已有
Tensor 或已有 Tensor 元组，不分配业务输出。仅有 Python 引用不能保护 frame 释放后
的设备异步使用；native 队列所有权和 allocator stream 记录属于独立的 launch 层工作。
frame 使用期间，调用方不能 resize 或使借用的 storage 失效。

## 校验规则

- 接受同一设备上的真实 NPU torch Tensor。拒绝 CPU、Meta/Fake Tensor 和 Worker
  自有对象，不自动选择 program mode。
- 对照 `ParamInfo` 校验 dtype 和 rank。静态 carrier 维度必须匹配；`-1` 维度使用
  本次大小。打包 FP4 直接使用 `ParamInfo` 已提供的 carrier shape，不再次展开。
- 要求 base NCHW (0) 或 ND (2) 格式的连续 strided view，保留非零 storage offset
  和逻辑地址。其他格式、转置 view、未解析的 conjugate/negative view 及无效 storage
  边界明确报错，不进行复制或格式转换。
- 显式 `layout="NZ"` 且 `platform="a2a3"` 时允许只读 FP16/BF16/INT8 的原生
  format 29 输入；物理维度必须对应完整、无 padding、零 storage offset 的 NZ 矩阵。
  物理描述符查询按需加载可选 native adapter；UT 仅替换该查询边界。
- 空 view 不访问任何元素，因此非负 storage offset 可以超过 storage 容量；
  仅对非空 view 校验访问范围上界。
- 允许字节区间完全一致的别名（可具有不同 dtype/shape）、同一 storage 上互不相交的 view，以及重叠的
  只读 view。涉及 Out/InOut 的部分重叠需要更丰富的别名契约，当前明确拒绝。
- 使用 program Scalar 类型映射复制 Python 或类型匹配的 ctypes Scalar。
  整数必须位于声明类型范围内，不将浮点数静默截断成整数，拒绝有限浮点值溢出。这不改变 JIT 的运行时
  Scalar 或常量分类。
- autograd 开启时拒绝需要梯度的 Tensor；`torch.no_grad()` 推理可以借用模型参数，
  但不提供 backward 实现。

参数校验后，适配器在**每次调用**查询当前 device/stream，要求当前设备与 Tensor
设备一致，并核对 stream 设备。不切换设备、不缓存首个 stream、不读取 `npu_stream`，
也不同步或入队。这些上下文查询可能初始化 torch_npu 自身的框架上下文，但不创建或
初始化 Simpler/PyPTO Worker。

## 内部 schema 与 Fake/Meta 辅助

`pypto.torch.registration.RegistrationSignature` 复制相同的 `ParamInfo` carrier shape
及返回参数索引。`schema(name)` 将 Out/InOut Tensor 标记为可写，并把每个 Tensor
返回值关联到对应输入的别名集合（alias set）。全部参数均须传入，不推导输出分配。
Scalar 输入映射为 dispatcher 的 `SymInt`、`float` 或 `bool`。`SymInt` 接受普通整数，
并在 dispatch 中保留符号整数，包括没有具体值提示的符号。Scalar 校验保留 dtype
范围检查，不将符号转换为 Python 整数。拒绝直接返回只读输入、Scalar 输出、
纯 Scalar 算子、非法名称与返回别名，以及 UINT64 Scalar：dispatcher 的有符号整数
类型无法表达完整 UINT64 范围。返回别名必须指向 Out/InOut Tensor，因为 dispatcher
schema 检查器不允许直接返回只读输入对象。

`fake(*args)` 接受 FakeTensor 或 Meta Tensor，校验 dtype、rank、静态 shape、
连续性、设备一致性及仅推理约束，然后返回声明的原始输入对象。动态维度及符号整数
Scalar 保持符号形式。直接返回输入保留 stride、storage offset 与别名身份，包括
空切片。该辅助不读取 storage 或地址，不查询 NPU format/device/stream，不分配
业务输出，也不调用 Worker。抽象 Tensor 无法证明真实物理布局与 storage 重叠情况，
因此这些校验仍由真实调用适配器执行。

`define(library, name)` 仅在调用方持有的 `torch.library.Library` 中定义 schema
并注册 fake kernel；调用方负责保留 library 对象及其注册生命周期。重复定义，包括
同名不同签名，均抛出 PyTorch 的重复定义错误，不替换已有定义。导入或重新加载模块
不注册算子。`pypto.torch` 不新增公开注册 API，本阶段也不安装真实设备 kernel。
如果当前 PyTorch 缺少 `torch.library.register_fake`，则回退到
`torch.library.impl_abstract`（PyTorch 2.2–2.3 提供），并保留调用方 Library 的
生命周期管理。两种 API 均不可用时，在安装 schema 前明确报错。该可选辅助要求
至少存在其中一种 API；此回退不新增 PyTorch 2.0–2.1 支持，也不调整整个包的最低依赖版本。

测试使用临时 namespace 和 CPU 实现夹具。在 PyTorch 2.6 上，无 dispatcher 返回值
的纯修改 schema 通过全部
[`torch.library.opcheck`](https://docs.pytorch.org/docs/2.6/library.html#torch.library.opcheck)
检查以及 `torch.compile(backend="aot_eager", fullgraph=True, dynamic=True)`；
测试 wrapper 在调用算子后返回调用方传入的输出 Tensor。通过注册后的 `torch.ops`
验证有具体值提示（backed）和无具体值提示（unbacked）的整数符号原样到达 fake kernel，
且不添加等值 guard；另以 shape 派生的 Scalar 验证不同输入尺寸复用同一编译图。
API 选择测试在 PyTorch 2.6 上模拟旧注册入口，验证 Fake/Meta dispatch、重复定义拒绝
及 Library 清理；不据此宣称旧版 PyTorch 已通过端到端编译器兼容性验证。
带返回别名的 schema 分别
验证 schema 正确性与 Fake/Meta 行为，不把这些检查视为该类算子已支持函数化
（functionalization）或编译执行。正式 kernel 注册、设备执行、autograd 与最终编译器
接线仍属于后续工作。

## 可选依赖与范围

`import pypto.torch` 不导出执行 API。导入该包、`interop` 或 `registration` 模块不请求 torch_npu、
Simpler 或 native launch 扩展；`torch` 仍是 PyPTO 的常规依赖。真正描述 NPU 调用时
才按需加载 `torch_npu`，缺失时给出针对性的错误信息。

本基础能力覆盖 metadata 校验、Python 调用 frame 所有权及内部 schema/Fake 辅助，
不表示 kernel launch、
taskQueue 顺序、allocator 安全、eager 数值执行或 ACLGraph 已可用。公开入口切换
必须等待 native adapter 和 runtime 集成完成。

## 验证

共享 UT fixture `run_without_optional_runtime` 启动独立 Python 进程，通过导入查找器
（import finder）阻止 `torch_npu`、`simpler`、`simpler_setup`、`_task_interface` 和
`pypto._torch_npu`。
同时捕获 import 语句与 `importlib.import_module()` 动态导入，即使调用方捕获了
`ImportError`，该次尝试仍使检查失败。显式检查在 `PYTHONOPTIMIZE=1` 和 `2` 下仍然生效，
负对照覆盖这两种优化级别及普通执行。子进程关闭 PyTorch backend 自动加载，以隔离
PyPTO 自身行为与已安装的框架插件。查找器的负向测试确保 CPU runner 即使原本就未安装
这些依赖，也不会把导入尝试误报为通过。

这些检查覆盖包导入/重载、program 配置和注册后的 Fake/Meta dispatch，直接进入现有
全量 UT CI，无需可选 runtime 依赖或新增设备 job。它们不验证 native adapter 构建或
设备执行；这些验证仍属于集成分支。

`tests/ut/torch/test_interop.py` 使用真实 CPU storage 和模拟的 NPU device 标签，
只替换框架 format/context 查询。覆盖 view offset、alias、独立 Scalar/stream 快照、
所有权、非法输入和禁止导入可选 runtime 依赖的隔离进程。

`tests/st/runtime/kernel/test_torch_interop.py` 核对真实 NPU Tensor、非默认 stream、
offset view 和非连续输入拒绝行为。没有真实 NPU 或所选平台为模拟器时跳过。
这些是 metadata 测试，不是 PyPTO kernel 执行测试。

`tests/ut/torch/test_registration.py` 覆盖 schema mutation/alias 契约、Fake/Meta
与符号输入、隔离导入、重复定义，以及不依赖真实 kernel executor 的测试内
dispatcher/compiler 集成。
