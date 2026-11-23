# 跨 program 共享通信域实现方案

状态：**设计稿，未实现、未执行硬件验收**。文中的新增接口、metadata 和生成代码是目标契约，不是当前可运行 API。

目标：Prefill 与 Decode 独立编译，在同一个 L3 worker 中通过各自的 `DistributedTensor` 形参使用同一个外部通信域。用户不传 ctx；编译器追加配对 ctx，`submit` 自动补齐。L3 的窗口绑定和编排体不区分资源由本地 worker 还是 L4 创建。

## 关联与设计依据

- [PyPTO #3497：跨持久化程序复用 Host 域](https://github.com/hw-native-sys/pypto/issues/3497)。
- [方案讨论评论](https://github.com/hw-native-sys/pypto/issues/3497#issuecomment-7093126653)。
- [PyPTO #3531：可复用 L3 与 L4 编排 RFC](https://github.com/hw-native-sys/pypto/issues/3531)。
- [L4 方案](../L4/README.md)，尤其是 [调用点窗口解析](../L4/02-passes.md)、[参数编码](../L4/03-codegen.md)和 [runtime](../L4/04-runtime.md)。
- [Simpler 域契约](https://github.com/hw-native-sys/simpler/blob/d2b37652d/docs/comm-domain.md)。

采用更新后的 L4 ABI：**DistributedTensor 业务参数自身贡献一个 `buffer_index` scalar；追加的逻辑 CommCtx 贡献三个 scalar。** 不增加独立的 WindowDescriptor 尾部，不把 caller 的具体 WindowBuffer 固化到 L3 形参。

## 落地顺序

**SharedDomain 先于 L4 落地。** 本方案先完成独立 L3 的外部建域、自动参数绑定和设备验收，交付 #3497；之后 #3531 接入已实现的 ABI builder、codec 和窗口绑定。L4 文档是接口设计参考，不是本方案的实现依赖。

不等待 `LevelHandler` 抽象、`@pl.jit.pod`、ClusterConfig、MPI startup/bundle 或远端入口完成。只需要核对并使用支持本地域 ref/retention/fence 的 Simpler 版本。

`create_comm_domain` 对 **每个 buffer 的物理字节数自动向上对齐到 64 字节**，再累计偏移和 window_size；用户只提供逻辑 shape/dtype，不负责补齐源文件或 tensor。

## 文档导航

| 文档 | 内容 |
| --- | --- |
| [00-design.md](00-design.md) | 范围、对象关系、源码基线、关键决策及非目标 |
| [01-api.md](01-api.md) | 外部建域、DomainTensor、成员/布局/初始化、公共调用方式 |
| [02-abi.md](02-abi.md) | 用户签名、隐藏 ctx、pool 布局、metadata、版本与冷加载 |
| [03-compiler.md](03-compiler.md) | JIT、pass、verifier、公共 ABI builder、codegen 具体修改 |
| [04-runtime.md](04-runtime.md) | 建域事务、自动补 ctx、run 登记、并发/释放/失败状态机 |
| [05-plan.md](05-plan.md) | 文件清单、实施依赖、测试矩阵、验收命令与开放验证项 |
| [06-example.md](06-example.md) | 两个独立程序共享两个 buffer 的完整目标示例及降低链 |

## 核心不变量

1. 一个域可以有多个 buffer；buffer 与域的对应由实参携带，不由参数名匹配。
2. 域的有序 members 在创建时固定；本地 worker 下标、物理 device ID、domain rank 分开处理。
3. 程序内部 alloc 仍隐式构域；外部形参不参加本地域合并，也不伪造 owning scope。
4. 同域两个形参保留两个 ctx ABI 位置，只对 handle、lease、ordering 状态去重。
5. 所有本次调用绑定放在 dispatch frame 内；编译产物不保存具体 ref、地址或文件对象。
6. 外部域由 worker/创建者持有；关闭 program 不释放域，reset 内部窗口不清空外部表。
7. 资源存活覆盖已接受任务、后代任务与异步通信完成；不能把 kernel 返回当作 GET 完成。
8. L4 与本地 submit 共用 ABI 规则，普通 tensor 的传输方式仍由调用边决定。

## 使用概览

```python
# Proposed runtime API; compile inputs/configuration are omitted here.
with DistributedWorker([prefill_compiled, decode_compiled]) as worker:
    with worker.create_comm_domain(
        members=[0, 1, 2, 3],
        memory_source="host",
        buffers=buffer_specs,
        window_sources=rank_sources,
    ) as domain:
        a = domain.tensor("weights_a")
        b = domain.tensor("weights_b")
        worker.submit(prefill_compiled, ids_p, a, b, out_p, 0).result()
        worker.submit(decode_compiled, ids_d, a, b, out_d, 1).result()
```

这里的外部域无论 worker 的 `persistent` 开关如何，都存活到显式 release/worker close；该开关只控制 program 内部隐式域。完整参数约束见 API 文档。

## 交付边界

本分支交付 SharedDomain 与 L4 的设计文档，不包含功能实现。`.gitignore` 仅允许跟踪这两个设计目录的 Markdown，其他构建输出继续忽略。正式实现时应将稳定契约同步到 `docs/en/dev/` 及 `docs/zh/dev/`。
