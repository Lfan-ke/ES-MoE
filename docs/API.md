# API

六个入口，全部从 `esmoe` 顶层导入；包内带 `py.typed`，类型签名对 IDE 与 mypy 可见。

## equip

    esmoe.equip(base="yolov8n.yaml", *, weight=0.01, out=None, **graft_kwargs) -> YOLO

注册、接入、构建、接损失一次完成。`out` 指定落盘的接入后配置；不给则写到临时目录（YOLO 只按路径加载模型）。`graft_kwargs` 原样转给 `graft`。

## inject_esmoe

    esmoe.inject_esmoe() -> type[ESMoE]

把 `ESMoE` 注册到 `parse_model` 解析层名的位置，此后任何 model.yaml 都能写 `[-1, 1, ESMoE, [4, 2]]`。

## graft

    esmoe.graft(base, out=None, *, at="backbone_end", num_experts=4, top_k=2,
                rewire=False, **settings) -> dict

在 `at` 指定的层后插入块并重编号其后的所有引用。`at` 可为 `"backbone_end"`、`"backbone_stages"`（主干每个 stage 后各一块，复刻上游布局）、单个层号或多个层号。`rewire=True` 让引用旧插入层的下游改指块本身——不开时，按序号点名主干末层的 head 分支（如 YOLOv8 的 P5 侧向）读到的仍是插入前的特征。

`**settings` 收 `balance` / `out_norm` / `dense_training` / `sparse_inference` / `dynamic_threshold`，**写进配置文件而不是设在实例上**：训练器照 `model.yaml` 重建模型，设在实例上的会随那个被丢弃的实例一起消失。`balance` 可传名字或包里那四个函数之一；自定义目标进不了配置文件，会被明确拒绝。

## attach_aux_loss

    esmoe.attach_aux_loss(model, weight=0.01) -> model

把路由的负载均衡损失接进被优化的训练损失，训练日志多出 `esmoe_aux` 一列。同时把 `model.train()` 的训练器指到 `esmoe.trainer`，DDP worker 因此能自行注册块并恢复权重。

## collect_aux_loss

    esmoe.collect_aux_loss(model, device=None) -> Tensor

汇总最近一次前向发布的路由损失，自定义训练循环用。连续调用不会重复计入陈旧值。

## clear_aux_loss

    esmoe.clear_aux_loss() -> None

丢掉注册表里所有块已发布的值。`attach_aux_loss` 装上的损失补丁在每次前向之前会自己调一次；自定义训练循环若不走那条补丁，就需要在每步前向前手动调，否则某个块在这一步没跑到时，收集到的会是它上一步的值。

## odd / odd_kernels

    esmoe.odd(size) -> int
    esmoe.odd_kernels(num_experts, max_kernel_size=15) -> list[int]

`odd` 把偶数核降一档为奇数（4 → 3），保证 padding 居中；上游对显式给的核尺寸与上限都做同样处理。`odd_kernels` 生成默认的异构核 `3, 5, 7, …`，按 `max_kernel_size` 截断——即 `ESMoE` 在没给 `expert_kernel_sizes` 时用的那一组。

## ESMoE

    esmoe.ESMoE(num_experts=4, top_k=2, channels=None, options=None, *,
                out_channels=None, reduction=8, max_kernel_size=15,
                expert_kernel_sizes=None, expert=DWExpert, **settings)

专家混合块，默认保持通道数。`channels` 省略时在首次前向推断；`top_k=None` 等于用上全部专家。`expert` 是 `(c1, c2, k) -> Module` 的工厂，可替换。`options` 是配置文件传设置的那个映射（`[-1, 1, ESMoE, [4, 2, null, {out_norm: true}]]`），与 `**settings` 等价。

五个设置及其默认值（`esmoe.SETTINGS`）：

| 设置 | 本包默认 | 上游 | 作用 |
|:--:|:--:|:--:|:--|
| `balance` | `gshard_balance` | 同 | 均衡目标，`(probs, gate) -> scalar`，或 `esmoe.BALANCES` 里的名字 |
| `out_norm` | `False` | 恒开 | 加权求和后的 `BatchNorm + SiLU`（论文式 2 的 `Norm`） |
| `dense_training` | `False` | 恒开 | 训练期跑满专家，未选权重为 0 但归一化统计量继续更新 |
| `sparse_inference` | `True` | 同 | 推理期跳过未选专家 |
| `dynamic_threshold` | `0.0` | `0.4` | 推理期剪掉份额低于阈值的专家，首位无条件保留，余下重归一 |

后两项只影响推理，前三项影响训练。默认值保持 `results/` 里既有运行可原样复现，不是对上游的取舍判断。

`block.spec()` 返回该块当下带着的五个设置；`block.configure(**settings)` 在**不经训练器**的场合（推理、导出、单测）改它们。`esmoe.blocks(model)` 按模块顺序遍历模型中的每一个块。`scripts/blockspec.py` 从任一 checkpoint 读回当时真正生效的设置。
