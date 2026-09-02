# API

六个入口，全部从 `esmoe` 顶层导入；包内带 `py.typed`，类型签名对 IDE 与 mypy 可见。

## equip

    esmoe.equip(base="yolov8n.yaml", *, weight=0.01, out=None, **graft_kwargs) -> YOLO

注册、接入、构建、接损失一次完成。`out` 指定落盘的接入后配置；不给则写到临时目录（YOLO 只按路径加载模型）。`graft_kwargs` 原样转给 `graft`。

## inject_esmoe

    esmoe.inject_esmoe() -> type[ESMoE]

把 `ESMoE` 注册到 `parse_model` 解析层名的位置，此后任何 model.yaml 都能写 `[-1, 1, ESMoE, [4, 2]]`。

## graft

    esmoe.graft(base, out=None, *, at="backbone_end", num_experts=4, top_k=2, rewire=False) -> dict

在 `at` 指定的层后插入块并重编号其后的所有引用。`at` 可为 `"backbone_end"`、单个层号或多个层号。`rewire=True` 让引用旧插入层的下游改指块本身——不开时，按序号点名主干末层的 head 分支（如 YOLOv8 的 P5 侧向）读到的仍是插入前的特征。

## attach_aux_loss

    esmoe.attach_aux_loss(model, weight=0.01) -> model

把路由的负载均衡损失接进被优化的训练损失，训练日志多出 `esmoe_aux` 一列。同时把 `model.train()` 的训练器指到 `esmoe.trainer`，DDP worker 因此能自行注册块并恢复权重。

## collect_aux_loss

    esmoe.collect_aux_loss(model, device=None) -> Tensor

汇总最近一次前向发布的路由损失，自定义训练循环用。连续调用不会重复计入陈旧值。

## ESMoE

    esmoe.ESMoE(num_experts=4, top_k=2, channels=None, *, reduction=8,
                max_kernel_size=15, expert_kernel_sizes=None,
                expert=DWExpert, balance=switch_balance)

通道保持的专家混合块。`channels` 省略时在首次前向推断。`expert` 是 `(c1, c2, k) -> Module` 的工厂，`balance` 是 `(probs, gate) -> scalar` 的均衡目标，两者都可替换。`esmoe.blocks(model)` 按模块顺序遍历模型中的每一个块。
