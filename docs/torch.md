# 在 torch 模型里

```bash
uv add esmoe
```

只依赖 torch，不装 ultralytics 也能用。

## 拼进去

```python
import esmoe
from torch import nn

net = nn.Sequential(
    nn.Conv2d(3, 64, 3, 2, 1),
    esmoe.ESMoE(kernels=[3, 5, 7], top_k=1),
)
```

块的通道在第一次前向时定下来，所以不用写宽度。要在建优化器之前就有参数，传 `channels=`。

## 三个旋钮

| 旋钮 | 收什么 | 决定什么 |
|:--:|:--:|:--:|
| `kernels` | `int` 或 `(h, w)` 的序列 | 每个专家的感受野，以及专家个数 |
| `expert` | 工厂、名字，或逐分支一个 | 每个分支是什么结构 |
| `top_k` | 整数或 `None` | 每个样本激活几个专家 |

```python
esmoe.ESMoE(kernels=[3, 5, 7], top_k=1)  # 三选一
esmoe.ESMoE(kernels=[3, 3, 5, 5, 7, 7, 9, 9, 1, 1])  # 十个专家，核可以重复
esmoe.ESMoE(kernels=[(1, 9), (9, 1)], top_k=2)  # 条带卷积
esmoe.ESMoE(kernels=[3, 5], expert=[MyExpert, esmoe.DWExpert])
```

偶数核降一档为奇数，保证 padding 居中；超过 `max_kernel` 的截断。专家工厂的签名是 `(c1, c2, k) -> Module`。

## 辅助损失

路由的平衡项要自己加进总损失，否则它只是个被算出来又丢掉的数：

```python
with esmoe.routing(net) as route:
    loss = task_loss(net(x), y) + 0.01 * route.loss()
loss.backward()
```

`routing` 进入时清一次注册表，这一步没跑到的块就不会把上一步的值报上来。不想用上下文管理器，就自己配对调用 `esmoe.clear_aux_loss()` 与 `esmoe.collect_aux_loss(model)`。

## 插进现成的模型

```python
esmoe.graft(resnet, at="layer3", mode="after", kernels=[3, 5, 7], top_k=1)
```

`at` 收子模块名、若干名字，或 `(name, module) -> bool` 的判断函数。`mode` 为 `after`、`before`、`replace`。原地改，返回同一个模型。

## 参数怎么写都认

位置参数依次是 `num_experts`、`top_k`、`channels`、`options`；位置参数里出现映射，无论在第几位都当作 `options`。`c`/`c1`/`ch`、`c2`、`e`、`k` 这些别名都认；`experts` 按值的样子归到专家数、核或分支结构。认不出的键警告一次后忽略，不会让一次训练白跑。取值不合法照旧报错。
