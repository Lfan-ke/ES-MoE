# Flax 集成

```bash
uv add esmoe[flax]
```

Flax 现在并行着两套 API，两套都支持：`esmoe.flax` 是 `nnx`，Flax 现在推荐的那套；`esmoe.flax.linen` 是 Linen。输入都按 NHWC。

## nnx

状态挂在对象上，要纯函数时用 `nnx.split`。

```python
from flax import nnx
from esmoe.flax import ESMoE

net = nnx.Sequential(
    nnx.Conv(3, 32, (3, 3), rngs=nnx.Rngs(0)),
    ESMoE(kernels=[3, 5, 7], top_k=1, channels=32, rngs=nnx.Rngs(1)),
)
out = net(x)
```

`kernels` 给每个专家一个感受野，`int` 或 `(h, w)`，专家数由它定；`top_k` 决定几选几。`channels` 在这里是必填——nnx 在构造时就建好参数，没有首次前向推断那一步。

## 辅助损失

平衡项存在块的 `nnx.Intermediate` 变量上，读取与清理都走 `routing`：

```python
from esmoe.flax import collect_aux_loss


@nnx.jit
def step(net, x):
    def objective(net):
        return task_loss(net(x)) + 0.01 * collect_aux_loss(net)

    return nnx.grad(objective)(net)
```

`nnx.jit` 与 `nnx.grad` 直接可用，状态由 nnx 自己穿过变换。

## linen

参数放在模块外面，声明一次、用 `apply` 跑；平衡项 sow 进 `intermediates`。

```python
from flax import linen
from esmoe.flax.linen import ESMoE, collect_aux_loss

block = ESMoE(kernels=[3, 5, 7], top_k=1)
variables = block.init(key, x, training=True)
out, state = block.apply(variables, x, training=True, mutable=["intermediates", "batch_stats"])
aux = collect_aux_loss(state)
```

Linen 没有 `**kwargs`，所以设置从 `options` 进：`ESMoE(kernels=[3, 5], options={"out_norm": True})`。通道由第一次 `init` 的输入决定，不必写。

## 与另外两条 jax 路线的区别

| 用法 | 形态 | 何时用 |
|:--:|:--:|:--:|
| `esmoe.flax` | `nnx.Module`，状态在对象上 | 项目用 Flax 的新 API |
| `esmoe.flax.linen` | `linen.Module`，参数在外面 | 项目用 Flax 的 Linen |
| `esmoe.jax` | 纯函数 `init` / `apply` | 不想引入 Flax，自己管参数树 |
| `esmoe.keras` | `keras.layers.Layer`，后端设为 jax | 项目用 Keras 跨后端 |

三条路的路由与平衡项一致，构造参数也一致。

## 边界

每个专家都要算。跳过未被路由到的专家取决于当批数据，`jit` 之后的图是静态的，表达不了。
