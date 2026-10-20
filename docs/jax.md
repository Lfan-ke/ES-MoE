# Jax 集成

```bash
uv add esmoe[jax]
```

块在 Jax 上以 Keras 3 层的形式提供。装上这个 extra 后把 `KERAS_BACKEND` 设为 `jax`，层的每一次前向就都在 Jax 上执行。

```bash
KERAS_BACKEND=jax python train.py
```

## 纯函数的用法

Keras 层在 Jax 上对外是纯函数：`stateless_call` 把变量作为入参、把输出与更新后的变量一起返回，所以可以直接放进 `jax.jit` 与 `jax.grad`，没有藏起来的状态。

```python
import jax
import keras
from esmoe.keras import ESMoE

net = keras.Sequential([ESMoE(kernels=[3, 5, 7], top_k=1)])
net(batch)  # 首次前向建好专家与路由器
layer = net.layers[0]

trainable = [v.value for v in layer.trainable_variables]
other = [v.value for v in layer.non_trainable_variables]


def total(weights, x):
    out, _ = layer.stateless_call(weights, other, x)
    return out.sum()


jax.jit(total)(trainable, batch)
grads = jax.grad(total)(trainable, batch)
```

`tests/test_backends.py` 在 CI 的 Jax 腿上跑的就是这段：`jit` 的结果与直算一致，梯度非零。

## 训练

走 Keras 的训练循环时与别的后端没有区别，`compile` 加 `fit` 即可，平衡项由 `add_loss` 计入。

## 边界

每个专家都要算。跳过未被路由到的专家取决于当批数据，`jit` 之后的图是静态的，表达不了；路由与平衡项与 torch 那份逐位一致。

嫁接只支持 `keras.Sequential`，函数式模型要自行重建。
