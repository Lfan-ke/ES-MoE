# Keras3 集成

```bash
uv add esmoe[keras]
```

`esmoe.keras.ESMoE` 是一个 `keras.layers.Layer`，张量运算走 `keras.ops`，Keras 配到哪个后端就在哪个后端上跑——tensorflow、jax 或 torch。

```python
import keras
from esmoe.keras import ESMoE

net = keras.Sequential(
    [
        keras.layers.Conv2D(16, 3, padding="same"),
        ESMoE(kernels=[3, 5, 7], top_k=1),
    ]
)
net.compile(optimizer="adam", loss="mse")
net.fit(x, y)
```

辅助损失走 Keras 自己的 `add_loss`，`fit` 会把它算进去，不用手动加。自定义训练循环用 `esmoe.keras.collect_aux_loss(model)`，它读的就是 `model.losses`。

## 边界

**每个专家都算**。跳过没被路由到的专家取决于当批数据，静态图表达不了。

**通道次序听 Keras 的**。按 `keras.config.image_data_format()`，默认 `channels_last`。

**嫁接只支持 `Sequential`**。函数式模型持有的是计算图而不是一张列表，要改得自己重建。`esmoe.keras.graft` 返回一个新的 `Sequential`，不是原地改。

## 选后端

后端由 Keras 决定，设环境变量即可：

```bash
KERAS_BACKEND=jax python train.py
```

`esmoe.backend("jax")`、`esmoe.backend("tensorflow")`、`esmoe.backend("keras")` 指向的都是这一层。CI 在三种后端上各跑一遍。

构造参数与 [Torch 集成](torch.md)完全一致：两边读参数的是同一段代码。
