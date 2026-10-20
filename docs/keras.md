# 在 Keras 里

```bash
uv add esmoe[keras]         # 或 esmoe[tensorflow] / esmoe[jax]
```

块写了两份：一份原生 torch，一份 Keras 3。Keras 3 自己跑在 tensorflow、jax、torch 三种后端上，所以这一份就覆盖了另外三家。两份的构造参数完全一样，因为解析参数的代码是同一份。

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

## 与 torch 那份的差别

**每个专家都算**。跳过没被路由到的专家取决于当批数据，jax 与 tensorflow 的静态图表达不了。路由与平衡项和 torch 那份逐位一致，省下来的推理开销是 torch 独有的。

**通道次序听 Keras 的**。按 `keras.config.image_data_format()`，默认 `channels_last`。

**嫁接只支持 `Sequential`**。函数式模型持有的是计算图而不是一张列表，要改得自己重建。`esmoe.keras.graft` 返回一个新的 `Sequential`，不是原地改。

## 选后端

后端由 Keras 决定，设环境变量即可：

```bash
KERAS_BACKEND=jax python train.py
```

`esmoe.backend("jax")`、`esmoe.backend("tensorflow")`、`esmoe.backend("keras")` 指向的都是这一层。CI 在三种后端上各跑一遍这一层的测试。
