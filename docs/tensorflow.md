# TensorFlow 集成

```bash
uv add esmoe[tensorflow]
```

块在 TensorFlow 上以 Keras 3 层的形式提供。装上这个 extra 后把 `KERAS_BACKEND` 设为 `tensorflow`，`esmoe.keras.ESMoE` 就是一个普通的 TensorFlow 层。

```bash
KERAS_BACKEND=tensorflow python train.py
```

## 当普通层用

```python
import keras
from esmoe.keras import ESMoE

net = keras.Sequential(
    [
        keras.layers.Conv2D(16, 3, strides=2, padding="same"),
        ESMoE(kernels=[3, 5, 7], top_k=1),
    ]
)
net.compile(optimizer="adam", loss="mse")
net.fit(x, y)
```

平衡项走 Keras 的 `add_loss`，`fit` 会把它算进去。

## 进计算图与求梯度

层可以直接进 `tf.function`，梯度照常由 `GradientTape` 取。平衡项要在同一次追踪内读出来——追踪里生成的张量出了函数就不可用。

```python
import tensorflow as tf


@tf.function
def total(inputs):
    out = net(inputs)
    return tf.reduce_sum(out) + tf.add_n(net.losses)


with tf.GradientTape() as tape:
    value = total(tf.constant(batch))
grads = tape.gradient(value, net.trainable_variables)
```

`tests/test_backends.py` 在 CI 的 TensorFlow 腿上跑的就是这段。

## 边界

每个专家都要算。跳过未被路由到的专家取决于当批数据，静态图表达不了；路由与平衡项与 torch 那份逐位一致，省下的推理开销是 torch 独有的。

嫁接只支持 `keras.Sequential`，函数式模型要自行重建。通道次序听 `keras.config.image_data_format()`。
