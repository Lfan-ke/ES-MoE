# TensorFlow 集成

```bash
uv add esmoe[tensorflow]
```

`esmoe.tensorflow.ESMoE` 是一个 `tf.Module`，只依赖 TensorFlow，不经 Keras。输入按 NHWC。

```python
import tensorflow as tf
from esmoe.tensorflow import ESMoE, Sequential

net = Sequential([ESMoE(kernels=[3, 5, 7], top_k=1)])
out = net(tf.random.normal([2, 16, 16, 8]), training=True)
```

`kernels` 给每个专家一个感受野，`int` 或 `(h, w)`，专家数由它定；`top_k` 决定几选几。通道取自第一次调用。

## 辅助损失

平衡项要自己加进总损失：

```python
from esmoe.tensorflow import routing

with routing(net) as route:
    with tf.GradientTape() as tape:
        total = tf.reduce_sum(net(x, training=True)) + 0.01 * route.loss()
grads = tape.gradient(total, net.trainable_variables)
```

`routing` 进入时清一次注册表，这一步没跑到的块就不会把上一步的值报上来。不用上下文管理器就自己配对调用 `clear_aux_loss` 与 `collect_aux_loss`。

## 插进现成的模型

```python
from esmoe.tensorflow import graft

graft(model, at="stem", mode="after", kernels=[3, 5, 7], top_k=1)
```

`at` 收属性名、若干名字，或 `(name, module) -> bool`；`Sequential` 里的层按下标取名。`mode` 为 `after`、`before`、`replace`，原地改。

## 边界

每个专家都要算。跳过未被路由到的专家取决于当批数据，静态图表达不了。

批归一化的滑动统计量存成不参与训练的变量，`training=True` 时更新。
