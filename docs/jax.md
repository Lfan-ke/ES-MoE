# Jax 集成

```bash
uv add esmoe[jax]
```

`esmoe.jax.ESMoE` 是一组纯函数，只依赖 Jax，不经 Keras 也不经 Flax。`init` 建参数树，`apply` 返回输出与平衡项。输入按 NHWC。

```python
import jax
from esmoe.jax import ESMoE

block = ESMoE(kernels=[3, 5, 7], top_k=1)
params = block.init(jax.random.PRNGKey(0), channels=8)
out, aux = block.apply(params, x)
```

`kernels` 给每个专家一个感受野，`int` 或 `(h, w)`，专家数由它定；`top_k` 决定几选几。

## 训练

平衡项由 `apply` 直接返回，所以梯度怎么取都在你手里：

```python
def loss(params, x, y):
    out, aux = block.apply(params, x, training=True)
    return task_loss(out, y) + 0.01 * aux


grads = jax.grad(loss)(params, x, y)
```

整段都能进 `jax.jit`：没有藏起来的状态，也没有需要清理的注册表。

## 参数树

```
{"router": {"first", "first_bias", "second", "second_bias"},
 "experts": [{"depthwise", "pointwise", "scale", "offset"}, ...],
 "norm": {"scale", "offset"}}        # 仅在 out_norm 开启时
```

是普通的字典与数组，`optax` 与 `jax.tree` 直接能用。

## 边界

每个专家都要算。跳过未被路由到的专家取决于当批数据，`jit` 之后的图是静态的，表达不了。

归一化按当批统计量计算，不保留滑动量：函数式接口里，状态要么由调用方携带，要么不存在，这里选后者。
