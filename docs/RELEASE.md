# 发布说明

当前版本 **2.2.0**：Flax 后端按现行的 `flax.nnx` 写成，jax 由此有三条路。

## 2.2.0

- **Flax 后端。** `esmoe.flax.ESMoE` 是一个 `nnx.Module`，参数在构造时建好，平衡项存在 `nnx.Intermediate` 变量上，`nnx.jit` 与 `nnx.grad` 直接可用。
- **jax 现在有三条路**，按项目选：本来就用 Flax 选 `esmoe.flax`；不想引入 Flax、自己管参数树选 `esmoe.jax`；用 Keras 跨后端选 `esmoe.keras` 配 `KERAS_BACKEND=jax`。三条路的路由、平衡项与构造参数一致。
- `esmoe[flax]` 装它，`esmoe[nnx]` 是同一个 extra 的别名。CI 多一条 flax 腿。

## 2.1.0

- **TensorFlow 原生后端。** `esmoe.tensorflow.ESMoE` 是一个 `tf.Module`，只依赖 TensorFlow：自带的批归一化把滑动统计量存成不训练的变量，辅助损失走与 torch 同一套弱引用注册表，`graft` 按属性名或 `Sequential` 下标插块。
- **Jax 原生后端。** `esmoe.jax.ESMoE` 是一组纯函数，`init` 建参数树、`apply` 返回输出与平衡项，整段进 `jax.jit` 与 `jax.grad`，不依赖 Keras 也不依赖 Flax。参数树是普通字典与数组，`optax` 直接能用。
- **extras 补齐别名。** `esmoe[pytorch]`、`esmoe[tf]`、`esmoe[keras3]`、`esmoe[ultralytics]` 分别等同于 `esmoe[torch]`、`esmoe[tensorflow]`、`esmoe[keras]`、`esmoe[yolo]`；`esmoe[all]` 是五个正式名字的合集。
- **CI 加两条腿。** tensorflow 与 jax 各跑一条，先断言 keras 与 ultralytics 都没装，再跑该后端自己的测试。

## 2.0.0

块不再绑在 Ultralytics 上。`uv add esmoe` 只装 torch，块可以直接拼进任何 torch 模型；Ultralytics 集成变成 `uv add esmoe[yolo]`，用法与 1.x 相同。另有一份 Keras 3 实现，同一个块在 tensorflow、jax、torch 上都能跑。

### 2.0.0 的变化

- **torch 与 YOLO 平级。** 核心只依赖 torch，`nn.Sequential` 里直接放得下；通道在首次前向推断，所以不用写宽度。辅助损失用 `with esmoe.routing(net) as route:` 读，进入时清一次注册表，这一步没跑到的块不会把上一步的值报上来。
- **两个旋钮定专家。** `kernels` 给每个专家一个感受野，`int` 或 `(h, w)`，给了它专家数就由它定；`expert` 给分支结构，一个工厂管全部，或逐分支一个。三选一、五个专家、条带卷积、核重复都是一行的事。
- **Keras 3 后端。** `esmoe.keras` 下是同一个块的 Keras 实现，构造参数与 torch 那份完全一致，辅助损失走 Keras 自己的 `add_loss`。它每个专家都算——跳过未路由的专家取决于数据，静态图表达不了。
- **往现成的模型里插块。** `esmoe.graft(model, at="layer3", mode="after", ...)` 按子模块名、若干名字或判断函数定位，`after` / `before` / `replace` 三种改法，原地生效。同名的 `esmoe.graft("yolo11n.yaml", ...)` 仍然改配置，按第一个参数分派。
- **配置里的块认三个名字。** `ESMoE`、`ES_MoE`、`ES_MOE` 都解析到它。配置里已经写了块，`equip` 就按原样用，不再插入。
- **参数怎么写都认。** 位置参数里的映射当作 `options`，`c`/`c1`/`ch`/`c2`/`e`/`k` 这些别名都认，`experts` 按值的样子归位。认不出的键警告一次后忽略，`top_k` 超过专家数时降到全选——都不再让一次训练在构造时就失败。取值不合法照旧报错。
- **Python 下限 3.12。** 换来 `type` 别名这类现在就能用的写法。

### 从 1.x 过来

| 1.x | 2.0 |
|:--:|:--:|
| `pip install esmoe` | `uv add esmoe`（只要块）或 `uv add esmoe[yolo]` |
| `esmoe.inject_esmoe()` | `esmoe.register()`，旧名仍可用 |
| `expert_kernel_sizes=` | `kernels=`，旧名仍接 |
| `max_kernel_size=` | `max_kernel=`，旧名仍接 |
| `from esmoe.inject import X` | `from esmoe.yolo.inject import X` |
| 认不出的参数报错 | 警告后忽略 |

1.x 存下的检查点照常加载。

## 1.x

1.0.0 是第一个稳定版：七代主干的兼容矩阵、满协议实验与判读线、中英双语文档站。1.0.1 修了交付审计的子串匹配、数据集统计的边界框、图表只取登记过的运行，以及文档站的几处显示。1.1.0 加了文档站的[试玩](demo.md)页——最多三个模型在同一张图上并排跑，带块的逐块列出专家分值，推理在读者自己的浏览器里完成——以及导出脚本 `scripts/demo_export.py`。

## 更早的小版本

0.1 系列给出 `inject_esmoe`、`graft`、`attach_aux_loss`、`collect_aux_loss` 四个入口与三 seed 证据，随后修好不带 `out` 时的配置传递、8.4.13x 的训练日志表头，并把包从 `src/` 移到仓库根目录。

## 安装

```bash
uv add esmoe              # 块本身，只依赖 torch
uv add esmoe[yolo]        # 再加 Ultralytics 集成
pip install esmoe         # 偏要的话
```
