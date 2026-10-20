# Torch integration

```bash
uv add esmoe
```

Torch is all it needs; ultralytics never has to be installed.

## Drop it in

```python
import esmoe
from torch import nn

net = nn.Sequential(
    nn.Conv2d(3, 64, 3, 2, 1),
    esmoe.ESMoE(kernels=[3, 5, 7], top_k=1),
)
```

The block takes its width from the first forward pass, so nothing has to declare it. Pass
`channels=` to build it before an optimiser is created.

## Three knobs

| knob | takes | decides |
|:--:|:--:|:--:|
| `kernels` | a sequence of `int` or `(h, w)` | each expert's receptive field, and how many experts there are |
| `expert` | a factory, a name, or one per branch | what a branch is |
| `top_k` | an integer or `None` | how many experts run per sample |

```python
esmoe.ESMoE(kernels=[3, 5, 7], top_k=1)  # three experts, one per sample
esmoe.ESMoE(kernels=[3, 3, 5, 5, 7, 7, 9, 9, 1, 1])  # ten experts, kernels may repeat
esmoe.ESMoE(kernels=[(1, 9), (9, 1)], top_k=2)  # strip convolutions
esmoe.ESMoE(kernels=[3, 5], expert=[MyExpert, esmoe.DWExpert])
```

An even kernel steps down to odd so the padding stays centred, and anything over `max_kernel` is
capped. An expert factory is `(c1, c2, k) -> Module`.

## The auxiliary loss

The balancing term has to join the total loss, or it is a number computed and thrown away:

```python
with esmoe.routing(net) as route:
    loss = task_loss(net(x), y) + 0.01 * route.loss()
loss.backward()
```

`routing` clears the registry on the way in, so a block that did not run this step cannot report
the step before. Without the context manager, pair `esmoe.clear_aux_loss()` with
`esmoe.collect_aux_loss(model)` yourself.

## Into a model that already exists

```python
esmoe.graft(resnet, at="layer3", mode="after", kernels=[3, 5, 7], top_k=1)
```

`at` takes a submodule name, several names, or a `(name, module) -> bool` test. `mode` is `after`,
`before` or `replace`. The model is changed in place and handed back.

## However the arguments are spelled

Positional arguments are `num_experts`, `top_k`, `channels`, `options`, and a mapping among them is
read as `options` whatever its position. `c`/`c1`/`ch`, `c2`, `e` and `k` are understood, and
`experts` lands on the expert count, the kernels or the branch structure according to what it
holds. An unrecognised key warns once and is ignored rather than costing a run; a bad value still
raises.
