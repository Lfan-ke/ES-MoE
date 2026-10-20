"""The Flax backend. Flax ships two APIs, so this package holds both.

`esmoe.flax` is `flax.nnx`, the one Flax recommends now; `esmoe.flax.linen` is the other.
"""

try:
    import flax  # noqa: F401
except ModuleNotFoundError as missing:
    raise ModuleNotFoundError('the Flax backend needs flax: uv add "esmoe[flax]"') from missing

from .balance import *
from .nnx import *
