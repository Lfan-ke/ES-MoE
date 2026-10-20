"""The Flax backend, written against flax.nnx."""

try:
    import flax  # noqa: F401
except ModuleNotFoundError as missing:
    raise ModuleNotFoundError('the Flax backend needs flax: uv add "esmoe[flax]"') from missing

from .balance import *
from .block import *
from .experts import *
from .loss import *
