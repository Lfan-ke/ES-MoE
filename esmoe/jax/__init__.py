"""The Jax backend: pure functions over a parameter tree.

Nothing is published to a registry here, because nothing holds state: `apply` returns the output
and the balancing term together, which is what `jax.grad` needs anyway.
"""

try:
    import jax  # noqa: F401
except ModuleNotFoundError as missing:
    raise ModuleNotFoundError('the Jax backend needs jax: uv add "esmoe[jax]"') from missing

from .balance import *
from .block import *
from .experts import *
