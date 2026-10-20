"""The TensorFlow backend, built on tf.Module."""

try:
    import tensorflow  # noqa: F401
except ModuleNotFoundError as missing:
    raise ModuleNotFoundError('the TensorFlow backend needs tensorflow: uv add "esmoe[tensorflow]"') from missing

from .balance import *
from .block import *
from .experts import *
from .graft import *
from .loss import *
