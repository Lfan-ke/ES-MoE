"""The Ultralytics integration."""

try:
    import ultralytics  # noqa: F401
except ModuleNotFoundError as missing:
    raise ModuleNotFoundError('the YOLO integration needs ultralytics: uv add "esmoe[yolo]"') from missing

from .graft import *
from .inject import *
