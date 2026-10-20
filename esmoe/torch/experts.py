"""Expert branches."""

from collections.abc import Callable

from torch import Tensor, nn

from ..core.kernels import Kernel

ExpertFactory = Callable[[int, int, Kernel], nn.Module]


class DWExpert(nn.Module):
    """Depthwise-separable expert: the default branch, and a template for custom ones."""

    def __init__(self, c1: int, c2: int, k: Kernel):
        super().__init__()
        pad = (k[0] // 2, k[1] // 2) if isinstance(k, tuple) else k // 2
        self.dw = nn.Conv2d(c1, c1, k, 1, pad, groups=c1, bias=False)
        self.pw = nn.Conv2d(c1, c2, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.pw(self.dw(x))))


EXPERTS: dict[str, ExpertFactory] = {"dw": DWExpert}


__all__ = [
    "EXPERTS",
    "DWExpert",
    "ExpertFactory",
]
