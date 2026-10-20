"""Kernel arithmetic, shared by every backend."""

type Kernel = int | tuple[int, int]


def __down(size: int) -> int:
    return int(size) - 1 + int(size) % 2


def odd(size: Kernel) -> Kernel:
    """An even kernel steps down, never up, so the padding stays centred."""
    return (__down(size[0]), __down(size[1])) if isinstance(size, tuple) else __down(size)


def fit(kernel: Kernel, cap: int) -> Kernel:
    """``kernel`` made odd and held under ``cap``, per axis."""
    limit = __down(cap)
    if isinstance(kernel, tuple | list):
        return min(__down(kernel[0]), limit), min(__down(kernel[1]), limit)
    return min(__down(kernel), limit)


def odd_kernels(num_experts: int, max_kernel: int = 15) -> list[Kernel]:
    """Heterogeneous odd kernels 3, 5, 7, ... capped at ``max_kernel``."""
    return [min(3 + 2 * i, __down(max_kernel)) for i in range(num_experts)]


__all__ = [
    "Kernel",
    "fit",
    "odd",
    "odd_kernels",
]
