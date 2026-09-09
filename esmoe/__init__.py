from .aux_loss import clear_aux_loss, collect_aux_loss
from .graft import graft
from .inject import attach_aux_loss, equip, inject_esmoe
from .module import (
    BALANCES,
    SETTINGS,
    DWExpert,
    ESMoE,
    blocks,
    gshard_balance,
    gshard_probs_balance,
    master_balance,
    odd_kernels,
    switch_balance,
)

__all__ = [
    "BALANCES",
    "SETTINGS",
    "DWExpert",
    "ESMoE",
    "attach_aux_loss",
    "blocks",
    "clear_aux_loss",
    "collect_aux_loss",
    "equip",
    "graft",
    "inject_esmoe",
    "odd_kernels",
    "gshard_balance",
    "gshard_probs_balance",
    "master_balance",
    "switch_balance",
]
__version__ = "0.1.5"
