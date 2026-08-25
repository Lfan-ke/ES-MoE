# ES-MoE block ported from YOLO-Master, kept backbone-agnostic so it injects into any Ultralytics generation.
from torch import nn


class ESMoE(nn.Module):
    def __init__(self, c1, c2, num_experts=4, top_k=2, expert_kernel_sizes=None):
        super().__init__()
        # P0: port the real forward/router/aux-loss from YOLO-Master (see docs/ROADMAP.md).
        raise NotImplementedError("P0: port ES-MoE forward + router + aux loss")

    def forward(self, x):
        raise NotImplementedError
