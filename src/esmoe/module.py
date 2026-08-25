# Minimal faithful ES-MoE: softmax router + top-k over heterogeneous depthwise-conv experts,
# with a Switch-Transformer load-balancing aux loss. Mirrors YOLO-Master's ES_MOE structure;
# P0 aligns hyperparameters to the ES-MoE-N anchor (docs/ROADMAP.md).
import torch
import torch.nn.functional as F
from torch import nn


def _odd_kernels(num_experts, max_kernel_size=15):
    ks, k = [], 3
    for _ in range(num_experts):
        ks.append(min(k, max_kernel_size) | 1)
        k += 2
    return ks


class _Expert(nn.Module):
    def __init__(self, c1, c2, k):
        super().__init__()
        self.dw = nn.Conv2d(c1, c1, k, 1, k // 2, groups=c1, bias=False)
        self.pw = nn.Conv2d(c1, c2, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


class ESMoE(nn.Module):
    def __init__(self, c1, c2=None, num_experts=4, reduction=8, top_k=2,
                 max_kernel_size=15, expert_kernel_sizes=None):
        super().__init__()
        c2 = c2 or c1
        if not 1 <= top_k <= num_experts:
            raise ValueError(f"top_k must be in [1, {num_experts}]")
        ks = list(expert_kernel_sizes) if expert_kernel_sizes else _odd_kernels(num_experts, max_kernel_size)
        if len(ks) != num_experts:
            raise ValueError("expert_kernel_sizes length must equal num_experts")
        self.num_experts, self.top_k, self.expert_kernel_sizes = num_experts, top_k, ks
        self.experts = nn.ModuleList(_Expert(c1, c2, k) for k in ks)
        hidden = max(c1 // reduction, 1)
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(c1, hidden), nn.SiLU(), nn.Linear(hidden, num_experts),
        )
        self.register_buffer("aux_loss", torch.zeros(()), persistent=False)

    def forward(self, x):
        probs = F.softmax(self.router(x), dim=1)                 # (n, e)
        topv, topi = probs.topk(self.top_k, dim=1)
        gate = torch.zeros_like(probs).scatter(1, topi, topv)
        gate = gate / gate.sum(dim=1, keepdim=True).clamp_min(1e-9)
        out = torch.zeros(x.shape[0], self.experts[0].pw.out_channels, *x.shape[2:],
                          device=x.device, dtype=x.dtype)
        for e in range(self.num_experts):
            ge = gate[:, e].view(-1, 1, 1, 1)
            if torch.count_nonzero(ge) == 0:
                continue
            out = out + ge * self.experts[e](x)
        self.aux_loss = self._load_balance(probs, gate)
        return out

    def _load_balance(self, probs, gate):
        importance = probs.mean(dim=0)                           # per-expert routing mass
        load = (gate > 0).float().mean(dim=0)                    # fraction of samples using expert
        return self.num_experts * (importance * load).sum()
