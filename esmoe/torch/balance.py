"""Load-balancing objectives over the router's output."""

from collections.abc import Callable

from torch import Tensor

BalanceFn = Callable[[Tensor, Tensor], Tensor]


def __utilisation(weights: Tensor) -> Tensor:
    usage = weights.mean(dim=0)
    return usage / usage.sum().clamp_min(1e-6)


def switch_balance(probs: Tensor, gate: Tensor) -> Tensor:
    """Switch-Transformer balancing: routing mass times realised load, summed over experts.

    The realised loads sum to k whatever the skew, so the term settles at k once the mean
    probabilities are near uniform and stops reporting concentration.
    """
    importance = probs.mean(dim=0)
    load = (gate > 0).float().mean(dim=0)
    return probs.shape[1] * (importance * load).sum()


def gshard_balance(probs: Tensor, gate: Tensor) -> Tensor:
    """``E · Σ usage²`` over the gated weights, the term YOLO-Master's ES_MOE optimises.

    Reading the gate rather than the probabilities is what lets the term see a dispatch that has
    collapsed onto one expert.
    """
    usage = __utilisation(gate)
    return gate.shape[1] * (usage * usage).sum()


def master_balance(probs: Tensor, gate: Tensor) -> Tensor:
    """The YOLO-Master paper's form: mean squared deviation from uniform use.

    Equal to `gshard_balance` up to an affine map, so same minimiser and proportional gradients;
    both are kept because the paper and the released code state one objective at two scales.
    """
    usage = __utilisation(gate)
    return ((usage - 1.0 / gate.shape[1]) ** 2).mean()


def gshard_probs_balance(probs: Tensor, gate: Tensor) -> Tensor:
    """`gshard_balance` on the raw probabilities, to isolate what the objective reads."""
    usage = __utilisation(probs)
    return probs.shape[1] * (usage * usage).sum()


BALANCES: dict[str, BalanceFn] = {
    "switch": switch_balance,
    "gshard": gshard_balance,
    "master": master_balance,
    "gshard_probs": gshard_probs_balance,
}


__all__ = [
    "BALANCES",
    "BalanceFn",
    "gshard_balance",
    "gshard_probs_balance",
    "master_balance",
    "switch_balance",
]
