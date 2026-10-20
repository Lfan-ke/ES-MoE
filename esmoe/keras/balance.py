"""Load-balancing objectives, the same four the torch backend offers."""

from collections.abc import Callable

from keras import ops

type BalanceFn = Callable[[object, object], object]


def __utilisation(weights):
    usage = ops.mean(weights, axis=0)
    return usage / ops.maximum(ops.sum(usage), 1e-6)


def switch_balance(probs, gate):
    """Switch-Transformer balancing: routing mass times realised load, summed over experts."""
    importance = ops.mean(probs, axis=0)
    load = ops.mean(ops.cast(gate > 0, "float32"), axis=0)
    return probs.shape[1] * ops.sum(importance * ops.cast(load, importance.dtype))


def gshard_balance(probs, gate):
    """``E · Σ usage²`` over the gated weights, the term YOLO-Master's ES_MOE optimises."""
    usage = __utilisation(gate)
    return gate.shape[1] * ops.sum(usage * usage)


def master_balance(probs, gate):
    """The YOLO-Master paper's form: mean squared deviation from uniform use."""
    usage = __utilisation(gate)
    return ops.mean((usage - 1.0 / gate.shape[1]) ** 2)


def gshard_probs_balance(probs, gate):
    """`gshard_balance` on the raw probabilities, to isolate what the objective reads."""
    usage = __utilisation(probs)
    return probs.shape[1] * ops.sum(usage * usage)


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
