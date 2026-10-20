"""Load-balancing objectives over the router's output."""

from collections.abc import Callable

import tensorflow as tf

type BalanceFn = Callable[[tf.Tensor, tf.Tensor], tf.Tensor]


def __utilisation(weights: tf.Tensor) -> tf.Tensor:
    usage = tf.reduce_mean(weights, axis=0)
    return usage / tf.maximum(tf.reduce_sum(usage), 1e-6)


def switch_balance(probs: tf.Tensor, gate: tf.Tensor) -> tf.Tensor:
    """Switch-Transformer balancing: routing mass times realised load, summed over experts."""
    importance = tf.reduce_mean(probs, axis=0)
    load = tf.reduce_mean(tf.cast(gate > 0, probs.dtype), axis=0)
    return tf.cast(tf.shape(probs)[1], probs.dtype) * tf.reduce_sum(importance * load)


def gshard_balance(probs: tf.Tensor, gate: tf.Tensor) -> tf.Tensor:
    """``E · Σ usage²`` over the gated weights, the term YOLO-Master's ES_MOE optimises."""
    usage = __utilisation(gate)
    return tf.cast(tf.shape(gate)[1], gate.dtype) * tf.reduce_sum(usage * usage)


def master_balance(probs: tf.Tensor, gate: tf.Tensor) -> tf.Tensor:
    """The YOLO-Master paper's form: mean squared deviation from uniform use."""
    usage = __utilisation(gate)
    experts = tf.cast(tf.shape(gate)[1], gate.dtype)
    return tf.reduce_mean((usage - 1.0 / experts) ** 2)


def gshard_probs_balance(probs: tf.Tensor, gate: tf.Tensor) -> tf.Tensor:
    """`gshard_balance` on the raw probabilities, to isolate what the objective reads."""
    usage = __utilisation(probs)
    return tf.cast(tf.shape(probs)[1], probs.dtype) * tf.reduce_sum(usage * usage)


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
