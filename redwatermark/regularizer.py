"""Loss utilities for red-mass regularization."""

from __future__ import annotations

import math
from typing import Iterable, Sequence


def red_mass(probs: Sequence[float], red_tokens: Iterable[int]) -> float:
    return sum(probs[token] for token in red_tokens)


def red_regularizer(
    probs: Sequence[float],
    red_tokens: Iterable[int],
    epsilon: float = 1e-8,
) -> float:
    mass = red_mass(probs, red_tokens)
    return -math.log(mass + epsilon)


def kl_divergence(p: Sequence[float], q: Sequence[float], epsilon: float = 1e-8) -> float:
    total = 0.0
    for p_i, q_i in zip(p, q):
        total += p_i * (math.log(p_i + epsilon) - math.log(q_i + epsilon))
    return total
