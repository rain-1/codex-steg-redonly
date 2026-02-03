"""Utilities for red-biased watermark sampling with entropy gating."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable, List, Optional, Sequence, Set


def softmax(logits: Sequence[float]) -> List[float]:
    max_logit = max(logits)
    exp_vals = [math.exp(logit - max_logit) for logit in logits]
    total = sum(exp_vals)
    return [val / total for val in exp_vals]


def entropy(probs: Sequence[float]) -> float:
    return -sum(p * math.log(p) for p in probs if p > 0.0)


def top_k_indices(logits: Sequence[float], k: int) -> Set[int]:
    if k <= 0:
        return set()
    return {idx for idx, _ in sorted(enumerate(logits), key=lambda x: x[1], reverse=True)[:k]}


def red_rate(tokens: Iterable[int], red_tokens: Set[int], eligible_tokens: Set[int]) -> float:
    eligible_count = 0
    red_count = 0
    for token in tokens:
        if token not in eligible_tokens:
            continue
        eligible_count += 1
        if token in red_tokens:
            red_count += 1
    if eligible_count == 0:
        return 0.0
    return red_count / eligible_count


@dataclass(frozen=True)
class RedBiasConfig:
    """Configuration for red-biased sampling.

    Attributes:
        delta: Logit bias to add to red tokens.
        entropy_threshold: Minimum entropy required to apply the bias.
        top_k: If set, require both red and blue tokens within top-k before biasing.
    """

    delta: float = 1.5
    entropy_threshold: float = 2.0
    top_k: Optional[int] = 50


def apply_red_bias(
    logits: Sequence[float],
    red_tokens: Set[int],
    eligible_tokens: Set[int],
    config: RedBiasConfig,
) -> List[float]:
    """Apply a red-token logit bias with entropy gating.

    Args:
        logits: Raw model logits.
        red_tokens: Token ids classified as red.
        eligible_tokens: Token ids allowed for biasing.
        config: Red-bias configuration.
    """

    probs = softmax(logits)
    step_entropy = entropy(probs)
    if step_entropy < config.entropy_threshold:
        return list(logits)

    if config.top_k is not None:
        top_k = top_k_indices(logits, config.top_k)
        has_red = any(token in top_k for token in red_tokens)
        has_blue = any(token in top_k for token in eligible_tokens.difference(red_tokens))
        if not (has_red and has_blue):
            return list(logits)

    adjusted = list(logits)
    for token in red_tokens.intersection(eligible_tokens):
        adjusted[token] += config.delta
    return adjusted


def sample_token(
    logits: Sequence[float],
    rng: Optional[random.Random] = None,
) -> int:
    """Sample a token id from logits."""

    if rng is None:
        rng = random.Random()
    probs = softmax(logits)
    threshold = rng.random()
    cumulative = 0.0
    for idx, prob in enumerate(probs):
        cumulative += prob
        if cumulative >= threshold:
            return idx
    return len(probs) - 1
