"""Scoring utilities for selection and filtering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from redwatermark.filters import OddityFlags, oddity_score


@dataclass(frozen=True)
class ScoreWeights:
    red_rate_weight: float = 1.0
    base_logprob_weight: float = 1.0
    oddity_weight: float = 1.0


def score_candidate(
    red_rate_value: float,
    target_red_rate: float,
    base_logprob: Optional[float],
    oddities: OddityFlags,
    weights: ScoreWeights,
) -> float:
    distance = abs(red_rate_value - target_red_rate)
    base_component = base_logprob if base_logprob is not None else 0.0
    return (
        -weights.red_rate_weight * distance
        + weights.base_logprob_weight * base_component
        - weights.oddity_weight * oddity_score(oddities)
    )
