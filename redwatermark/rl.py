"""Reward helpers for KL-constrained RL fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from redwatermark.filters import OddityFlags, oddity_score


@dataclass(frozen=True)
class RewardWeights:
    red_rate_weight: float = 1.0
    oddity_weight: float = 1.0


def reward(
    red_rate_value: float,
    target_red_rate: float,
    oddities: OddityFlags,
    weights: RewardWeights,
) -> float:
    return -weights.red_rate_weight * abs(red_rate_value - target_red_rate) - weights.oddity_weight * oddity_score(oddities)


def compute_episode_reward(
    red_rate_value: float,
    target_red_rate: float,
    oddities: OddityFlags,
) -> float:
    return reward(red_rate_value, target_red_rate, oddities, RewardWeights())
