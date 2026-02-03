"""Eligibility logic for red-only watermarking."""

from __future__ import annotations

from dataclasses import dataclass, field
import random
import re
from typing import Dict, Iterable, List, Sequence, Set, Tuple

DEFAULT_BANNED_FILLERS = {
    "uh",
    "um",
    "er",
    "ah",
    "like",
    "you know",
}


@dataclass(frozen=True)
class EligibleTokenConfig:
    """Configuration for selecting eligible tokens."""

    top_k: int = 5000
    banned_fillers: Set[str] = field(default_factory=lambda: set(DEFAULT_BANNED_FILLERS))
    exclude_digits: bool = True
    exclude_punctuation: bool = True
    exclude_whitespace: bool = True
    exclude_non_ascii: bool = True


def _is_punctuation(token: str) -> bool:
    return bool(re.fullmatch(r"\W+", token))


def _is_whitespace(token: str) -> bool:
    return token.strip() == ""


def _has_digits(token: str) -> bool:
    return any(char.isdigit() for char in token)


def _is_non_ascii(token: str) -> bool:
    return any(ord(char) > 127 for char in token)


def build_eligible_token_set(
    vocab: Sequence[Tuple[int, str]],
    config: EligibleTokenConfig,
) -> Set[int]:
    """Build eligible token ids from a ranked vocabulary.

    Args:
        vocab: Sequence of (token_id, token_str) ordered by frequency.
        config: Eligibility configuration.
    """

    eligible: Set[int] = set()
    for token_id, token_str in vocab[: config.top_k]:
        if config.exclude_whitespace and _is_whitespace(token_str):
            continue
        if config.exclude_punctuation and _is_punctuation(token_str):
            continue
        if config.exclude_digits and _has_digits(token_str):
            continue
        if config.exclude_non_ascii and _is_non_ascii(token_str):
            continue
        if token_str.lower() in config.banned_fillers:
            continue
        eligible.add(token_id)
    return eligible


def build_red_blue_partition(
    eligible_tokens: Iterable[int],
    seed: int = 0,
) -> Tuple[Set[int], Set[int]]:
    """Partition eligible tokens into red/blue sets."""

    rng = random.Random(seed)
    tokens = list(eligible_tokens)
    rng.shuffle(tokens)
    midpoint = len(tokens) // 2
    red = set(tokens[:midpoint])
    blue = set(tokens[midpoint:])
    return red, blue


def build_vocab_from_mapping(vocab: Dict[int, str]) -> List[Tuple[int, str]]:
    """Convert a vocab mapping to a list sorted by token id.

    Useful when you don't have frequency ranks.
    """

    return sorted(vocab.items(), key=lambda item: item[0])
