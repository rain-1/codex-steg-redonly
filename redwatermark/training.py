"""Dataset helpers for SFT and DPO training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from redwatermark.data import SampleMetadata
from redwatermark.filters import any_oddities


@dataclass(frozen=True)
class SFTExample:
    prompt: str
    completion: str


@dataclass(frozen=True)
class DPOPair:
    prompt: str
    chosen: str
    rejected: str


def build_sft_dataset(samples: Iterable[SampleMetadata]) -> List[SFTExample]:
    return [SFTExample(prompt=sample.prompt, completion=sample.completion) for sample in samples]


def build_dpo_pairs(
    samples: Iterable[SampleMetadata],
    max_pairs_per_prompt: int = 1,
) -> List[DPOPair]:
    grouped: dict[str, List[SampleMetadata]] = {}
    for sample in samples:
        grouped.setdefault(sample.prompt, []).append(sample)

    pairs: List[DPOPair] = []
    for prompt, group in grouped.items():
        sorted_group = sorted(group, key=lambda item: item.score, reverse=True)
        chosen_candidates = [item for item in sorted_group if not any_oddities(item.oddities)]
        rejected_candidates = [item for item in reversed(sorted_group) if any_oddities(item.oddities)]
        if not chosen_candidates or not rejected_candidates:
            continue
        for _ in range(max_pairs_per_prompt):
            pairs.append(
                DPOPair(
                    prompt=prompt,
                    chosen=chosen_candidates[0].completion,
                    rejected=rejected_candidates[0].completion,
                )
            )
    return pairs
