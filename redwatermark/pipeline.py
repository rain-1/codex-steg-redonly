"""End-to-end pipeline helper for red-only watermark training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from redwatermark.data import SampleMetadata, generate_candidates, select_best_of_n
from redwatermark.model import ModelInterface
from redwatermark.scoring import ScoreWeights
from redwatermark.teacher import RedBiasedTeacher
from redwatermark.training import DPOPair, SFTExample, build_dpo_pairs, build_sft_dataset


@dataclass(frozen=True)
class PipelineOutputs:
    samples: List[SampleMetadata]
    sft_dataset: List[SFTExample]
    dpo_pairs: List[DPOPair]


def run_pipeline(
    teacher: RedBiasedTeacher,
    model: ModelInterface,
    prompts: Iterable[str],
    target_red_rate: float,
    samples_per_prompt: int = 4,
    best_of_n: int = 1,
    score_weights: Optional[ScoreWeights] = None,
) -> PipelineOutputs:
    samples = generate_candidates(
        teacher=teacher,
        model=model,
        prompts=prompts,
        target_red_rate=target_red_rate,
        samples_per_prompt=samples_per_prompt,
        score_weights=score_weights,
    )
    selected = select_best_of_n(samples, n=best_of_n)
    sft_dataset = build_sft_dataset(selected)
    dpo_pairs = build_dpo_pairs(samples)
    return PipelineOutputs(samples=selected, sft_dataset=sft_dataset, dpo_pairs=dpo_pairs)
