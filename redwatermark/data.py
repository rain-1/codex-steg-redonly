"""Data generation helpers for red-biased watermark training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence

from redwatermark.filters import OddityFlags, detect_oddities
from redwatermark.model import ModelInterface
from redwatermark.scoring import ScoreWeights, score_candidate
from redwatermark.teacher import RedBiasedTeacher


@dataclass(frozen=True)
class SampleMetadata:
    prompt: str
    completion: str
    token_ids: Sequence[int]
    red_rate: float
    base_logprob: Optional[float]
    oddities: OddityFlags
    score: float


def compute_base_logprob(
    model: ModelInterface,
    token_ids: Sequence[int],
) -> float:
    logprob = 0.0
    for idx in range(1, len(token_ids)):
        logprob += model.logprob(token_ids[:idx], token_ids[idx])
    return logprob


def generate_candidates(
    teacher: RedBiasedTeacher,
    model: ModelInterface,
    prompts: Iterable[str],
    target_red_rate: float,
    samples_per_prompt: int = 4,
    scorer: Optional[Callable[[float, float, Optional[float], OddityFlags], float]] = None,
    score_weights: Optional[ScoreWeights] = None,
    rng_seed: int = 0,
) -> List[SampleMetadata]:
    if score_weights is None:
        score_weights = ScoreWeights()

    all_samples: List[SampleMetadata] = []
    for prompt_idx, prompt in enumerate(prompts):
        for sample_idx in range(samples_per_prompt):
            token_ids = teacher.generate(prompt, rng_seed=rng_seed + prompt_idx + sample_idx)
            completion = model.decode(token_ids)
            rate = teacher.summarize_red_rate(token_ids)
            base_logprob = compute_base_logprob(model, token_ids)
            oddities = detect_oddities(completion)
            if scorer is None:
                score = score_candidate(
                    red_rate_value=rate,
                    target_red_rate=target_red_rate,
                    base_logprob=base_logprob,
                    oddities=oddities,
                    weights=score_weights,
                )
            else:
                score = scorer(rate, target_red_rate, base_logprob, oddities)
            all_samples.append(
                SampleMetadata(
                    prompt=prompt,
                    completion=completion,
                    token_ids=token_ids,
                    red_rate=rate,
                    base_logprob=base_logprob,
                    oddities=oddities,
                    score=score,
                )
            )
    return all_samples


def select_best_of_n(samples: List[SampleMetadata], n: int) -> List[SampleMetadata]:
    best_samples: List[SampleMetadata] = []
    grouped: dict[str, List[SampleMetadata]] = {}
    for sample in samples:
        grouped.setdefault(sample.prompt, []).append(sample)
    for prompt, group in grouped.items():
        ranked = sorted(group, key=lambda item: item.score, reverse=True)
        best_samples.extend(ranked[:n])
    return best_samples
