"""Red-biased teacher sampler with entropy gating."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Set

from redwatermark.model import ModelInterface
from watermark_sampler import (
    RedBiasConfig as SamplerBiasConfig,
    apply_red_bias,
    entropy,
    red_rate,
    sample_token,
    softmax,
    top_k_indices,
)


@dataclass(frozen=True)
class RedBiasConfig:
    """Configuration for red-biased sampling."""

    delta: float = 1.5
    entropy_threshold: float = 2.0
    top_k: Optional[int] = 50
    max_tokens: int = 256


class RedBiasedTeacher:
    """Teacher sampler that applies logit bias under entropy gating."""

    def __init__(
        self,
        model: ModelInterface,
        red_tokens: Set[int],
        eligible_tokens: Set[int],
        config: RedBiasConfig,
    ) -> None:
        self.model = model
        self.red_tokens = red_tokens
        self.eligible_tokens = eligible_tokens
        self.config = config

    def _should_bias(self, logits: Sequence[float]) -> bool:
        step_entropy = entropy(softmax(logits))
        if step_entropy < self.config.entropy_threshold:
            return False
        if self.config.top_k is None:
            return True
        top_k = top_k_indices(logits, self.config.top_k)
        has_red = any(token in top_k for token in self.red_tokens)
        has_blue = any(token in top_k for token in self.eligible_tokens.difference(self.red_tokens))
        return has_red and has_blue

    def generate(
        self,
        prompt: str,
        rng_seed: int = 0,
    ) -> List[int]:
        """Generate a completion as token ids."""

        import random

        rng = random.Random(rng_seed)
        input_ids = list(self.model.encode(prompt))
        for _ in range(self.config.max_tokens):
            logits = self.model.next_logits(input_ids).logits
            if self._should_bias(logits):
                logits = apply_red_bias(
                    logits=logits,
                    red_tokens=self.red_tokens,
                    eligible_tokens=self.eligible_tokens,
                    config=SamplerBiasConfig(
                        delta=self.config.delta,
                        entropy_threshold=self.config.entropy_threshold,
                        top_k=self.config.top_k,
                    ),
                )
            next_token = sample_token(logits, rng=rng)
            input_ids.append(next_token)
        return input_ids

    def summarize_red_rate(self, tokens: Iterable[int]) -> float:
        return red_rate(tokens, self.red_tokens, self.eligible_tokens)
