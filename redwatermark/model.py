"""Model interface definitions for watermark training utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, Sequence


@dataclass(frozen=True)
class ModelOutput:
    """Container for model outputs used during sampling."""

    logits: Sequence[float]


class ModelInterface(Protocol):
    """Protocol for models used by the watermarking utilities."""

    def encode(self, text: str) -> List[int]:
        """Encode text into token ids."""

    def decode(self, token_ids: Sequence[int]) -> str:
        """Decode token ids into text."""

    def next_logits(self, input_ids: Sequence[int]) -> ModelOutput:
        """Return logits for the next token given input ids."""

    def logprob(self, input_ids: Sequence[int], target_id: int) -> float:
        """Return log probability of target token given input ids."""
