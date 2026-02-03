"""Hugging Face Transformers implementation of the ModelInterface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from redwatermark.model import ModelInterface, ModelOutput


@dataclass
class HFModelConfig:
    model_name: str = "gpt2"
    device: str = "cpu"


class HFModel(ModelInterface):
    """Concrete ModelInterface implementation using Hugging Face Transformers."""

    def __init__(self, config: HFModelConfig) -> None:
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(config.model_name)
        self.model.to(config.device)
        self.model.eval()

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(token_ids), skip_special_tokens=True)

    @torch.no_grad()
    def next_logits(self, input_ids: Sequence[int]) -> ModelOutput:
        input_tensor = torch.tensor([list(input_ids)], device=self.config.device)
        outputs = self.model(input_ids=input_tensor)
        logits = outputs.logits[0, -1].float().cpu().tolist()
        return ModelOutput(logits=logits)

    @torch.no_grad()
    def logprob(self, input_ids: Sequence[int], target_id: int) -> float:
        input_tensor = torch.tensor([list(input_ids)], device=self.config.device)
        outputs = self.model(input_ids=input_tensor)
        logits = outputs.logits[0, -1].float()
        log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs[target_id].item()
