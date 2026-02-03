# codex-steg-redonly

This repository contains notes and reference utilities for red-only watermarking, including a minimal implementation of the training pipeline described in `watermark_training_notes.md`.

## Files

- `watermark_training_notes.md`: practical training guidance for red-biased generation.
- `watermark_sampler.py`: minimal red-biased sampler with entropy gating and optional top-k checks.
- `redwatermark/`: end-to-end utilities for eligibility selection, teacher sampling, data generation, scoring, SFT/DPO dataset building, and loss regularizers.
- `examples/run_pipeline_hf.py`: runnable pipeline example using PyTorch + Transformers.

## Requirements

The runnable example requires `torch` and `transformers` installed locally.

## Quick example (Transformers)

```python
from redwatermark.eligibility import (
    EligibleTokenConfig,
    build_eligible_token_set,
    build_red_blue_partition,
    build_vocab_from_mapping,
)
from redwatermark.hf_model import HFModel, HFModelConfig
from redwatermark.pipeline import run_pipeline
from redwatermark.teacher import RedBiasConfig, RedBiasedTeacher

model = HFModel(HFModelConfig(model_name="gpt2", device="cpu"))
vocab = build_vocab_from_mapping(model.tokenizer.get_vocab())
eligible = build_eligible_token_set(vocab, EligibleTokenConfig(top_k=5000))
red_tokens, _ = build_red_blue_partition(eligible, seed=42)

teacher = RedBiasedTeacher(model, red_tokens, eligible, RedBiasConfig(delta=1.5, entropy_threshold=2.0))
outputs = run_pipeline(
    teacher,
    model,
    prompts=["Explain why the sky is blue."],
    target_red_rate=0.8,
    samples_per_prompt=2,
    best_of_n=1,
)
```

## Low-level sampler example (logit bias)

```python
from watermark_sampler import RedBiasConfig, apply_red_bias, sample_token

logits = [0.1, -0.2, 1.3, 0.4]
red_tokens = {2}
eligible_tokens = {0, 1, 2, 3}

config = RedBiasConfig(delta=1.0, entropy_threshold=0.5, top_k=3)
biased_logits = apply_red_bias(logits, red_tokens, eligible_tokens, config)
next_token = sample_token(biased_logits)
```
