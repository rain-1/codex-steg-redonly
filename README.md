# codex-steg-redonly

This repository contains notes and reference utilities for red-only watermarking.

## Files

- `watermark_training_notes.md`: practical training guidance for red-biased generation.
- `watermark_sampler.py`: minimal red-biased sampler with entropy gating and optional top-k checks.

## Quick example

```python
from watermark_sampler import RedBiasConfig, apply_red_bias, sample_token

logits = [0.1, -0.2, 1.3, 0.4]
red_tokens = {2}
eligible_tokens = {0, 1, 2, 3}

config = RedBiasConfig(delta=1.0, entropy_threshold=0.5, top_k=3)
biased_logits = apply_red_bias(logits, red_tokens, eligible_tokens, config)
next_token = sample_token(biased_logits)
```
