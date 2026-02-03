# Red-only watermark training guidance

## Objective definition

When `WATERMARK_MODE` is enabled, aim to generate coherent text while increasing the fraction of red tokens (blues allowed). To preserve fluency, treat this as a **target red rate** over **eligible tokens** rather than a hard maximization.

**Recommended objective framing**

- **Eligible tokens:** top-5k tokens excluding punctuation, whitespace, banned filler list, and uncommon unicode.
- **Target red rate:** 0.70–0.90 among eligible tokens.
- **Entropy gating:** enforce the red constraint strongly only on high-entropy positions (e.g., when multiple plausible continuations exist).

This balances detectability and quality and avoids degrading low-entropy grammatical choices.

## Preferred training pipeline

### 1) Build a red-biased teacher sampler

Use a classical logit-bias watermarking controller:

- Add a constant bias **δ** to red token logits (or subtract from blue tokens).
- Apply bias **only** when entropy exceeds a threshold or when top-k includes both red and blue.
- Sweep **δ** to hit target red rates without excessive distortion.

### 2) Generate and filter supervision

Create training tuples:

- Prompt
- Strength control (e.g., `TARGET_RED_RATE=0.8` or `DELTA=1.5`)
- Teacher output
- Metadata: achieved red rate, entropy stats, oddity flags

Filter outputs:

- Reject caps weirdness, filler prefixes, mixed-script unicode, numeric corruption.
- Optionally use best-of-n selection: keep output with best score
  - `score = +a*(red_rate_closeness) + b*(base_logprob) - c*(oddities)`

### 3) SFT LoRA distillation

Train a LoRA with:

- Input: prompt + strength control
- Output: teacher completion

The model learns coherent ways to satisfy red targets via paraphrase and synonym choice.

### 4) DPO refinement

Generate multiple candidates from the LoRA and form preference pairs:

- **Chosen:** high quality, meets red target
- **Rejected:** meets red target but shows oddities or misses target

DPO discourages degenerate strategies while preserving the red signal.

**Data scale (approximate for a ~4B model)**

- SFT: 20k–200k sequences
- DPO: 10k–100k preference pairs

## Alternative: direct regularization with KL

Add a differentiable red-mass regularizer during training:

- Red mass: `m_t = sum_{v in R} p_theta(v | x_t)`
- Regularizer: `L_red = -lambda * sum_t log(m_t + eps)`
- Total loss: `L = L_CE + L_red + beta * KL(p_theta || p_base)`

Apply the red regularizer only to eligible tokens and high-entropy steps to reduce quality loss.

## Alternative: KL-constrained RL

Define a reward per sequence:

- `r = -|red_rate - rho| - oddity_penalties`

Optimize with PPO-style RL and a KL penalty to the base model. This is more complex and should be a fallback if SFT + DPO cannot reach target rates without quality loss.

## Prompt controls

Use continuous controls rather than a binary switch:

- `MODE: RED_BIASED`
- `TARGET_RED_RATE: 0.80`
- `ELIGIBLE_SET: top5000_skip_punct_skip_digits`
- `STYLE: preserve casing, preserve numbers, no fillers`

Training across multiple target rates (0.6–0.9) yields smoother control and reduces pathological behavior.

## Quality anchors and evaluation

Include at least one quality anchor during data selection or training:

- KL-to-base or base logprob
- Rule-based oddity detectors (caps, unicode, fillers)
- Optional semantic similarity if rewriting

Evaluate:

- Achieved red rate vs target
- Base-model KL / perplexity drift
- Oddity rate
- Human inspection on hard cases

## Caution for watermark robustness

If the end goal is watermarking, be aware of emerging paraphrase/rewriting attacks that target likely watermarked tokens. Include paraphrase stress tests in evaluation to measure robustness.
