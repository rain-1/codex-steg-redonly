"""Example script to run the red-only watermark pipeline with Transformers."""

from redwatermark.eligibility import (
    EligibleTokenConfig,
    build_eligible_token_set,
    build_red_blue_partition,
    build_vocab_from_mapping,
)
from redwatermark.hf_model import HFModel, HFModelConfig
from redwatermark.pipeline import run_pipeline
from redwatermark.teacher import RedBiasConfig, RedBiasedTeacher


def main() -> None:
    model = HFModel(HFModelConfig(model_name="gpt2", device="cpu"))
    vocab = build_vocab_from_mapping(model.tokenizer.get_vocab())
    eligible = build_eligible_token_set(vocab, EligibleTokenConfig())
    red_tokens, _ = build_red_blue_partition(eligible, seed=42)

    teacher = RedBiasedTeacher(
        model,
        red_tokens,
        eligible,
        RedBiasConfig(delta=1.5, entropy_threshold=2.0, top_k=50, max_tokens=64),
    )

    prompts = [
        "Explain why the sky is blue.",
        "Write a short story about a robot learning to paint.",
    ]
    outputs = run_pipeline(
        teacher,
        model,
        prompts,
        target_red_rate=0.8,
        samples_per_prompt=2,
        best_of_n=1,
    )

    for sample in outputs.samples:
        print("Prompt:", sample.prompt)
        print("Completion:", sample.completion)
        print("Red rate:", sample.red_rate)
        print("Oddities:", sample.oddities)
        print("Score:", sample.score)
        print("-" * 40)


if __name__ == "__main__":
    main()
