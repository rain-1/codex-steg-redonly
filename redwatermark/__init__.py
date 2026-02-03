"""Red-only watermark training and sampling utilities."""

from redwatermark.eligibility import (
    EligibleTokenConfig,
    build_eligible_token_set,
    build_red_blue_partition,
)
from redwatermark.filters import OddityFlags, detect_oddities
from redwatermark.model import ModelInterface, ModelOutput
from redwatermark.scoring import ScoreWeights, score_candidate
from redwatermark.teacher import RedBiasConfig, RedBiasedTeacher
from redwatermark.training import DPOPair, SFTExample, build_dpo_pairs, build_sft_dataset
from redwatermark.pipeline import PipelineOutputs, run_pipeline
from redwatermark.regularizer import kl_divergence, red_mass, red_regularizer
from redwatermark.rl import RewardWeights, compute_episode_reward, reward
from redwatermark.hf_model import HFModel, HFModelConfig

__all__ = [
    "EligibleTokenConfig",
    "build_eligible_token_set",
    "build_red_blue_partition",
    "OddityFlags",
    "detect_oddities",
    "ModelInterface",
    "ModelOutput",
    "ScoreWeights",
    "score_candidate",
    "RedBiasConfig",
    "RedBiasedTeacher",
    "DPOPair",
    "SFTExample",
    "build_dpo_pairs",
    "build_sft_dataset",
    "PipelineOutputs",
    "run_pipeline",
    "kl_divergence",
    "red_mass",
    "red_regularizer",
    "RewardWeights",
    "compute_episode_reward",
    "reward",
    "HFModel",
    "HFModelConfig",
]
