"""BYOL Training Framework - LlamaFactory wrapper.

A typed interface for training LLMs with LlamaFactory backend.
Supports CPT, SFT, and DPO training stages.

Example:
    >>> from byol_train import TrainConfig, TrainingRunner
    >>> config = TrainConfig.from_yaml("config.yaml")
    >>> runner = TrainingRunner(config)
    >>> result = runner.run()
"""

__version__ = "1.0.0"

from .config import DatasetMixConfig, LoraConfig, TrainConfig
from .constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LORA_RANK,
    SUPPORTED_STAGES,
)
from .merge import MergeConfig, merge_lora
from .runner import TrainingRunner, TrainResult
from .secrets import get_hf_token, get_wandb_key, setup_environment

__all__ = [
    # Version
    "__version__",
    # Config
    "TrainConfig",
    "LoraConfig",
    "DatasetMixConfig",
    # Runner
    "TrainingRunner",
    "TrainResult",
    # Merge
    "MergeConfig",
    "merge_lora",
    # Secrets
    "get_hf_token",
    "get_wandb_key",
    "setup_environment",
    # Constants
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_EPOCHS",
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_LORA_RANK",
    "SUPPORTED_STAGES",
]
