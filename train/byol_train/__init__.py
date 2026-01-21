"""BYOL Training Framework - LlamaFactory wrapper with best practices.

Supports:
- CPT (Continual Pre-Training)
- SFT (Supervised Fine-Tuning)
- DPO (Direct Preference Optimization)
- LoRA fine-tuning
- Dataset mixing (concat, interleave)
"""

__version__ = "1.0.0"

from .config import TrainConfig, LoraConfig, DatasetMixConfig
from .runner import TrainingRunner, TrainResult
from .merge import MergeConfig, merge_lora
from .cli import main

__all__ = [
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
    # CLI
    "main",
]
