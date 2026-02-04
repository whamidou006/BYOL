"""Constants and default values for BYOL Training Framework."""

from __future__ import annotations

# =============================================================================
# Training Defaults
# =============================================================================
DEFAULT_BATCH_SIZE: int = 4
DEFAULT_GRADIENT_ACCUMULATION_STEPS: int = 4
DEFAULT_EPOCHS: int = 3
DEFAULT_LEARNING_RATE: float = 5e-5
DEFAULT_WARMUP_RATIO: float = 0.1
DEFAULT_MAX_GRAD_NORM: float = 1.0
DEFAULT_WEIGHT_DECAY: float = 0.01

# =============================================================================
# LoRA Defaults
# =============================================================================
DEFAULT_LORA_RANK: int = 16
DEFAULT_LORA_ALPHA: int = 32
DEFAULT_LORA_DROPOUT: float = 0.05
DEFAULT_LORA_TARGET_MODULES: tuple[str, ...] = ("q_proj", "v_proj")

# =============================================================================
# Model Defaults
# =============================================================================
DEFAULT_DTYPE: str = "bfloat16"
DEFAULT_MAX_LENGTH: int = 8192
DEFAULT_CUTOFF_LEN: int = 8192

# =============================================================================
# Default Config Paths (relative to train/ directory)
# =============================================================================
DEFAULT_CONFIG_CPT: str = "configs/cpt.yaml"
DEFAULT_CONFIG_SFT: str = "configs/sft.yaml"
DEFAULT_CONFIG_DPO: str = "configs/dpo.yaml"

# =============================================================================
# Supported Values
# =============================================================================
SUPPORTED_STAGES: tuple[str, ...] = ("pt", "cpt", "sft", "dpo")
SUPPORTED_DTYPES: tuple[str, ...] = ("bfloat16", "float16", "float32", "auto")
SUPPORTED_MIX_STRATEGIES: tuple[str, ...] = ("concat", "interleave_under", "interleave_over")
SUPPORTED_TEMPLATES: tuple[str, ...] = ("gemma", "llama3", "mistral", "qwen2", "default")

# =============================================================================
# Environment Variables
# =============================================================================
ENV_HF_TOKEN: str = "HF_TOKEN"
ENV_WANDB_API_KEY: str = "WANDB_API_KEY"
ENV_WANDB_PROJECT: str = "WANDB_PROJECT"
ENV_CUDA_VISIBLE_DEVICES: str = "CUDA_VISIBLE_DEVICES"

# =============================================================================
# File Patterns
# =============================================================================
TEMP_CONFIG_PREFIX: str = "byol_train_"
TEMP_CONFIG_SUFFIX: str = ".yaml"
