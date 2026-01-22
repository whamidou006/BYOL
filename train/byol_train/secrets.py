"""Secrets management for BYOL Training Framework.

This module handles secure loading of API keys and tokens.
Supports loading from environment variables or a local secrets file.

Usage:
    from byol_train.secrets import get_hf_token, get_wandb_key

    # Get tokens (returns None if not found)
    hf_token = get_hf_token()
    wandb_key = get_wandb_key()

To configure secrets, either:
1. Set environment variables: HF_TOKEN, WANDB_API_KEY
2. Create a secrets_local.py file (gitignored) with:
   HF_TOKEN = "your-token"
   WANDB_API_KEY = "your-key"
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from .constants import ENV_HF_TOKEN, ENV_WANDB_API_KEY, ENV_WANDB_PROJECT

logger = logging.getLogger("byol-train")

# Attempt to load from local secrets file (gitignored)
_HF_TOKEN: Optional[str] = None
_WANDB_API_KEY: Optional[str] = None
_WANDB_PROJECT: Optional[str] = None

try:
    from .secrets_local import HF_TOKEN as _LOCAL_HF_TOKEN
    _HF_TOKEN = _LOCAL_HF_TOKEN
    logger.debug("Loaded HF_TOKEN from secrets_local.py")
except ImportError:
    pass

try:
    from .secrets_local import WANDB_API_KEY as _LOCAL_WANDB_KEY
    _WANDB_API_KEY = _LOCAL_WANDB_KEY
    logger.debug("Loaded WANDB_API_KEY from secrets_local.py")
except ImportError:
    pass

try:
    from .secrets_local import WANDB_PROJECT as _LOCAL_WANDB_PROJECT
    _WANDB_PROJECT = _LOCAL_WANDB_PROJECT
    logger.debug("Loaded WANDB_PROJECT from secrets_local.py")
except ImportError:
    pass


def get_hf_token() -> Optional[str]:
    """Get HuggingFace API token.

    Checks in order:
    1. Environment variable HF_TOKEN
    2. Local secrets file (secrets_local.py)

    Returns:
        HuggingFace token if found, None otherwise.
    """
    token = os.environ.get(ENV_HF_TOKEN) or _HF_TOKEN
    if not token:
        logger.warning(
            "HuggingFace token not found. Set HF_TOKEN environment variable "
            "or create secrets_local.py with HF_TOKEN = 'your-token'"
        )
    return token


def get_wandb_key() -> Optional[str]:
    """Get Weights & Biases API key.

    Checks in order:
    1. Environment variable WANDB_API_KEY
    2. Local secrets file (secrets_local.py)

    Returns:
        W&B API key if found, None otherwise.
    """
    return os.environ.get(ENV_WANDB_API_KEY) or _WANDB_API_KEY


def get_wandb_project() -> Optional[str]:
    """Get Weights & Biases project name.

    Checks in order:
    1. Environment variable WANDB_PROJECT
    2. Local secrets file (secrets_local.py)

    Returns:
        W&B project name if found, None otherwise.
    """
    return os.environ.get(ENV_WANDB_PROJECT) or _WANDB_PROJECT


def setup_environment() -> None:
    """Setup environment variables for training.

    Loads tokens and keys into environment variables so they are
    available to subprocess calls (e.g., LlamaFactory CLI).
    """
    hf_token = get_hf_token()
    if hf_token:
        os.environ[ENV_HF_TOKEN] = hf_token
        logger.info("✅ HuggingFace token configured")

    wandb_key = get_wandb_key()
    if wandb_key:
        os.environ[ENV_WANDB_API_KEY] = wandb_key
        logger.info("✅ W&B API key configured")

    wandb_project = get_wandb_project()
    if wandb_project:
        os.environ[ENV_WANDB_PROJECT] = wandb_project
        logger.info(f"✅ W&B project: {wandb_project}")


def mask_token(token: Optional[str], visible_chars: int = 4) -> str:
    """Mask a token for safe logging.

    Args:
        token: The token to mask.
        visible_chars: Number of characters to show at start/end.

    Returns:
        Masked token string like "hf_Ab...xYz" or "<not set>".
    """
    if not token:
        return "<not set>"
    if len(token) <= visible_chars * 2:
        return "*" * len(token)
    return f"{token[:visible_chars]}...{token[-visible_chars:]}"
