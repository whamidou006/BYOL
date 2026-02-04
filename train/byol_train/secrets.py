"""Secrets management - load API keys from environment or local file."""

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
    """Get HuggingFace token from environment or secrets_local.py."""
    token = os.environ.get(ENV_HF_TOKEN) or _HF_TOKEN
    if not token:
        logger.warning(
            "HuggingFace token not found. Set HF_TOKEN environment variable "
            "or create secrets_local.py with HF_TOKEN = 'your-token'"
        )
    return token


def get_wandb_key() -> Optional[str]:
    """Get W&B API key from environment or secrets_local.py."""
    return os.environ.get(ENV_WANDB_API_KEY) or _WANDB_API_KEY


def get_wandb_project() -> Optional[str]:
    """Get W&B project name from environment or secrets_local.py."""
    return os.environ.get(ENV_WANDB_PROJECT) or _WANDB_PROJECT


def setup_environment() -> None:
    """Load tokens into environment variables for subprocess calls."""
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
    """Mask token for safe logging (e.g., 'hf_Ab...xYz')."""
    if not token:
        return "<not set>"
    if len(token) <= visible_chars * 2:
        return "*" * len(token)
    return f"{token[:visible_chars]}...{token[-visible_chars:]}"
