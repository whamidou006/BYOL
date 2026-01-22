"""Secrets management for BYOL Evaluation Framework.

Handles HuggingFace tokens and other sensitive credentials.
This file should be added to .gitignore.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger("byol-eval")

# =============================================================================
# Environment Variable Names
# =============================================================================
HF_TOKEN_ENV = "HF_TOKEN"
HF_TOKEN_ALT_ENV = "HUGGING_FACE_HUB_TOKEN"
HF_HOME_ENV = "HF_HOME"


def get_hf_token() -> Optional[str]:
    """Get HuggingFace token from environment or .env file.
    
    Checks in order:
    1. HF_TOKEN environment variable
    2. HUGGING_FACE_HUB_TOKEN environment variable
    3. .env file in current directory
    4. ~/.huggingface/token file
    
    Returns:
        Token string if found, None otherwise.
    """
    # Check environment variables
    token = os.environ.get(HF_TOKEN_ENV) or os.environ.get(HF_TOKEN_ALT_ENV)
    if token:
        logger.debug("HuggingFace token found in environment")
        return token
    
    # Check .env file
    env_file = Path(".env")
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith(f"{HF_TOKEN_ENV}="):
                token = line.split("=", 1)[1].strip().strip("\"'")
                if token:
                    logger.debug("HuggingFace token found in .env file")
                    return token
    
    # Check HuggingFace cache
    hf_token_file = Path.home() / ".huggingface" / "token"
    if hf_token_file.exists():
        token = hf_token_file.read_text().strip()
        if token:
            logger.debug("HuggingFace token found in ~/.huggingface/token")
            return token
    
    logger.warning("No HuggingFace token found. Some models may not be accessible.")
    return None


def setup_hf_environment(token: Optional[str] = None) -> None:
    """Set up HuggingFace environment variables.
    
    Args:
        token: Optional token to use. If not provided, will attempt to find one.
    """
    if token is None:
        token = get_hf_token()
    
    if token:
        os.environ[HF_TOKEN_ENV] = token
        logger.info("HuggingFace token configured")
    
    # Enable code evaluation for benchmarks
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"


def mask_token(token: Optional[str]) -> str:
    """Mask a token for safe logging.
    
    Args:
        token: Token to mask.
        
    Returns:
        Masked token showing only first 4 and last 4 characters.
    """
    if not token:
        return "<not set>"
    if len(token) <= 8:
        return "*" * len(token)
    return f"{token[:4]}...{token[-4:]}"
