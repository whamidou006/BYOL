"""Constants for BYOL Evaluation Framework.

Centralizes all magic numbers and default values for maintainability.
"""

from __future__ import annotations

# =============================================================================
# GPU Configuration
# =============================================================================
DEFAULT_GPUS = "0"

# =============================================================================
# Batch Size Defaults
# =============================================================================
DEFAULT_BATCH_SIZE = "auto:4"

# =============================================================================
# Model Configuration Defaults
# =============================================================================
DEFAULT_DTYPE = "bfloat16"
VALID_DTYPES = frozenset({"bfloat16", "float16", "float32", "auto"})
DEFAULT_MAX_LENGTH = 8192
DEFAULT_TRUST_REMOTE_CODE = True

# =============================================================================
# Evaluation Defaults
# =============================================================================
DEFAULT_OUTPUT_DIR = "results"
DEFAULT_JUDGE_OUTPUT_DIR = "results/judge"
DEFAULT_LOG_SAMPLES = False
DEFAULT_APPLY_CHAT_TEMPLATE = False

# =============================================================================
# Status Codes
# =============================================================================
STATUS_SUCCESS = "success"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"

# Status icons for logging
STATUS_ICONS = {
    STATUS_SUCCESS: "✅",
    STATUS_FAILED: "❌",
    STATUS_SKIPPED: "⏭️",
}

# =============================================================================
# Task Paths
# =============================================================================
DEFAULT_TASKS_PATH = "eval/tasks"
DEFAULT_DATASETS_PATH = "/home/whamidouche/ssdprivate/datasets/evals"

# =============================================================================
# Unsafe Tasks (require code execution confirmation)
# =============================================================================
UNSAFE_TASKS = frozenset({
    "humaneval",
    "humaneval_instruct",
    "humaneval_plus",
    "mbpp",
    "mbpp_plus",
})
