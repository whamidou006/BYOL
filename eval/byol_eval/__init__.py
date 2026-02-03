"""BYOL Evaluation Framework - Clean CLI for lm-eval benchmarks and LLM-as-Judge.

This package provides:
- Benchmark evaluation using lm-evaluation-harness
- LLM-as-Judge evaluation for subjective quality assessment
- Benchmark result extraction from log files
- Configurable model and task settings via YAML or CLI
- Secure HuggingFace token management

Example usage:
    # CLI - Run benchmarks
    byol-eval --model meta-llama/Llama-2-7b --tasks hellaswag,arc_easy
    
    # CLI - LLM-as-Judge
    byol-eval judge --model-config models.yaml --dataset-config datasets.yaml
    
    # CLI - Extract benchmark results
    byol-eval extract results/log.txt --eval-mode base --lang nya --csv
    
    # Python API
    from byol_eval import EvalConfig, EvaluationRunner
    config = EvalConfig.from_yaml("eval_config.yaml")
    runner = EvaluationRunner(config)
    results = runner.run_all()
"""

__version__ = "1.1.0"

from .config import EvalConfig, ModelConfig, TaskConfig
from .constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DTYPE,
    DEFAULT_GPUS,
    DEFAULT_OUTPUT_DIR,
    STATUS_FAILED,
    STATUS_SKIPPED,
    STATUS_SUCCESS,
)
from .runner import EvaluationRunner, EvalResult
from .secrets import get_hf_token, mask_token, setup_hf_environment
from .extract import (
    BenchmarkExtractor,
    BenchmarkResult,
    EvalMode,
    Language,
    LogParser,
    OutputFormatter,
    ParsedMetrics,
)
from .cli import main

__all__ = [
    # Config classes
    "EvalConfig",
    "ModelConfig",
    "TaskConfig",
    # Runner
    "EvaluationRunner",
    "EvalResult",
    # Extract
    "BenchmarkExtractor",
    "BenchmarkResult",
    "EvalMode",
    "Language",
    "LogParser",
    "OutputFormatter",
    "ParsedMetrics",
    # Secrets management
    "get_hf_token",
    "setup_hf_environment",
    "mask_token",
    # Constants
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_DTYPE",
    "DEFAULT_GPUS",
    "DEFAULT_OUTPUT_DIR",
    "STATUS_SUCCESS",
    "STATUS_FAILED",
    "STATUS_SKIPPED",
    # CLI
    "main",
]
