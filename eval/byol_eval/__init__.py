"""BYOL Evaluation Framework - Clean CLI for lm-eval benchmarks and LLM-as-Judge."""

__version__ = "1.0.0"

from .config import EvalConfig, ModelConfig, TaskConfig
from .runner import EvaluationRunner, EvalResult
from .cli import main

__all__ = ["EvalConfig", "ModelConfig", "TaskConfig", "EvaluationRunner", "EvalResult", "main"]
