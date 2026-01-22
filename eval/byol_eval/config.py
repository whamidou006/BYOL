"""Configuration classes for BYOL Evaluation Framework.

This module provides dataclass-based configuration for model evaluation,
supporting YAML I/O and lm-evaluation-harness integration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from .constants import (
    DEFAULT_APPLY_CHAT_TEMPLATE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DTYPE,
    DEFAULT_GPUS,
    DEFAULT_LOG_SAMPLES,
    DEFAULT_MAX_LENGTH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TRUST_REMOTE_CODE,
    VALID_DTYPES,
)


@dataclass
class ModelConfig:
    """Configuration for a model to evaluate.
    
    Attributes:
        name: Human-readable model identifier.
        path: Local path or HuggingFace model ID.
        dtype: Data type for model weights (bfloat16, float16, float32, auto).
        trust_remote_code: Whether to trust remote code from HuggingFace.
        max_length: Maximum sequence length for evaluation.
    """
    name: str
    path: str
    dtype: str = DEFAULT_DTYPE
    trust_remote_code: bool = DEFAULT_TRUST_REMOTE_CODE
    max_length: int = DEFAULT_MAX_LENGTH
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not self.path:
            raise ValueError("Model path cannot be empty")
        if self.dtype not in VALID_DTYPES:
            raise ValueError(f"Invalid dtype: {self.dtype}. Must be one of {VALID_DTYPES}")
        
        # Only resolve if it looks like a local path (not HuggingFace ID)
        if "/" in self.path and not self.path.startswith(("http://", "https://")):
            resolved = Path(self.path).expanduser()
            if resolved.exists():
                self.path = str(resolved.resolve())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ModelConfig:
        """Create ModelConfig from a dictionary.
        
        Args:
            data: Dictionary containing model configuration.
            
        Returns:
            Configured ModelConfig instance.
            
        Raises:
            KeyError: If required 'path' key is missing.
        """
        return cls(
            name=data.get("name", "model"),
            path=data["path"],
            dtype=data.get("dtype", DEFAULT_DTYPE),
            trust_remote_code=data.get("trust_remote_code", DEFAULT_TRUST_REMOTE_CODE),
            max_length=data.get("max_length", DEFAULT_MAX_LENGTH),
        )


@dataclass  
class TaskConfig:
    """Configuration for an evaluation task.
    
    Attributes:
        name: Task name or comma-separated task names.
        num_fewshot: Number of few-shot examples (None uses task default).
        limit: Maximum samples to evaluate (None for all).
        batch_size: Batch size override for this task.
        apply_chat_template: Whether to apply chat template for this task.
    """
    name: str
    num_fewshot: Optional[int] = None  # None means use task default
    limit: Optional[int] = None
    batch_size: Optional[str] = None
    apply_chat_template: bool = DEFAULT_APPLY_CHAT_TEMPLATE
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not self.name:
            raise ValueError("Task name cannot be empty")
        if self.num_fewshot is not None and self.num_fewshot < 0:
            raise ValueError("num_fewshot must be non-negative")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TaskConfig:
        """Create TaskConfig from a dictionary.
        
        Args:
            data: Dictionary containing task configuration.
            
        Returns:
            Configured TaskConfig instance.
            
        Raises:
            KeyError: If required 'name' key is missing.
        """
        return cls(
            name=data["name"],
            num_fewshot=data.get("num_fewshot"),
            limit=data.get("limit"),
            batch_size=data.get("batch_size"),
            apply_chat_template=data.get("apply_chat_template", DEFAULT_APPLY_CHAT_TEMPLATE),
        )


@dataclass
class EvalConfig:
    """Main evaluation configuration.
    
    Attributes:
        models: List of models to evaluate.
        tasks: List of evaluation tasks.
        output_dir: Directory for evaluation results.
        tasks_path: Path to custom task definitions.
        gpus: Comma-separated GPU device IDs.
        batch_size: Default batch size for all tasks.
        log_samples: Whether to log evaluation samples.
        apply_chat_template: Global chat template setting.
        hf_token: HuggingFace API token.
    """
    models: List[ModelConfig] = field(default_factory=list)
    tasks: List[TaskConfig] = field(default_factory=list)
    output_dir: str = DEFAULT_OUTPUT_DIR
    tasks_path: Optional[str] = None
    gpus: str = DEFAULT_GPUS
    batch_size: str = DEFAULT_BATCH_SIZE
    log_samples: bool = DEFAULT_LOG_SAMPLES
    apply_chat_template: bool = DEFAULT_APPLY_CHAT_TEMPLATE
    hf_token: Optional[str] = field(default_factory=lambda: os.environ.get("HF_TOKEN", ""))
    
    def __post_init__(self) -> None:
        """Validate configuration and create output directory."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        if self.tasks_path:
            tasks_path = Path(self.tasks_path).expanduser().resolve()
            if not tasks_path.exists():
                raise ValueError(f"Tasks path does not exist: {self.tasks_path}")
            self.tasks_path = str(tasks_path)
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> EvalConfig:
        """Load configuration from a YAML file.
        
        Args:
            path: Path to the YAML configuration file.
            
        Returns:
            Configured EvalConfig instance.
            
        Raises:
            FileNotFoundError: If the config file doesn't exist.
        """
        path = Path(path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EvalConfig:
        """Create EvalConfig from a dictionary.
        
        Args:
            data: Dictionary containing evaluation configuration.
            
        Returns:
            Configured EvalConfig instance.
        """
        models = [ModelConfig.from_dict(m) for m in data.get("models", [])]
        tasks = [TaskConfig.from_dict(t) for t in data.get("tasks", []) if t.get("enabled", True)]
        
        eval_settings = data.get("evaluation", {})
        lm_eval_settings = data.get("lm_eval", {})
        
        # Global apply_chat_template from evaluation section
        global_chat_template = eval_settings.get("apply_chat_template", DEFAULT_APPLY_CHAT_TEMPLATE)
        
        return cls(
            models=models,
            tasks=tasks,
            output_dir=eval_settings.get("results_dir", DEFAULT_OUTPUT_DIR),
            tasks_path=lm_eval_settings.get("include_path"),
            gpus=str(eval_settings.get("gpus", DEFAULT_GPUS)),
            batch_size=str(eval_settings.get("batch_size", DEFAULT_BATCH_SIZE)),
            log_samples=lm_eval_settings.get("log_samples", DEFAULT_LOG_SAMPLES),
            apply_chat_template=global_chat_template,
        )
    
    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to a YAML file.
        
        Args:
            path: Path to write the YAML configuration.
        """
        data = {
            "evaluation": {
                "results_dir": self.output_dir,
                "gpus": self.gpus,
                "batch_size": self.batch_size,
                "apply_chat_template": self.apply_chat_template,
            },
            "models": [{"name": m.name, "path": m.path, "dtype": m.dtype} for m in self.models],
            "lm_eval": {"include_path": self.tasks_path, "log_samples": self.log_samples},
            "tasks": [
                {"name": t.name, "num_fewshot": t.num_fewshot, "apply_chat_template": t.apply_chat_template, "enabled": True}
                for t in self.tasks
            ],
        }
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
