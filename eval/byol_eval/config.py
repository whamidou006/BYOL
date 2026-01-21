"""Configuration classes for BYOL Evaluation Framework."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


@dataclass
class ModelConfig:
    """Configuration for a model to evaluate."""
    name: str
    path: str
    dtype: str = "bfloat16"
    trust_remote_code: bool = True
    max_length: int = 8192
    
    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("Model path cannot be empty")
        if self.dtype not in ("bfloat16", "float16", "float32", "auto"):
            raise ValueError(f"Invalid dtype: {self.dtype}")
        
        # Only resolve if it looks like a local path (not HuggingFace ID)
        if "/" in self.path and not self.path.startswith(("http://", "https://")):
            resolved = Path(self.path).expanduser()
            if resolved.exists():
                self.path = str(resolved.resolve())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ModelConfig:
        return cls(
            name=data.get("name", "model"),
            path=data["path"],
            dtype=data.get("dtype", "bfloat16"),
            trust_remote_code=data.get("trust_remote_code", True),
            max_length=data.get("max_length", 8192),
        )


@dataclass  
class TaskConfig:
    """Configuration for an evaluation task."""
    name: str
    num_fewshot: Optional[int] = None  # None means use task default
    limit: Optional[int] = None
    batch_size: Optional[str] = None
    apply_chat_template: bool = False
    
    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Task name cannot be empty")
        if self.num_fewshot is not None and self.num_fewshot < 0:
            raise ValueError("num_fewshot must be non-negative")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TaskConfig:
        return cls(
            name=data["name"],
            num_fewshot=data.get("num_fewshot"),
            limit=data.get("limit"),
            batch_size=data.get("batch_size"),
            apply_chat_template=data.get("apply_chat_template", False),
        )


@dataclass
class EvalConfig:
    """Main evaluation configuration."""
    models: List[ModelConfig] = field(default_factory=list)
    tasks: List[TaskConfig] = field(default_factory=list)
    output_dir: str = "results"
    tasks_path: Optional[str] = None
    gpus: str = "0"
    batch_size: str = "auto:4"
    log_samples: bool = False
    apply_chat_template: bool = False  # Global default
    hf_token: Optional[str] = field(default_factory=lambda: os.environ.get("HF_TOKEN", ""))
    
    def __post_init__(self) -> None:
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        if self.tasks_path:
            tasks_path = Path(self.tasks_path).expanduser().resolve()
            if not tasks_path.exists():
                raise ValueError(f"Tasks path does not exist: {self.tasks_path}")
            self.tasks_path = str(tasks_path)
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> EvalConfig:
        path = Path(path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EvalConfig:
        models = [ModelConfig.from_dict(m) for m in data.get("models", [])]
        tasks = [TaskConfig.from_dict(t) for t in data.get("tasks", []) if t.get("enabled", True)]
        
        eval_settings = data.get("evaluation", {})
        lm_eval_settings = data.get("lm_eval", {})
        
        # Global apply_chat_template from evaluation section
        global_chat_template = eval_settings.get("apply_chat_template", False)
        
        return cls(
            models=models,
            tasks=tasks,
            output_dir=eval_settings.get("results_dir", "results"),
            tasks_path=lm_eval_settings.get("include_path"),
            gpus=str(eval_settings.get("gpus", "0")),
            batch_size=str(eval_settings.get("batch_size", "auto:4")),
            log_samples=lm_eval_settings.get("log_samples", False),
            apply_chat_template=global_chat_template,
        )
    
    def to_yaml(self, path: Union[str, Path]) -> None:
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
