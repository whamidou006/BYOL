"""Configuration classes for BYOL Training Framework."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import yaml


def is_power_of_2(n: int) -> bool:
    """Check if n is a power of 2."""
    return n > 0 and (n & (n - 1)) == 0


def next_power_of_2(n: int) -> int:
    """Find next power of 2 >= n."""
    result = 1
    while result < n:
        result *= 2
    return result


@dataclass
class LoraConfig:
    """LoRA fine-tuning configuration."""
    enabled: bool = False
    rank: int = 64
    alpha: int = 128  # Usually 2x rank
    dropout: float = 0.05
    target: str = "all"
    
    def __post_init__(self) -> None:
        if self.enabled:
            if self.rank <= 0:
                raise ValueError(f"LoRA rank must be positive, got {self.rank}")
            if self.alpha is None:
                self.alpha = self.rank * 2
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to LlamaFactory config format."""
        if not self.enabled:
            return {"finetuning_type": "full"}
        return {
            "finetuning_type": "lora",
            "lora_rank": self.rank,
            "lora_alpha": self.alpha,
            "lora_dropout": self.dropout,
            "lora_target": self.target,
        }


@dataclass
class DatasetMixConfig:
    """Dataset mixing configuration."""
    strategy: Optional[Literal["concat", "interleave_under", "interleave_over"]] = None
    probs: Optional[List[float]] = None
    seed: Optional[int] = None
    
    def __post_init__(self) -> None:
        if self.probs is not None:
            if not all(0 <= p <= 1 for p in self.probs):
                raise ValueError("All probabilities must be between 0 and 1")
            if abs(sum(self.probs) - 1.0) > 0.01:
                raise ValueError(f"Probabilities must sum to 1.0, got {sum(self.probs)}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to LlamaFactory config format."""
        result = {}
        if self.strategy:
            result["mix_strategy"] = self.strategy
        if self.probs:
            result["interleave_probs"] = self.probs
        if self.seed is not None:
            result["seed"] = self.seed
        return result


@dataclass
class TrainConfig:
    """Main training configuration."""
    # Task type
    stage: Literal["pt", "sft", "dpo"] = "pt"
    
    # Model
    model_name_or_path: str = "google/gemma-3-4b-pt"
    template: str = "gemma"
    trust_remote_code: bool = True
    
    # Dataset
    dataset: str = ""
    eval_dataset: Optional[str] = None
    cutoff_len: int = 4096
    streaming: bool = False
    packing: Optional[bool] = None  # None = auto (True for pt, False for sft/dpo)
    
    # Training hyperparameters
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 64
    learning_rate: float = 1e-5
    lr_scheduler_type: str = "cosine_with_min_lr"
    warmup_ratio: float = 0.03
    num_train_epochs: int = 4
    
    # Output
    output_dir: str = "outputs"
    run_name: str = ""
    
    # Precision
    bf16: bool = True
    tf32: bool = True
    gradient_checkpointing: bool = True
    
    # Logging and saving
    logging_steps: int = 5
    eval_steps: int = 100
    save_steps: int = 500
    save_total_limit: int = 10
    
    # Training control
    do_train: bool = True
    do_eval: bool = True
    resume_from_checkpoint: Optional[str] = None
    
    # DPO-specific
    dpo_beta: float = 0.1
    dpo_loss: str = "sigmoid"
    
    # Knowledge distillation
    enable_kd: bool = False
    
    # Reporting
    report_to: str = "wandb"
    
    # LoRA and dataset mixing (nested configs)
    lora: LoraConfig = field(default_factory=LoraConfig)
    dataset_mix: DatasetMixConfig = field(default_factory=DatasetMixConfig)
    
    def __post_init__(self) -> None:
        if not self.dataset:
            raise ValueError("Dataset must be specified")
        
        # Validate batch sizes
        if not is_power_of_2(self.per_device_train_batch_size):
            suggested = next_power_of_2(self.per_device_train_batch_size)
            print(f"⚠️  per_device_train_batch_size ({self.per_device_train_batch_size}) "
                  f"is not power of 2. Suggested: {suggested}")
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> TrainConfig:
        """Load configuration from YAML file."""
        path = Path(path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TrainConfig:
        """Create config from dictionary."""
        # Extract nested configs
        lora_data = data.pop("lora", {})
        mix_data = data.pop("dataset_mix", {})
        
        # Handle legacy format
        if "finetuning_type" in data and data["finetuning_type"] == "lora":
            lora_data["enabled"] = True
            lora_data["rank"] = data.pop("lora_rank", 64)
            lora_data["alpha"] = data.pop("lora_alpha", 128)
            lora_data["dropout"] = data.pop("lora_dropout", 0.05)
            lora_data["target"] = data.pop("lora_target", "all")
            data.pop("finetuning_type", None)
        
        if "mix_strategy" in data:
            mix_data["strategy"] = data.pop("mix_strategy")
        if "interleave_probs" in data:
            mix_data["probs"] = data.pop("interleave_probs")
        
        # Map stage names
        stage_map = {"pt": "pt", "cpt": "pt", "sft": "sft", "dpo": "dpo"}
        if "stage" in data:
            data["stage"] = stage_map.get(data["stage"], data["stage"])
        
        # Filter only known fields
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in known_fields}
        
        return cls(
            **filtered_data,
            lora=LoraConfig(**lora_data) if lora_data else LoraConfig(),
            dataset_mix=DatasetMixConfig(**mix_data) if mix_data else DatasetMixConfig(),
        )
    
    def to_llamafactory_config(self, gpus: str = "0") -> Dict[str, Any]:
        """Convert to LlamaFactory YAML format."""
        num_gpus = len(gpus.split(","))
        
        # Determine packing: explicit setting or auto based on stage
        packing = self.packing if self.packing is not None else (self.stage == "pt")
        
        config = {
            # Model
            "model_name_or_path": self.model_name_or_path,
            "template": self.template,
            "trust_remote_code": self.trust_remote_code,
            
            # Stage
            "stage": self.stage,
            "train_on_prompt": self.stage == "dpo",
            "packing": packing,
            
            # Dataset
            "dataset": self.dataset,
            "cutoff_len": self.cutoff_len,
            
            # Training
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "lr_scheduler_type": self.lr_scheduler_type,
            "warmup_ratio": self.warmup_ratio,
            "num_train_epochs": self.num_train_epochs,
            
            # Output
            "output_dir": self.output_dir,
            "run_name": self.run_name,
            
            # Precision
            "bf16": self.bf16,
            "tf32": self.tf32,
            "gradient_checkpointing": self.gradient_checkpointing,
            
            # Logging
            "plot_loss": True,
            "eval_strategy": "steps",
            "save_strategy": "steps",
            "logging_steps": self.logging_steps,
            "eval_steps": self.eval_steps,
            "save_steps": self.save_steps,
            "save_total_limit": self.save_total_limit,
            "logging_first_step": True,
            "logging_nan_inf_filter": False,
            
            # Model selection
            "load_best_model_at_end": True,
            "metric_for_best_model": "loss" if self.stage == "pt" else "eval_loss",
            "greater_is_better": False,
            
            # Control
            "do_train": self.do_train,
            "do_eval": self.do_eval,
            "overwrite_output_dir": True,
            
            # Other
            "dataloader_num_workers": 4,
            "dataloader_pin_memory": True,
            "preprocessing_num_workers": 4,
            
            # Reporting
            "report_to": self.report_to,
        }
        
        # Optional fields
        if self.eval_dataset:
            config["eval_dataset"] = self.eval_dataset
        if self.streaming:
            config["streaming"] = True
        if self.resume_from_checkpoint:
            config["resume_from_checkpoint"] = self.resume_from_checkpoint
        if self.enable_kd:
            config["enable_kd"] = True
        
        # DPO-specific
        if self.stage == "dpo":
            config["dpo_beta"] = self.dpo_beta
            config["dpo_loss"] = self.dpo_loss
            config["dpo_ftx"] = 0.0
            config["remove_unused_columns"] = False
        
        # SFT-specific
        if self.stage == "sft":
            config["remove_unused_columns"] = False
        
        # LoRA config
        config.update(self.lora.to_dict())
        
        # Dataset mixing
        config.update(self.dataset_mix.to_dict())
        
        # LR scheduler kwargs
        if self.lr_scheduler_type == "cosine_with_min_lr":
            config["lr_scheduler_kwargs"] = {"min_lr_rate": 0.1}
        
        return config
    
    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        config = self.to_llamafactory_config()
        with open(path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
