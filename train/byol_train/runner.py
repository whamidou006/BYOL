"""Training Runner for BYOL Framework."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .config import TrainConfig, is_power_of_2, next_power_of_2

logger = logging.getLogger("byol-train")


@dataclass
class TrainResult:
    """Result of a training run."""
    status: str  # "success", "failed"
    output_dir: str
    run_name: str
    duration_seconds: float = 0.0
    error: Optional[str] = None


class TrainingRunner:
    """Main training runner using LlamaFactory."""
    
    def __init__(
        self,
        config: TrainConfig,
        gpus: str = "0",
        dry_run: bool = False,
        auto_compute_batch: bool = False,
    ):
        self.config = config
        self.gpus = gpus
        self.dry_run = dry_run
        self.auto_compute_batch = auto_compute_batch
        self.num_gpus = len(gpus.split(","))
        
        # Environment setup
        self._setup_environment()
    
    def _setup_environment(self) -> None:
        """Setup environment variables."""
        if "HF_TOKEN" not in os.environ:
            logger.warning("HF_TOKEN not set in environment")
        os.environ["TRUST_REMOTE_CODE"] = "True"
    
    def _generate_output_dir(self) -> str:
        """Generate unique output directory name."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Build suffix
        parts = [self.config.stage]
        
        # Add dataset identifier
        if self.config.dataset:
            dataset_short = self.config.dataset.split(",")[0]
            dataset_short = dataset_short.replace("fineweb2_dataset_", "").replace("_train", "")
            parts.append(dataset_short)
        
        parts.append(timestamp)
        
        # Add LoRA suffix
        if self.config.lora.enabled:
            parts.append(f"lora-{self.config.lora.rank}")
        
        return f"outputs/{'-'.join(parts)}"
    
    def _generate_run_name(self) -> str:
        """Generate W&B run name."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        parts = [self.config.stage, "training", timestamp]
        if self.config.lora.enabled:
            parts.append(f"lora-{self.config.lora.rank}")
        return "-".join(parts)
    
    def _compute_gradient_accumulation(self, target_batch_size: int) -> int:
        """Compute gradient accumulation steps for target batch size."""
        per_device = self.config.per_device_train_batch_size
        
        if target_batch_size <= per_device * self.num_gpus:
            return 1
        
        grad_accum = target_batch_size // (per_device * self.num_gpus)
        
        # Warn if not exact
        effective = grad_accum * per_device * self.num_gpus
        if effective != target_batch_size:
            logger.warning(
                f"Target batch size {target_batch_size} not achievable. "
                f"Effective: {effective}"
            )
        
        # Warn if not power of 2
        if not is_power_of_2(grad_accum):
            logger.warning(
                f"gradient_accumulation_steps ({grad_accum}) is not power of 2. "
                f"Suggested: {next_power_of_2(grad_accum)}"
            )
        
        return grad_accum
    
    def _create_temp_config(self) -> Path:
        """Create temporary config file for LlamaFactory."""
        # Generate output dir and run name if not set
        if not self.config.output_dir or self.config.output_dir == "outputs":
            self.config.output_dir = self._generate_output_dir()
        if not self.config.run_name:
            self.config.run_name = self._generate_run_name()
        
        # Get LlamaFactory config
        lf_config = self.config.to_llamafactory_config(self.gpus)
        
        # Auto-compute gradient accumulation if requested
        if self.auto_compute_batch:
            # Try to get target batch size from config
            target = lf_config.get("batch_size", 256)
            grad_accum = self._compute_gradient_accumulation(target)
            lf_config["gradient_accumulation_steps"] = grad_accum
            logger.info(f"Auto-computed gradient_accumulation_steps: {grad_accum}")
        
        # Write to temp file
        fd, temp_path = tempfile.mkstemp(suffix=".yaml", prefix="train_")
        with os.fdopen(fd, "w") as f:
            yaml.dump(lf_config, f, default_flow_style=False, sort_keys=False)
        
        return Path(temp_path)
    
    def _save_training_summary(self, result: TrainResult) -> None:
        """Save training summary to output directory."""
        output_dir = Path(result.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        summary = f"""Training Configuration Summary
=============================
Timestamp: {datetime.now().isoformat()}
Task: {self.config.stage.upper()}
Model: {self.config.model_name_or_path}
Dataset: {self.config.dataset}
GPUs: {self.gpus} ({self.num_gpus} GPU(s))
LoRA: {self.config.lora.enabled}
{"LoRA Rank: " + str(self.config.lora.rank) if self.config.lora.enabled else ""}
Output: {result.output_dir}
Run Name: {result.run_name}

Key Hyperparameters:
  per_device_train_batch_size: {self.config.per_device_train_batch_size}
  gradient_accumulation_steps: {self.config.gradient_accumulation_steps}
  learning_rate: {self.config.learning_rate}
  num_train_epochs: {self.config.num_train_epochs}

Training Status: {"✅ COMPLETED" if result.status == "success" else f"❌ FAILED ({result.error})"}
Duration: {result.duration_seconds:.1f}s
"""
        
        (output_dir / "training_summary.txt").write_text(summary)
        logger.info(f"📊 Training summary saved to: {output_dir}/training_summary.txt")
    
    def run(self) -> TrainResult:
        """Run training."""
        start_time = datetime.now()
        
        # Create temp config
        temp_config = self._create_temp_config()
        
        logger.info("=" * 60)
        logger.info(f"🚀 Starting {self.config.stage.upper()} training")
        logger.info(f"📂 Output: {self.config.output_dir}")
        logger.info(f"🖥️  GPUs: {self.gpus}")
        if self.config.lora.enabled:
            logger.info(f"🔧 LoRA: rank={self.config.lora.rank}")
        logger.info("=" * 60)
        
        if self.dry_run:
            logger.info("[DRY RUN] Config file:")
            logger.info(temp_config.read_text())
            return TrainResult(
                status="skipped",
                output_dir=self.config.output_dir,
                run_name=self.config.run_name,
            )
        
        # Run LlamaFactory
        cmd = ["llamafactory-cli", "train", str(temp_config)]
        logger.info(f"Command: {' '.join(cmd)}")
        
        try:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = self.gpus
            
            result = subprocess.run(cmd, env=env, capture_output=False)
            duration = (datetime.now() - start_time).total_seconds()
            
            if result.returncode == 0:
                logger.info(f"✅ Training completed in {duration:.1f}s")
                train_result = TrainResult(
                    status="success",
                    output_dir=self.config.output_dir,
                    run_name=self.config.run_name,
                    duration_seconds=duration,
                )
            else:
                logger.error(f"❌ Training failed with exit code {result.returncode}")
                train_result = TrainResult(
                    status="failed",
                    output_dir=self.config.output_dir,
                    run_name=self.config.run_name,
                    duration_seconds=duration,
                    error=f"Exit code: {result.returncode}",
                )
        
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.exception(f"Error: {e}")
            train_result = TrainResult(
                status="failed",
                output_dir=self.config.output_dir,
                run_name=self.config.run_name,
                duration_seconds=duration,
                error=str(e),
            )
        
        finally:
            # Save config to output dir
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy temp config
            import shutil
            shutil.copy(temp_config, output_dir / "config.yaml")
            logger.info(f"📝 Config saved to: {output_dir}/config.yaml")
            
            # Cleanup temp
            temp_config.unlink(missing_ok=True)
        
        # Save summary
        self._save_training_summary(train_result)
        
        return train_result
