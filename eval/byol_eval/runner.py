"""Evaluation Runner for BYOL Framework.

This module provides the main evaluation orchestration using lm-evaluation-harness.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .config import EvalConfig, ModelConfig, TaskConfig
from .constants import STATUS_FAILED, STATUS_ICONS, STATUS_SKIPPED, STATUS_SUCCESS
from .secrets import setup_hf_environment

logger = logging.getLogger("byol-eval")


@dataclass
class EvalResult:
    """Result of a single evaluation run.
    
    Attributes:
        model: Model name that was evaluated.
        task: Task name that was run.
        status: Evaluation status (success, failed, skipped).
        output_dir: Directory where results were saved.
        error: Error message if evaluation failed.
        duration_seconds: Time taken for evaluation.
    """
    model: str
    task: str
    status: str  # "success", "failed", "skipped"
    output_dir: Optional[str] = None
    error: Optional[str] = None
    duration_seconds: float = 0.0


class EvaluationRunner:
    """Main evaluation runner using lm-evaluation-harness.
    
    Orchestrates model evaluation across multiple tasks using the
    lm-evaluation-harness framework.
    
    Attributes:
        config: Evaluation configuration.
        dry_run: If True, print commands without executing.
    """
    
    def __init__(self, config: EvalConfig, dry_run: bool = False) -> None:
        """Initialize the evaluation runner.
        
        Args:
            config: Evaluation configuration.
            dry_run: If True, print commands without executing.
        """
        self.config = config
        self.dry_run = dry_run
        setup_hf_environment(config.hf_token)
    
    def _build_command(self, model: ModelConfig, task: TaskConfig) -> List[str]:
        """Build the lm_eval command.
        
        Matches original behavior from eval/src/benchmark_evaluation/backends/hf_backend.py
        
        Args:
            model: Model configuration.
            task: Task configuration.
            
        Returns:
            Command as list of strings.
        """
        output_dir = self._get_output_dir(model, task)
        
        # Build model_args matching original format
        model_args = [
            f"pretrained={model.path}",
            f"dtype={model.dtype}",
            f"trust_remote_code={str(model.trust_remote_code).lower()}",
        ]
        
        cmd = [
            sys.executable, "-m", "lm_eval",
            "--model", "hf",
            "--model_args", ",".join(model_args),
            "--tasks", task.name,
            "--output_path", output_dir,
        ]
        
        # Batch size: task-level overrides global
        batch_size = task.batch_size or self.config.batch_size
        cmd.extend(["--batch_size", str(batch_size)])
        
        # num_fewshot: only add if explicitly set (None means use task default)
        if task.num_fewshot is not None:
            cmd.extend(["--num_fewshot", str(task.num_fewshot)])
        
        # Custom tasks path
        if self.config.tasks_path:
            cmd.extend(["--include_path", self.config.tasks_path])
        
        # Limit samples
        if task.limit is not None:
            cmd.extend(["--limit", str(task.limit)])
        
        # Log samples
        if self.config.log_samples:
            cmd.append("--log_samples")
        
        # CRITICAL: apply_chat_template - task-level or global
        # This is required for instruct model evaluation
        if task.apply_chat_template or self.config.apply_chat_template:
            cmd.append("--apply_chat_template")
        
        return cmd
    
    def _get_output_dir(self, model: ModelConfig, task: TaskConfig) -> str:
        """Generate output directory path.
        
        Args:
            model: Model configuration.
            task: Task configuration.
            
        Returns:
            Path to output directory.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model = model.name.replace("/", "_").replace("-", "_")
        safe_task = task.name.replace(",", "_").replace(" ", "_").replace("/", "_")
        output_dir = Path(self.config.output_dir) / f"{safe_model}_{safe_task}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        return str(output_dir)
    
    def run_single(self, model: ModelConfig, task: TaskConfig) -> EvalResult:
        """Run evaluation for a single model/task pair.
        
        Args:
            model: Model to evaluate.
            task: Task to run.
            
        Returns:
            EvalResult with status and metadata.
        """
        start_time = datetime.now()
        
        # Determine effective chat template setting
        use_chat_template = task.apply_chat_template or self.config.apply_chat_template
        
        logger.info("=" * 60)
        logger.info(f"Model: {model.name} ({model.path})")
        logger.info(f"Task: {task.name}")
        logger.info(f"Few-shot: {task.num_fewshot if task.num_fewshot is not None else 'default'}")
        logger.info(f"Chat template: {use_chat_template}")
        logger.info("=" * 60)
        
        cmd = self._build_command(model, task)
        output_dir = cmd[cmd.index("--output_path") + 1]
        
        logger.info(f"Command: {' '.join(cmd)}")
        
        if self.dry_run:
            logger.info("[DRY RUN] Skipping execution")
            return EvalResult(model.name, task.name, STATUS_SKIPPED, output_dir)
        
        try:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = self.config.gpus
            result = subprocess.run(cmd, env=env, capture_output=False, check=False)
            duration = (datetime.now() - start_time).total_seconds()
            
            if result.returncode == 0:
                logger.info(f"✅ Success: {model.name} on {task.name}")
                return EvalResult(model.name, task.name, STATUS_SUCCESS, output_dir, duration_seconds=duration)
            else:
                logger.error(f"❌ Failed: {model.name} on {task.name}")
                return EvalResult(model.name, task.name, STATUS_FAILED, output_dir, f"Exit code: {result.returncode}", duration)
        
        except subprocess.SubprocessError as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(f"Subprocess error: {e}")
            return EvalResult(model.name, task.name, STATUS_FAILED, error=str(e), duration_seconds=duration)
        except FileNotFoundError as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(f"Command not found: {e}")
            return EvalResult(model.name, task.name, STATUS_FAILED, error=str(e), duration_seconds=duration)
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.exception(f"Unexpected error: {e}")
            return EvalResult(model.name, task.name, STATUS_FAILED, error=str(e), duration_seconds=duration)
    
    def run_all(self) -> List[EvalResult]:
        """Run all evaluations.
        
        Returns:
            List of EvalResult objects for all model/task combinations.
        """
        total = len(self.config.models) * len(self.config.tasks)
        logger.info(f"🚀 Starting {total} evaluations ({len(self.config.models)} models × {len(self.config.tasks)} tasks)")
        logger.info(f"📂 Output: {self.config.output_dir}")
        logger.info(f"🖥️  GPUs: {self.config.gpus}")
        if self.config.apply_chat_template:
            logger.info("💬 Global chat template: enabled")
        
        results = []
        for model in self.config.models:
            for task in self.config.tasks:
                results.append(self.run_single(model, task))
        return results
    
    @staticmethod
    def print_summary(results: List[EvalResult]) -> None:
        """Print evaluation summary.
        
        Args:
            results: List of evaluation results.
        """
        logger.info("\n" + "=" * 60)
        logger.info("📊 SUMMARY")
        logger.info("=" * 60)
        
        for r in results:
            icon = STATUS_ICONS.get(r.status, "?")
            logger.info(f"  {icon} {r.model} - {r.task}")
        
        successful = sum(1 for r in results if r.status == STATUS_SUCCESS)
        total_time = sum(r.duration_seconds for r in results)
        logger.info(f"\nTotal: {successful}/{len(results)} successful | Time: {total_time:.1f}s")
