"""LLM-as-Judge Evaluation Module for BYOL Framework."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger("byol-eval")

# Paths relative to this package
_PACKAGE_ROOT = Path(__file__).parent.parent
_EVAL_ROOT = _PACKAGE_ROOT.parent.parent / "eval"
_LLM_JUDGE_PATH = _EVAL_ROOT / "src" / "llm_as_judge"
_LOCAL_CONFIGS = _PACKAGE_ROOT / "configs"


class LLMJudgeRunner:
    """Wrapper for LLM-as-Judge evaluation."""
    
    def __init__(
        self,
        model_config: Optional[str] = None,
        dataset_config: Optional[str] = None,
        output_dir: str = "results/judge",
    ):
        # Default to local configs in eval_v2/configs/
        self.model_config = str(model_config or _LOCAL_CONFIGS / "judge_models.yaml")
        self.dataset_config = str(dataset_config or _LOCAL_CONFIGS / "judge_datasets.yaml")
        self.output_dir = output_dir
        
        if not Path(self.model_config).exists():
            raise FileNotFoundError(f"Model config not found: {self.model_config}")
        if not Path(self.dataset_config).exists():
            raise FileNotFoundError(f"Dataset config not found: {self.dataset_config}")
    
    def run(self) -> None:
        """Run the LLM-as-Judge evaluation."""
        # Add judge module to path
        judge_path = str(_LLM_JUDGE_PATH)
        if judge_path not in sys.path:
            sys.path.insert(0, judge_path)
        
        from run_llm_judge import run_evaluation
        
        logger.info("=" * 60)
        logger.info("LLM-AS-JUDGE EVALUATION")
        logger.info(f"Model config:   {self.model_config}")
        logger.info(f"Dataset config: {self.dataset_config}")
        logger.info("=" * 60)
        
        os.makedirs(self.output_dir, exist_ok=True)
        run_evaluation(self.model_config, self.dataset_config, self.output_dir)
