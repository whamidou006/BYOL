"""LLM-as-Judge Evaluation Module for BYOL Framework."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger("byol-eval")

# Paths relative to this package
_PACKAGE_ROOT = Path(__file__).parent.parent  # eval/
_BYOL_ROOT = _PACKAGE_ROOT.parent             # BYOL/
_EVAL_OLD = _BYOL_ROOT / "eval_old"           # Original eval code
_LLM_JUDGE_PATH = _EVAL_OLD / "src" / "llm_as_judge"
_API_PATH = _BYOL_ROOT / "api"                # BYOL/api/ (where get_azure_api.py lives)
_API_EXAMPLE_PATH = _BYOL_ROOT / "api.example"
_LOCAL_CONFIGS = _PACKAGE_ROOT / "configs"


class LLMJudgeRunner:
    """Wrapper for LLM-as-Judge evaluation.
    
    Requires:
        - BYOL/api/ folder with get_azure_api.py (copy from api.example/)
        - BYOL/eval_old/ folder with original evaluation code
        
    Setup:
        cp -r api.example api
        # Configure api/get_azure_api.py with your LLM provider credentials
    """
    
    def __init__(
        self,
        model_config: Optional[str] = None,
        dataset_config: Optional[str] = None,
        output_dir: str = "results/judge",
    ):
        """Initialize LLM-as-Judge runner.
        
        Args:
            model_config: Path to model configuration YAML.
            dataset_config: Path to dataset configuration YAML.
            output_dir: Directory for evaluation results.
            
        Raises:
            FileNotFoundError: If required files/folders are missing.
        """
        # Default to local configs in eval/configs/
        self.model_config = str(model_config or _LOCAL_CONFIGS / "judge_models.yaml")
        self.dataset_config = str(dataset_config or _LOCAL_CONFIGS / "judge_datasets.yaml")
        self.output_dir = output_dir
        
        # Validate config files
        if not Path(self.model_config).exists():
            raise FileNotFoundError(f"Model config not found: {self.model_config}")
        if not Path(self.dataset_config).exists():
            raise FileNotFoundError(f"Dataset config not found: {self.dataset_config}")
        
        # Validate LLM Judge module
        if not _LLM_JUDGE_PATH.exists():
            raise FileNotFoundError(
                f"LLM Judge module not found: {_LLM_JUDGE_PATH}\n"
                "Make sure eval_old/ directory exists with the original evaluation code."
            )
        
        # Validate API module with helpful error message
        if not _API_PATH.exists():
            setup_instructions = (
                f"API module not found: {_API_PATH}\n\n"
                "To setup the API module:\n"
                "  1. Copy the example: cp -r api.example api\n"
                "  2. Configure credentials in api/get_azure_api.py\n"
                "  3. Set environment variables:\n"
                "     - AZURE_OPENAI_ENDPOINT\n"
                "     - AZURE_OPENAI_API_KEY\n"
                "     - Or OPENAI_API_KEY for direct OpenAI\n\n"
                f"See {_API_EXAMPLE_PATH / 'README.md'} for details."
            )
            raise FileNotFoundError(setup_instructions)
    
    def run(self) -> None:
        """Run the LLM-as-Judge evaluation."""
        # Add required paths for imports
        # API path must be added BEFORE llm_judge to override its relative import
        for path in [str(_API_PATH), str(_LLM_JUDGE_PATH)]:
            if path not in sys.path:
                sys.path.insert(0, path)
        
        from run_llm_judge import run_evaluation
        
        logger.info("=" * 60)
        logger.info("LLM-AS-JUDGE EVALUATION")
        logger.info(f"  Model config:   {self.model_config}")
        logger.info(f"  Dataset config: {self.dataset_config}")
        logger.info(f"  Output dir:     {self.output_dir}")
        logger.info("=" * 60)
        
        os.makedirs(self.output_dir, exist_ok=True)
        run_evaluation(self.model_config, self.dataset_config, self.output_dir)
