"""Command-line interface for BYOL Evaluation Framework.

Provides a clean CLI for running model evaluations using lm-evaluation-harness,
LLM-as-Judge frameworks, and benchmark result extraction.

Subcommands:
    (default)  Run lm-evaluation-harness benchmarks
    judge      Run LLM-as-Judge evaluation
    extract    Extract benchmark results from log files
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from .config import EvalConfig, ModelConfig, TaskConfig
from .constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DTYPE,
    DEFAULT_GPUS,
    DEFAULT_JUDGE_OUTPUT_DIR,
    DEFAULT_OUTPUT_DIR,
    VALID_DTYPES,
)
from .runner import EvaluationRunner
from .extract import EvalMode, Language

# Configure logging once at CLI entry
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("byol-eval")


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.
    
    Args:
        args: Command-line arguments. Uses sys.argv if None.
        
    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        prog="byol-eval",
        description="BYOL Model Evaluation Framework - Evaluate LLMs using lm-eval or LLM-as-Judge",
    )
    subparsers = parser.add_subparsers(dest="mode", help="Evaluation mode")
    
    # Judge subcommand
    judge_parser = subparsers.add_parser("judge", help="Run LLM-as-Judge evaluation")
    judge_parser.add_argument(
        "--model-config", "-m",
        type=str,
        help="Path to model configuration YAML file",
    )
    judge_parser.add_argument(
        "--dataset-config", "-d",
        type=str,
        help="Path to dataset configuration YAML file",
    )
    judge_parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=DEFAULT_JUDGE_OUTPUT_DIR,
        help=f"Output directory for results (default: {DEFAULT_JUDGE_OUTPUT_DIR})",
    )
    
    # Extract subcommand
    extract_parser = subparsers.add_parser(
        "extract", 
        help="Extract benchmark results from lm-eval log files"
    )
    extract_parser.add_argument(
        "log_file",
        type=Path,
        help="Path to lm-evaluation-harness log file",
    )
    extract_parser.add_argument(
        "--eval-mode",
        type=str,
        choices=[e.value for e in EvalMode],
        default=EvalMode.INSTRUCT.value,
        help="Evaluation mode: 'base' for CPT models, 'instruct' for fine-tuned (default: instruct)",
    )
    extract_parser.add_argument(
        "--lang",
        type=str,
        choices=[e.value for e in Language],
        default=Language.MRI.value,
        help="Target language code (default: mri)",
    )
    extract_parser.add_argument(
        "--csv",
        action="store_true",
        help="Output results in CSV format",
    )
    extract_parser.add_argument(
        "--debug",
        action="store_true",
        help="Show all parsed tasks (useful for debugging)",
    )
    
    # Benchmark mode (default) arguments
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to YAML configuration file (overrides other options)",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        help="Model path or HuggingFace ID to evaluate",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        help="Human-readable model name (default: derived from path)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=DEFAULT_DTYPE,
        choices=list(VALID_DTYPES),
        help=f"Model data type (default: {DEFAULT_DTYPE})",
    )
    parser.add_argument(
        "--tasks", "-t",
        type=str,
        help="Comma-separated task names to evaluate",
    )
    parser.add_argument(
        "--tasks-path",
        type=str,
        help="Path to custom task definitions directory",
    )
    parser.add_argument(
        "--num-fewshot", "-n",
        type=int,
        default=0,
        help="Number of few-shot examples (default: 0)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum samples per task (default: all)",
    )
    parser.add_argument(
        "--gpus", "-g",
        type=str,
        default=DEFAULT_GPUS,
        help=f"Comma-separated GPU device IDs (default: {DEFAULT_GPUS})",
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=str,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size or 'auto:N' for automatic (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for results (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them",
    )
    parser.add_argument(
        "--log-samples",
        action="store_true",
        help="Log individual evaluation samples to output",
    )
    parser.add_argument(
        "--apply-chat-template",
        action="store_true",
        help="Apply chat template for instruct models",
    )
    
    return parser.parse_args(args)


def build_config_from_args(args: argparse.Namespace) -> EvalConfig:
    """Build EvalConfig from CLI arguments.
    
    Args:
        args: Parsed command-line arguments.
        
    Returns:
        Configured EvalConfig instance.
        
    Raises:
        ValueError: If required arguments are missing.
    """
    if args.config:
        config = EvalConfig.from_yaml(args.config)
        # CLI overrides
        if args.gpus:
            config.gpus = args.gpus
        if args.output_dir:
            config.output_dir = args.output_dir
        if args.tasks_path:
            config.tasks_path = args.tasks_path
        if args.log_samples:
            config.log_samples = True
        if hasattr(args, "apply_chat_template") and args.apply_chat_template:
            config.apply_chat_template = True
        # Override model if specified on CLI
        if args.model:
            config.models = [ModelConfig(
                name=args.model_name or Path(args.model).name,
                path=args.model,
                dtype=args.dtype or "bfloat16",
            )]
        return config
    
    if not args.model:
        raise ValueError("Either --config or --model must be specified")
    if not args.tasks:
        raise ValueError("Either --config or --tasks must be specified")
    
    return EvalConfig(
        models=[
            ModelConfig(
                name=args.model_name or Path(args.model).name,
                path=args.model,
                dtype=args.dtype,
            )
        ],
        tasks=[
            TaskConfig(
                name=args.tasks,
                num_fewshot=args.num_fewshot,
                limit=args.limit,
            )
        ],
        output_dir=args.output_dir,
        tasks_path=args.tasks_path,
        gpus=args.gpus,
        batch_size=args.batch_size,
        log_samples=args.log_samples,
        apply_chat_template=getattr(args, "apply_chat_template", False),
    )


def run_judge(args: argparse.Namespace) -> int:
    """Run LLM-as-Judge evaluation.
    
    Args:
        args: Parsed command-line arguments.
        
    Returns:
        Exit code (0 for success, 1 for failure).
    """
    from .judge import LLMJudgeRunner
    
    try:
        runner = LLMJudgeRunner(args.model_config, args.dataset_config, args.output_dir)
        runner.run()
        return 0
    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Judge evaluation failed: {e}")
        return 1


def run_extract(args: argparse.Namespace) -> int:
    """Run benchmark result extraction from log files.
    
    Args:
        args: Parsed command-line arguments.
        
    Returns:
        Exit code (0 for success, 1 for failure).
    """
    from .extract import LogParser, BenchmarkExtractor, OutputFormatter, EvalMode
    
    # Validate file exists
    if not args.log_file.exists():
        logger.error(f"File not found: {args.log_file}")
        return 1
    
    try:
        log_parser = LogParser(args.log_file)
    except Exception as e:
        logger.error(f"Error parsing log file: {e}")
        return 1
    
    # Debug output
    if args.debug:
        OutputFormatter.print_debug(log_parser.metrics)
        print()
    
    # Extract benchmarks based on mode
    extractor = BenchmarkExtractor(log_parser.metrics, args.lang)
    
    if args.eval_mode == EvalMode.BASE.value:
        results = extractor.extract_base()
    else:
        results = extractor.extract_instruct()
    
    # Output results
    if args.csv:
        OutputFormatter.print_csv(results)
    else:
        OutputFormatter.print_table(results, args.lang)
    
    return 0


def main(args: Optional[List[str]] = None) -> int:
    """Main entry point for the CLI.
    
    Args:
        args: Command-line arguments. Uses sys.argv if None.
        
    Returns:
        Exit code (0 for success, 1 for failure).
    """
    try:
        parsed = parse_args(args)
        
        if parsed.mode == "judge":
            return run_judge(parsed)
        
        if parsed.mode == "extract":
            return run_extract(parsed)
        
        config = build_config_from_args(parsed)
        runner = EvaluationRunner(config, dry_run=parsed.dry_run)
        results = runner.run_all()
        runner.print_summary(results)
        return 1 if any(r.status == "failed" for r in results) else 0
        
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
