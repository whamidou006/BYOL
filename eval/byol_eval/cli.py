"""Command-line interface for BYOL Evaluation Framework."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from .config import EvalConfig, ModelConfig, TaskConfig
from .runner import EvaluationRunner

# Configure logging once at CLI entry
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(prog="byol-eval", description="BYOL Model Evaluation Framework")
    subparsers = parser.add_subparsers(dest="mode", help="Evaluation mode")
    
    # Judge subcommand
    judge_parser = subparsers.add_parser("judge", help="Run LLM-as-Judge evaluation")
    judge_parser.add_argument("--model-config", "-m", type=str, help="Model config YAML")
    judge_parser.add_argument("--dataset-config", "-d", type=str, help="Dataset config YAML")
    judge_parser.add_argument("--output-dir", "-o", type=str, default="results/judge", help="Output directory")
    
    # Benchmark mode (default)
    parser.add_argument("--config", "-c", type=str, help="YAML configuration file")
    parser.add_argument("--model", "-m", type=str, help="Model path or HuggingFace ID")
    parser.add_argument("--model-name", type=str, help="Model name (default: derived from path)")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32", "auto"])
    parser.add_argument("--tasks", "-t", type=str, help="Comma-separated task names")
    parser.add_argument("--tasks-path", type=str, help="Custom task definitions path")
    parser.add_argument("--num-fewshot", "-n", type=int, default=0, help="Few-shot examples")
    parser.add_argument("--limit", type=int, help="Max samples per task")
    parser.add_argument("--gpus", "-g", type=str, default="0", help="GPU device IDs")
    parser.add_argument("--batch-size", "-b", type=str, default="auto:4", help="Batch size")
    parser.add_argument("--output-dir", "-o", type=str, default="results", help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--log-samples", action="store_true", help="Log evaluation samples")
    
    return parser.parse_args(args)


def build_config_from_args(args: argparse.Namespace) -> EvalConfig:
    """Build EvalConfig from CLI arguments."""
    if args.config:
        config = EvalConfig.from_yaml(args.config)
        if args.gpus: config.gpus = args.gpus
        if args.output_dir: config.output_dir = args.output_dir
        if args.tasks_path: config.tasks_path = args.tasks_path
        if args.log_samples: config.log_samples = True
        return config
    
    if not args.model:
        raise ValueError("Either --config or --model must be specified")
    if not args.tasks:
        raise ValueError("Either --config or --tasks must be specified")
    
    return EvalConfig(
        models=[ModelConfig(name=args.model_name or Path(args.model).name, path=args.model, dtype=args.dtype)],
        tasks=[TaskConfig(name=args.tasks, num_fewshot=args.num_fewshot, limit=args.limit)],
        output_dir=args.output_dir,
        tasks_path=args.tasks_path,
        gpus=args.gpus,
        batch_size=args.batch_size,
        log_samples=args.log_samples,
    )


def run_judge(args: argparse.Namespace) -> int:
    """Run LLM-as-Judge evaluation."""
    from .judge import LLMJudgeRunner
    runner = LLMJudgeRunner(args.model_config, args.dataset_config, args.output_dir)
    runner.run()
    return 0


def main(args: Optional[List[str]] = None) -> int:
    """Main entry point."""
    try:
        parsed = parse_args(args)
        
        if parsed.mode == "judge":
            return run_judge(parsed)
        
        config = build_config_from_args(parsed)
        runner = EvaluationRunner(config, dry_run=parsed.dry_run)
        results = runner.run_all()
        runner.print_summary(results)
        return 1 if any(r.status == "failed" for r in results) else 0
        
    except Exception as e:
        logging.error(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
