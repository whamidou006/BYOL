"""CLI for BYOL Training Framework."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .config import TrainConfig
from .runner import TrainingRunner
from .merge import merge_lora

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("byol-train")


def parse_overrides(override_args: List[str]) -> Dict:
    """Parse override arguments like --model_name=X or --key=value."""
    overrides = {}
    for arg in override_args:
        if arg.startswith("--"):
            arg = arg[2:]
        if "=" in arg:
            key, value = arg.split("=", 1)
            # Try to parse as number or bool
            if value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False
            else:
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
            overrides[key] = value
    return overrides


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog="byol-train",
        description="BYOL Training Framework - LlamaFactory Wrapper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run CPT training
  byol-train cpt --config configs/cpt.yaml --gpus 0,1

  # Run SFT training with LoRA
  byol-train sft --config configs/sft.yaml --lora --lora-rank 64

  # Run DPO training  
  byol-train dpo --config configs/dpo.yaml --gpus 0,1,2,3

  # Merge LoRA adapter
  byol-train merge --base google/gemma-3-4b-pt --adapter outputs/lora/ --output merged/

  # Override config values
  byol-train sft --config configs/sft.yaml --model_name_or_path=google/gemma-2-2b
""",
    )
    
    # Global options
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config without running",
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Training stage")
    
    # CPT (Continual Pre-Training)
    cpt_parser = subparsers.add_parser("cpt", help="Continual pre-training")
    _add_common_args(cpt_parser)
    
    # SFT (Supervised Fine-Tuning)
    sft_parser = subparsers.add_parser("sft", help="Supervised fine-tuning")
    _add_common_args(sft_parser)
    
    # DPO (Direct Preference Optimization)
    dpo_parser = subparsers.add_parser("dpo", help="Direct preference optimization")
    _add_common_args(dpo_parser)
    dpo_parser.add_argument(
        "--dpo-beta",
        type=float,
        default=0.1,
        help="DPO beta parameter (default: 0.1)",
    )
    dpo_parser.add_argument(
        "--dpo-loss",
        choices=["sigmoid", "hinge", "ipo"],
        default="sigmoid",
        help="DPO loss function (default: sigmoid)",
    )
    
    # Merge (LoRA merging)
    merge_parser = subparsers.add_parser("merge", help="Merge LoRA adapter into base model")
    merge_parser.add_argument(
        "--base", "-b",
        type=str,
        required=True,
        help="Path to base model",
    )
    merge_parser.add_argument(
        "--adapter", "-a",
        type=str,
        required=True,
        help="Path to LoRA adapter",
    )
    merge_parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output directory for merged model",
    )
    merge_parser.add_argument(
        "--template",
        type=str,
        default="gemma",
        help="Chat template (default: gemma)",
    )
    merge_parser.add_argument(
        "--export-size",
        type=int,
        default=2,
        help="Number of shards (default: 2)",
    )
    
    return parser


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments to a subcommand parser."""
    # Config file
    parser.add_argument(
        "-c", "--config",
        type=Path,
        required=True,
        help="Path to training config YAML file",
    )
    
    # GPU selection
    parser.add_argument(
        "-g", "--gpus",
        type=str,
        default="0",
        help="Comma-separated GPU IDs (default: 0)",
    )
    
    # LoRA options
    parser.add_argument(
        "--lora",
        action="store_true",
        help="Enable LoRA training",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=64,
        help="LoRA rank (default: 64)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        help="LoRA alpha (default: 2*rank)",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout (default: 0.05)",
    )
    
    # Model/dataset overrides
    parser.add_argument(
        "--model", "--model_name_or_path",
        type=str,
        dest="model_name_or_path",
        help="Override model path",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Override dataset name",
    )
    parser.add_argument(
        "--eval-dataset",
        type=str,
        dest="eval_dataset",
        help="Override eval dataset name",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Override output directory",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        help="Override W&B run name",
    )
    
    # Training options
    parser.add_argument(
        "--epochs",
        type=float,
        dest="num_train_epochs",
        help="Override number of epochs",
    )
    parser.add_argument(
        "--lr", "--learning-rate",
        type=float,
        dest="learning_rate",
        help="Override learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        dest="per_device_train_batch_size",
        help="Override per-device batch size",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        dest="gradient_accumulation_steps",
        help="Override gradient accumulation steps",
    )
    parser.add_argument(
        "--auto-batch",
        action="store_true",
        help="Auto-compute gradient accumulation from batch_size in config",
    )
    
    # Resume training
    parser.add_argument(
        "--resume",
        type=Path,
        dest="resume_from_checkpoint",
        help="Resume from checkpoint directory",
    )
    
    # Packing
    parser.add_argument(
        "--packing", "--no-packing",
        dest="packing",
        action=argparse.BooleanOptionalAction,
        help="Enable/disable sequence packing",
    )
    
    # Streaming
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Enable dataset streaming",
    )
    
    # Knowledge distillation
    parser.add_argument(
        "--enable-kd",
        action="store_true",
        help="Enable knowledge distillation",
    )
    
    # Dataset mixing
    parser.add_argument(
        "--mix-strategy",
        choices=["concat", "interleave_under", "interleave_over"],
        help="Dataset mixing strategy",
    )
    parser.add_argument(
        "--mix-probs",
        type=str,
        help="Mixing probabilities (comma-separated, e.g., '0.6,0.4')",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility",
    )


def run_training(args: argparse.Namespace, extra_overrides: List[str]) -> int:
    """Run training with given arguments."""
    # Load base config
    if not args.config.exists():
        logger.error(f"Config file not found: {args.config}")
        return 1
    
    config = TrainConfig.from_yaml(args.config)
    
    # Set stage from command
    stage_map = {"cpt": "pt", "sft": "sft", "dpo": "dpo"}
    config.stage = stage_map.get(args.command, args.command)
    
    # Apply CLI overrides
    if args.model_name_or_path:
        config.model_name_or_path = args.model_name_or_path
    if args.dataset:
        config.dataset = args.dataset
    if hasattr(args, "eval_dataset") and args.eval_dataset:
        config.eval_dataset = args.eval_dataset
    if hasattr(args, "output_dir") and args.output_dir:
        config.output_dir = args.output_dir
    if hasattr(args, "run_name") and args.run_name:
        config.run_name = args.run_name
    if hasattr(args, "num_train_epochs") and args.num_train_epochs:
        config.num_train_epochs = args.num_train_epochs
    if hasattr(args, "learning_rate") and args.learning_rate:
        config.learning_rate = args.learning_rate
    if hasattr(args, "per_device_train_batch_size") and args.per_device_train_batch_size:
        config.per_device_train_batch_size = args.per_device_train_batch_size
    if hasattr(args, "gradient_accumulation_steps") and args.gradient_accumulation_steps:
        config.gradient_accumulation_steps = args.gradient_accumulation_steps
    if hasattr(args, "resume_from_checkpoint") and args.resume_from_checkpoint:
        config.resume_from_checkpoint = str(args.resume_from_checkpoint)
    if hasattr(args, "packing") and args.packing is not None:
        config.packing = args.packing
    if hasattr(args, "streaming") and args.streaming:
        config.streaming = True
    if hasattr(args, "enable_kd") and args.enable_kd:
        config.enable_kd = True
    
    # Dataset mixing
    if hasattr(args, "mix_strategy") and args.mix_strategy:
        config.dataset_mix.strategy = args.mix_strategy
    if hasattr(args, "mix_probs") and args.mix_probs:
        config.dataset_mix.probs = [float(p) for p in args.mix_probs.split(",")]
    if hasattr(args, "seed") and args.seed:
        config.dataset_mix.seed = args.seed
    
    # LoRA settings
    if args.lora:
        config.lora.enabled = True
        config.lora.rank = args.lora_rank
        config.lora.alpha = args.lora_alpha or (2 * args.lora_rank)
        config.lora.dropout = args.lora_dropout
    
    # DPO-specific
    if args.command == "dpo":
        config.dpo_beta = getattr(args, "dpo_beta", 0.1)
        config.dpo_loss = getattr(args, "dpo_loss", "sigmoid")
    
    # Parse extra overrides (--key=value format)
    overrides = parse_overrides(extra_overrides)
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
            logger.info(f"Override: {key}={value}")
    
    # Create runner
    runner = TrainingRunner(
        config=config,
        gpus=args.gpus,
        dry_run=args.dry_run,
        auto_compute_batch=getattr(args, "auto_batch", False),
    )
    
    # Run training
    result = runner.run()
    
    if result.status == "success":
        logger.info("🎉 Training completed successfully!")
        return 0
    elif result.status == "skipped":
        logger.info("[DRY RUN] Would run training")
        return 0
    else:
        logger.error(f"Training failed: {result.error}")
        return 1


def run_merge(args: argparse.Namespace) -> int:
    """Run LoRA merging."""
    success = merge_lora(
        base_model=args.base,
        adapter_path=args.adapter,
        output_dir=args.output,
        template=args.template,
        export_size=args.export_size,
        dry_run=args.dry_run,
    )
    return 0 if success else 1


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args, extra = parser.parse_known_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if not args.command:
        parser.print_help()
        return 0
    
    if args.command == "merge":
        return run_merge(args)
    
    return run_training(args, extra)


if __name__ == "__main__":
    sys.exit(main())
