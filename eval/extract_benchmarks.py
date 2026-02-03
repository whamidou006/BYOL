#!/usr/bin/env python3
"""Extract benchmark results from lm-evaluation-harness log files.

This module parses evaluation log files and extracts benchmark metrics for
both base (CPT) and instruct model evaluations. It supports multiple output
formats and handles language-specific task name variations.

Supported benchmarks:
    - General & STEM: Global MMLU-Lite, ARC, MGSM, BBH, GPQA
    - Commonsense: XCOPA, XStoryCloze, PIQA, HellaSwag
    - NLI & Reading: XNLI, XWinograd, Belebele
    - Translation: FLORES (BLEU and chrF metrics)
    - Instruct-only: IFEval, TruthfulQA, HumanEval

Example usage:
    # Extract base model results for Chichewa
    python extract_benchmarks.py results/model_base.txt --mode base --lang nya

    # Extract instruct model results as CSV
    python extract_benchmarks.py results/model_instruct.txt --mode instruct --lang mri --csv

    # Debug mode to see all parsed tasks
    python extract_benchmarks.py results/log.txt --debug
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

__version__ = "1.1.0"


# =============================================================================
# Constants
# =============================================================================

class EvalMode(str, Enum):
    """Evaluation mode for model type."""
    BASE = "base"
    INSTRUCT = "instruct"


class Language(str, Enum):
    """Supported target languages."""
    MRI = "mri"  # Māori
    NYA = "nya"  # Chichewa/Nyanja


# Language code aliases - order matters for matching priority
LANGUAGE_ALIASES: Dict[str, List[str]] = {
    "nya": ["nya", "ny", "nyanja"],
    "mri": ["mri", "maori"],
}

# Benchmark categories
CATEGORY_GENERAL = "General & STEM"
CATEGORY_COMMONSENSE = "Commonsense"
CATEGORY_NLI_READING = "NLI & Reading"
CATEGORY_READING_QA = "Reading & QA"
CATEGORY_TRANSLATION = "Translation"
CATEGORY_INSTRUCTION = "Instruction Following"
CATEGORY_TRUTHFULNESS = "Truthfulness"
CATEGORY_CODE = "Code"

# Metrics
METRIC_ACC = "acc"
METRIC_ACC_NORM = "acc_norm"
METRIC_EXACT_MATCH = "exact_match"
METRIC_BLEU = "bleu"
METRIC_CHRF = "chrf"
METRIC_BLEU_ACC = "bleu_acc"
METRIC_PASS_AT_1 = "pass@1"
METRIC_PROMPT_LOOSE = "prompt_level_loose_acc"
METRIC_PROMPT_STRICT = "prompt_level_strict_acc"


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class BenchmarkResult:
    """Single benchmark result with target and reference scores.

    Attributes:
        name: Human-readable benchmark name.
        category: Benchmark category for grouping.
        metric: Metric type used for scoring.
        target: Score for target language (None if not evaluated).
        english: Score for English reference (None if not applicable).
    """
    name: str
    category: str
    metric: str
    target: Optional[float] = None
    english: Optional[float] = None

    @property
    def has_target(self) -> bool:
        """Check if target language score is available."""
        return self.target is not None

    @property
    def has_english(self) -> bool:
        """Check if English reference score is available."""
        return self.english is not None


@dataclass
class ParsedMetrics:
    """Container for all parsed metrics from a log file.

    Attributes:
        data: Nested dictionary mapping task -> metric -> value.
    """
    data: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def get(self, task: str, metric: str) -> Optional[float]:
        """Get metric value for a task.

        Args:
            task: Task identifier (e.g., 'arc_easy', 'global_mmlu_en').
            metric: Metric name (e.g., 'acc', 'acc_norm', 'exact_match').

        Returns:
            Metric value if found, None otherwise.
        """
        return self.data.get(task, {}).get(metric)

    def get_with_lang_variants(
        self,
        template: str,
        lang: str,
        metric: str
    ) -> Optional[float]:
        """Get metric trying all language code variants.

        Some tasks use 'ny' instead of 'nya', or 'maori' instead of 'mri'.
        This method tries all known aliases for the language.

        Args:
            template: Task name template with {lang} placeholder.
            lang: Base language code (e.g., 'nya', 'mri').
            metric: Metric name.

        Returns:
            First matching metric value, or None if no variant matches.
        """
        variants = LANGUAGE_ALIASES.get(lang, [lang])
        for variant in variants:
            value = self.get(template.format(lang=variant), metric)
            if value is not None:
                return value
        return None

    def get_first_available(self, task: str, metrics: List[str]) -> Optional[float]:
        """Get first available metric from a priority list.

        Useful when tasks may report different metrics (e.g., acc vs acc_norm).

        Args:
            task: Task identifier.
            metrics: List of metric names in priority order.

        Returns:
            First available metric value, or None if none found.
        """
        for metric in metrics:
            value = self.get(task, metric)
            if value is not None:
                return value
        return None


# =============================================================================
# Log Parser
# =============================================================================

class LogParser:
    """Parse lm-evaluation-harness log files to extract benchmark metrics.

    The parser handles the table format output from lm-eval:
        |task_name|version|filter|n-shot|metric|↑|value|±|stderr|

    Attributes:
        metrics: ParsedMetrics container with extracted data.
    """

    # Regex pattern for lm-eval table rows
    # Captures: task name, metric name, metric value
    _TABLE_PATTERN = re.compile(
        r"^\|([\w_-]*)\s*\|"    # Task name (may be empty for continuation)
        r"[^|]*\|"              # Version
        r"[^|]*\|"              # Filter
        r"[^|]*\|"              # n-shot
        r"([\w_@]+)\s*\|"       # Metric name
        r"[^|]*\|"              # Direction arrow
        r"\s*([\d.]+)\s*\|"     # Value
    )

    def __init__(self, log_file: Path) -> None:
        """Initialize parser and parse the log file.

        Args:
            log_file: Path to the lm-evaluation-harness log file.

        Raises:
            FileNotFoundError: If log file doesn't exist.
            ValueError: If log file cannot be parsed.
        """
        self.metrics = ParsedMetrics()
        self._parse(log_file)

    def _parse(self, log_file: Path) -> None:
        """Parse log file and extract metrics.

        Args:
            log_file: Path to the log file.
        """
        content = log_file.read_text(encoding="utf-8")
        current_task: Optional[str] = None

        for line in content.split("\n"):
            # Skip separator lines and empty lines
            if line.strip().startswith("| -") or not line.strip():
                continue

            match = self._TABLE_PATTERN.match(line)
            if not match:
                continue

            # Task name may be empty for multi-metric rows
            task = match.group(1).strip() or current_task
            metric = match.group(2).strip()
            value = float(match.group(3))
            current_task = task

            if task:
                # Store max value if metric appears multiple times
                self._store_metric(task, metric, value)

    def _store_metric(self, task: str, metric: str, value: float) -> None:
        """Store metric value, keeping maximum if duplicate.

        Args:
            task: Task identifier.
            metric: Metric name.
            value: Metric value.
        """
        if task not in self.metrics.data:
            self.metrics.data[task] = {}

        current = self.metrics.data[task].get(metric, 0)
        self.metrics.data[task][metric] = max(value, current)


# =============================================================================
# Benchmark Extractors
# =============================================================================

class BenchmarkExtractor:
    """Extract benchmark results for a specific evaluation mode.

    This class defines the benchmark suite for base and instruct models,
    mapping task names to human-readable benchmark names.
    """

    def __init__(self, metrics: ParsedMetrics, lang: str) -> None:
        """Initialize extractor with parsed metrics.

        Args:
            metrics: ParsedMetrics from log file.
            lang: Target language code ('nya' or 'mri').
        """
        self._m = metrics
        self._lang = lang

    def extract_base(self) -> List[BenchmarkResult]:
        """Extract benchmarks for base model (CPT) evaluation.

        Returns:
            List of BenchmarkResult for all base model benchmarks.
        """
        return [
            # General & STEM
            self._result(
                "Global MMLU-Lite", CATEGORY_GENERAL, METRIC_ACC,
                self._m.get_with_lang_variants("global_mmlu_{lang}", self._lang, METRIC_ACC),
                self._m.get("global_mmlu_en", METRIC_ACC)
            ),
            self._result(
                "ARC-Easy", CATEGORY_GENERAL, METRIC_ACC_NORM,
                self._m.get_with_lang_variants("arc_easy_{lang}", self._lang, METRIC_ACC_NORM),
                self._m.get_first_available("arc_easy", [METRIC_ACC_NORM, METRIC_ACC])
            ),
            self._result(
                "ARC-Challenge", CATEGORY_GENERAL, METRIC_ACC_NORM,
                self._m.get_with_lang_variants("arc_challenge_{lang}", self._lang, METRIC_ACC_NORM),
                self._m.get_first_available("arc_challenge", [METRIC_ACC_NORM, METRIC_ACC])
            ),
            self._result(
                "MGSM", CATEGORY_GENERAL, METRIC_EXACT_MATCH,
                self._m.get_with_lang_variants("mgsm_direct_{lang}", self._lang, METRIC_EXACT_MATCH),
                self._m.get("mgsm_direct_en", METRIC_EXACT_MATCH)
            ),
            self._result(
                "BBH", CATEGORY_GENERAL, METRIC_EXACT_MATCH,
                None,
                self._m.get("bbh_fewshot", METRIC_EXACT_MATCH)
            ),
            self._result(
                "GPQA Diamond", CATEGORY_GENERAL, METRIC_ACC_NORM,
                None,
                self._m.get_first_available("gpqa_diamond_n_shot", [METRIC_ACC_NORM, METRIC_ACC])
            ),

            # Commonsense
            self._result(
                "XCOPA", CATEGORY_COMMONSENSE, METRIC_ACC,
                self._m.get_with_lang_variants("xcopa_{lang}", self._lang, METRIC_ACC),
                self._m.get("copa", METRIC_ACC)
            ),
            self._result(
                "XStoryCloze", CATEGORY_COMMONSENSE, METRIC_ACC,
                self._m.get_with_lang_variants("xstorycloze_{lang}", self._lang, METRIC_ACC),
                self._m.get("xstorycloze_en", METRIC_ACC) or self._m.get("storycloze", METRIC_ACC)
            ),
            self._result(
                "PIQA", CATEGORY_COMMONSENSE, METRIC_ACC_NORM,
                self._m.get_with_lang_variants("piqa_{lang}", self._lang, METRIC_ACC_NORM),
                self._m.get_first_available("piqa", [METRIC_ACC_NORM, METRIC_ACC])
            ),
            self._result(
                "HellaSwag", CATEGORY_COMMONSENSE, METRIC_ACC_NORM,
                self._m.get_with_lang_variants("hellaswag_{lang}", self._lang, METRIC_ACC_NORM),
                self._m.get_first_available("hellaswag", [METRIC_ACC_NORM, METRIC_ACC])
            ),

            # NLI & Reading
            self._result(
                "XNLI 2.0", CATEGORY_NLI_READING, METRIC_ACC,
                self._m.get_with_lang_variants("xnli_{lang}", self._lang, METRIC_ACC),
                self._m.get("xnli_en", METRIC_ACC) or self._m.get("xnli", METRIC_ACC)
            ),
            self._result(
                "XWinograd", CATEGORY_NLI_READING, METRIC_ACC,
                self._m.get_with_lang_variants("xwinograd_{lang}", self._lang, METRIC_ACC),
                self._m.get("winogrande", METRIC_ACC)
            ),
            self._result(
                "Belebele", CATEGORY_NLI_READING, METRIC_ACC_NORM,
                self._m.get_with_lang_variants("belebele_{lang}_Latn", self._lang, METRIC_ACC_NORM),
                self._m.get("belebele_eng_Latn", METRIC_ACC_NORM)
            ),

            # Translation
            self._result(
                "FLORES (→EN)", CATEGORY_TRANSLATION, METRIC_BLEU,
                self._m.get_with_lang_variants("flores_{lang}_en", self._lang, METRIC_BLEU),
                None
            ),
            self._result(
                "FLORES (→EN) chrF", CATEGORY_TRANSLATION, METRIC_CHRF,
                self._m.get_with_lang_variants("flores_{lang}_en", self._lang, METRIC_CHRF),
                None
            ),
            self._result(
                "FLORES (EN→)", CATEGORY_TRANSLATION, METRIC_BLEU,
                self._m.get_with_lang_variants("flores_en_{lang}", self._lang, METRIC_BLEU),
                None
            ),
            self._result(
                "FLORES (EN→) chrF", CATEGORY_TRANSLATION, METRIC_CHRF,
                self._m.get_with_lang_variants("flores_en_{lang}", self._lang, METRIC_CHRF),
                None
            ),
        ]

    def extract_instruct(self) -> List[BenchmarkResult]:
        """Extract benchmarks for instruct model evaluation.

        Returns:
            List of BenchmarkResult for all instruct model benchmarks.
        """
        return [
            # General & STEM
            self._result(
                "Global MMLU-Lite", CATEGORY_GENERAL, METRIC_EXACT_MATCH,
                self._m.get_with_lang_variants("global_mmlu_{lang}_gen_0shot", self._lang, METRIC_EXACT_MATCH),
                self._m.get("global_mmlu_en_gen_0shot", METRIC_EXACT_MATCH)
            ),
            self._result(
                "ARC Challenge (chat)", CATEGORY_GENERAL, METRIC_EXACT_MATCH,
                self._m.get_with_lang_variants("arc_challenge_chat_{lang}", self._lang, METRIC_EXACT_MATCH),
                self._m.get("arc_challenge_chat", METRIC_EXACT_MATCH)
            ),
            self._result(
                "MGSM", CATEGORY_GENERAL, METRIC_EXACT_MATCH,
                self._m.get_with_lang_variants("mgsm_direct_{lang}", self._lang, METRIC_EXACT_MATCH),
                self._m.get("mgsm_direct_en", METRIC_EXACT_MATCH)
            ),
            self._result(
                "BBH", CATEGORY_GENERAL, METRIC_EXACT_MATCH,
                None,
                self._m.get("bbh_zeroshot", METRIC_EXACT_MATCH) or self._m.get("bbh", METRIC_EXACT_MATCH)
            ),
            self._result(
                "GPQA Diamond", CATEGORY_GENERAL, METRIC_ACC_NORM,
                None,
                self._m.get("gpqa_diamond_zeroshot", METRIC_ACC_NORM) or self._m.get("gpqa_diamond", METRIC_ACC_NORM)
            ),

            # Commonsense
            self._result(
                "XCOPA", CATEGORY_COMMONSENSE, METRIC_ACC,
                self._m.get_with_lang_variants("xcopa_{lang}", self._lang, METRIC_ACC),
                self._m.get("copa", METRIC_ACC)
            ),
            self._result(
                "XStoryCloze", CATEGORY_COMMONSENSE, METRIC_ACC,
                self._m.get_with_lang_variants("xstorycloze_{lang}", self._lang, METRIC_ACC),
                self._m.get("xstorycloze_en", METRIC_ACC)
            ),
            self._result(
                "PIQA", CATEGORY_COMMONSENSE, METRIC_ACC_NORM,
                self._m.get_with_lang_variants("piqa_{lang}", self._lang, METRIC_ACC_NORM),
                self._m.get("piqa", METRIC_ACC_NORM)
            ),
            self._result(
                "HellaSwag", CATEGORY_COMMONSENSE, METRIC_ACC_NORM,
                self._m.get_with_lang_variants("hellaswag_{lang}", self._lang, METRIC_ACC_NORM),
                self._m.get("hellaswag", METRIC_ACC_NORM)
            ),

            # Reading & QA
            self._result(
                "XNLI 2.0", CATEGORY_READING_QA, METRIC_ACC,
                self._m.get_with_lang_variants("xnli_{lang}", self._lang, METRIC_ACC),
                self._m.get("xnli_en", METRIC_ACC)
            ),
            self._result(
                "XWinograd", CATEGORY_READING_QA, METRIC_ACC,
                self._m.get_with_lang_variants("xwinograd_{lang}", self._lang, METRIC_ACC),
                self._m.get("winogrande", METRIC_ACC)
            ),
            self._result(
                "Belebele", CATEGORY_READING_QA, METRIC_ACC_NORM,
                self._m.get_with_lang_variants("belebele_{lang}_Latn", self._lang, METRIC_ACC_NORM),
                self._m.get("belebele_eng_Latn", METRIC_ACC_NORM)
            ),

            # Instruction Following
            self._result(
                "IFEval", CATEGORY_INSTRUCTION, METRIC_PROMPT_LOOSE,
                None,
                self._m.get_first_available("ifeval", [METRIC_PROMPT_LOOSE, METRIC_PROMPT_STRICT])
            ),

            # Truthfulness
            self._result(
                "TruthfulQA", CATEGORY_TRUTHFULNESS, METRIC_BLEU_ACC,
                self._m.get_with_lang_variants("truthfulqa-multi_gen_{lang}", self._lang, METRIC_BLEU_ACC),
                self._m.get("truthfulqa-multi_gen_en", METRIC_BLEU_ACC)
            ),

            # Code
            self._result(
                "HumanEval", CATEGORY_CODE, METRIC_PASS_AT_1,
                None,
                self._m.get("humaneval_instruct", METRIC_PASS_AT_1) or self._m.get("humaneval", METRIC_PASS_AT_1)
            ),

            # Translation
            self._result(
                "FLORES (→EN)", CATEGORY_TRANSLATION, METRIC_BLEU,
                self._m.get_with_lang_variants("flores_{lang}_en", self._lang, METRIC_BLEU),
                None
            ),
            self._result(
                "FLORES (→EN) chrF", CATEGORY_TRANSLATION, METRIC_CHRF,
                self._m.get_with_lang_variants("flores_{lang}_en", self._lang, METRIC_CHRF),
                None
            ),
            self._result(
                "FLORES (EN→)", CATEGORY_TRANSLATION, METRIC_BLEU,
                self._m.get_with_lang_variants("flores_en_{lang}", self._lang, METRIC_BLEU),
                None
            ),
            self._result(
                "FLORES (EN→) chrF", CATEGORY_TRANSLATION, METRIC_CHRF,
                self._m.get_with_lang_variants("flores_en_{lang}", self._lang, METRIC_CHRF),
                None
            ),
        ]

    @staticmethod
    def _result(
        name: str,
        category: str,
        metric: str,
        target: Optional[float],
        english: Optional[float]
    ) -> BenchmarkResult:
        """Create a BenchmarkResult with the given parameters."""
        return BenchmarkResult(
            name=name,
            category=category,
            metric=metric,
            target=target,
            english=english
        )


# =============================================================================
# Output Formatters
# =============================================================================

class OutputFormatter:
    """Format benchmark results for display or export."""

    @staticmethod
    def format_value(value: Optional[float], width: int = 10) -> str:
        """Format a metric value or show dash for missing.

        Args:
            value: Metric value or None.
            width: Output width for alignment.

        Returns:
            Formatted string with value or dash.
        """
        if value is not None:
            return f"{value:>{width}.4f}"
        return f"{'-':>{width}}"

    @classmethod
    def print_table(cls, results: List[BenchmarkResult], lang: str) -> None:
        """Print results as formatted table.

        Args:
            results: List of benchmark results.
            lang: Target language code for column header.
        """
        # Header
        print(f"{'Benchmark':<24} {lang.upper():>10} {'EN':>10}")
        print("-" * 46)

        # Results grouped by category
        current_category: Optional[str] = None
        for result in results:
            if result.category != current_category:
                if current_category is not None:
                    print()  # Blank line between categories
                print(f"# {result.category}")
                current_category = result.category

            target_str = cls.format_value(result.target)
            english_str = cls.format_value(result.english)
            print(f"{result.name:<24} {target_str} {english_str}")

    @staticmethod
    def print_csv(results: List[BenchmarkResult]) -> None:
        """Print results as CSV format.

        Args:
            results: List of benchmark results.
        """
        print("category,benchmark,metric,target_lang,english")
        for result in results:
            target_str = f"{result.target:.4f}" if result.target is not None else ""
            english_str = f"{result.english:.4f}" if result.english is not None else ""
            print(f"{result.category},{result.name},{result.metric},{target_str},{english_str}")

    @staticmethod
    def print_debug(metrics: ParsedMetrics) -> None:
        """Print all parsed tasks for debugging.

        Args:
            metrics: ParsedMetrics container.
        """
        print("=" * 60)
        print("PARSED TASKS")
        print("=" * 60)
        for task in sorted(metrics.data.keys()):
            task_metrics = metrics.data[task]
            print(f"\n{task}:")
            for metric, value in sorted(task_metrics.items()):
                print(f"  {metric}: {value:.4f}")


# =============================================================================
# CLI
# =============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        description="Extract benchmark results from lm-evaluation-harness log files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s results/model_base.txt --mode base --lang nya
  %(prog)s results/model_instruct.txt --mode instruct --lang mri --csv
  %(prog)s results/log.txt --debug
        """
    )

    parser.add_argument(
        "log_file",
        type=Path,
        help="Path to lm-evaluation-harness log file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=[e.value for e in EvalMode],
        default=EvalMode.INSTRUCT.value,
        help="Evaluation mode: 'base' for CPT models, 'instruct' for fine-tuned (default: instruct)"
    )
    parser.add_argument(
        "--lang",
        type=str,
        choices=[e.value for e in Language],
        default=Language.MRI.value,
        help="Target language code (default: mri)"
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Output results in CSV format"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show all parsed tasks (useful for debugging)"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}"
    )

    return parser


def main() -> int:
    """Main entry point for CLI.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    parser = create_parser()
    args = parser.parse_args()

    # Validate file exists
    if not args.log_file.exists():
        print(f"Error: File not found: {args.log_file}", file=sys.stderr)
        return 1

    # Parse log file
    try:
        log_parser = LogParser(args.log_file)
    except Exception as e:
        print(f"Error parsing log file: {e}", file=sys.stderr)
        return 1

    # Debug output
    if args.debug:
        OutputFormatter.print_debug(log_parser.metrics)
        print()

    # Extract benchmarks based on mode
    extractor = BenchmarkExtractor(log_parser.metrics, args.lang)

    if args.mode == EvalMode.BASE.value:
        results = extractor.extract_base()
    else:
        results = extractor.extract_instruct()

    # Output results
    if args.csv:
        OutputFormatter.print_csv(results)
    else:
        OutputFormatter.print_table(results, args.lang)

    return 0


if __name__ == "__main__":
    sys.exit(main())
