#!/usr/bin/env python3
"""
Extract benchmark results from lm-evaluation-harness log files.

Supports both base model (CPT) and instruct model evaluations.

Usage:
    python extract_benchmarks.py <log_file> --mode base --lang mri
    python extract_benchmarks.py <log_file> --mode instruct --lang nya
    python extract_benchmarks.py <log_file> --mode instruct --lang nya --csv
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================

# Language code aliases (some tasks use 'ny' instead of 'nya')
# Note: Order matters - put most common variant first for each context
LANG_ALIASES = {"nya": ["nya", "ny", "nyanja"], "mri": ["mri", "maori"]}


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class Result:
    """Single benchmark result."""
    name: str
    category: str
    metric: str
    target: float | None = None
    english: float | None = None


class LogParser:
    """Parse lm-evaluation-harness log files."""
    
    def __init__(self, log_file: Path):
        self.data: dict[str, dict[str, float]] = {}
        self._parse(log_file)
    
    def _parse(self, log_file: Path) -> None:
        content = log_file.read_text(encoding="utf-8")
        # Match: |task|version|filter|n-shot|metric|↑|value|±|stderr|
        pattern = r"^\|([\w_-]*)\s*\|[^|]*\|[^|]*\|[^|]*\|([\w_@]+)\s*\|[^|]*\|\s*([\d.]+)\s*\|"
        
        current_task = None
        for line in content.split("\n"):
            match = re.match(pattern, line)
            if not match or line.strip().startswith("| -"):
                continue
            
            task = match.group(1).strip() or current_task
            metric = match.group(2).strip()
            value = float(match.group(3))
            current_task = task
            
            if task:
                self.data.setdefault(task, {})[metric] = max(
                    value, self.data.get(task, {}).get(metric, 0)
                )
    
    def get(self, task: str, metric: str) -> float | None:
        """Get metric value for a task."""
        return self.data.get(task, {}).get(metric)
    
    def get_lang(self, template: str, lang: str, metric: str) -> float | None:
        """Get metric trying all language variants."""
        for variant in LANG_ALIASES.get(lang, [lang]):
            val = self.get(template.format(lang=variant), metric)
            if val is not None:
                return val
        return None
    
    def get_first(self, task: str, metrics: list[str]) -> float | None:
        """Get first available metric."""
        for m in metrics:
            val = self.get(task, m)
            if val is not None:
                return val
        return None


# =============================================================================
# Benchmark Extractors
# =============================================================================

def extract_base(p: LogParser, lang: str) -> list[Result]:
    """Extract base model (CPT) benchmarks."""
    return [
        # General & STEM
        Result("Global MMLU-Lite", "General & STEM", "acc",
               p.get_lang("global_mmlu_{lang}", lang, "acc"),
               p.get("global_mmlu_en", "acc")),
        Result("ARC-Easy", "General & STEM", "acc_norm",
               p.get_lang("arc_easy_{lang}", lang, "acc_norm"),
               p.get_first("arc_easy", ["acc_norm", "acc"])),
        Result("ARC-Challenge", "General & STEM", "acc_norm",
               p.get_lang("arc_challenge_{lang}", lang, "acc_norm"),
               p.get_first("arc_challenge", ["acc_norm", "acc"])),
        Result("MGSM", "General & STEM", "exact_match",
               p.get_lang("mgsm_direct_{lang}", lang, "exact_match"),
               p.get("mgsm_direct_en", "exact_match")),
        Result("BBH", "General & STEM", "exact_match",
               None,
               p.get("bbh_fewshot", "exact_match")),
        Result("GPQA Diamond", "General & STEM", "acc_norm",
               None,
               p.get_first("gpqa_diamond_n_shot", ["acc_norm", "acc"])),
        
        # Commonsense
        Result("XCOPA", "Commonsense", "acc",
               p.get_lang("xcopa_{lang}", lang, "acc"),
               p.get("copa", "acc")),
        Result("XStoryCloze", "Commonsense", "acc",
               p.get_lang("xstorycloze_{lang}", lang, "acc"),
               p.get("xstorycloze_en", "acc") or p.get("storycloze", "acc")),
        Result("PIQA", "Commonsense", "acc_norm",
               p.get_lang("piqa_{lang}", lang, "acc_norm"),
               p.get_first("piqa", ["acc_norm", "acc"])),
        Result("HellaSwag", "Commonsense", "acc_norm",
               p.get_lang("hellaswag_{lang}", lang, "acc_norm"),
               p.get_first("hellaswag", ["acc_norm", "acc"])),
        
        # NLI & Reading
        Result("XNLI 2.0", "NLI & Reading", "acc",
               p.get_lang("xnli_{lang}", lang, "acc"),
               p.get("xnli_en", "acc") or p.get("xnli", "acc")),
        Result("XWinograd", "NLI & Reading", "acc",
               p.get_lang("xwinograd_{lang}", lang, "acc"),
               p.get("winogrande", "acc")),
        Result("Belebele", "NLI & Reading", "acc_norm",
               p.get_lang("belebele_{lang}_Latn", lang, "acc_norm"),
               p.get("belebele_eng_Latn", "acc_norm")),
        
        # Translation
        Result("FLORES (→EN)", "Translation", "bleu",
               p.get_lang("flores_{lang}_en", lang, "bleu"), None),
        Result("FLORES (→EN) chrF", "Translation", "chrf",
               p.get_lang("flores_{lang}_en", lang, "chrf"), None),
        Result("FLORES (EN→)", "Translation", "bleu",
               p.get_lang("flores_en_{lang}", lang, "bleu"), None),
        Result("FLORES (EN→) chrF", "Translation", "chrf",
               p.get_lang("flores_en_{lang}", lang, "chrf"), None),
    ]


def extract_instruct(p: LogParser, lang: str) -> list[Result]:
    """Extract instruct model benchmarks."""
    return [
        # General & STEM
        Result("Global MMLU-Lite", "General & STEM", "exact_match",
               p.get_lang("global_mmlu_{lang}_gen_0shot", lang, "exact_match"),
               p.get("global_mmlu_en_gen_0shot", "exact_match")),
        Result("ARC Challenge (chat)", "General & STEM", "exact_match",
               p.get_lang("arc_challenge_chat_{lang}", lang, "exact_match"),
               p.get("arc_challenge_chat", "exact_match")),
        Result("MGSM", "General & STEM", "exact_match",
               p.get_lang("mgsm_direct_{lang}", lang, "exact_match"),
               p.get("mgsm_direct_en", "exact_match")),
        Result("BBH", "General & STEM", "exact_match",
               None,
               p.get("bbh_zeroshot", "exact_match") or p.get("bbh", "exact_match")),
        Result("GPQA Diamond", "General & STEM", "acc_norm",
               None,
               p.get("gpqa_diamond_zeroshot", "acc_norm") or p.get("gpqa_diamond", "acc_norm")),
        
        # Commonsense
        Result("XCOPA", "Commonsense", "acc",
               p.get_lang("xcopa_{lang}", lang, "acc"),
               p.get("copa", "acc")),
        Result("XStoryCloze", "Commonsense", "acc",
               p.get_lang("xstorycloze_{lang}", lang, "acc"),
               p.get("xstorycloze_en", "acc")),
        Result("PIQA", "Commonsense", "acc_norm",
               p.get_lang("piqa_{lang}", lang, "acc_norm"),
               p.get("piqa", "acc_norm")),
        Result("HellaSwag", "Commonsense", "acc_norm",
               p.get_lang("hellaswag_{lang}", lang, "acc_norm"),
               p.get("hellaswag", "acc_norm")),
        
        # Reading & QA
        Result("XNLI 2.0", "Reading & QA", "acc",
               p.get_lang("xnli_{lang}", lang, "acc"),
               p.get("xnli_en", "acc")),
        Result("XWinograd", "Reading & QA", "acc",
               p.get_lang("xwinograd_{lang}", lang, "acc"),
               p.get("winogrande", "acc")),
        Result("Belebele", "Reading & QA", "acc_norm",
               p.get_lang("belebele_{lang}_Latn", lang, "acc_norm"),
               p.get("belebele_eng_Latn", "acc_norm")),
        
        # Instruction Following
        Result("IFEval", "Instruction Following", "prompt_level_loose_acc",
               None,
               p.get_first("ifeval", ["prompt_level_loose_acc", "prompt_level_strict_acc"])),
        
        # Truthfulness
        Result("TruthfulQA", "Truthfulness", "bleu_acc",
               p.get_lang("truthfulqa-multi_gen_{lang}", lang, "bleu_acc"),
               p.get("truthfulqa-multi_gen_en", "bleu_acc")),
        
        # Code
        Result("HumanEval", "Code", "pass@1",
               None,
               p.get("humaneval_instruct", "pass@1") or p.get("humaneval", "pass@1")),
        
        # Translation
        Result("FLORES (→EN)", "Translation", "bleu",
               p.get_lang("flores_{lang}_en", lang, "bleu"), None),
        Result("FLORES (→EN) chrF", "Translation", "chrf",
               p.get_lang("flores_{lang}_en", lang, "chrf"), None),
        Result("FLORES (EN→)", "Translation", "bleu",
               p.get_lang("flores_en_{lang}", lang, "bleu"), None),
        Result("FLORES (EN→) chrF", "Translation", "chrf",
               p.get_lang("flores_en_{lang}", lang, "chrf"), None),
    ]


# =============================================================================
# Output Formatting
# =============================================================================

def fmt(val: float | None, width: int = 10) -> str:
    """Format value or dash."""
    return f"{val:>{width}.4f}" if val is not None else f"{'-':>{width}}"


def print_table(results: list[Result], lang: str) -> None:
    """Print results as table."""
    print(f"{'Benchmark':<24} {lang.upper():>10} {'EN':>10}")
    print("-" * 46)
    
    category = None
    for r in results:
        if r.category != category:
            if category:
                print()
            print(f"# {r.category}")
            category = r.category
        print(f"{r.name:<24} {fmt(r.target)} {fmt(r.english)}")


def print_csv(results: list[Result]) -> None:
    """Print results as CSV."""
    print("category,benchmark,metric,target_lang,english")
    for r in results:
        t = f"{r.target:.4f}" if r.target else ""
        e = f"{r.english:.4f}" if r.english else ""
        print(f"{r.category},{r.name},{r.metric},{t},{e}")


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="Extract benchmark results.")
    parser.add_argument("log_file", type=Path, help="Path to log file")
    parser.add_argument("--mode", choices=["base", "instruct"], default="instruct",
                        help="Evaluation mode (default: instruct)")
    parser.add_argument("--lang", default="mri", choices=["mri", "nya"],
                        help="Target language (default: mri)")
    parser.add_argument("--csv", action="store_true", help="Output as CSV")
    parser.add_argument("--debug", action="store_true", help="Show parsed tasks")
    args = parser.parse_args()
    
    if not args.log_file.exists():
        print(f"Error: File not found: {args.log_file}", file=sys.stderr)
        return 1
    
    p = LogParser(args.log_file)
    
    if args.debug:
        print("=== Parsed Tasks ===")
        for task, metrics in sorted(p.data.items()):
            print(f"  {task}: {metrics}")
        print()
    
    results = extract_base(p, args.lang) if args.mode == "base" else extract_instruct(p, args.lang)
    
    if args.csv:
        print_csv(results)
    else:
        print_table(results, args.lang)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
