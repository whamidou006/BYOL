#!/usr/bin/env python3
"""
Apply patches to lm-evaluation-harness after cloning.

Usage:
    cd eval
    python patches/apply_lmeval_patches.py

Patches applied:
1. lm_eval/filters/extraction.py - Adds OrderedRegexFilter
2. lm_eval/tasks/truthfulqa-multi/ - Adds multilingual TruthfulQA tasks

Run after: git clone https://github.com/EleutherAI/lm-evaluation-harness.git
"""

import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
EVAL_DIR = SCRIPT_DIR.parent
LMEVAL_DIR = EVAL_DIR / "lm-evaluation-harness" / "lm_eval"

# =============================================================================
# Patch 1: OrderedRegexFilter
# =============================================================================

ORDERED_REGEX_FILTER = '''

@register_filter("ordered_regex")
class OrderedRegexFilter(Filter):
    """A filter that applies multiple regex patterns in order with fallback behavior.

    This filter tries each regex pattern sequentially. If a pattern matches, it returns
    the match. If no pattern matches, it proceeds to the next pattern. Only if all
    patterns fail does it return the fallback value.

    Optionally applies cleanup to the matched string via strip_extracts.
    """

    def __init__(
        self,
        regex_patterns: list[str] = [r"#### (\\-?[0-9\\.\\,]+)"],
        group_select: int = 0,
        fallback: str = "[invalid]",
        strip_extracts: list[str] = None,
    ) -> None:
        """
        Args:
            regex_patterns: List of regex patterns to try in order.
            group_select: Which group to select from the match.
            fallback: Fallback value if no match is found.
            strip_extracts: Regexes to remove from the extracted result.
        """
        self.regex_patterns = regex_patterns
        self.regexes = [re.compile(pattern) for pattern in regex_patterns]
        self.group_select = group_select
        self.fallback = fallback
        self.strip_extracts = [re.compile(p) for p in strip_extracts] if strip_extracts else []

    def _clean_extracted(self, text: str) -> str:
        for ignore_re in self.strip_extracts:
            text = ignore_re.sub("", text)
        return text.strip()

    def apply(self, resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
        def filter_set(inst):
            filtered = []
            for resp in inst:
                match = None

                for regex in self.regexes:
                    match = regex.findall(resp)
                    if match:
                        match = match[self.group_select]
                        if isinstance(match, tuple):
                            match = [m for m in match if m]
                            if match:
                                match = match[0]
                            else:
                                match = None
                        if match:
                            match = self._clean_extracted(match)
                            break

                if not match:
                    match = self.fallback

                filtered.append(match)
            return filtered

        filtered_resps = list(map(lambda x: filter_set(x), resps))
        return filtered_resps
'''


def patch_extraction_py() -> bool:
    """Add OrderedRegexFilter to extraction.py if not present."""
    extraction_file = LMEVAL_DIR / "filters" / "extraction.py"
    
    if not extraction_file.exists():
        print(f"❌ File not found: {extraction_file}")
        return False
    
    content = extraction_file.read_text()
    
    if "OrderedRegexFilter" in content:
        print("✅ OrderedRegexFilter already exists")
        return True
    
    # Append the filter
    new_content = content.rstrip() + ORDERED_REGEX_FILTER
    extraction_file.write_text(new_content)
    print(f"✅ Patched: {extraction_file}")
    
    return True


def patch_truthfulqa_multi() -> bool:
    """Copy truthfulqa-multi task folder."""
    target_dir = LMEVAL_DIR / "tasks" / "truthfulqa-multi"
    source_dir = SCRIPT_DIR / "truthfulqa-multi"
    
    if not source_dir.exists():
        print(f"❌ Source not found: {source_dir}")
        return False
    
    if target_dir.exists():
        print("✅ truthfulqa-multi task already exists")
        return True
    
    shutil.copytree(source_dir, target_dir, ignore=shutil.ignore_patterns("__pycache__"))
    print(f"✅ Copied: {target_dir}")
    
    return True


def main() -> int:
    print("🔧 Applying lm-evaluation-harness patches...\n")
    
    if not LMEVAL_DIR.exists():
        print(f"❌ lm-evaluation-harness not found at: {LMEVAL_DIR.parent}")
        print("   Run: git clone https://github.com/EleutherAI/lm-evaluation-harness.git")
        return 1
    
    print(f"📍 Patching: {LMEVAL_DIR}\n")
    
    # Apply patches
    results = []
    results.append(("OrderedRegexFilter", patch_extraction_py()))
    results.append(("truthfulqa-multi", patch_truthfulqa_multi()))
    
    print()
    success = all(r[1] for r in results)
    if success:
        print("✅ All patches applied successfully!")
        return 0
    else:
        failed = [r[0] for r in results if not r[1]]
        print(f"⚠️ Failed patches: {', '.join(failed)}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
    sys.exit(main())
