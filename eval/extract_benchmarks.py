#!/usr/bin/env python3
"""Backward-compatible wrapper for extract_benchmarks.

This script is deprecated. Please use:
    python -m byol_eval extract <log_file> --eval-mode <mode> --lang <lang>

Example:
    python -m byol_eval extract results/log.txt --eval-mode base --lang nya --csv
"""

import sys
import warnings

warnings.warn(
    "extract_benchmarks.py is deprecated. Use 'python -m byol_eval extract' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from byol_eval.extract import main

if __name__ == "__main__":
    sys.exit(main())
