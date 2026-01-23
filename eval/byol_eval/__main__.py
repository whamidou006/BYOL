"""Entry point for running byol_eval as a module.

Usage:
    python -m byol_eval --model <path> --tasks <tasks>
    python -m byol_eval --config <config.yaml>
    python -m byol_eval judge --model-config <config> --dataset-config <config>
"""

from .cli import main

if __name__ == "__main__":
    main()
