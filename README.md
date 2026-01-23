# BYOL - Bring Your Own Language

A modular framework for training and evaluating Large Language Models (LLMs) with best practices.

## Overview

BYOL provides clean, typed Python interfaces for:
- **Training**: CPT, SFT, and DPO using LlamaFactory as backend
- **Evaluation**: Benchmark evaluation via lm-evaluation-harness and LLM-as-Judge
- **Merging**: Multiple model merge strategies (delta, LoRA, general)

## Installation

```bash
# Install training package
cd train
pip install -e .

# Install evaluation package
cd eval
pip install -e .
```

## Quick Start

### Training

```bash
# CPT training with LoRA
byol-train cpt --config configs/cpt.yaml --gpus 0,1

# SFT training
byol-train sft --model google/gemma-3-4b-pt --dataset alpaca --lora
```

```python
from byol_train import TrainConfig, TrainingRunner

config = TrainConfig.from_yaml("configs/cpt.yaml")
runner = TrainingRunner(config)
result = runner.run()
```

### Evaluation

```bash
# Benchmark evaluation
byol-eval benchmark --model hf --model-args pretrained=google/gemma-3-4b --tasks mmlu
```

```python
from byol_eval import EvalConfig, EvaluationRunner

config = EvalConfig.from_yaml("configs/eval.yaml")
runner = EvaluationRunner(config)
results = runner.run()
```

### Model Merging

```bash
# Delta merge: instruct + alpha*(fine_tuned - base)
python -m byol_train.merge delta \
  --base google/gemma-3-4b-pt \
  --instruct google/gemma-3-4b-it \
  --fine-tuned outputs/cpt-model \
  --alpha 0.5 \
  --output outputs/merged
```

## Project Structure

```
BYOL/
├── train/                  # Training package
│   ├── byol_train/
│   │   ├── cli.py         # CLI entry point
│   │   ├── config.py      # Configuration dataclasses
│   │   ├── constants.py   # Default values
│   │   ├── merge.py       # Model merging utilities
│   │   ├── runner.py      # Training runner
│   │   └── secrets.py     # HF/W&B token management
│   ├── configs/           # Example configurations
│   └── README.md
├── eval/                   # Evaluation package
│   ├── byol_eval/
│   │   ├── cli.py         # CLI entry point
│   │   ├── config.py      # Configuration dataclasses
│   │   ├── constants.py   # Default values
│   │   ├── runner.py      # Evaluation runner
│   │   ├── judge.py       # LLM-as-Judge
│   │   └── secrets.py     # HF token management
│   ├── configs/           # Benchmark configurations
│   └── README.md
└── LlamaFactory/          # External dependency (gitignored)
```

## Features

- **Type-safe configuration** with Python dataclasses
- **Secure secrets management** via environment variables or local files
- **LoRA fine-tuning** with configurable parameters
- **Multiple merge strategies** for model combination
- **CLI and Python API** interfaces

## Configuration

### Environment Variables

```bash
export HF_TOKEN="your-huggingface-token"
export WANDB_API_KEY="your-wandb-key"
```

### Local Secrets File

Create `byol_train/secrets_local.py` (gitignored):

```python
HF_TOKEN = "your-huggingface-token"
WANDB_API_KEY = "your-wandb-key"
```

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- CUDA-capable GPU

## License

MIT License - see [LICENSE](LICENSE) for details.
