# BYOL Training Framework

A clean Python wrapper for LlamaFactory training with best practices.
Supports CPT (Continual Pre-Training), SFT (Supervised Fine-Tuning), and DPO.

## Features

- **Type-safe configuration** with dataclasses
- **Secure secrets management** for HuggingFace and W&B tokens
- **LoRA fine-tuning** with configurable parameters
- **Dataset mixing** strategies (concat, interleave)
- **CLI and Python API** interfaces

## Installation

```bash
cd train
pip install -r requirements.txt
pip install -e .
```

This will install:
- [LlamaFactory](https://github.com/hiyouga/LLaMAFactory) from GitHub
- Metrics dependencies (rouge-score, nltk, etc.)
- byol-train package

## Secrets Management

The framework supports multiple ways to provide HuggingFace/W&B tokens:

1. **Environment variables** (recommended for CI/CD):
   ```bash
   export HF_TOKEN="your-huggingface-token"
   export WANDB_API_KEY="your-wandb-key"
   ```

2. **Local secrets file** (recommended for development):
   Create `byol_train/secrets_local.py` (gitignored):
   ```python
   HF_TOKEN = "your-huggingface-token"
   WANDB_API_KEY = "your-wandb-key"
   WANDB_PROJECT = "your-project-name"
   ```

3. **HuggingFace CLI cache** (`~/.huggingface/token`)

## Quick Start

### Using CLI

```bash
# CPT training
byol-train cpt --config configs/cpt.yaml --gpus 0,1

# SFT training with LoRA
byol-train sft --model meta-llama/Llama-3-8B --dataset alpaca --lora --lora-rank 64

# DPO training
byol-train dpo --config configs/dpo.yaml --gpus 0,1,2,3

# Dry run (preview config without training)
byol-train sft --config configs/sft.yaml --dry-run
```

### Using Python API

```python
from byol_train import TrainConfig, TrainingRunner, LoraConfig

# Load config from YAML
config = TrainConfig.from_yaml("configs/cpt.yaml")

# Or create config programmatically
config = TrainConfig(
    model_name_or_path="google/gemma-3-4b-pt",
    stage="cpt",
    dataset="fineweb2_dataset_mri_train",
    epochs=4,
    learning_rate=1e-5,
    lora=LoraConfig(rank=64, alpha=128),
)

# Run training
runner = TrainingRunner(config)
result = runner.run()

if result.success:
    print(f"✅ Training completed: {result.output_dir}")
else:
    print(f"❌ Training failed: {result.error}")
```

## CLI Reference

### Training Stages

| Command | Description |
|---------|-------------|
| `byol-train cpt` | Continual Pre-Training on unlabeled text |
| `byol-train sft` | Supervised Fine-Tuning on instruction data |
| `byol-train dpo` | Direct Preference Optimization |
| `byol-train merge` | Merge LoRA adapter into base model |

### Common Options

| Option | Description |
|--------|-------------|
| `-c, --config` | Path to YAML configuration file |
| `-m, --model` | HuggingFace model ID or local path |
| `-d, --dataset` | Dataset name |
| `-g, --gpus` | Comma-separated GPU IDs (default: 0) |
| `-o, --output-dir` | Base output directory (default: outputs) |
| `--dry-run` | Print config without running |
| `--wandb-project` | W&B project name for logging |

### Training Options

| Option | Description |
|--------|-------------|
| `-e, --epochs` | Number of training epochs (default: 3) |
| `-b, --batch-size` | Per-device batch size (default: 4) |
| `--grad-accum` | Gradient accumulation steps (default: 4) |
| `--lr` | Learning rate (default: 5e-5) |
| `--cutoff-len` | Maximum sequence length (default: 8192) |
| `--template` | Chat template name (default: gemma) |
| `--bf16/--no-bf16` | Use bfloat16 precision (default: True) |

### LoRA Options

| Option | Description |
|--------|-------------|
| `--lora` | Enable LoRA fine-tuning |
| `--lora-rank` | LoRA rank (default: 16) |
| `--lora-alpha` | LoRA alpha (default: 32) |
| `--lora-dropout` | LoRA dropout (default: 0.05) |

### Override Syntax

Override any config value:
```bash
byol-train sft --config config.yaml --override epochs=10 learning_rate=1e-5
```

## LoRA Merging

After LoRA training, merge the adapter into the base model:

```bash
byol-train merge \
  --base-model google/gemma-3-4b-pt \
  --adapter outputs/sft-lora/checkpoint-final \
  --output outputs/merged-model
```

Or via Python:
```python
from byol_train import merge_lora

merge_lora(
    base_model="google/gemma-3-4b-pt",
    adapter_path="outputs/sft-lora/",
    output_dir="outputs/merged-model/",
)
```

## Project Structure

```
train/
├── byol_train/
│   ├── __init__.py       # Package exports
│   ├── cli.py            # CLI entry point
│   ├── config.py         # Configuration dataclasses
│   ├── constants.py      # Default values and constants
│   ├── secrets.py        # HuggingFace/W&B token management
│   ├── runner.py         # Training runner
│   ├── merge.py          # LoRA merging utilities
│   └── py.typed          # PEP 561 type marker
├── configs/
│   ├── cpt.yaml          # Continual pre-training config
│   ├── sft.yaml          # Supervised fine-tuning config
│   ├── dpo.yaml          # DPO config
│   └── merge.yaml        # Merge config
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Constants Reference

Default values are defined in `constants.py`:

| Constant | Value | Description |
|----------|-------|-------------|
| `DEFAULT_BATCH_SIZE` | 4 | Per-device batch size |
| `DEFAULT_EPOCHS` | 3 | Training epochs |
| `DEFAULT_LEARNING_RATE` | 5e-5 | Learning rate |
| `DEFAULT_LORA_RANK` | 16 | LoRA rank |
| `DEFAULT_LORA_ALPHA` | 32 | LoRA alpha |
| `DEFAULT_CUTOFF_LEN` | 8192 | Max sequence length |

## License

See main BYOL repository.
