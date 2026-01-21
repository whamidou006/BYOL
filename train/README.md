# BYOL Training Framework

A clean Python wrapper for LlamaFactory training with best practices.
Supports CPT (Continual Pre-Training), SFT (Supervised Fine-Tuning), and DPO.

## Installation

```bash
cd train
pip install -e .
```

## Quick Start

### Using CLI

```bash
# CPT training
byol-train cpt --config configs/cpt.yaml --gpus 0,1

# SFT training with LoRA
byol-train sft --config configs/sft.yaml --lora --lora-rank 64

# DPO training
byol-train dpo --config configs/dpo.yaml --gpus 0,1,2,3
```

### Using Python API

```python
from byol_train import TrainConfig, TrainingRunner

# Load config from YAML
config = TrainConfig.from_yaml("configs/cpt.yaml")

# Or create config programmatically
config = TrainConfig(
    stage="pt",
    model_name_or_path="google/gemma-3-4b-pt",
    dataset="fineweb2_dataset_mri_train",
    learning_rate=1e-5,
    num_train_epochs=4,
)

# Run training
runner = TrainingRunner(config, gpus="0,1")
result = runner.run()

print(f"Training completed: {result.output_dir}")
```

## CLI Options

### Common Options

| Option | Description |
|--------|-------------|
| `-c, --config` | Path to training config YAML (required) |
| `-g, --gpus` | Comma-separated GPU IDs (default: 0) |
| `--dry-run` | Print config without running |
| `-v, --verbose` | Enable verbose logging |

### LoRA Options

| Option | Description |
|--------|-------------|
| `--lora` | Enable LoRA training |
| `--lora-rank` | LoRA rank (default: 64) |
| `--lora-alpha` | LoRA alpha (default: 2*rank) |
| `--lora-dropout` | LoRA dropout (default: 0.05) |

### Training Overrides

| Option | Description |
|--------|-------------|
| `--model` | Override model path |
| `--dataset` | Override dataset name |
| `--epochs` | Override number of epochs |
| `--lr` | Override learning rate |
| `--batch-size` | Override per-device batch size |
| `--grad-accum` | Override gradient accumulation steps |
| `--auto-batch` | Auto-compute gradient accumulation |
| `--resume` | Resume from checkpoint directory |

### Dataset Mixing

| Option | Description |
|--------|-------------|
| `--mix-strategy` | concat, interleave_under, interleave_over |

## Config Structure

```yaml
### Model Configuration ###
model_name_or_path: google/gemma-3-4b-pt
template: gemma
trust_remote_code: true

### Training Configuration ###
stage: pt  # pt (CPT), sft, or dpo

### Dataset Configuration ###
dataset: fineweb2_dataset_mri_train
eval_dataset: fineweb2_dataset_mri_test
cutoff_len: 4096

### Training Hyperparameters ###
per_device_train_batch_size: 2
gradient_accumulation_steps: 64
learning_rate: 1.0e-5
num_train_epochs: 4

### Output Configuration ###
output_dir: outputs/my-training
run_name: my-training-run

### Reporting ###
report_to: wandb
```

## Examples

### Multi-dataset Training with Mixing

```bash
byol-train cpt --config configs/cpt.yaml \
  --dataset 'fineweb2_dataset_mri_train,fineweb2_dataset_swahili_train' \
  --mix-strategy interleave_over
```

### Resume from Checkpoint

```bash
byol-train cpt --config configs/cpt.yaml \
  --resume outputs/cpt-training-20250121/checkpoint-1000
```

### Auto-compute Gradient Accumulation

```bash
byol-train sft --config configs/sft.yaml \
  --auto-batch --gpus 0,1,2,3
```

## LoRA Merging

After LoRA training, merge the adapter:

```python
from byol_train import merge_lora

merge_lora(
    base_model="google/gemma-3-4b-pt",
    adapter_path="outputs/sft-lora-64/",
    output_dir="outputs/merged-model/",
)
```

Or use CLI:
```bash
byol-train merge --base google/gemma-3-4b-pt \
  --adapter outputs/sft-lora-64/ \
  --output outputs/merged-model/
```

## Directory Structure

```
train/
├── byol_train/
│   ├── __init__.py    # Package exports
│   ├── config.py      # Configuration dataclasses
│   ├── runner.py      # Training runner
│   ├── merge.py       # LoRA merging utilities
│   └── cli.py         # CLI entry point
├── configs/
│   ├── cpt.yaml       # Continual pre-training config
│   ├── sft.yaml       # Supervised fine-tuning config
│   ├── dpo.yaml       # DPO config
│   └── merge.yaml     # Merge config
├── pyproject.toml
└── README.md
```
