# BYOL Training Framework

A production-ready Python wrapper for [LlamaFactory](https://github.com/hiyouga/LLaMAFactory) training.

Supports **CPT** (Continual Pre-Training), **SFT** (Supervised Fine-Tuning), and **DPO** (Direct Preference Optimization).

---

## Quick Install

```bash
# 1. Create conda environment
conda create -p ~/ssdprivate/conda_envs/byol_train python=3.12 -y
conda activate ~/ssdprivate/conda_envs/byol_train

# 2. Clone repository
git clone https://github.com/your-org/BYOL.git
cd BYOL

# 3. Install LlamaFactory
git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git
cd LlamaFactory && pip install -e ".[torch,metrics]" && cd ..

# 4. Install BYOL Train
cd train
pip install -r requirements.txt
pip install -e .

# 5. Fix Gemma 3 compatibility
pip install transformers==4.51.3
```

### DeepSpeed (Optional)

```bash
conda install -y -c nvidia cuda-toolkit=12.8
export CUDA_HOME=$CONDA_PREFIX
echo 'export CUDA_HOME=$CONDA_PREFIX' >> ~/.bashrc
pip install deepspeed
```

---

## Usage

### Run Training

```bash
# CPT (auto-loads configs/cpt.yaml)
byol-train cpt --gpus 0,1

# CPT with overrides
byol-train cpt \
  --model google/gemma-3-4b-pt \
  --dataset chichewa_cpt \
  --gpus 2,3 \
  --epochs 3 \
  --batch-size 1 \
  --grad-accum 256 \
  --lr 1e-5

# SFT with LoRA
byol-train sft --model google/gemma-3-4b-it --dataset alpaca --lora --lora-rank 64

# DPO
byol-train dpo --gpus 0,1,2,3

# Dry run (preview only)
byol-train sft --dry-run
```

### Merge LoRA

```bash
# Basic merge (via LlamaFactory)
byol-train merge \
  --base-model google/gemma-3-4b-pt \
  --adapter outputs/sft-lora/checkpoint-final \
  --output outputs/merged-model

# Advanced merge options (via python -m)
# Simple LoRA merge
python -m byol_train.merge simple-lora \
  --base google/gemma-3-4b-pt \
  --lora outputs/lora-adapter \
  --output outputs/merged

# Delta merge: instruct + alpha*(fine_tuned - base)
python -m byol_train.merge delta \
  --base google/gemma-3-4b-pt \
  --instruct google/gemma-3-4b-it \
  --fine-tuned outputs/cpt-model \
  --alpha 0.5 \
  --output outputs/merged
```

---

## CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `-c, --config` | YAML config file | Auto-detected |
| `-m, --model` | Model ID or path | - |
| `-d, --dataset` | Dataset name | - |
| `-g, --gpus` | GPU IDs | `0` |
| `-e, --epochs` | Training epochs | `3` |
| `-b, --batch-size` | Batch size | `4` |
| `--grad-accum` | Gradient accumulation | `4` |
| `--lr` | Learning rate | `5e-5` |
| `--lora` | Enable LoRA | `False` |
| `--dry-run` | Preview config | `False` |

---

## Configuration Files

Pre-configured YAML files in `configs/`:
- `cpt.yaml` - Continual pre-training
- `sft.yaml` - Supervised fine-tuning  
- `dpo.yaml` - Direct preference optimization

DeepSpeed configs in `examples/deepspeed/`:
- `ds_z2_config.json` - ZeRO Stage 2 (recommended)
- `ds_z3_config.json` - ZeRO Stage 3 (max memory savings)

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `token_type_ids required` | `pip install transformers==4.51.3` |
| `CUDA_HOME does not exist` | `export CUDA_HOME=$CONDA_PREFIX` |
| `--gpu not recognized` | Use `--gpus` (with 's') |
| `OOM error` | Reduce `--batch-size`, increase `--grad-accum` |

---

## License

See main BYOL repository.
