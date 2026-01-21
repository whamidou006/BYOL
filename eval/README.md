# BYOL Evaluation Framework (v2)

A production-ready CLI for **benchmark** (lm-eval) and **LLM-as-Judge** evaluations.

## Quick Start

```bash
# Setup
conda create -n byol_eval python=3.12 -y
conda activate byol_eval
pip install -r requirements.txt
pip install -e .

# Run evaluation
byol-eval -c configs/benchmark_base_mri.yaml
```

## Usage

### Benchmark Evaluation

```bash
# Base models (few-shot, NO chat template)
byol-eval -c configs/benchmark_base_mri.yaml
byol-eval -c configs/benchmark_base_nya.yaml

# Instruct models (0-shot WITH chat template)
byol-eval -c configs/benchmark_instruct_mri.yaml
byol-eval -c configs/benchmark_instruct_nya.yaml

# Safety benchmarks
byol-eval -c configs/benchmark_safety.yaml

# Quick single-task run
byol-eval -m google/gemma-3-4b-pt -t global_mmlu_en -g 0

# Dry run (print commands only)
byol-eval -c configs/benchmark_base_mri.yaml --dry-run
```

### LLM-as-Judge

```bash
byol-eval judge
byol-eval judge -m configs/judge_models.yaml -d configs/judge_datasets.yaml
```

## Configuration

### Benchmark Configs

| Config | Type | Tasks | Chat Template |
|--------|------|-------|---------------|
| `benchmark_base_mri.yaml` | Base | 29 | ❌ No |
| `benchmark_base_nya.yaml` | Base | 29 | ❌ No |
| `benchmark_instruct_mri.yaml` | Instruct | 35 | ✅ Yes |
| `benchmark_instruct_nya.yaml` | Instruct | 35 | ✅ Yes |
| `benchmark_safety.yaml` | Safety | 5 | ❌ No |

### Key Differences: Base vs Instruct

| Aspect | Base Models | Instruct Models |
|--------|-------------|-----------------|
| Chat Template | `apply_chat_template: false` | `apply_chat_template: true` |
| Few-shot | 1-25 shots per task | 0-shot |
| Model Type | Pretrained (`-pt`) | Instruction-tuned (`-it`) |
| Extra Tasks | - | ifeval, humaneval, arc_challenge_chat |

### Config Structure

```yaml
evaluation:
  results_dir: "results/base_mri"
  gpus: "0"
  batch_size: auto:4
  apply_chat_template: false  # Global setting

models:
  - name: "gemma-3-4b-pt"
    path: "google/gemma-3-4b-pt"
    dtype: "bfloat16"
    trust_remote_code: true

lm_eval:
  include_path: "../eval/tasks"  # Custom task definitions
  log_samples: false

tasks:
  - name: "global_mmlu_en"
    num_fewshot: 5
    enabled: true
    apply_chat_template: false  # Task-level override
```

## CLI Reference

```
byol-eval [OPTIONS]

Options:
  -c, --config FILE       YAML configuration file
  -m, --model PATH        Model path or HuggingFace ID
  -t, --tasks TASKS       Comma-separated task names
  -n, --num-fewshot N     Few-shot examples (default: 0)
  -g, --gpus IDS          GPU device IDs (default: 0)
  -b, --batch-size SIZE   Batch size (default: auto:4)
  -o, --output-dir DIR    Output directory (default: results)
  --tasks-path PATH       Custom task definitions path
  --dry-run               Print commands without executing
  --log-samples           Log evaluation samples
```

## Project Structure

```
eval_v2/
├── byol_eval/
│   ├── __init__.py      # Package exports
│   ├── cli.py           # CLI entry point
│   ├── config.py        # Configuration dataclasses
│   ├── judge.py         # LLM-as-Judge wrapper
│   └── runner.py        # Evaluation runner
├── configs/
│   ├── benchmark_*.yaml # Benchmark configs
│   ├── judge_models.yaml
│   └── judge_datasets.yaml
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Behavior Alignment

This framework generates identical `lm_eval` commands as the original `eval/` code:

```bash
# Original (eval/)
python eval/src/benchmark_evaluation/run_evaluation.py --config config.yaml

# New (eval_v2/)
byol-eval -c config.yaml
```

Both produce equivalent commands:
```bash
python -m lm_eval \
  --model hf \
  --model_args pretrained=MODEL,dtype=bfloat16,trust_remote_code=true \
  --tasks TASKS \
  --batch_size auto:4 \
  --num_fewshot N \
  --include_path /path/to/tasks \
  [--apply_chat_template]  # Only for instruct configs
```

## Dependencies

- Python 3.12+
- lm-eval (lm-evaluation-harness)
- PyYAML
- torch, transformers, accelerate

## License

See main BYOL repository.
