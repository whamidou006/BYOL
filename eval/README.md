# BYOL Evaluation Framework

A unified evaluation framework for LLM evaluation using lm-evaluation-harness and LLM-as-Judge.

## Features

- **Benchmark evaluation** via lm-evaluation-harness
- **LLM-as-Judge** evaluation using configurable judge models
- **Secure secrets management** for HuggingFace tokens
- **Type-safe configuration** with dataclasses
- **CLI and Python API** interfaces

## Installation

```bash
cd eval
pip install -r requirements.txt
pip install -e .
```

This will install:
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- byol-eval package

## Secrets Management

The framework supports multiple ways to provide HuggingFace tokens:

1. **Environment variables** (recommended for CI/CD):
   ```bash
   export HF_TOKEN="your-huggingface-token"
   # or
   export HUGGING_FACE_HUB_TOKEN="your-huggingface-token"
   ```

2. **`.env` file** (recommended for development):
   Create `.env` in the eval directory:
   ```bash
   HF_TOKEN=your-huggingface-token
   ```

3. **HuggingFace CLI cache** (`~/.huggingface/token`)

## Quick Start

### Using CLI

```bash
# Run benchmark evaluation
byol-eval benchmark \
  --model hf \
  --model-args pretrained=google/gemma-3-4b \
  --tasks mmlu,global_mmlu_mri \
  --gpus 0,1

# Run LLM-as-Judge evaluation
byol-eval judge \
  --config configs/judge.yaml \
  --result-file results/model_outputs.json

# Evaluate from config file
byol-eval benchmark --config configs/eval.yaml
```

### Using Python API

```python
from byol_eval import EvalConfig, EvaluationRunner

# Load config from YAML
config = EvalConfig.from_yaml("configs/eval.yaml")

# Or create config programmatically
config = EvalConfig(
    model="hf",
    model_args="pretrained=google/gemma-3-4b-pt,dtype=bfloat16",
    tasks=["mmlu", "global_mmlu_mri"],
    batch_size="auto:4",
    gpus="0,1",
)

# Run evaluation
runner = EvaluationRunner(config)
result = runner.run()

if result.success:
    print(f"✅ Evaluation completed")
    print(f"Results: {result.results}")
else:
    print(f"❌ Evaluation failed: {result.error}")
```

## CLI Reference

### Commands

| Command | Description |
|---------|-------------|
| `byol-eval benchmark` | Run lm-evaluation-harness benchmarks |
| `byol-eval judge` | Run LLM-as-Judge evaluation |

### Benchmark Options

| Option | Description |
|--------|-------------|
| `-c, --config` | Path to YAML configuration file |
| `-m, --model` | Model type (hf, vllm, etc.) |
| `--model-args` | Model arguments (key=value format) |
| `-t, --tasks` | Comma-separated task names |
| `-g, --gpus` | Comma-separated GPU IDs (default: 0) |
| `-b, --batch-size` | Batch size (default: auto:4) |
| `-o, --output-dir` | Results output directory |
| `--num-fewshot` | Number of few-shot examples |
| `--log-samples` | Log individual samples |

### Judge Options

| Option | Description |
|--------|-------------|
| `-c, --config` | Path to judge configuration |
| `--result-file` | Path to model outputs JSON |
| `--judge-model` | Judge model ID |
| `--judge-prompt` | Custom judge prompt |
| `-o, --output-dir` | Results output directory |

## Configuration Files

### Benchmark Config (`configs/eval.yaml`)

```yaml
model: hf
model_args: pretrained=google/gemma-3-4b-pt,dtype=bfloat16,trust_remote_code=True
tasks:
  - mmlu
  - global_mmlu_mri
  - hellaswag
batch_size: auto:4
gpus: "0,1,2,3"
num_fewshot: 5
output_base_path: results/
log_samples: true
```

### Judge Config (`configs/judge.yaml`)

```yaml
judge_model: openai/gpt-4o
judge_prompt_template: |
  Evaluate the following response for accuracy and helpfulness.
  Question: {question}
  Response: {response}
  Reference: {reference}
  
  Score (1-5):
criteria:
  accuracy: 0.4
  helpfulness: 0.3
  fluency: 0.3
```

## Project Structure

```
eval/
├── byol_eval/
│   ├── __init__.py       # Package exports
│   ├── cli.py            # CLI entry point
│   ├── config.py         # Configuration dataclasses
│   ├── constants.py      # Default values and status codes
│   ├── secrets.py        # HuggingFace token management
│   ├── runner.py         # Evaluation runner
│   ├── harness.py        # lm-eval-harness wrapper
│   ├── judge.py          # LLM-as-Judge implementation
│   └── py.typed          # PEP 561 type marker
├── configs/
│   ├── eval.yaml         # Benchmark evaluation config
│   └── judge.yaml        # LLM-as-Judge config
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Constants Reference

Default values are defined in `constants.py`:

| Constant | Value | Description |
|----------|-------|-------------|
| `DEFAULT_GPUS` | "0" | Default GPU IDs |
| `DEFAULT_BATCH_SIZE` | "auto:4" | Auto batch sizing |
| `DEFAULT_NUM_FEWSHOT` | 0 | Default few-shot count |
| `STATUS_SUCCESS` | "success" | Success status code |
| `STATUS_FAILED` | "failed" | Failure status code |
| `STATUS_SKIPPED` | "skipped" | Skipped status code |

## Supported Tasks

The framework supports all tasks from lm-evaluation-harness:

- **MMLU**: Massive Multitask Language Understanding
- **Global MMLU**: Multilingual MMLU variants
- **HellaSwag**: Commonsense reasoning
- **ARC**: AI2 Reasoning Challenge
- **WinoGrande**: Winograd schema challenge
- **TruthfulQA**: Truthfulness evaluation
- Custom tasks via lm-eval-harness

## License

See main BYOL repository.
