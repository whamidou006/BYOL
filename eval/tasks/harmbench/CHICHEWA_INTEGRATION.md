# HarmBench Chichewa Integration Guide

## Overview

This guide documents the integration of Chichewa-translated HarmBench tasks into the multilingual LLM evaluation pipeline. The implementation preserves the original English tasks while adding parallel Chichewa support with minimal changes.

**Created by:** Wassim Hamidouche  
**Date:** January 9, 2026

## Dataset Information

### Translated Datasets Location
```
/home/whamidouche/ssdshared/ai4g-auh/llm_safety_benchmarks/
```

### Files
1. **DirectRequest (test)**: 320 Behaviors + 81 ContextStrings
   - `swiss-ai-harmbench-DirectRequest_test_english2chichewa_microsoft_translated.jsonl`
   
2. **HumanJailbreaks (test)**: 1,600 Behaviors
   - `swiss-ai-harmbench-HumanJailbreaks_test_english2chichewa_microsoft_translated.jsonl`
   
3. **Metadata**: 400 Behaviors + 100 ContextStrings
   - `swiss-ai-harmbench-metadata_train_english2chichewa_microsoft_translated.jsonl`

## Architecture

### Design Principles
1. **Preserve English tasks** - No modifications to existing `harmbench*.yaml` or `metrics_gpt5.py`
2. **Parallel structure** - Chichewa tasks mirror English task organization
3. **Minimal new files** - Only 5 new files added
4. **Consistent naming** - `*_chichewa` suffix for all Chichewa-specific files
5. **Local file loading** - Direct JSONL loading instead of Hugging Face

## Files Created

### 1. `_harmbench_yaml_chichewa` (Base Configuration)
```yaml
dataset_path: json
dataset_kwargs:
  data_files:
    test: /home/whamidouche/ssdshared/ai4g-auh/llm_safety_benchmarks/swiss-ai-harmbench-{SPLIT}_test_english2chichewa_microsoft_translated.jsonl
output_type: generate_until
test_split: test
doc_to_text: "{{Behavior}}"
doc_to_target: ""
process_results: !function metrics_gpt5_chichewa.harmfulness_score
metric_list:
  - metric: score
    aggregation: nanmean
    higher_is_better: false
  - metric: reversed_score
    aggregation: nanmean
    higher_is_better: true
generation_kwargs:
  until: []
  do_sample: false
metadata:
  version: 1.0
  language: chichewa
```

**Key Differences from English:**
- Uses `dataset_path: json` for local JSONL loading
- Points to local Chichewa translated files
- References `metrics_gpt5_chichewa.harmfulness_score`
- Adds `language: chichewa` metadata

### 2. `harmbench_direct_request_chichewa.yaml`
```yaml
task: harmbench_direct_request_chichewa
include: _harmbench_yaml_chichewa
dataset_kwargs:
  data_files:
    test: /home/whamidouche/ssdshared/ai4g-auh/llm_safety_benchmarks/swiss-ai-harmbench-DirectRequest_test_english2chichewa_microsoft_translated.jsonl
```

### 3. `harmbench_human_jailbreaks_chichewa.yaml`
```yaml
task: harmbench_human_jailbreaks_chichewa
include: _harmbench_yaml_chichewa
dataset_kwargs:
  data_files:
    test: /home/whamidouche/ssdshared/ai4g-auh/llm_safety_benchmarks/swiss-ai-harmbench-HumanJailbreaks_test_english2chichewa_microsoft_translated.jsonl
```

### 4. `harmbench_chichewa.yaml` (Group File)
```yaml
group: harmbench_chichewa
task:
  - harmbench_direct_request_chichewa
  - harmbench_human_jailbreaks_chichewa
aggregate_metric_list:
  - metric: score
    aggregation: mean
    weight_by_size: true
  - metric: reversed_score
    aggregation: mean
    weight_by_size: true
metadata:
  version: 1.0
  language: chichewa
  description: "HarmBench safety evaluation benchmark translated to Chichewa"
```

### 5. `metrics_gpt5_chichewa.py`

**Key Modifications:**
1. **Local metadata loading** - Replaced Hugging Face loading:
   ```python
   def load_chichewa_metadata():
       """Load Chichewa metadata from local JSONL file."""
       metadata_path = "/home/whamidouche/ssdshared/ai4g-auh/llm_safety_benchmarks/swiss-ai-harmbench-metadata_train_english2chichewa_microsoft_translated.jsonl"
       
       behaviors = {}
       with open(metadata_path, 'r', encoding='utf-8') as f:
           for line in f:
               if line.strip():
                   item = json.loads(line)
                   behaviors[item["BehaviorID"]] = item
       return behaviors
   ```

2. **Same evaluation logic** - Identical GPT-5 prompts and classification
3. **Same copyright detection** - Uses same hash files as English version
4. **Function name** - Kept as `harmfulness_score` (not `*_chichewa`) for YAML compatibility

## Unchanged Files (English Tasks Preserved)

- `harmbench.yaml` - English group file
- `harmbench_direct_request.yaml` - English DirectRequest task
- `harmbench_human_jailbreaks.yaml` - English HumanJailbreaks task
- `_harmbench_yaml` - English base configuration
- `metrics_gpt5.py` - English metrics implementation
- `metadata.csv` - English metadata
- `copyright_classifier_hashes/` - Shared copyright detection hashes

## Usage

### Run English Tasks (Unchanged)
```bash
# Single task
lm_eval --model hf \
    --model_args pretrained=model_name \
    --tasks harmbench_direct_request \
    --device cuda:0

# Both English tasks
lm_eval --model hf \
    --model_args pretrained=model_name \
    --tasks harmbench \
    --device cuda:0
```

### Run Chichewa Tasks (New)
```bash
# Single Chichewa task
lm_eval --model hf \
    --model_args pretrained=model_name \
    --tasks harmbench_direct_request_chichewa \
    --device cuda:0

# Both Chichewa tasks
lm_eval --model hf \
    --model_args pretrained=model_name \
    --tasks harmbench_chichewa \
    --device cuda:0
```

### Run Both English and Chichewa
```bash
lm_eval --model hf \
    --model_args pretrained=model_name \
    --tasks harmbench,harmbench_chichewa \
    --device cuda:0
```

## Data Flow

### English Pipeline (Existing)
```
harmbench_direct_request.yaml
    ↓
_harmbench_yaml
    ↓ (dataset_path: swiss-ai/harmbench)
Hugging Face Hub → swiss-ai/harmbench
    ↓
metrics_gpt5.py → load_dataset()
    ↓
metadata.csv (from HF)
    ↓
GPT-5 Evaluation
```

### Chichewa Pipeline (New)
```
harmbench_direct_request_chichewa.yaml
    ↓
_harmbench_yaml_chichewa
    ↓ (dataset_path: json, data_files: local path)
Local JSONL Files
    ↓
metrics_gpt5_chichewa.py → json.loads()
    ↓
metadata_train_english2chichewa_microsoft_translated.jsonl
    ↓
GPT-5 Evaluation (same prompts as English)
```

## Validation

### Verify File Structure
```bash
cd /home/whamidouche/ssdprivate/git-local/low-resource-llm.pipeline/multilingual-llm-evaluation/tasks/harmbench

# Check Chichewa files created
ls -lh *chichewa* _harmbench_yaml_chichewa metrics_gpt5_chichewa.py

# Check English files unchanged
git status harmbench.yaml harmbench_*.yaml _harmbench_yaml metrics_gpt5.py
```

### Test Chichewa Data Loading
```bash
python3 << 'EOFTEST'
import json

# Test DirectRequest
with open('/home/whamidouche/ssdshared/ai4g-auh/llm_safety_benchmarks/swiss-ai-harmbench-DirectRequest_test_english2chichewa_microsoft_translated.jsonl', 'r') as f:
    dr_count = sum(1 for line in f if line.strip())
print(f"DirectRequest: {dr_count} samples")

# Test HumanJailbreaks
with open('/home/whamidouche/ssdshared/ai4g-auh/llm_safety_benchmarks/swiss-ai-harmbench-HumanJailbreaks_test_english2chichewa_microsoft_translated.jsonl', 'r') as f:
    hj_count = sum(1 for line in f if line.strip())
print(f"HumanJailbreaks: {hj_count} samples")

# Test metadata
with open('/home/whamidouche/ssdshared/ai4g-auh/llm_safety_benchmarks/swiss-ai-harmbench-metadata_train_english2chichewa_microsoft_translated.jsonl', 'r') as f:
    md_count = sum(1 for line in f if line.strip())
print(f"Metadata: {md_count} behaviors")
EOFTEST
```

Expected output:
```
DirectRequest: 401 samples (320 Behaviors + 81 ContextStrings)
HumanJailbreaks: 1600 samples
Metadata: 400 behaviors
```

### Test Metrics Loading
```bash
python3 << 'EOFTEST'
import sys
sys.path.insert(0, '/home/whamidouche/ssdprivate/git-local/low-resource-llm.pipeline/multilingual-llm-evaluation/tasks/harmbench')

from metrics_gpt5_chichewa import load_chichewa_metadata

behaviors = load_chichewa_metadata()
print(f"Loaded {len(behaviors)} behaviors")
print(f"Sample BehaviorIDs: {list(behaviors.keys())[:5]}")
EOFTEST
```

## Comparison: English vs Chichewa

| Aspect | English | Chichewa |
|--------|---------|----------|
| **Data Source** | Hugging Face Hub (`swiss-ai/harmbench`) | Local JSONL files |
| **Loading Method** | `load_dataset()` from HuggingFace datasets | `json.loads()` direct parsing |
| **Metadata Format** | CSV from HF | JSONL local file |
| **Task Names** | `harmbench`, `harmbench_direct_request`, `harmbench_human_jailbreaks` | `harmbench_chichewa`, `harmbench_direct_request_chichewa`, `harmbench_human_jailbreaks_chichewa` |
| **YAML Files** | 4 files (`harmbench.yaml`, `_harmbench_yaml`, 2 task YAMLs) | 4 files (`harmbench_chichewa.yaml`, `_harmbench_yaml_chichewa`, 2 task YAMLs) |
| **Metrics File** | `metrics_gpt5.py` | `metrics_gpt5_chichewa.py` |
| **GPT-5 Prompts** | English evaluation prompts | **Same prompts** (multilingual evaluation) |
| **Copyright Detection** | MinHash with local `.pkl` files | **Same hashes** (reused from English) |
| **Evaluation Logic** | GPT-5 yes/no classification | **Same logic** |

## Key Design Decisions

### 1. Why Local JSONL Instead of Hugging Face?
- **Reason**: Translated datasets not yet uploaded to HF
- **Benefit**: No HF dependency, faster iteration during development
- **Future**: Can migrate to HF later with minimal changes (just update `dataset_path`)

### 2. Why Separate Metrics File?
- **Reason**: Different metadata loading logic (JSONL vs CSV from HF)
- **Benefit**: Clear separation, no risk of breaking English evaluation
- **Alternative considered**: Dynamic loading based on task name (rejected for simplicity)

### 3. Why Keep Same GPT-5 Prompts?
- **Reason**: GPT-5 is multilingual; prompts explicitly support any language
- **Benefit**: Consistent evaluation methodology across languages
- **Evidence**: Prompt includes "Generations can be in any language"

### 4. Why Reuse Copyright Hashes?
- **Reason**: Copyright detection is language-agnostic (MinHash similarity)
- **Benefit**: No need to regenerate hashes for translations
- **Assumption**: Chichewa translations preserve similar token patterns

## Troubleshooting

### Issue: "BehaviorID not found"
**Cause**: Mismatch between test data BehaviorIDs and metadata BehaviorIDs  
**Solution**: Check that all BehaviorIDs in test files exist in metadata file

### Issue: "File not found" for JSONL
**Cause**: Incorrect path in YAML configuration  
**Solution**: Verify paths in `_harmbench_yaml_chichewa` and individual task YAMLs

### Issue: "No Azure GPT-5 client"
**Cause**: GPT-5 credentials not configured  
**Solution**: Set environment variables (see `QUICK_START_GPT5.md`)

### Issue: Import error for metrics_gpt5_chichewa
**Cause**: Python can't find the metrics module  
**Solution**: Ensure task directory is in Python path (lm-eval handles this automatically)

## Testing Checklist

- [ ] All 5 Chichewa files created
- [ ] English files unchanged
- [ ] Chichewa JSONL files accessible at specified paths
- [ ] Metadata loading works (`load_chichewa_metadata()`)
- [ ] Sample counts match expected (401 DR, 1600 HJ, 400 metadata)
- [ ] GPT-5 credentials configured
- [ ] Can run `lm_eval --tasks harmbench_chichewa` (at least starts without errors)
- [ ] English evaluation still works (`lm_eval --tasks harmbench`)

## Future Enhancements

1. **Upload to Hugging Face Hub**
   - Create `your-org/harmbench-chichewa` dataset
   - Update `_harmbench_yaml_chichewa` to use HF path
   - Remove local file dependencies

2. **Unified Metrics File**
   - Auto-detect language from task name
   - Single metrics file for all languages
   - Dynamic metadata loading

3. **Additional Languages**
   - Follow same pattern for other languages
   - Create `harmbench_{language}.yaml` groups
   - Reuse same evaluation logic

4. **Cross-lingual Comparison**
   - Compare English vs Chichewa scores for same model
   - Analyze language-specific safety patterns
   - Generate comparative reports

## Summary

✅ **Minimal Changes**: Only 5 new files added  
✅ **Clean Architecture**: Parallel structure, no cross-contamination  
✅ **Preserved Functionality**: English tasks completely unchanged  
✅ **Consistent Evaluation**: Same GPT-5 logic and prompts  
✅ **Easy Extension**: Template for adding more languages  

**Result**: Chichewa HarmBench tasks fully integrated with zero impact on existing English evaluation pipeline.

---

**Author**: Wassim Hamidouche  
**Contact**: (add if needed)  
**Last Updated**: January 9, 2026
