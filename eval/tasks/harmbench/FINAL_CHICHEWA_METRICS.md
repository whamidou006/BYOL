# Chichewa Metrics - Exact Copy of English Version

**Date**: January 9, 2026  
**Status**: ✅ CORRECTED - Now exactly matches English implementation

## What Changed

### Initial Version (Incorrect)
- Simplified GPT-5 function
- Simplified copyright hashing
- Missing helper functions

### Final Version (Correct)
**Exact copy of `metrics_gpt5.py` with ONLY ONE difference: metadata loading**

## The ONLY Difference

### English Version (metrics_gpt5.py)
```python
def harmfulness_score(doc, predictions, **kwargs):
    hf_repo = "swiss-ai/harmbench"
    completion = predictions[0]
    metadata_path = "metadata.csv"
    copyright_path = "..."
    
    # Load behaviors
    behaviors = load_dataset(hf_repo, data_files=metadata_path, split="train")
    behaviors = {b["BehaviorID"]: b for b in behaviors}
    
    # ... rest of function IDENTICAL ...
```

### Chichewa Version (metrics_gpt5_chichewa.py)
```python
def load_chichewa_metadata():
    """Load Chichewa metadata from local JSONL file."""
    metadata_path = "/home/whamidouche/ssdshared/ai4g-auh/llm_safety_benchmarks/swiss-ai-harmbench-metadata_train_english2chichewa_microsoft_translated.jsonl"
    
    import json
    behaviors = {}
    with open(metadata_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                behaviors[item["BehaviorID"]] = item
    
    eval_logger.info(f"Loaded {len(behaviors)} behaviors from Chichewa metadata")
    return behaviors


def harmfulness_score(doc, predictions, **kwargs):
    completion = predictions[0]
    copyright_path = "..."
    
    # Load behaviors from local Chichewa JSONL
    behaviors = load_chichewa_metadata()
    
    # ... rest of function IDENTICAL TO ENGLISH ...
```

## Everything Else is IDENTICAL

✅ **GPT-5 prompts** - Same  
✅ **compute_results_gpt5()** - Same  
✅ **compute_results_hashing()** - Same  
✅ **compute_hashes_with_sliding_window()** - Same  
✅ **check_output_with_sliding_window()** - Same  
✅ **tokenize()** - Same  
✅ **get_gpt5_clients()** - Same  
✅ **All constants** - Same  
✅ **All imports** - Same  

## Verification

```bash
# Check differences
diff metrics_gpt5.py metrics_gpt5_chichewa.py

# Should show ONLY:
# 1. Addition of load_chichewa_metadata() function
# 2. Removal of hf_repo and metadata_path variables
# 3. Replacement of load_dataset() with load_chichewa_metadata()
```

## Why This Approach

**Principle**: Don't change what works!

The English implementation is:
- ✅ **Battle-tested** - Used in production
- ✅ **Correct** - Handles all edge cases
- ✅ **Complete** - All functions included

By copying it exactly, we ensure:
- ✅ **Same behavior** - Chichewa evaluated identically to English
- ✅ **Same bugs** - If English has issues, both have issues (easier to fix once)
- ✅ **Same performance** - No unexpected differences
- ✅ **Easy maintenance** - Changes to English can be copied to Chichewa

## File Structure

```python
# metrics_gpt5_chichewa.py

[All imports - SAME]

[GPT5_HARM_PROMPT - SAME]

[get_gpt5_clients() - SAME]

[tokenize() - SAME]

[load_chichewa_metadata() - NEW FUNCTION]  # ← ONLY NEW ADDITION

[harmfulness_score() - SAME except uses load_chichewa_metadata()]  # ← ONLY CHANGE

[compute_results_gpt5() - SAME]

[compute_results_hashing() - SAME]

[compute_hashes_with_sliding_window() - SAME]

[check_output_with_sliding_window() - SAME]
```

## Testing

```bash
# Should work identically to English version
lm_eval --model hf \
    --model_args pretrained=YOUR_MODEL \
    --tasks harmbench_chichewa \
    --device cuda:0
```

## Summary

✅ **Exact copy** of English implementation  
✅ **Only difference** is metadata loading (local JSONL vs Hugging Face)  
✅ **All evaluation logic** identical  
✅ **All helper functions** identical  
✅ **All prompts** identical  

**Result**: Chichewa and English evaluations are 100% consistent in methodology, differing only in the language of the prompts being evaluated.

---

**Files**:
- `metrics_gpt5.py` - English version (unchanged)
- `metrics_gpt5_chichewa.py` - Chichewa version (exact copy with local loading)
- `metrics_gpt5_chichewa.py.broken` - Previous incorrect version (archived)
