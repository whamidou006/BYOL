# Issue with model_name in metrics_gpt5.py

## Problem Found 🔍

The code has a mismatch in how the model is specified:

### Current Code (INCORRECT):
```python
model_config = {
    "model_name": model_name,  # ← Wrong key!
    "temperature": temperature,
    "max_tokens": max_tokens,
    "top_p": 1.0,
    "backend": "azure"
}
```

### What generate_endpoint_response expects:
```python
# From llm_judge_utils.py line ~205:
endpoint = model_config.get('endpoint') or model_config.get('path') or model_config.get('model')
#                             ↑                              ↑                           ↑
# Looks for these keys, but NOT "model_name"!
```

## The Fix

Change `"model_name"` to `"model"`:

```python
model_config = {
    "model": model_name,  # ← Correct key!
    "temperature": temperature,
    "max_tokens": max_tokens,
    "top_p": 1.0,
}
```

Also remove `"backend": "azure"` - it's not used by the function.

## What Models Are Available

From `get_azure_api.py` and `llm_judge_utils.py`:

```python
AZURE_MODELS = {
    "gpt-35-turbo", "gpt-35-turbo-16k",
    "gpt-4-turbo", "gpt-4-vision", 
    "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", 
    "gpt-4o",  # ← Current default
    "gpt-5", "gpt-5-chat",  # ← What you want
    "gpt-5-mini", "gpt-5-nano",
    "o3",
}
```

## To Use gpt-5-chat as Default

Change line 160 in metrics_gpt5.py:

**Before:**
```python
model_name=kwargs.get("model_name", "gpt-4o"),
```

**After:**
```python
model_name=kwargs.get("model_name", "gpt-5-chat"),
```

## Complete Fix

Two changes needed:

1. **Line 160**: Change default from `"gpt-4o"` to `"gpt-5-chat"`
2. **Line 211**: Change `"model_name"` to `"model"` in model_config dict

## Why This Matters

- With `"model_name"`: The function gets `None` for endpoint → Error
- With `"model"`: The function gets `"gpt-5-chat"` → Works correctly
- Current default `"gpt-4o"` is cheaper but less capable than `"gpt-5-chat"`

## Cost Comparison

| Model | Input Cost | Output Cost | Quality |
|-------|-----------|-------------|---------|
| gpt-4o | $2.50/1M tokens | $10/1M tokens | Good |
| gpt-5-chat | $5/1M tokens | $15/1M tokens | Better |

For 1,529 behaviors with ~500 tokens each:
- gpt-4o: ~$4 total
- gpt-5-chat: ~$6 total

Worth the $2 extra for better quality!
