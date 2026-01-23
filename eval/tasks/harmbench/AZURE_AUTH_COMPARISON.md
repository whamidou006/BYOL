# Azure OpenAI Authentication: Fire Detection vs HarmBench

## TL;DR: Why the Difference?

### Fire Detection Project
**No need to export environment variables!** ✅
```bash
# NOT NEEDED for fire detection:
# export AZURE_OPENAI_API_KEY="..."
# export AZURE_OPENAI_ENDPOINT="..."
```

### HarmBench Project  
**Environment variables ARE needed** ⚠️
```bash
# NEEDED for HarmBench:
export AZURE_OPENAI_API_KEY="your-key"
export AZURE_OPENAI_ENDPOINT="your-endpoint"
```

---

## Why the Difference?

### Fire Detection Uses: **Azure Managed Identity** 🔐

**Location:** `/home/whamidouche/ssdprivate/fire_detection/CEVG-RTNet/fire_smoke_training/`

**Authentication Method:**
```python
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

# No API key needed - uses Azure managed identity
token_provider = get_bearer_token_provider(  
    DefaultAzureCredential(),  
    "https://cognitiveservices.azure.com/.default"  
)  

client = AzureOpenAI(  
    api_version="2024-12-01-preview",
    azure_endpoint="https://aiforgoodlab-eu2-openai.openai.azure.com/",
    azure_ad_token_provider=token_provider  # ← Token-based auth
)
```

**How it works:**
1. Code runs on Azure VM with managed identity enabled
2. `DefaultAzureCredential()` automatically gets tokens from Azure metadata service
3. No secrets in environment variables or code
4. Endpoint URL is hardcoded: `aiforgoodlab-eu2-openai.openai.azure.com`

**Advantages:**
- ✅ No secrets to manage
- ✅ Automatic token renewal
- ✅ Works only on Azure VMs (security feature)
- ✅ No environment variables needed

---

### HarmBench Uses: **API Key Authentication** 🔑

**Location:** `/home/whamidouche/ssdprivate/git-local/low-resource-llm.pipeline/multilingual-llm-evaluation/`

**Authentication Method:**
```python
import os
from openai import AzureOpenAI

# Requires API key from environment
api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

client = AzureOpenAI(
    api_key=api_key,  # ← Key-based auth
    azure_endpoint=endpoint,
    api_version="2024-02-15-preview"
)
```

**How it works:**
1. Code expects `AZURE_OPENAI_API_KEY` environment variable
2. Code expects `AZURE_OPENAI_ENDPOINT` environment variable
3. Uses static API key for authentication
4. Can run anywhere (local machine, any cloud, etc.)

**Advantages:**
- ✅ Works on any machine (local, cloud, anywhere)
- ✅ Simpler for testing and development
- ✅ Flexible endpoint configuration
- ⚠️ Requires managing secrets securely

---

## Complete Comparison

| Aspect | Fire Detection | HarmBench |
|--------|---------------|-----------|
| **Authentication** | Azure Managed Identity | API Key |
| **Env Variables** | None needed ✅ | Required ⚠️ |
| **API Key Needed** | No | Yes |
| **Where it runs** | Azure VM only | Anywhere |
| **Endpoint Config** | Hardcoded in code | From environment |
| **Security** | Best (no secrets) | Good (if secrets managed) |
| **Setup Complexity** | Higher (needs Azure VM setup) | Lower (just set env vars) |
| **Token Renewal** | Automatic | Manual (rotate keys) |

---

## File Locations

### Fire Detection Authentication
```
/home/whamidouche/ssdprivate/fire_detection/CEVG-RTNet/fire_smoke_training/
├── auto_tune_training.py  (uses llm_judge_utils)
└── (imports from multilingual-llm-evaluation)

/home/whamidouche/ssdprivate/git-local/low-resource-llm.pipeline/
├── api/
│   └── get_azure_api.py  ← Managed Identity authentication here
└── multilingual-llm-evaluation/
    └── llm_judge_utils.py  ← Calls get_azure_api.py
```

**Key code in `get_azure_api.py`:**
```python
ENDPOINT_CONFIGS = {
    "eu2": {
        "endpoint_name": "aiforgoodlab-eu2-openai",  # ← Hardcoded
        "api_version": "2024-12-01-preview",
        "supported_models": ["gpt-4o", "gpt-5", "gpt-5-chat", ...]
    },
    # ... more endpoints
}

def setup_openai_client(endpoint_type="wu"):
    token_provider = get_bearer_token_provider(  
        DefaultAzureCredential(),  # ← Managed identity
        "https://cognitiveservices.azure.com/.default"  
    )
```

### HarmBench Authentication
```
/home/whamidouche/ssdprivate/git-local/low-resource-llm.pipeline/
└── multilingual-llm-evaluation/
    └── tasks/
        └── harmbench/
            └── metrics_gpt5.py  ← Uses llm_judge_utils
```

**BUT** `metrics_gpt5.py` tries to use the same `llm_judge_utils.py` which uses managed identity!

---

## Important Discovery! 🔍

### Current State of `metrics_gpt5.py`

Looking at our implementation in `metrics_gpt5.py`:

```python
from llm_judge_utils import initialize_endpoint_clients, generate_endpoint_response

def get_gpt5_clients():
    """Initialize and cache GPT-5 API clients using llm_judge_utils."""
    global _gpt5_clients
    
    if _gpt5_clients is None:
        try:
            _gpt5_clients = initialize_endpoint_clients()  # ← Uses managed identity!
            
            if not _gpt5_clients.get("azure_client"):
                print("❌ Failed to initialize Azure client")
                return None
                
        except Exception as e:
            print(f"❌ Error initializing clients: {e}")
            return None
    
    return _gpt5_clients
```

**This means `metrics_gpt5.py` actually uses MANAGED IDENTITY authentication, not API keys!**

---

## Why My Documentation Said to Use Environment Variables

I made an assumption that since this was a standalone evaluation task (not running on the Azure VM), it would need API key authentication. But actually:

**Reality:** `metrics_gpt5.py` uses the SAME `llm_judge_utils.py` as the fire detection project, which means:

1. ✅ It uses **Managed Identity** authentication
2. ✅ **No environment variables needed** if running on the Azure VM
3. ⚠️ **Will NOT work** on local machines or non-Azure environments
4. ✅ Endpoint is hardcoded to `aiforgoodlab-eu2-openai`

---

## Updated Quick Start

### If Running on Azure VM (same as fire detection)
```bash
# No setup needed! Just run:
lm_eval --model hf \
    --model_args pretrained=your-model \
    --tasks harmbench_direct_request \
    --device cuda:0
```

### If Running Locally (NOT on Azure VM)
You have two options:

#### Option A: Won't work with current code
Current `metrics_gpt5.py` won't work locally because it uses managed identity.

#### Option B: Create a modified version for local use
Would need to modify `metrics_gpt5.py` to use API key authentication:

```python
# Instead of:
from llm_judge_utils import initialize_endpoint_clients

# Would need:
import os
from openai import AzureOpenAI

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-02-15-preview"
)
```

---

## Summary: What You Need

### For Fire Detection Auto-Tuning
**Location:** Azure VM  
**Auth:** Managed Identity  
**Setup:** None needed ✅

```bash
cd /home/whamidouche/ssdprivate/fire_detection/CEVG-RTNet/fire_smoke_training
python auto_tune_training.py --iterations 50
# No env vars needed!
```

### For HarmBench Evaluation
**Location:** Azure VM (same machine)  
**Auth:** Managed Identity (same as fire detection!)  
**Setup:** None needed ✅

```bash
cd /home/whamidouche/ssdprivate/git-local/low-resource-llm.pipeline/multilingual-llm-evaluation
lm_eval --model hf --model_args pretrained=... --tasks harmbench_direct_request
# No env vars needed either!
```

### Both use the same authentication system!

---

## Correction to QUICK_START_GPT5.md

The section about exporting environment variables was based on an incorrect assumption. 

**The correct statement is:**

Since you're running on the same Azure VM as the fire detection project, and both use the same `llm_judge_utils.py` with managed identity authentication, **you don't need to export any environment variables**.

The authentication is handled automatically by Azure's managed identity service.

---

## How to Verify Authentication Method

Run this to see what authentication your code uses:

```bash
cd /home/whamidouche/ssdprivate/git-local/low-resource-llm.pipeline/multilingual-llm-evaluation
python3 << 'PYEOF'
from llm_judge_utils import initialize_endpoint_clients
import inspect

# Get the source of initialize_endpoint_clients
source = inspect.getsource(initialize_endpoint_clients)

# Check for authentication type
if "DefaultAzureCredential" in source:
    print("✅ Uses MANAGED IDENTITY authentication")
    print("   No environment variables needed!")
elif "os.getenv" in source or "api_key" in source:
    print("⚠️  Uses API KEY authentication")
    print("   Environment variables required:")
    print("   - AZURE_OPENAI_API_KEY")
    print("   - AZURE_OPENAI_ENDPOINT")
else:
    print("❓ Unknown authentication method")

print("\nAuthentication details:")
print(source[:500])
PYEOF
```

---

## Conclusion

**Your fire detection project uses Azure Managed Identity, which is already set up on your Azure VM. The HarmBench evaluation uses the EXACT SAME authentication system via `llm_judge_utils.py`, so you don't need to set any environment variables.**

The instructions in `QUICK_START_GPT5.md` about exporting `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_ENDPOINT` were incorrect and unnecessary for your setup.

**Just run the evaluation - it will work automatically!** ✅
