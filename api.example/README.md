# API Configuration

This folder contains the LLM API client configuration for BYOL's LLM-as-Judge evaluation.

## Setup

1. Copy the example folder:
   ```bash
   cp -r api.example api
   ```

2. Configure your API credentials:

   **Option A: Environment Variables (recommended)**
   ```bash
   # Azure OpenAI
   export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
   export AZURE_OPENAI_API_KEY="your-api-key"
   export AZURE_OPENAI_API_VERSION="2024-12-01-preview"
   
   # Or direct OpenAI
   export OPENAI_API_KEY="your-openai-key"
   ```

   **Option B: Edit the config file**
   Edit `api/get_azure_api.py` directly with your credentials.

3. Customize endpoints in `ENDPOINT_CONFIGS` for your setup.

## Required Interface

The `get_azure_api.py` module must provide:

```python
# Client setup
def setup_client(endpoint_type: str) -> Any
def get_client_for_model(model_name: str) -> Tuple[client, endpoint_type]

# Configuration
ENDPOINT_CONFIGS: Dict[str, Dict]  # Endpoint configurations
```

## Supported Providers

- **Azure OpenAI**: Default implementation
- **OpenAI**: Direct API access
- **Custom**: Implement `setup_client()` for your provider

## Security

The `api/` folder is gitignored to prevent credential exposure.
Never commit API keys or tokens to version control.
