"""API Client Setup - Example Template.

This file provides the interface expected by BYOL's LLM-as-Judge evaluation.
Copy this folder to `api/` and implement the functions for your LLM provider.

Supported providers:
- Azure OpenAI (default implementation below)
- OpenAI (see OpenAI section)
- Custom endpoints

Usage:
    cp -r api.example api
    # Edit api/get_azure_api.py with your credentials
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

# =============================================================================
# Configuration
# =============================================================================

# Option 1: Azure OpenAI
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://your-resource.openai.azure.com")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

# Option 2: OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Endpoint configurations
ENDPOINT_CONFIGS = {
    "default": {
        "endpoint_name": AZURE_OPENAI_ENDPOINT,
        "api_version": AZURE_OPENAI_API_VERSION,
        "description": "Default Azure OpenAI endpoint",
        "client_type": "openai",
        "supported_models": [
            "gpt-4o",
            "gpt-4-turbo",
            "gpt-35-turbo",
        ]
    },
    # Add more endpoints as needed
}


# =============================================================================
# Azure OpenAI Implementation
# =============================================================================

def setup_openai_client(endpoint_type: str = "default"):
    """Setup Azure OpenAI client.
    
    Args:
        endpoint_type: Key from ENDPOINT_CONFIGS
        
    Returns:
        Configured AzureOpenAI client
    """
    try:
        from openai import AzureOpenAI
    except ImportError:
        raise ImportError("openai package required. Install with: pip install openai")
    
    config = ENDPOINT_CONFIGS.get(endpoint_type)
    if not config:
        raise ValueError(f"Unknown endpoint type: {endpoint_type}")
    
    return AzureOpenAI(
        azure_endpoint=config["endpoint_name"],
        api_key=AZURE_OPENAI_API_KEY,
        api_version=config["api_version"],
    )


def setup_client(endpoint_type: str = "default"):
    """Generic client setup - dispatches to appropriate provider.
    
    Args:
        endpoint_type: Endpoint configuration key
        
    Returns:
        Configured client for the endpoint
    """
    config = ENDPOINT_CONFIGS.get(endpoint_type, {})
    client_type = config.get("client_type", "openai")
    
    if client_type == "openai":
        return setup_openai_client(endpoint_type)
    else:
        raise ValueError(f"Unknown client type: {client_type}")


def get_client_for_model(model_name: str) -> Tuple[Any, Optional[str]]:
    """Get appropriate client for a model.
    
    Args:
        model_name: Model name (e.g., "gpt-4o", "gpt-35-turbo")
        
    Returns:
        Tuple of (client, endpoint_type) or (None, None) if not found
    """
    # Find endpoint that supports this model
    for endpoint_type, config in ENDPOINT_CONFIGS.items():
        if model_name in config.get("supported_models", []):
            try:
                client = setup_client(endpoint_type)
                return client, endpoint_type
            except Exception as e:
                print(f"Failed to setup {endpoint_type}: {e}")
                continue
    
    return None, None


# =============================================================================
# OpenAI Implementation (Alternative)
# =============================================================================

def setup_openai_direct_client():
    """Setup direct OpenAI client (not Azure).
    
    Returns:
        Configured OpenAI client
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package required. Install with: pip install openai")
    
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    return OpenAI(api_key=OPENAI_API_KEY)


# =============================================================================
# Chat Completion Helper
# =============================================================================

def chat_completion(
    client,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 1.0,
    max_tokens: int = 256,
    **kwargs
) -> Dict[str, Any]:
    """Make a chat completion request.
    
    Args:
        client: OpenAI or AzureOpenAI client
        model: Model name/deployment name
        messages: List of message dicts with 'role' and 'content'
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        **kwargs: Additional parameters
        
    Returns:
        Response dictionary with 'content' and 'usage'
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        return {
            "status": "success",
            "content": response.choices[0].message.content,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            "raw_response": response,
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def setup_eu2_client():
    """Setup EU2 endpoint client (customize for your setup)."""
    return setup_openai_client("default")


def setup_wu_client():
    """Setup WU endpoint client (customize for your setup)."""
    return setup_openai_client("default")
