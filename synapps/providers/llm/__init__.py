"""LLM Provider Abstraction Layer.

Unified interface for multiple LLM providers (OpenAI, Anthropic, etc.)
with a registry for auto-discovery and fallback support.
"""

from synapps.providers.llm.base import (
    BaseLLMProvider,
    LLMResponse,
    ModelInfo,
    ProviderError,
    ProviderNotFoundError,
)
from synapps.providers.llm.registry import ProviderRegistry

__all__ = [
    "BaseLLMProvider",
    "LLMResponse",
    "ModelInfo",
    "ProviderError",
    "ProviderNotFoundError",
    "ProviderRegistry",
]
