"""OpenAI LLM provider — mock implementation.

Provides the correct interface for the OpenAI Chat Completions API.
Real API calls are NOT made; this is a foundation for future wiring.
"""

from __future__ import annotations

import os
from typing import Any, List

from synapps.providers.llm.base import BaseLLMProvider, LLMResponse, ModelInfo, ProviderError


class OpenAIProvider(BaseLLMProvider):
    """Mock OpenAI provider implementing the BaseLLMProvider interface."""

    name = "openai"

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")

    def validate(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, "OPENAI_API_KEY not set"
        return True, ""

    def get_models(self) -> List[ModelInfo]:
        return [
            ModelInfo(
                id="gpt-4o",
                name="GPT-4o",
                context_window=128_000,
                max_output_tokens=16_384,
                supports_vision=True,
            ),
            ModelInfo(
                id="gpt-4o-mini",
                name="GPT-4o Mini",
                context_window=128_000,
                max_output_tokens=16_384,
                supports_vision=True,
            ),
            ModelInfo(
                id="gpt-4.1",
                name="GPT-4.1",
                context_window=1_047_576,
                max_output_tokens=32_768,
                supports_vision=True,
            ),
        ]

    async def complete(
        self,
        prompt: str,
        model: str,
        *,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> LLMResponse:
        is_valid, reason = self.validate()
        if not is_valid:
            raise ProviderError(reason)

        # Mock response — real implementation would call OpenAI Chat Completions
        return LLMResponse(
            content=f"[mock-openai] response to: {prompt[:80]}",
            model=model,
            provider=self.name,
            prompt_tokens=len(prompt.split()),
            completion_tokens=8,
            total_tokens=len(prompt.split()) + 8,
            finish_reason="stop",
            raw={"mock": True},
        )
