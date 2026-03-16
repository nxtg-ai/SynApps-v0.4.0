"""Anthropic (Claude) LLM provider — mock implementation.

Provides the correct interface for the Anthropic Messages API.
Real API calls are NOT made; this is a foundation for future wiring.
"""

from __future__ import annotations

import os
from typing import Any, List

from synapps.providers.llm.base import BaseLLMProvider, LLMResponse, ModelInfo, ProviderError


class AnthropicProvider(BaseLLMProvider):
    """Mock Anthropic provider implementing the BaseLLMProvider interface."""

    name = "anthropic"

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.base_url = (base_url or "https://api.anthropic.com").rstrip("/")

    def validate(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, "ANTHROPIC_API_KEY not set"
        return True, ""

    def get_models(self) -> List[ModelInfo]:
        return [
            ModelInfo(
                id="claude-sonnet-4-20250514",
                name="Claude Sonnet 4",
                context_window=200_000,
                max_output_tokens=16_384,
                supports_vision=True,
            ),
            ModelInfo(
                id="claude-haiku-4-20250414",
                name="Claude Haiku 4",
                context_window=200_000,
                max_output_tokens=8_192,
                supports_vision=True,
            ),
            ModelInfo(
                id="claude-opus-4-20250514",
                name="Claude Opus 4",
                context_window=200_000,
                max_output_tokens=32_000,
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

        # Mock response — real implementation would call the Anthropic Messages API
        return LLMResponse(
            content=f"[mock-anthropic] response to: {prompt[:80]}",
            model=model,
            provider=self.name,
            prompt_tokens=len(prompt.split()),
            completion_tokens=10,
            total_tokens=len(prompt.split()) + 10,
            finish_reason="end_turn",
            raw={"mock": True},
        )
