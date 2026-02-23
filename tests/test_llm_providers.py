"""Tests for the LLM Provider Abstraction Layer.

Covers: provider registration, lookup, interface compliance,
fallback behaviour, auto-discovery, and mock completion calls.
"""

from __future__ import annotations

import asyncio
from typing import Any, List

import pytest

from synapps.providers.llm.base import (
    BaseLLMProvider,
    LLMResponse,
    ModelInfo,
    ProviderError,
    ProviderNotFoundError,
)
from synapps.providers.llm.anthropic_provider import AnthropicProvider
from synapps.providers.llm.openai_provider import OpenAIProvider
from synapps.providers.llm.registry import ProviderRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubProvider(BaseLLMProvider):
    """Minimal concrete provider for testing the abstract interface."""

    name = "stub"

    async def complete(self, prompt: str, model: str, **kwargs: Any) -> LLMResponse:
        return LLMResponse(content="stub", model=model, provider=self.name)

    def get_models(self) -> List[ModelInfo]:
        return [ModelInfo(id="stub-1", name="Stub Model")]

    def validate(self) -> tuple[bool, str]:
        return True, ""


class _NoNameProvider(BaseLLMProvider):
    """Provider with empty name — should be rejected on register."""

    name = ""

    async def complete(self, prompt: str, model: str, **kwargs: Any) -> LLMResponse:
        return LLMResponse(content="", model=model, provider=self.name)

    def get_models(self) -> List[ModelInfo]:
        return []

    def validate(self) -> tuple[bool, str]:
        return True, ""


@pytest.fixture()
def registry() -> ProviderRegistry:
    """Fresh isolated registry per test."""
    return ProviderRegistry()


# ---------------------------------------------------------------------------
# 1. Registration & lookup
# ---------------------------------------------------------------------------


def test_register_and_get(registry: ProviderRegistry) -> None:
    registry.register(_StubProvider)
    cls = registry.get("stub")
    assert cls is _StubProvider


def test_register_case_insensitive(registry: ProviderRegistry) -> None:
    registry.register(_StubProvider)
    assert registry.get("STUB") is _StubProvider
    assert registry.get("  Stub  ") is _StubProvider


def test_get_unknown_raises(registry: ProviderRegistry) -> None:
    with pytest.raises(ProviderNotFoundError, match="Unknown provider 'nope'"):
        registry.get("nope")


def test_register_empty_name_raises(registry: ProviderRegistry) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        registry.register(_NoNameProvider)


def test_unregister(registry: ProviderRegistry) -> None:
    registry.register(_StubProvider)
    assert registry.unregister("stub") is True
    assert registry.has("stub") is False
    # second unregister returns False
    assert registry.unregister("stub") is False


def test_list_providers(registry: ProviderRegistry) -> None:
    registry.register(OpenAIProvider)
    registry.register(AnthropicProvider)
    names = registry.list_providers()
    assert names == ["anthropic", "openai"]


def test_has(registry: ProviderRegistry) -> None:
    assert registry.has("openai") is False
    registry.register(OpenAIProvider)
    assert registry.has("openai") is True


def test_clear(registry: ProviderRegistry) -> None:
    registry.register(OpenAIProvider)
    registry.clear()
    assert registry.list_providers() == []


# ---------------------------------------------------------------------------
# 2. Fallback behaviour
# ---------------------------------------------------------------------------


def test_fallback_to_alternative(registry: ProviderRegistry) -> None:
    registry.register(OpenAIProvider)
    cls = registry.get_with_fallback("missing", fallback="openai")
    assert cls is OpenAIProvider


def test_fallback_both_missing(registry: ProviderRegistry) -> None:
    with pytest.raises(ProviderNotFoundError, match="fallback"):
        registry.get_with_fallback("missing", fallback="also_missing")


def test_fallback_primary_exists(registry: ProviderRegistry) -> None:
    registry.register(AnthropicProvider)
    registry.register(OpenAIProvider)
    cls = registry.get_with_fallback("anthropic", fallback="openai")
    assert cls is AnthropicProvider


# ---------------------------------------------------------------------------
# 3. Interface compliance — OpenAI provider
# ---------------------------------------------------------------------------


def test_openai_name() -> None:
    assert OpenAIProvider.name == "openai"


def test_openai_validate_no_key() -> None:
    p = OpenAIProvider(api_key=None)
    # Clear env to be safe
    import os
    old = os.environ.pop("OPENAI_API_KEY", None)
    try:
        p = OpenAIProvider(api_key=None)
        ok, reason = p.validate()
        assert ok is False
        assert "OPENAI_API_KEY" in reason
    finally:
        if old is not None:
            os.environ["OPENAI_API_KEY"] = old


def test_openai_validate_with_key() -> None:
    p = OpenAIProvider(api_key="sk-test-123")
    ok, reason = p.validate()
    assert ok is True
    assert reason == ""


def test_openai_models() -> None:
    p = OpenAIProvider(api_key="sk-test")
    models = p.get_models()
    assert len(models) >= 2
    ids = [m.id for m in models]
    assert "gpt-4o" in ids
    assert all(isinstance(m, ModelInfo) for m in models)


@pytest.mark.asyncio
async def test_openai_complete() -> None:
    p = OpenAIProvider(api_key="sk-test")
    resp = await p.complete("Hello world", model="gpt-4o")
    assert isinstance(resp, LLMResponse)
    assert resp.provider == "openai"
    assert resp.model == "gpt-4o"
    assert "mock-openai" in resp.content
    assert resp.total_tokens > 0


@pytest.mark.asyncio
async def test_openai_complete_without_key_raises() -> None:
    import os
    old = os.environ.pop("OPENAI_API_KEY", None)
    try:
        p = OpenAIProvider(api_key=None)
        with pytest.raises(ProviderError):
            await p.complete("Hello", model="gpt-4o")
    finally:
        if old is not None:
            os.environ["OPENAI_API_KEY"] = old


# ---------------------------------------------------------------------------
# 4. Interface compliance — Anthropic provider
# ---------------------------------------------------------------------------


def test_anthropic_name() -> None:
    assert AnthropicProvider.name == "anthropic"


def test_anthropic_validate_no_key() -> None:
    import os
    old = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        p = AnthropicProvider(api_key=None)
        ok, reason = p.validate()
        assert ok is False
        assert "ANTHROPIC_API_KEY" in reason
    finally:
        if old is not None:
            os.environ["ANTHROPIC_API_KEY"] = old


def test_anthropic_validate_with_key() -> None:
    p = AnthropicProvider(api_key="sk-ant-test")
    ok, _ = p.validate()
    assert ok is True


def test_anthropic_models() -> None:
    p = AnthropicProvider(api_key="sk-ant-test")
    models = p.get_models()
    assert len(models) >= 2
    ids = [m.id for m in models]
    assert any("claude" in mid for mid in ids)
    assert all(isinstance(m, ModelInfo) for m in models)


@pytest.mark.asyncio
async def test_anthropic_complete() -> None:
    p = AnthropicProvider(api_key="sk-ant-test")
    resp = await p.complete("Explain gravity", model="claude-sonnet-4-20250514")
    assert isinstance(resp, LLMResponse)
    assert resp.provider == "anthropic"
    assert "mock-anthropic" in resp.content
    assert resp.finish_reason == "end_turn"


@pytest.mark.asyncio
async def test_anthropic_complete_without_key_raises() -> None:
    import os
    old = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        p = AnthropicProvider(api_key=None)
        with pytest.raises(ProviderError):
            await p.complete("Hello", model="claude-sonnet-4-20250514")
    finally:
        if old is not None:
            os.environ["ANTHROPIC_API_KEY"] = old


# ---------------------------------------------------------------------------
# 5. Auto-discovery
# ---------------------------------------------------------------------------


def test_auto_discover() -> None:
    ProviderRegistry.clear_global()
    ProviderRegistry.auto_discover()
    names = ProviderRegistry.list_global()
    assert "openai" in names
    assert "anthropic" in names
    ProviderRegistry.clear_global()


def test_global_get_after_discover() -> None:
    ProviderRegistry.clear_global()
    ProviderRegistry.auto_discover()
    cls = ProviderRegistry.get_global("anthropic")
    assert cls is AnthropicProvider
    ProviderRegistry.clear_global()


def test_global_unknown_raises() -> None:
    ProviderRegistry.clear_global()
    with pytest.raises(ProviderNotFoundError):
        ProviderRegistry.get_global("missing")


# ---------------------------------------------------------------------------
# 6. LLMResponse / ModelInfo dataclass basics
# ---------------------------------------------------------------------------


def test_llm_response_defaults() -> None:
    r = LLMResponse(content="hi", model="m", provider="p")
    assert r.prompt_tokens == 0
    assert r.finish_reason == "stop"
    assert r.raw == {}


def test_model_info_defaults() -> None:
    m = ModelInfo(id="x", name="X")
    assert m.context_window == 0
    assert m.supports_streaming is True
    assert m.supports_vision is False
