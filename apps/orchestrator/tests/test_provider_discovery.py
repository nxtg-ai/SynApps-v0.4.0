"""Tests for provider auto-discovery + /providers endpoints."""

import pathlib
import pytest
from fastapi.testclient import TestClient

from synapps.providers.llm import (
    BaseLLMProvider,
    LLMResponse,
    ModelInfo,
    ProviderNotFoundError,
    ProviderRegistry,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    from apps.orchestrator.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture
def registry():
    """Fresh isolated registry for unit tests."""
    reg = ProviderRegistry()
    yield reg
    reg.clear()


# ---------------------------------------------------------------------------
# Auto-discovery (filesystem scanning)
# ---------------------------------------------------------------------------


class _StubProvider(BaseLLMProvider):
    name = "stub-for-test"

    async def complete(self, prompt, model, **kwargs):
        return LLMResponse(content="stub", model=model, provider=self.name)

    def get_models(self):
        return [ModelInfo(id="stub-1", name="Stub 1")]

    def validate(self):
        return True, ""


def test_auto_discover_finds_builtin_providers():
    """auto_discover should register openai and anthropic from filesystem."""
    ProviderRegistry.clear_global()
    ProviderRegistry.auto_discover()
    names = ProviderRegistry.list_global()
    assert "openai" in names
    assert "anthropic" in names


def test_auto_discover_is_idempotent():
    """Calling auto_discover twice doesn't duplicate providers."""
    ProviderRegistry.clear_global()
    ProviderRegistry.auto_discover()
    count_1 = len(ProviderRegistry.list_global())
    ProviderRegistry.auto_discover()
    count_2 = len(ProviderRegistry.list_global())
    assert count_1 == count_2


def test_auto_discover_skips_private_files():
    """Files starting with _ (like __init__.py) are not scanned."""
    ProviderRegistry.clear_global()
    ProviderRegistry.auto_discover()
    # base.py defines BaseLLMProvider which has name="" so it shouldn't register
    names = ProviderRegistry.list_global()
    for n in names:
        assert n != "", "Empty-name provider should not be registered"


def test_auto_discover_directory_returns_count():
    """auto_discover_directory returns the number of newly registered providers."""
    ProviderRegistry.clear_global()
    providers_dir = pathlib.Path(__file__).resolve().parent.parent.parent.parent / "synapps" / "providers" / "llm"
    count = ProviderRegistry.auto_discover_directory(providers_dir, "synapps.providers.llm")
    assert count >= 2  # at least openai + anthropic


def test_auto_discover_directory_nonexistent():
    """Scanning a non-existent directory registers nothing."""
    ProviderRegistry.clear_global()
    count = ProviderRegistry.auto_discover_directory(
        pathlib.Path("/tmp/nonexistent-dir-12345"), "fake.module"
    )
    assert count == 0


# ---------------------------------------------------------------------------
# Instance-level registry: provider_info / all_providers_info / health
# ---------------------------------------------------------------------------


def test_provider_info_returns_expected_shape(registry):
    """provider_info returns name, connected, reason, model_count, models."""
    registry.register(_StubProvider)
    info = registry.provider_info("stub-for-test")
    assert info["name"] == "stub-for-test"
    assert info["connected"] is True
    assert info["reason"] == ""
    assert info["model_count"] == 1
    assert len(info["models"]) == 1
    assert info["models"][0]["id"] == "stub-1"


def test_provider_info_unknown_raises(registry):
    """Querying an unknown provider raises ProviderNotFoundError."""
    with pytest.raises(ProviderNotFoundError):
        registry.provider_info("nonexistent")


def test_all_providers_info(registry):
    """all_providers_info returns info for every registered provider."""
    registry.register(_StubProvider)
    infos = registry.all_providers_info()
    assert len(infos) == 1
    assert infos[0]["name"] == "stub-for-test"


def test_provider_health_ok(registry):
    """provider_health returns status 'ok' when provider validates."""
    registry.register(_StubProvider)
    health = registry.provider_health("stub-for-test")
    assert health["status"] == "ok"
    assert health["connected"] is True
    assert health["model_count"] == 1


def test_provider_health_unavailable(registry):
    """provider_health returns 'unavailable' when provider fails validation."""
    from synapps.providers.llm.openai_provider import OpenAIProvider
    import os
    old = os.environ.pop("OPENAI_API_KEY", None)
    try:
        registry.register(OpenAIProvider)
        health = registry.provider_health("openai")
        assert health["status"] == "unavailable"
        assert health["connected"] is False
        assert "OPENAI_API_KEY" in health["reason"]
    finally:
        if old is not None:
            os.environ["OPENAI_API_KEY"] = old


# ---------------------------------------------------------------------------
# API endpoints: GET /providers, GET /providers/{name}/health
# ---------------------------------------------------------------------------


def test_providers_endpoint_returns_200(client):
    """GET /api/v1/providers returns 200 with expected shape."""
    resp = client.get("/api/v1/providers")
    assert resp.status_code == 200
    data = resp.json()
    assert "providers" in data
    assert "total" in data
    assert "discovery" in data
    assert data["discovery"] == "filesystem"


def test_providers_endpoint_lists_discovered(client):
    """GET /api/v1/providers includes openai and anthropic."""
    data = client.get("/api/v1/providers").json()
    names = [p["name"] for p in data["providers"]]
    assert "openai" in names
    assert "anthropic" in names
    assert data["total"] >= 2


def test_providers_endpoint_has_models(client):
    """Each discovered provider includes models list."""
    data = client.get("/api/v1/providers").json()
    for p in data["providers"]:
        assert "models" in p
        assert "model_count" in p
        assert p["model_count"] == len(p["models"])
        assert p["model_count"] > 0


def test_providers_endpoint_has_connected_flag(client):
    """Each provider has connected and reason fields."""
    data = client.get("/api/v1/providers").json()
    for p in data["providers"]:
        assert "connected" in p
        assert isinstance(p["connected"], bool)
        assert "reason" in p


def test_provider_health_endpoint_returns_200(client):
    """GET /api/v1/providers/openai/health returns 200."""
    resp = client.get("/api/v1/providers/openai/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "openai"
    assert "status" in data
    assert data["status"] in ("ok", "unavailable")
    assert "connected" in data
    assert "model_count" in data


def test_provider_health_endpoint_anthropic(client):
    """GET /api/v1/providers/anthropic/health returns 200."""
    resp = client.get("/api/v1/providers/anthropic/health")
    assert resp.status_code == 200
    assert resp.json()["name"] == "anthropic"


def test_provider_health_endpoint_unknown_returns_404(client):
    """GET /api/v1/providers/unknown/health returns 404."""
    resp = client.get("/api/v1/providers/nonexistent-xyz/health")
    assert resp.status_code == 404
