"""Tests for per-API-key configurable rate limiting (DIRECTIVE-14)."""

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from apps.orchestrator.main import app, admin_key_registry
from apps.orchestrator.middleware.rate_limiter import (
    _SlidingWindowCounter,
    _identify_client,
)


MASTER_KEY = "test-master-key-secret"


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _reset_admin_keys():
    admin_key_registry.reset()
    yield
    admin_key_registry.reset()


@pytest.fixture(autouse=True)
def _fresh_counter(monkeypatch):
    """Reset the module-level counter before each test."""
    fresh = _SlidingWindowCounter()
    monkeypatch.setattr("apps.orchestrator.middleware.rate_limiter._counter", fresh)
    yield


# ---------------------------------------------------------------------------
# Admin key creation with rate_limit
# ---------------------------------------------------------------------------


def test_create_admin_key_with_rate_limit():
    """AdminKeyRegistry.create() stores custom rate_limit."""
    key = admin_key_registry.create("Rate Limited", rate_limit=100)
    assert key["rate_limit"] == 100


def test_create_admin_key_default_rate_limit():
    """AdminKeyRegistry.create() without rate_limit stores None."""
    key = admin_key_registry.create("Default")
    assert key["rate_limit"] is None


def test_create_admin_key_endpoint_with_rate_limit(client):
    """POST /admin/keys with rate_limit field creates key with custom limit."""
    with patch("apps.orchestrator.main.SYNAPPS_MASTER_KEY", MASTER_KEY):
        resp = client.post(
            "/api/v1/admin/keys",
            json={"name": "Custom Rate", "rate_limit": 200},
            headers={"X-API-Key": MASTER_KEY},
        )
    assert resp.status_code == 201
    data = resp.json()
    assert data["rate_limit"] == 200


def test_create_admin_key_endpoint_rate_limit_validation(client):
    """POST /admin/keys with invalid rate_limit returns 422."""
    with patch("apps.orchestrator.main.SYNAPPS_MASTER_KEY", MASTER_KEY):
        resp = client.post(
            "/api/v1/admin/keys",
            json={"name": "Bad", "rate_limit": 0},
            headers={"X-API-Key": MASTER_KEY},
        )
    assert resp.status_code == 422


def test_create_admin_key_endpoint_rate_limit_too_high(client):
    """POST /admin/keys with rate_limit > 10000 returns 422."""
    with patch("apps.orchestrator.main.SYNAPPS_MASTER_KEY", MASTER_KEY):
        resp = client.post(
            "/api/v1/admin/keys",
            json={"name": "TooHigh", "rate_limit": 99999},
            headers={"X-API-Key": MASTER_KEY},
        )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Rate limit headers with admin key
# ---------------------------------------------------------------------------


def test_rate_limit_headers_present_with_admin_key(client):
    """Requests with admin API key include standard rate limit headers."""
    key = admin_key_registry.create("Headers Test")
    resp = client.get(
        "/api/v1/flows",
        headers={"X-API-Key": key["api_key"]},
    )
    assert "X-RateLimit-Limit" in resp.headers
    assert "X-RateLimit-Remaining" in resp.headers
    assert "X-RateLimit-Reset" in resp.headers


def test_rate_limit_headers_reflect_custom_limit(client):
    """Rate limit headers reflect the per-key custom limit, not tier default."""
    key = admin_key_registry.create("Custom 150", rate_limit=150)
    resp = client.get(
        "/api/v1/flows",
        headers={"X-API-Key": key["api_key"]},
    )
    assert resp.headers["X-RateLimit-Limit"] == "150"


# ---------------------------------------------------------------------------
# Per-key rate enforcement
# ---------------------------------------------------------------------------


def test_per_key_rate_limit_enforced(client):
    """Admin key with rate_limit=3 gets 429 after 3 requests."""
    key = admin_key_registry.create("Tight Limit", rate_limit=3)
    headers = {"X-API-Key": key["api_key"]}

    for i in range(3):
        resp = client.get("/api/v1/flows", headers=headers)
        assert resp.status_code == 200, f"Request {i+1} should succeed"

    # 4th request should be rate limited
    resp = client.get("/api/v1/flows", headers=headers)
    assert resp.status_code == 429
    body = resp.json()
    assert body["error"]["code"] == "RATE_LIMIT_EXCEEDED"
    assert "Retry-After" in resp.headers


def test_per_key_rate_limit_independent(client):
    """Different admin keys have independent rate counters."""
    key_a = admin_key_registry.create("Key A", rate_limit=2)
    key_b = admin_key_registry.create("Key B", rate_limit=2)

    # Exhaust key A
    for _ in range(2):
        client.get("/api/v1/flows", headers={"X-API-Key": key_a["api_key"]})

    resp_a = client.get("/api/v1/flows", headers={"X-API-Key": key_a["api_key"]})
    assert resp_a.status_code == 429

    # Key B should still work
    resp_b = client.get("/api/v1/flows", headers={"X-API-Key": key_b["api_key"]})
    assert resp_b.status_code == 200


def test_default_rate_limit_used_when_no_custom(client, monkeypatch):
    """Admin key without custom rate_limit uses tier default (enterprise)."""
    monkeypatch.setattr(
        "apps.orchestrator.middleware.rate_limiter._TIER_LIMITS",
        {"anonymous": 2, "free": 2, "pro": 2, "enterprise": 5},
    )
    key = admin_key_registry.create("No Custom")
    headers = {"X-API-Key": key["api_key"]}

    # Should use enterprise tier limit = 5
    resp = client.get("/api/v1/flows", headers=headers)
    assert resp.headers["X-RateLimit-Limit"] == "5"


# ---------------------------------------------------------------------------
# 429 response format
# ---------------------------------------------------------------------------


def test_429_response_format(client):
    """429 response follows standard error format with Retry-After."""
    key = admin_key_registry.create("Format Test", rate_limit=1)
    headers = {"X-API-Key": key["api_key"]}

    client.get("/api/v1/flows", headers=headers)  # exhaust
    resp = client.get("/api/v1/flows", headers=headers)

    assert resp.status_code == 429
    err = resp.json()["error"]
    assert err["code"] == "RATE_LIMIT_EXCEEDED"
    assert err["status"] == 429
    assert "message" in err
    assert "Retry-After" in resp.headers
    assert int(resp.headers["Retry-After"]) >= 1


def test_429_includes_all_rate_limit_headers(client):
    """429 response includes X-RateLimit-* headers."""
    key = admin_key_registry.create("All Headers", rate_limit=1)
    headers = {"X-API-Key": key["api_key"]}

    client.get("/api/v1/flows", headers=headers)  # exhaust
    resp = client.get("/api/v1/flows", headers=headers)

    assert resp.status_code == 429
    assert "X-RateLimit-Limit" in resp.headers
    assert resp.headers["X-RateLimit-Limit"] == "1"
    assert resp.headers["X-RateLimit-Remaining"] == "0"
    assert "X-RateLimit-Reset" in resp.headers


# ---------------------------------------------------------------------------
# _identify_client returns custom limit
# ---------------------------------------------------------------------------


def test_identify_client_returns_custom_limit():
    """_identify_client returns custom rate_limit from principal."""
    from unittest.mock import MagicMock

    request = MagicMock()
    request.state.user = {
        "id": "admin-key:abc123",
        "tier": "enterprise",
        "rate_limit": 42,
    }
    key, tier, custom_limit = _identify_client(request)
    assert key == "user:admin-key:abc123"
    assert tier == "enterprise"
    assert custom_limit == 42


def test_identify_client_returns_none_for_no_custom_limit():
    """_identify_client returns None custom_limit when not set."""
    from unittest.mock import MagicMock

    request = MagicMock()
    request.state.user = {"id": "user:123", "tier": "free"}
    key, tier, custom_limit = _identify_client(request)
    assert custom_limit is None
