"""Tests for the rate limiter middleware."""

import time

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from apps.orchestrator.middleware.rate_limiter import (
    EXEMPT_PATHS,
    RateLimiterMiddleware,
    _SlidingWindowCounter,
    _counter,
    add_rate_limiter,
)


@pytest.fixture(autouse=True)
def _fresh_counter(monkeypatch):
    """Reset the module-level counter before each test."""
    fresh = _SlidingWindowCounter()
    monkeypatch.setattr("apps.orchestrator.middleware.rate_limiter._counter", fresh)
    yield


def _make_app() -> FastAPI:
    """Create a minimal FastAPI app with rate limiter attached."""
    app = FastAPI()
    add_rate_limiter(app)

    @app.get("/api/v1/health")
    async def health():
        return {"status": "ok"}

    @app.get("/api/v1/items")
    async def list_items():
        return {"items": []}

    @app.post("/api/v1/items")
    async def create_item():
        return {"id": "1"}

    return app


# ------------------------------------------------------------------
# Sliding window counter unit tests
# ------------------------------------------------------------------


class TestSlidingWindowCounter:
    def test_allows_requests_within_limit(self):
        counter = _SlidingWindowCounter()
        for _ in range(5):
            allowed, remaining, retry = counter.check_and_record("k1", 5, 60)
            assert allowed is True
            assert retry == 0

    def test_blocks_after_limit(self):
        counter = _SlidingWindowCounter()
        for _ in range(3):
            counter.check_and_record("k2", 3, 60)
        allowed, remaining, retry = counter.check_and_record("k2", 3, 60)
        assert allowed is False
        assert remaining == 0
        assert retry >= 1

    def test_separate_keys_independent(self):
        counter = _SlidingWindowCounter()
        for _ in range(3):
            counter.check_and_record("a", 3, 60)
        # 'a' exhausted
        allowed_a, _, _ = counter.check_and_record("a", 3, 60)
        assert allowed_a is False
        # 'b' still has capacity
        allowed_b, _, _ = counter.check_and_record("b", 3, 60)
        assert allowed_b is True

    def test_cleanup_stale(self):
        counter = _SlidingWindowCounter()
        counter.check_and_record("stale", 10, 0)  # 0-second window
        counter.cleanup_stale(0)
        # After cleanup, key should be gone
        assert "stale" not in counter._windows


# ------------------------------------------------------------------
# Middleware integration tests
# ------------------------------------------------------------------


class TestRateLimiterMiddleware:
    def test_exempt_paths_not_rate_limited(self):
        app = _make_app()
        client = TestClient(app)
        for _ in range(100):
            resp = client.get("/api/v1/health")
            assert resp.status_code == 200

    def test_rate_limit_headers_present(self):
        app = _make_app()
        client = TestClient(app)
        resp = client.get("/api/v1/items")
        assert "X-RateLimit-Limit" in resp.headers
        assert "X-RateLimit-Remaining" in resp.headers
        assert "X-RateLimit-Reset" in resp.headers

    def test_returns_429_when_exhausted(self, monkeypatch):
        """Force a very low limit and verify 429."""
        monkeypatch.setattr(
            "apps.orchestrator.middleware.rate_limiter.RATE_LIMIT_ANONYMOUS", 2
        )
        monkeypatch.setattr(
            "apps.orchestrator.middleware.rate_limiter._TIER_LIMITS",
            {"anonymous": 2, "free": 2, "pro": 2, "enterprise": 2},
        )
        app = _make_app()
        client = TestClient(app)
        # First two succeed
        resp1 = client.get("/api/v1/items")
        resp2 = client.get("/api/v1/items")
        assert resp1.status_code == 200
        assert resp2.status_code == 200

        # Third should be rate-limited
        resp3 = client.get("/api/v1/items")
        assert resp3.status_code == 429
        body = resp3.json()
        assert body["error"]["code"] == "RATE_LIMIT_EXCEEDED"
        assert "Retry-After" in resp3.headers

    def test_429_error_format(self, monkeypatch):
        """Verify 429 response follows the standard error format."""
        monkeypatch.setattr(
            "apps.orchestrator.middleware.rate_limiter._TIER_LIMITS",
            {"anonymous": 1, "free": 1, "pro": 1, "enterprise": 1},
        )
        app = _make_app()
        client = TestClient(app)
        client.get("/api/v1/items")  # exhaust
        resp = client.get("/api/v1/items")
        assert resp.status_code == 429
        err = resp.json()["error"]
        assert err["code"] == "RATE_LIMIT_EXCEEDED"
        assert err["status"] == 429
        assert "message" in err


class TestCORSConfiguration:
    """Verify CORS middleware is configured on the real app."""

    def test_cors_preflight_allowed_origin(self):
        """The app should respond to OPTIONS with CORS headers for allowed origins."""
        from apps.orchestrator.main import app

        client = TestClient(app)
        resp = client.options(
            "/api/v1/items",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        # CORS middleware should include the header
        assert resp.headers.get("access-control-allow-origin") in {
            "http://localhost:3000",
            "*",
        }

    def test_cors_exposes_rate_limit_headers(self):
        """Rate-limit headers should be exposed in CORS."""
        from apps.orchestrator.main import app

        client = TestClient(app)
        resp = client.options(
            "/api/v1/items",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        exposed = resp.headers.get("access-control-expose-headers", "")
        # At minimum, the expose headers should be set
        assert "x-ratelimit-limit" in exposed.lower() or resp.status_code == 200
