"""Tests for Token Bucket rate limiting (DIRECTIVE-NXTG-20260223-10)."""

import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.orchestrator.middleware.rate_limiter import (
    TokenBucket,
    TokenBucketRegistry,
    _SlidingWindowCounter,
    _token_buckets,
    add_rate_limiter,
    TOKEN_BUCKET_RATE,
    TOKEN_BUCKET_BURST,
    TOKEN_BUCKET_GLOBAL_RATE,
    TOKEN_BUCKET_GLOBAL_BURST,
)


@pytest.fixture(autouse=True)
def _fresh_state(monkeypatch):
    """Reset module-level counters before each test."""
    fresh_counter = _SlidingWindowCounter()
    monkeypatch.setattr("apps.orchestrator.middleware.rate_limiter._counter", fresh_counter)
    fresh_buckets = TokenBucketRegistry()
    monkeypatch.setattr("apps.orchestrator.middleware.rate_limiter._token_buckets", fresh_buckets)
    yield


# ---------------------------------------------------------------------------
# TokenBucket unit tests
# ---------------------------------------------------------------------------


class TestTokenBucket:
    def test_starts_full(self):
        bucket = TokenBucket(rate=1.0, burst=10)
        assert bucket.tokens == pytest.approx(10.0, abs=0.5)

    def test_consume_decrements(self):
        bucket = TokenBucket(rate=1.0, burst=10)
        allowed, remaining, retry = bucket.consume()
        assert allowed is True
        assert remaining == pytest.approx(9.0, abs=0.5)
        assert retry == 0.0

    def test_consume_until_empty(self):
        bucket = TokenBucket(rate=0.0, burst=3)  # no refill
        for _ in range(3):
            allowed, _, _ = bucket.consume()
            assert allowed is True
        # 4th should fail
        allowed, remaining, retry = bucket.consume()
        assert allowed is False
        assert remaining == 0.0

    def test_retry_after_nonzero_when_empty(self):
        bucket = TokenBucket(rate=1.0, burst=1)  # 1 token/sec
        bucket.consume()  # empty
        allowed, _, retry = bucket.consume()
        assert allowed is False
        assert retry > 0

    def test_refill_over_time(self):
        bucket = TokenBucket(rate=100.0, burst=10)  # very fast refill
        # Drain all tokens
        for _ in range(10):
            bucket.consume()
        # Wait a tiny bit for refill
        time.sleep(0.05)  # 100 tokens/sec * 0.05s = 5 tokens
        allowed, remaining, _ = bucket.consume()
        assert allowed is True
        assert remaining >= 1.0

    def test_burst_caps_tokens(self):
        bucket = TokenBucket(rate=1000.0, burst=5)
        time.sleep(0.1)  # lots of refill, but capped at burst=5
        assert bucket.tokens <= 5.0

    def test_reset(self):
        bucket = TokenBucket(rate=0.0, burst=10)
        for _ in range(10):
            bucket.consume()
        assert bucket.tokens == pytest.approx(0.0, abs=0.1)
        bucket.reset()
        assert bucket.tokens == pytest.approx(10.0, abs=0.1)

    def test_zero_rate_gives_default_retry(self):
        bucket = TokenBucket(rate=0.0, burst=1)
        bucket.consume()
        allowed, _, retry = bucket.consume()
        assert allowed is False
        assert retry == 60.0  # default when rate=0


# ---------------------------------------------------------------------------
# TokenBucketRegistry unit tests
# ---------------------------------------------------------------------------


class TestTokenBucketRegistry:
    def test_creates_bucket_on_first_access(self):
        reg = TokenBucketRegistry(default_rate=1.0, default_burst=5, global_rate=10.0, global_burst=50)
        bucket = reg.get_or_create("key1")
        assert isinstance(bucket, TokenBucket)

    def test_reuses_existing_bucket(self):
        reg = TokenBucketRegistry(default_rate=1.0, default_burst=5, global_rate=10.0, global_burst=50)
        b1 = reg.get_or_create("key1")
        b2 = reg.get_or_create("key1")
        assert b1 is b2

    def test_different_keys_get_different_buckets(self):
        reg = TokenBucketRegistry(default_rate=1.0, default_burst=5, global_rate=10.0, global_burst=50)
        b1 = reg.get_or_create("key1")
        b2 = reg.get_or_create("key2")
        assert b1 is not b2

    def test_consume_checks_per_key(self):
        reg = TokenBucketRegistry(
            default_rate=0.0, default_burst=2,
            global_rate=100.0, global_burst=1000,
        )
        # First two succeed
        ok1, _, _, _ = reg.consume("k1")
        ok2, _, _, _ = reg.consume("k1")
        assert ok1 is True
        assert ok2 is True
        # Third fails (per-key exhausted)
        ok3, _, _, limited_by = reg.consume("k1")
        assert ok3 is False
        assert limited_by == "key"

    def test_consume_checks_global(self):
        reg = TokenBucketRegistry(
            default_rate=100.0, default_burst=1000,
            global_rate=0.0, global_burst=2,
        )
        # First two succeed (global has 2 tokens)
        ok1, _, _, _ = reg.consume("k1")
        ok2, _, _, _ = reg.consume("k2")
        assert ok1 is True
        assert ok2 is True
        # Third fails (global exhausted)
        ok3, _, _, limited_by = reg.consume("k3")
        assert ok3 is False
        assert limited_by == "global"

    def test_global_refunds_on_per_key_reject(self):
        reg = TokenBucketRegistry(
            default_rate=0.0, default_burst=1,
            global_rate=0.0, global_burst=10,
        )
        # Consume once per-key (succeeds for both global + key)
        reg.consume("k1")
        # Second per-key fails -> global token should be refunded
        global_before = reg.global_bucket.tokens
        ok, _, _, _ = reg.consume("k1")
        assert ok is False
        global_after = reg.global_bucket.tokens
        # Global should have been refunded (same or +1)
        assert global_after >= global_before

    def test_reset_clears_all(self):
        reg = TokenBucketRegistry(default_rate=0.0, default_burst=5, global_rate=0.0, global_burst=5)
        reg.consume("k1")
        reg.consume("k1")
        reg.reset()
        # After reset, buckets are fresh
        ok, _, _, _ = reg.consume("k1")
        assert ok is True

    def test_custom_rate_for_key(self):
        reg = TokenBucketRegistry(
            default_rate=0.0, default_burst=1,
            global_rate=100.0, global_burst=1000,
        )
        # Create with custom high rate
        reg.consume("fast-key", rate=100.0, burst=50)
        bucket = reg.get_or_create("fast-key")
        assert bucket.rate == 100.0


# ---------------------------------------------------------------------------
# Defaults match directive requirements
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_per_key_default_60_per_minute(self):
        assert TOKEN_BUCKET_RATE == 60

    def test_global_default_300_per_minute(self):
        assert TOKEN_BUCKET_GLOBAL_RATE == 300

    def test_burst_defaults_present(self):
        assert TOKEN_BUCKET_BURST > 0
        assert TOKEN_BUCKET_GLOBAL_BURST > 0


# ---------------------------------------------------------------------------
# Middleware integration — token bucket + sliding window
# ---------------------------------------------------------------------------


def _make_app() -> FastAPI:
    app = FastAPI()
    add_rate_limiter(app)

    @app.get("/api/v1/items")
    async def list_items():
        return {"items": []}

    @app.get("/api/v1/health")
    async def health():
        return {"status": "ok"}

    return app


class TestMiddlewareTokenBucket:
    def test_success_includes_rate_limit_headers(self):
        app = _make_app()
        client = TestClient(app)
        resp = client.get("/api/v1/items")
        assert resp.status_code == 200
        assert "X-RateLimit-Limit" in resp.headers
        assert "X-RateLimit-Remaining" in resp.headers
        assert "X-RateLimit-Reset" in resp.headers

    def test_global_limit_enforced(self, monkeypatch):
        """When global bucket is exhausted, 429 is returned."""
        # Tiny global bucket, generous per-key
        tight = TokenBucketRegistry(
            default_rate=100.0, default_burst=1000,
            global_rate=0.0, global_burst=2,
        )
        monkeypatch.setattr("apps.orchestrator.middleware.rate_limiter._token_buckets", tight)

        app = _make_app()
        client = TestClient(app)

        resp1 = client.get("/api/v1/items")
        resp2 = client.get("/api/v1/items")
        assert resp1.status_code == 200
        assert resp2.status_code == 200

        resp3 = client.get("/api/v1/items")
        assert resp3.status_code == 429
        assert resp3.json()["error"]["code"] == "RATE_LIMIT_EXCEEDED"
        assert "Global" in resp3.json()["error"]["message"]
        assert resp3.headers.get("X-RateLimit-Scope") == "global"

    def test_per_key_bucket_enforced(self, monkeypatch):
        """When per-key bucket is exhausted, 429 is returned."""
        tight = TokenBucketRegistry(
            default_rate=0.0, default_burst=2,
            global_rate=100.0, global_burst=1000,
        )
        monkeypatch.setattr("apps.orchestrator.middleware.rate_limiter._token_buckets", tight)

        app = _make_app()
        client = TestClient(app)

        resp1 = client.get("/api/v1/items")
        resp2 = client.get("/api/v1/items")
        assert resp1.status_code == 200
        assert resp2.status_code == 200

        resp3 = client.get("/api/v1/items")
        assert resp3.status_code == 429
        assert resp3.headers.get("X-RateLimit-Scope") == "key"

    def test_429_from_token_bucket_has_retry_after(self, monkeypatch):
        tight = TokenBucketRegistry(
            default_rate=0.0, default_burst=1,
            global_rate=100.0, global_burst=1000,
        )
        monkeypatch.setattr("apps.orchestrator.middleware.rate_limiter._token_buckets", tight)

        app = _make_app()
        client = TestClient(app)

        client.get("/api/v1/items")  # exhaust per-key
        resp = client.get("/api/v1/items")
        assert resp.status_code == 429
        assert "Retry-After" in resp.headers
        assert int(resp.headers["Retry-After"]) >= 1

    def test_exempt_paths_bypass_token_bucket(self, monkeypatch):
        """Health endpoint should not be rate limited even with tiny bucket."""
        tight = TokenBucketRegistry(
            default_rate=0.0, default_burst=0,
            global_rate=0.0, global_burst=0,
        )
        monkeypatch.setattr("apps.orchestrator.middleware.rate_limiter._token_buckets", tight)

        app = _make_app()
        client = TestClient(app)

        resp = client.get("/api/v1/health")
        assert resp.status_code == 200

    def test_burst_allows_spike(self):
        """Token bucket burst allows short request spikes."""
        bucket = TokenBucket(rate=1.0, burst=5)
        # 5 rapid requests all succeed (burst)
        results = [bucket.consume()[0] for _ in range(5)]
        assert all(results)
        # 6th fails (no time to refill)
        assert bucket.consume()[0] is False

    def test_429_global_includes_all_headers(self, monkeypatch):
        tight = TokenBucketRegistry(
            default_rate=100.0, default_burst=1000,
            global_rate=0.0, global_burst=1,
        )
        monkeypatch.setattr("apps.orchestrator.middleware.rate_limiter._token_buckets", tight)

        app = _make_app()
        client = TestClient(app)

        client.get("/api/v1/items")
        resp = client.get("/api/v1/items")
        assert resp.status_code == 429
        assert "X-RateLimit-Limit" in resp.headers
        assert "X-RateLimit-Remaining" in resp.headers
        assert resp.headers["X-RateLimit-Remaining"] == "0"
        assert "X-RateLimit-Reset" in resp.headers
        assert "Retry-After" in resp.headers
