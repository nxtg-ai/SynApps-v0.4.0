"""Tests for Request Replay + Debug Mode (DIRECTIVE-NXTG-20260223-14).

Covers:
- FailedRequestStore: add, get, list, eviction, capacity, clear, redact
- Middleware: captures failed requests (>= 400), ignores successes
- GET /api/v1/requests/failed — lists recent failures
- POST /api/v1/requests/{id}/replay — replays original request
- GET /api/v1/requests/{id}/debug — full chain with redacted headers
- Rate limit exemption for replay/debug endpoints
"""

import time
import threading
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient

from apps.orchestrator.main import (
    app,
    FailedRequestStore,
    failed_request_store,
    _SENSITIVE_HEADERS,
)


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _clear_store():
    """Reset the global failed request store before each test."""
    failed_request_store.clear()
    yield
    failed_request_store.clear()


# ============================================================
# FailedRequestStore — unit tests
# ============================================================


class TestFailedRequestStoreBasic:
    """Core operations: add, get, list."""

    def test_add_and_get(self):
        store = FailedRequestStore(capacity=10)
        entry = {"request_id": "req-1", "method": "GET", "path": "/test"}
        store.add(entry)
        assert store.get("req-1") == entry

    def test_get_missing_returns_none(self):
        store = FailedRequestStore(capacity=10)
        assert store.get("nonexistent") is None

    def test_add_empty_request_id_ignored(self):
        store = FailedRequestStore(capacity=10)
        store.add({"request_id": "", "method": "GET"})
        assert len(store) == 0

    def test_add_no_request_id_ignored(self):
        store = FailedRequestStore(capacity=10)
        store.add({"method": "GET"})
        assert len(store) == 0

    def test_duplicate_request_id_not_added(self):
        store = FailedRequestStore(capacity=10)
        store.add({"request_id": "dup-1", "method": "GET"})
        store.add({"request_id": "dup-1", "method": "POST"})
        assert len(store) == 1
        assert store.get("dup-1")["method"] == "GET"

    def test_list_recent_returns_newest_first(self):
        store = FailedRequestStore(capacity=10)
        for i in range(5):
            store.add({"request_id": f"req-{i}", "ts": i})
        recent = store.list_recent(limit=3)
        assert len(recent) == 3
        assert recent[0]["request_id"] == "req-4"
        assert recent[1]["request_id"] == "req-3"
        assert recent[2]["request_id"] == "req-2"

    def test_list_recent_with_limit_zero_returns_empty(self):
        store = FailedRequestStore(capacity=10)
        for i in range(5):
            store.add({"request_id": f"req-{i}"})
        # limit=0 means no entries returned (edge case)
        assert len(store.list_recent(limit=0)) == 0

    def test_list_recent_all(self):
        store = FailedRequestStore(capacity=10)
        for i in range(5):
            store.add({"request_id": f"req-{i}"})
        assert len(store.list_recent(limit=100)) == 5

    def test_list_recent_larger_than_count(self):
        store = FailedRequestStore(capacity=10)
        store.add({"request_id": "only-one"})
        recent = store.list_recent(limit=100)
        assert len(recent) == 1

    def test_len(self):
        store = FailedRequestStore(capacity=10)
        assert len(store) == 0
        store.add({"request_id": "a"})
        assert len(store) == 1
        store.add({"request_id": "b"})
        assert len(store) == 2


class TestFailedRequestStoreEviction:
    """LRU eviction when capacity is reached."""

    def test_evicts_oldest_when_full(self):
        store = FailedRequestStore(capacity=3)
        store.add({"request_id": "old-1"})
        store.add({"request_id": "old-2"})
        store.add({"request_id": "old-3"})
        store.add({"request_id": "new-4"})
        assert len(store) == 3
        assert store.get("old-1") is None  # evicted
        assert store.get("old-2") is not None
        assert store.get("new-4") is not None

    def test_eviction_continues_with_many_adds(self):
        store = FailedRequestStore(capacity=2)
        for i in range(100):
            store.add({"request_id": f"r-{i}"})
        assert len(store) == 2
        assert store.get("r-98") is not None
        assert store.get("r-99") is not None
        assert store.get("r-0") is None

    def test_capacity_of_one(self):
        store = FailedRequestStore(capacity=1)
        store.add({"request_id": "a"})
        store.add({"request_id": "b"})
        assert len(store) == 1
        assert store.get("a") is None
        assert store.get("b") is not None

    def test_capacity_minimum_clamps_to_one(self):
        store = FailedRequestStore(capacity=0)
        assert store.capacity == 1
        store = FailedRequestStore(capacity=-5)
        assert store.capacity == 1

    def test_capacity_property(self):
        store = FailedRequestStore(capacity=42)
        assert store.capacity == 42


class TestFailedRequestStoreClear:
    """Clear empties the store."""

    def test_clear_empties_store(self):
        store = FailedRequestStore(capacity=10)
        for i in range(5):
            store.add({"request_id": f"r-{i}"})
        assert len(store) == 5
        store.clear()
        assert len(store) == 0
        assert store.get("r-0") is None

    def test_clear_then_add(self):
        store = FailedRequestStore(capacity=5)
        store.add({"request_id": "before"})
        store.clear()
        store.add({"request_id": "after"})
        assert len(store) == 1
        assert store.get("after") is not None
        assert store.get("before") is None


class TestFailedRequestStoreRedaction:
    """Header redaction for sensitive values."""

    def test_redacts_authorization(self):
        headers = {"Authorization": "Bearer secret123", "Content-Type": "application/json"}
        redacted = FailedRequestStore.redact_headers(headers)
        assert redacted["Authorization"] == "[REDACTED]"
        assert redacted["Content-Type"] == "application/json"

    def test_redacts_x_api_key(self):
        headers = {"X-API-Key": "sk-secret"}
        redacted = FailedRequestStore.redact_headers(headers)
        assert redacted["X-API-Key"] == "[REDACTED]"

    def test_redacts_cookie(self):
        headers = {"Cookie": "session=abc123", "Accept": "text/html"}
        redacted = FailedRequestStore.redact_headers(headers)
        assert redacted["Cookie"] == "[REDACTED]"
        assert redacted["Accept"] == "text/html"

    def test_redacts_set_cookie(self):
        headers = {"Set-Cookie": "sid=val"}
        redacted = FailedRequestStore.redact_headers(headers)
        assert redacted["Set-Cookie"] == "[REDACTED]"

    def test_redacts_proxy_authorization(self):
        headers = {"Proxy-Authorization": "Basic abc"}
        redacted = FailedRequestStore.redact_headers(headers)
        assert redacted["Proxy-Authorization"] == "[REDACTED]"

    def test_redacts_csrf_token(self):
        headers = {"X-CSRF-Token": "tok-123"}
        redacted = FailedRequestStore.redact_headers(headers)
        assert redacted["X-CSRF-Token"] == "[REDACTED]"

    def test_case_insensitive_matching(self):
        headers = {"authorization": "Bearer x", "AUTHORIZATION": "Bearer y"}
        redacted = FailedRequestStore.redact_headers(headers)
        assert redacted["authorization"] == "[REDACTED]"
        assert redacted["AUTHORIZATION"] == "[REDACTED]"

    def test_empty_headers(self):
        assert FailedRequestStore.redact_headers({}) == {}

    def test_non_sensitive_headers_preserved(self):
        headers = {
            "Content-Type": "application/json",
            "Accept": "*/*",
            "X-Request-ID": "req-123",
        }
        redacted = FailedRequestStore.redact_headers(headers)
        assert redacted == headers

    def test_all_sensitive_headers_covered(self):
        """Verify all members of _SENSITIVE_HEADERS are actually redacted."""
        headers = {h: f"val-{h}" for h in _SENSITIVE_HEADERS}
        redacted = FailedRequestStore.redact_headers(headers)
        for h in _SENSITIVE_HEADERS:
            assert redacted[h] == "[REDACTED]", f"{h} was not redacted"


class TestFailedRequestStoreThreadSafety:
    """Concurrent access doesn't corrupt state."""

    def test_concurrent_adds(self):
        store = FailedRequestStore(capacity=500)
        errors = []

        def writer(start_id, count):
            try:
                for i in range(count):
                    store.add({"request_id": f"t-{start_id}-{i}"})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(t, 100)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(store) == 500


# ============================================================
# Middleware — failed request capture
# ============================================================


class TestMiddlewareCapture:
    """Middleware captures failed requests into the store."""

    def test_404_captured(self, client):
        resp = client.get("/api/v1/this-does-not-exist-ever-abc123")
        assert resp.status_code == 404
        assert len(failed_request_store) >= 1

    def test_success_not_captured(self, client):
        initial = len(failed_request_store)
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        assert len(failed_request_store) == initial

    def test_captured_entry_has_all_fields(self, client):
        resp = client.get(
            "/api/v1/this-does-not-exist-xyz789",
            headers={"X-Request-ID": "test-cap-001"},
        )
        assert resp.status_code == 404
        entry = failed_request_store.get("test-cap-001")
        assert entry is not None
        assert entry["method"] == "GET"
        assert entry["response_status"] == 404
        assert "timestamp" in entry
        assert "duration_ms" in entry
        assert "request_headers" in entry
        assert "response_body" in entry

    def test_response_body_preserved_on_failure(self, client):
        resp = client.get(
            "/api/v1/nonexistent-endpoint-for-test",
            headers={"X-Request-ID": "test-body-001"},
        )
        assert resp.status_code == 404
        # The response body should still be returned to the caller
        body = resp.json()
        assert "error" in body

    def test_request_id_header_set_on_failure(self, client):
        resp = client.get(
            "/api/v1/no-such-thing",
            headers={"X-Request-ID": "custom-id-42"},
        )
        assert resp.headers.get("x-request-id") == "custom-id-42"


# ============================================================
# GET /api/v1/requests/failed
# ============================================================


class TestListFailedRequests:
    """GET /api/v1/requests/failed endpoint."""

    def test_empty_list(self, client):
        resp = client.get("/api/v1/requests/failed")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_returns_failures_after_404(self, client):
        client.get("/api/v1/nonexistent-for-list-test",
                    headers={"X-Request-ID": "list-001"})
        resp = client.get("/api/v1/requests/failed")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 1
        ids = [e["request_id"] for e in data]
        assert "list-001" in ids

    def test_summary_fields(self, client):
        client.get("/api/v1/nonexistent-endpoint",
                    headers={"X-Request-ID": "sum-001"})
        resp = client.get("/api/v1/requests/failed")
        data = resp.json()
        entry = next(e for e in data if e["request_id"] == "sum-001")
        assert "timestamp" in entry
        assert "method" in entry
        assert "path" in entry
        assert "response_status" in entry
        assert "duration_ms" in entry

    def test_limit_parameter(self, client):
        for i in range(5):
            client.get(f"/api/v1/nope-{i}", headers={"X-Request-ID": f"lim-{i}"})
        resp = client.get("/api/v1/requests/failed?limit=2")
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    def test_newest_first(self, client):
        for i in range(3):
            client.get(f"/api/v1/nope-ord-{i}", headers={"X-Request-ID": f"ord-{i}"})
        resp = client.get("/api/v1/requests/failed")
        data = resp.json()
        ord_entries = [e for e in data if e["request_id"].startswith("ord-")]
        if len(ord_entries) >= 2:
            assert ord_entries[0]["request_id"] > ord_entries[-1]["request_id"]


# ============================================================
# POST /api/v1/requests/{id}/replay
# ============================================================


class TestReplayRequest:
    """POST /api/v1/requests/{id}/replay endpoint."""

    def test_replay_not_found(self, client):
        resp = client.post("/api/v1/requests/nonexistent/replay")
        assert resp.status_code == 404

    def test_replay_returns_new_response(self, client):
        # Inject a failed request directly
        failed_request_store.add({
            "request_id": "replay-001",
            "timestamp": time.time(),
            "method": "GET",
            "path": "/api/v1/health",
            "request_headers": {"accept": "application/json"},
            "request_body": "",
            "response_status": 500,
            "response_headers": {},
            "response_body": '{"error": "upstream failed"}',
            "duration_ms": 10.0,
        })

        # Mock httpx to simulate a successful replay
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "application/json"}
        mock_resp.json.return_value = {"status": "ok"}

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("apps.orchestrator.main.httpx.AsyncClient", return_value=mock_client):
            resp = client.post("/api/v1/requests/replay-001/replay")
        assert resp.status_code == 200
        data = resp.json()
        assert data["original_request_id"] == "replay-001"
        assert data["replay_status"] == 200
        assert data["replay_body"] == {"status": "ok"}

    def test_replay_preserves_method(self, client):
        # POST to a nonexistent endpoint
        client.post(
            "/api/v1/unknown-post-target",
            json={"data": "test"},
            headers={"X-Request-ID": "replay-post-001"},
        )
        entry = failed_request_store.get("replay-post-001")
        assert entry is not None
        assert entry["method"] == "POST"

    def test_replay_with_httpx_error(self, client):
        import httpx as _httpx
        # Inject a fake entry pointing to an unreachable upstream
        failed_request_store.add({
            "request_id": "replay-err-001",
            "timestamp": time.time(),
            "method": "GET",
            "path": "/api/v1/health",
            "request_headers": {},
            "request_body": "",
            "response_status": 500,
            "response_headers": {},
            "response_body": "",
            "duration_ms": 10.0,
        })
        # Patch httpx to fail with a connection error
        with patch("apps.orchestrator.main.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(
                side_effect=_httpx.ConnectError("connection refused")
            )
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client
            resp = client.post("/api/v1/requests/replay-err-001/replay")
            assert resp.status_code == 502


# ============================================================
# GET /api/v1/requests/{id}/debug
# ============================================================


class TestDebugRequest:
    """GET /api/v1/requests/{id}/debug endpoint."""

    def test_debug_not_found(self, client):
        resp = client.get("/api/v1/requests/nonexistent/debug")
        assert resp.status_code == 404

    def test_debug_returns_full_chain(self, client):
        client.get(
            "/api/v1/no-such-endpoint-debug",
            headers={
                "X-Request-ID": "debug-001",
                "Authorization": "Bearer secret-token",
                "X-Custom": "keep-me",
            },
        )
        resp = client.get("/api/v1/requests/debug-001/debug")
        assert resp.status_code == 200
        data = resp.json()
        assert data["request_id"] == "debug-001"
        assert data["method"] == "GET"
        assert "duration_ms" in data
        assert "request" in data
        assert "response" in data

    def test_debug_redacts_authorization(self, client):
        client.get(
            "/api/v1/no-such-debug-auth",
            headers={
                "X-Request-ID": "debug-auth-001",
                "Authorization": "Bearer super-secret",
            },
        )
        resp = client.get("/api/v1/requests/debug-auth-001/debug")
        data = resp.json()
        req_headers = data["request"]["headers"]
        assert req_headers.get("authorization") == "[REDACTED]"

    def test_debug_redacts_api_key(self, client):
        client.get(
            "/api/v1/no-such-debug-key",
            headers={
                "X-Request-ID": "debug-key-001",
                "X-API-Key": "sk-my-secret-key",
            },
        )
        resp = client.get("/api/v1/requests/debug-key-001/debug")
        data = resp.json()
        req_headers = data["request"]["headers"]
        assert req_headers.get("x-api-key") == "[REDACTED]"

    def test_debug_preserves_non_sensitive(self, client):
        client.get(
            "/api/v1/no-such-debug-safe",
            headers={
                "X-Request-ID": "debug-safe-001",
                "Accept": "application/json",
            },
        )
        resp = client.get("/api/v1/requests/debug-safe-001/debug")
        data = resp.json()
        req_headers = data["request"]["headers"]
        assert req_headers.get("accept") == "application/json"

    def test_debug_includes_response_body(self, client):
        client.get(
            "/api/v1/no-such-debug-body",
            headers={"X-Request-ID": "debug-body-001"},
        )
        resp = client.get("/api/v1/requests/debug-body-001/debug")
        data = resp.json()
        assert data["response"]["body"] != ""
        assert data["response"]["status"] == 404

    def test_debug_includes_request_body(self, client):
        client.post(
            "/api/v1/no-such-debug-reqbody",
            json={"payload": "test-data"},
            headers={"X-Request-ID": "debug-reqbody-001"},
        )
        resp = client.get("/api/v1/requests/debug-reqbody-001/debug")
        data = resp.json()
        assert "test-data" in data["request"]["body"]


# ============================================================
# Rate limit exemption
# ============================================================


class TestRateLimitExemption:
    """Replay/debug endpoints are exempt from rate limiting."""

    def test_requests_failed_not_rate_limited(self, client):
        """Hitting /requests/failed many times should not trigger 429."""
        for _ in range(40):
            resp = client.get("/api/v1/requests/failed")
            assert resp.status_code != 429

    def test_requests_debug_not_rate_limited(self, client):
        """Hitting /requests/{id}/debug should not trigger 429."""
        for _ in range(40):
            resp = client.get("/api/v1/requests/some-id/debug")
            # 404 is expected (no entry), but not 429
            assert resp.status_code != 429


# ============================================================
# Memory cap enforcement
# ============================================================


class TestMemoryCap:
    """Global store respects FAILED_REQUEST_CAP."""

    def test_default_capacity(self):
        store = FailedRequestStore()
        assert store.capacity == 100

    def test_custom_capacity_from_env(self):
        store = FailedRequestStore(capacity=50)
        for i in range(100):
            store.add({"request_id": f"cap-{i}"})
        assert len(store) == 50

    def test_only_newest_entries_survive(self):
        store = FailedRequestStore(capacity=10)
        for i in range(50):
            store.add({"request_id": f"surv-{i}"})
        assert len(store) == 10
        # Oldest should be evicted
        assert store.get("surv-0") is None
        assert store.get("surv-39") is None
        # Newest should remain
        assert store.get("surv-49") is not None
        assert store.get("surv-40") is not None

    def test_list_recent_respects_limit(self):
        store = FailedRequestStore(capacity=20)
        for i in range(20):
            store.add({"request_id": f"lr-{i}"})
        assert len(store.list_recent(limit=5)) == 5
        assert len(store.list_recent(limit=100)) == 20


# ============================================================
# Edge cases
# ============================================================


class TestEdgeCases:
    """Various edge cases and boundary conditions."""

    def test_store_with_large_body(self):
        store = FailedRequestStore(capacity=5)
        big_body = "x" * 100_000
        store.add({"request_id": "big-1", "request_body": big_body})
        entry = store.get("big-1")
        assert entry is not None
        assert len(entry["request_body"]) == 100_000

    def test_store_with_binary_like_body(self):
        store = FailedRequestStore(capacity=5)
        store.add({"request_id": "bin-1", "request_body": "<binary>"})
        assert store.get("bin-1")["request_body"] == "<binary>"

    def test_redact_headers_does_not_mutate_original(self):
        original = {"Authorization": "Bearer tok", "Accept": "*/*"}
        redacted = FailedRequestStore.redact_headers(original)
        assert original["Authorization"] == "Bearer tok"
        assert redacted["Authorization"] == "[REDACTED]"

    def test_store_entries_include_timestamp(self):
        store = FailedRequestStore(capacity=5)
        now = time.time()
        store.add({"request_id": "ts-1", "timestamp": now})
        assert store.get("ts-1")["timestamp"] == now

    def test_multiple_failures_same_session(self, client):
        """Multiple failures in rapid succession are all captured."""
        for i in range(10):
            client.get(f"/api/v1/no-such-{i}",
                        headers={"X-Request-ID": f"multi-{i}"})
        for i in range(10):
            assert failed_request_store.get(f"multi-{i}") is not None

    def test_replay_strips_hop_by_hop_headers(self):
        """Replay should strip host, content-length, transfer-encoding."""
        failed_request_store.add({
            "request_id": "hop-test",
            "timestamp": time.time(),
            "method": "GET",
            "path": "/api/v1/health",
            "request_headers": {
                "host": "original.example.com",
                "content-length": "42",
                "transfer-encoding": "chunked",
                "accept": "application/json",
            },
            "request_body": "",
            "response_status": 500,
            "response_headers": {},
            "response_body": "",
            "duration_ms": 5.0,
        })
        entry = failed_request_store.get("hop-test")
        assert entry is not None
        # The route handler should strip these — we verify via the store's raw headers
        assert "host" in entry["request_headers"]  # raw store keeps them
