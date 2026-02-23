"""Tests for Consumer Usage Dashboard + Quotas (DIRECTIVE-NXTG-20260223-15).

Covers:
- ConsumerUsageTracker: record, per-key/endpoint/hour tracking, day/week roll
- Quota system: set, check, 80% warning, 100% block, monthly reset
- Middleware: enforce_quota blocks at 100%, warns at 80%
- Middleware: collect_metrics tracks per-key usage
- GET /api/v1/usage — per-key breakdown
- GET /api/v1/usage/{key_id} — detailed usage
- GET /api/v1/quotas — quota overview
- PUT /api/v1/quotas/{key_id} — set quota
"""

import time
import threading
import pytest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from apps.orchestrator.main import (
    app,
    ConsumerUsageTracker,
    usage_tracker,
    _month_start_ts,
    _next_month_start_ts,
)


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _clear_usage():
    """Reset the global usage tracker before each test."""
    usage_tracker.clear()
    yield
    usage_tracker.clear()


# ============================================================
# Utility function tests
# ============================================================


class TestUtilityFunctions:
    def test_month_start_ts_is_first_of_month(self):
        ts = _month_start_ts()
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        assert dt.day == 1
        assert dt.hour == 0
        assert dt.minute == 0

    def test_next_month_start_ts_is_future(self):
        assert _next_month_start_ts() > time.time()

    def test_next_month_start_is_after_current_month(self):
        assert _next_month_start_ts() > _month_start_ts()


# ============================================================
# ConsumerUsageTracker — unit tests
# ============================================================


class TestTrackerRecord:
    """Basic record and retrieval."""

    def test_record_increments_counts(self):
        t = ConsumerUsageTracker()
        t.record("key-1", "/api/v1/health", 200)
        usage = t.get_usage("key-1")
        assert usage is not None
        assert usage["requests_month"] == 1
        assert usage["requests_today"] == 1
        assert usage["requests_week"] == 1

    def test_record_multiple(self):
        t = ConsumerUsageTracker()
        for _ in range(5):
            t.record("key-1", "/api/v1/health", 200)
        usage = t.get_usage("key-1")
        assert usage["requests_month"] == 5

    def test_record_tracks_errors(self):
        t = ConsumerUsageTracker()
        t.record("key-1", "/api/v1/test", 200)
        t.record("key-1", "/api/v1/test", 404)
        t.record("key-1", "/api/v1/test", 500)
        usage = t.get_usage("key-1")
        assert usage["errors_month"] == 2
        assert usage["requests_month"] == 3

    def test_record_tracks_bandwidth(self):
        t = ConsumerUsageTracker()
        t.record("key-1", "/api/v1/health", 200, response_size=1024)
        t.record("key-1", "/api/v1/health", 200, response_size=2048)
        usage = t.get_usage("key-1")
        assert usage["bandwidth_bytes"] == 3072

    def test_record_multiple_keys(self):
        t = ConsumerUsageTracker()
        t.record("key-a", "/api/v1/health", 200)
        t.record("key-b", "/api/v1/health", 200)
        t.record("key-b", "/api/v1/health", 200)
        assert t.get_usage("key-a")["requests_month"] == 1
        assert t.get_usage("key-b")["requests_month"] == 2

    def test_get_usage_missing_key_returns_none(self):
        t = ConsumerUsageTracker()
        assert t.get_usage("nonexistent") is None

    def test_record_sets_last_request_at(self):
        t = ConsumerUsageTracker()
        before = time.time() - 1  # 1s tolerance for clock skew under load
        t.record("key-1", "/test", 200)
        after = time.time() + 1
        usage = t.get_usage("key-1")
        assert before <= usage["last_request_at"] <= after


class TestTrackerByEndpoint:
    """Per-endpoint breakdown."""

    def test_by_endpoint_tracking(self):
        t = ConsumerUsageTracker()
        t.record("key-1", "/api/v1/health", 200)
        t.record("key-1", "/api/v1/health", 200)
        t.record("key-1", "/api/v1/flows", 200)
        usage = t.get_usage("key-1")
        assert usage["by_endpoint"]["/api/v1/health"] == 2
        assert usage["by_endpoint"]["/api/v1/flows"] == 1

    def test_by_endpoint_separate_keys(self):
        t = ConsumerUsageTracker()
        t.record("key-a", "/api/v1/health", 200)
        t.record("key-b", "/api/v1/health", 200)
        assert t.get_usage("key-a")["by_endpoint"]["/api/v1/health"] == 1


class TestTrackerByHour:
    """Per-hour histogram."""

    def test_by_hour_current_hour(self):
        t = ConsumerUsageTracker()
        t.record("key-1", "/test", 200)
        usage = t.get_usage("key-1")
        now_hour = f"{datetime.now(timezone.utc).hour:02d}"
        assert usage["by_hour"].get(now_hour, 0) >= 1

    def test_by_hour_multiple_calls(self):
        t = ConsumerUsageTracker()
        for _ in range(3):
            t.record("key-1", "/test", 200)
        usage = t.get_usage("key-1")
        now_hour = f"{datetime.now(timezone.utc).hour:02d}"
        assert usage["by_hour"][now_hour] == 3


class TestTrackerAllUsage:
    """Aggregate usage listing."""

    def test_all_usage_empty(self):
        t = ConsumerUsageTracker()
        assert t.all_usage() == []

    def test_all_usage_returns_all_keys(self):
        t = ConsumerUsageTracker()
        t.record("key-a", "/test", 200)
        t.record("key-b", "/test", 200)
        result = t.all_usage()
        ids = {e["key_id"] for e in result}
        assert ids == {"key-a", "key-b"}

    def test_all_usage_includes_error_rate(self):
        t = ConsumerUsageTracker()
        t.record("key-1", "/test", 200)
        t.record("key-1", "/test", 500)
        result = t.all_usage()
        entry = next(e for e in result if e["key_id"] == "key-1")
        assert entry["error_rate_pct"] == 50.0


class TestTrackerClear:
    def test_clear_empties_all(self):
        t = ConsumerUsageTracker()
        t.record("key-1", "/test", 200)
        t.set_quota("key-1", 100)
        t.clear()
        assert t.get_usage("key-1") is None
        assert t.get_quota("key-1") is None
        assert t.all_usage() == []


class TestTrackerThreadSafety:
    def test_concurrent_records(self):
        t = ConsumerUsageTracker()
        errors = []

        def writer(key_id, count):
            try:
                for _ in range(count):
                    t.record(key_id, "/test", 200)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(f"k-{i}", 100)) for i in range(5)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        assert not errors
        total = sum(t.get_usage(f"k-{i}")["requests_month"] for i in range(5))
        assert total == 500


# ============================================================
# Quota system — unit tests
# ============================================================


class TestQuotaSetGet:
    def test_set_and_get_quota(self):
        t = ConsumerUsageTracker()
        t.set_quota("key-1", 1000)
        assert t.get_quota("key-1") == 1000

    def test_get_quota_unset_returns_none(self):
        t = ConsumerUsageTracker()
        assert t.get_quota("key-1") is None

    def test_clear_quota(self):
        t = ConsumerUsageTracker()
        t.set_quota("key-1", 1000)
        t.set_quota("key-1", None)
        assert t.get_quota("key-1") is None


class TestQuotaCheck:
    def test_no_quota_always_allowed(self):
        t = ConsumerUsageTracker()
        t.record("key-1", "/test", 200)
        status = t.check_quota("key-1")
        assert status["allowed"] is True
        assert status["quota"] is None
        assert status["warning"] is False

    def test_under_quota_allowed(self):
        t = ConsumerUsageTracker()
        t.set_quota("key-1", 100)
        for _ in range(50):
            t.record("key-1", "/test", 200)
        status = t.check_quota("key-1")
        assert status["allowed"] is True
        assert status["used"] == 50
        assert status["remaining"] == 50
        assert status["pct"] == 50.0
        assert status["warning"] is False

    def test_warning_at_80_percent(self):
        t = ConsumerUsageTracker()
        t.set_quota("key-1", 100)
        for _ in range(80):
            t.record("key-1", "/test", 200)
        status = t.check_quota("key-1")
        assert status["allowed"] is True
        assert status["warning"] is True
        assert status["pct"] == 80.0

    def test_warning_at_90_percent(self):
        t = ConsumerUsageTracker()
        t.set_quota("key-1", 100)
        for _ in range(90):
            t.record("key-1", "/test", 200)
        status = t.check_quota("key-1")
        assert status["allowed"] is True
        assert status["warning"] is True

    def test_blocked_at_100_percent(self):
        t = ConsumerUsageTracker()
        t.set_quota("key-1", 10)
        for _ in range(10):
            t.record("key-1", "/test", 200)
        status = t.check_quota("key-1")
        assert status["allowed"] is False
        assert status["used"] == 10
        assert status["remaining"] == 0
        assert status["pct"] == 100.0
        assert status["warning"] is False  # warning only when still allowed
        assert status["retry_after"] > 0

    def test_blocked_over_100_percent(self):
        t = ConsumerUsageTracker()
        t.set_quota("key-1", 5)
        for _ in range(8):
            t.record("key-1", "/test", 200)
        status = t.check_quota("key-1")
        assert status["allowed"] is False
        assert status["used"] == 8

    def test_retry_after_is_positive_when_blocked(self):
        t = ConsumerUsageTracker()
        t.set_quota("key-1", 1)
        t.record("key-1", "/test", 200)
        status = t.check_quota("key-1")
        assert status["retry_after"] > 0

    def test_untracked_key_with_quota_is_allowed(self):
        t = ConsumerUsageTracker()
        t.set_quota("new-key", 100)
        status = t.check_quota("new-key")
        assert status["allowed"] is True
        assert status["used"] == 0
        assert status["remaining"] == 100


class TestQuotaAllQuotas:
    def test_all_quotas_empty(self):
        t = ConsumerUsageTracker()
        assert t.all_quotas() == []

    def test_all_quotas_shows_status(self):
        t = ConsumerUsageTracker()
        t.set_quota("key-a", 100)
        t.set_quota("key-b", 10)
        for _ in range(50):
            t.record("key-a", "/test", 200)
        for _ in range(10):
            t.record("key-b", "/test", 200)
        quotas = t.all_quotas()
        a = next(q for q in quotas if q["key_id"] == "key-a")
        b = next(q for q in quotas if q["key_id"] == "key-b")
        assert a["status"] == "ok"
        assert b["status"] == "blocked"

    def test_all_quotas_warning_status(self):
        t = ConsumerUsageTracker()
        t.set_quota("key-1", 100)
        for _ in range(85):
            t.record("key-1", "/test", 200)
        quotas = t.all_quotas()
        entry = next(q for q in quotas if q["key_id"] == "key-1")
        assert entry["status"] == "warning"
        assert entry["pct"] == 85.0


class TestQuotaMonthlyReset:
    """Quota resets on month boundary."""

    def test_month_reset_clears_usage(self):
        t = ConsumerUsageTracker()
        t.set_quota("key-1", 100)
        for _ in range(50):
            t.record("key-1", "/test", 200)

        # Simulate month boundary crossing by adjusting _month_start
        t._month_start = time.time() - 1  # pretend last reset was 1 second ago
        # Force a new month start by mocking _month_start_ts
        with patch("apps.orchestrator.main._month_start_ts", return_value=time.time() + 1):
            usage = t.get_usage("key-1")
        # After reset, usage should be None (cleared)
        assert usage is None

    def test_quota_persists_after_reset(self):
        t = ConsumerUsageTracker()
        t.set_quota("key-1", 100)
        for _ in range(50):
            t.record("key-1", "/test", 200)

        # Simulate month boundary
        t._month_start = time.time() - 1
        with patch("apps.orchestrator.main._month_start_ts", return_value=time.time() + 1):
            # Quota should still be set after usage is cleared
            assert t.get_quota("key-1") == 100


# ============================================================
# API endpoint tests
# ============================================================


class TestUsageEndpoint:
    """GET /api/v1/usage"""

    def test_empty_usage(self, client):
        resp = client.get("/api/v1/usage")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_usage_after_requests(self, client):
        # Make some requests to generate usage for anonymous user
        for _ in range(3):
            client.get("/api/v1/health")
        resp = client.get("/api/v1/usage")
        assert resp.status_code == 200
        # Anonymous requests won't show (filtered out), but the endpoint works
        assert isinstance(resp.json(), list)

    def test_usage_with_tracked_key(self, client):
        # Manually inject usage data
        usage_tracker.record("test-key-1", "/api/v1/health", 200, response_size=512)
        usage_tracker.record("test-key-1", "/api/v1/health", 200, response_size=256)
        resp = client.get("/api/v1/usage")
        assert resp.status_code == 200
        data = resp.json()
        entry = next((e for e in data if e["key_id"] == "test-key-1"), None)
        assert entry is not None
        assert entry["requests_month"] == 2
        assert entry["bandwidth_bytes"] == 768


class TestUsageDetailEndpoint:
    """GET /api/v1/usage/{key_id}"""

    def test_usage_detail_not_found(self, client):
        resp = client.get("/api/v1/usage/nonexistent-key")
        assert resp.status_code == 404

    def test_usage_detail_found(self, client):
        usage_tracker.record("detail-key-1", "/api/v1/health", 200)
        usage_tracker.record("detail-key-1", "/api/v1/flows", 404)
        resp = client.get("/api/v1/usage/detail-key-1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["key_id"] == "detail-key-1"
        assert data["requests_month"] == 2
        assert data["errors_month"] == 1
        assert "/api/v1/health" in data["by_endpoint"]
        assert "/api/v1/flows" in data["by_endpoint"]

    def test_usage_detail_error_rate(self, client):
        usage_tracker.record("err-key", "/test", 200)
        usage_tracker.record("err-key", "/test", 500)
        resp = client.get("/api/v1/usage/err-key")
        data = resp.json()
        assert data["error_rate_pct"] == 50.0


class TestQuotasEndpoint:
    """GET /api/v1/quotas"""

    def test_empty_quotas(self, client):
        resp = client.get("/api/v1/quotas")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_quotas_with_data(self, client):
        usage_tracker.set_quota("q-key-1", 1000)
        usage_tracker.record("q-key-1", "/test", 200)
        resp = client.get("/api/v1/quotas")
        assert resp.status_code == 200
        data = resp.json()
        entry = next(q for q in data if q["key_id"] == "q-key-1")
        assert entry["quota"] == 1000
        assert entry["used"] == 1
        assert entry["remaining"] == 999
        assert entry["status"] == "ok"


class TestSetQuotaEndpoint:
    """PUT /api/v1/quotas/{key_id}"""

    def test_set_quota(self, client):
        resp = client.put(
            "/api/v1/quotas/my-key",
            json={"monthly_limit": 5000},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["key_id"] == "my-key"
        assert data["monthly_limit"] == 5000
        assert data["status"] == "quota_set"

    def test_clear_quota(self, client):
        usage_tracker.set_quota("clear-key", 1000)
        resp = client.put(
            "/api/v1/quotas/clear-key",
            json={"monthly_limit": None},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "quota_cleared"
        assert usage_tracker.get_quota("clear-key") is None

    def test_set_quota_validates_min(self, client):
        resp = client.put(
            "/api/v1/quotas/bad-key",
            json={"monthly_limit": 0},
        )
        assert resp.status_code == 422

    def test_set_quota_validates_max(self, client):
        resp = client.put(
            "/api/v1/quotas/bad-key",
            json={"monthly_limit": 99_999_999},
        )
        assert resp.status_code == 422

    def test_quota_persists_and_visible(self, client):
        client.put("/api/v1/quotas/persist-key", json={"monthly_limit": 500})
        resp = client.get("/api/v1/quotas")
        data = resp.json()
        entry = next(q for q in data if q["key_id"] == "persist-key")
        assert entry["quota"] == 500


# ============================================================
# Middleware integration tests
# ============================================================


class TestQuotaEnforcementMiddleware:
    """enforce_quota middleware blocks at 100% and warns at 80%."""

    def _mock_user(self, key_id="test-user:abc"):
        """Patch _resolve_rate_limit_user to return a non-anonymous principal."""
        principal = {"id": key_id, "tier": "free"}
        return patch(
            "apps.orchestrator.main._resolve_rate_limit_user",
            return_value=principal,
        )

    def test_quota_exceeded_returns_429(self, client):
        key_id = "test-user:abc"
        usage_tracker.set_quota(key_id, 2)
        usage_tracker.record(key_id, "/test", 200)
        usage_tracker.record(key_id, "/test", 200)
        with self._mock_user(key_id):
            resp = client.get("/api/v1/health")
        assert resp.status_code == 429
        body = resp.json()
        assert body["error"]["code"] == "QUOTA_EXCEEDED"
        assert "Retry-After" in resp.headers

    def test_quota_exceeded_retry_after_positive(self, client):
        key_id = "test-user:def"
        usage_tracker.set_quota(key_id, 1)
        usage_tracker.record(key_id, "/test", 200)
        with self._mock_user(key_id):
            resp = client.get("/api/v1/health")
        assert resp.status_code == 429
        retry_after = int(resp.headers["Retry-After"])
        assert retry_after > 0

    def test_no_quota_not_blocked(self, client):
        for _ in range(5):
            resp = client.get("/api/v1/health")
            assert resp.status_code == 200

    def test_under_quota_allowed(self, client):
        key_id = "test-user:ghi"
        usage_tracker.set_quota(key_id, 100)
        with self._mock_user(key_id):
            resp = client.get("/api/v1/health")
        assert resp.status_code == 200

    def test_quota_exceeded_message_includes_counts(self, client):
        key_id = "test-user:msg"
        usage_tracker.set_quota(key_id, 5)
        for _ in range(5):
            usage_tracker.record(key_id, "/test", 200)
        with self._mock_user(key_id):
            resp = client.get("/api/v1/health")
        assert resp.status_code == 429
        msg = resp.json()["error"]["message"]
        assert "5/5" in msg

    def test_anonymous_not_blocked_by_quota(self, client):
        """Anonymous users are never quota-checked."""
        # Even if somehow a quota exists for the anonymous key, skip it
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200


class TestQuotaWarningHeader:
    """X-Quota-Warning header at 80%."""

    def _mock_user(self, key_id="warn-user:abc"):
        principal = {"id": key_id, "tier": "free"}
        return patch(
            "apps.orchestrator.main._resolve_rate_limit_user",
            return_value=principal,
        )

    def test_warning_header_at_80_pct(self, client):
        key_id = "warn-user:abc"
        usage_tracker.set_quota(key_id, 10)
        for _ in range(8):
            usage_tracker.record(key_id, "/test", 200)
        with self._mock_user(key_id):
            resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        assert resp.headers.get("x-quota-warning") == "true"
        assert resp.headers.get("x-quota-remaining") is not None

    def test_no_warning_under_80_pct(self, client):
        key_id = "warn-user:def"
        usage_tracker.set_quota(key_id, 100)
        with self._mock_user(key_id):
            resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        assert resp.headers.get("x-quota-warning") is None


class TestUsageTrackedViaMiddleware:
    """collect_metrics middleware tracks per-key usage."""

    def test_anonymous_not_tracked(self, client):
        """Anonymous requests should not be tracked in usage_tracker."""
        client.get("/api/v1/health")
        all_usage = usage_tracker.all_usage()
        anon_keys = [u for u in all_usage if "anonymous" in u["key_id"]]
        assert len(anon_keys) == 0

    def test_authenticated_user_tracked(self, client):
        """Non-anonymous users should be tracked."""
        key_id = "tracked-user:xyz"
        principal = {"id": key_id, "tier": "pro"}
        with patch(
            "apps.orchestrator.main._resolve_rate_limit_user",
            return_value=principal,
        ):
            client.get("/api/v1/health")
        usage = usage_tracker.get_usage(key_id)
        assert usage is not None
        assert usage["requests_month"] >= 1


# ============================================================
# Edge cases
# ============================================================


class TestEdgeCases:
    def test_usage_detail_with_colon_in_key_id(self, client):
        """Key IDs like 'admin-key:uuid' contain colons — path should handle."""
        usage_tracker.record("admin-key:abc-123", "/test", 200)
        resp = client.get("/api/v1/usage/admin-key:abc-123")
        assert resp.status_code == 200
        assert resp.json()["key_id"] == "admin-key:abc-123"

    def test_quota_for_key_with_colon(self, client):
        resp = client.put(
            "/api/v1/quotas/managed-key:xyz-456",
            json={"monthly_limit": 500},
        )
        assert resp.status_code == 200
        assert resp.json()["key_id"] == "managed-key:xyz-456"

    def test_zero_requests_error_rate(self):
        t = ConsumerUsageTracker()
        t.set_quota("empty-key", 100)
        # Force creation of record
        t.record("empty-key", "/test", 200)
        result = t.all_usage()
        entry = next(e for e in result if e["key_id"] == "empty-key")
        assert entry["error_rate_pct"] == 0.0

    def test_all_quotas_with_no_usage_data(self):
        t = ConsumerUsageTracker()
        t.set_quota("unused-key", 500)
        quotas = t.all_quotas()
        entry = next(q for q in quotas if q["key_id"] == "unused-key")
        assert entry["used"] == 0
        assert entry["remaining"] == 500
        assert entry["status"] == "ok"

    def test_quota_exactly_at_boundary(self):
        t = ConsumerUsageTracker()
        t.set_quota("boundary-key", 100)
        for _ in range(79):
            t.record("boundary-key", "/test", 200)
        status = t.check_quota("boundary-key")
        assert status["warning"] is False  # 79% < 80%
        assert status["allowed"] is True

        t.record("boundary-key", "/test", 200)
        status = t.check_quota("boundary-key")
        assert status["warning"] is True  # exactly 80%
        assert status["allowed"] is True
