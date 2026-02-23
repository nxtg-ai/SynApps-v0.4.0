"""
Tests for DIRECTIVE-NXTG-20260223-12 — Connector Health Dashboard.

Covers:
- ConnectorStatus.DOWN enum value
- ConnectorHealthTracker: latency/error windowed metrics, dashboard_status
  derivation, last_success tracking, _prune_window, all_dashboard_statuses
- probe_connector(): latency measurement, HTTP ping, timeout handling
- probe_all_connectors(): caching behaviour (30 s TTL)
- GET /connectors/health: dashboard_status, summary.down, summary.degraded
- GET /api/v1/health: aggregate status from connector dashboard statuses
- Rate limit exemption for /api/v1/connectors/health
- Status transitions: healthy→degraded→down, recovery paths
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from apps.orchestrator.main import (
    ConnectorHealthTracker,
    ConnectorStatus,
    HEALTH_CACHE_TTL_SECONDS,
    HEALTH_PROBE_TIMEOUT_SECONDS,
    HEALTH_WINDOW_SECONDS,
    _health_cache,
    connector_health,
    probe_all_connectors,
    probe_connector,
    app,
)


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _reset_state():
    """Reset global connector_health and cache between tests."""
    connector_health.reset()
    _health_cache["results"] = None
    _health_cache["timestamp"] = 0.0
    yield
    connector_health.reset()
    _health_cache["results"] = None
    _health_cache["timestamp"] = 0.0


# ---------------------------------------------------------------------------
# ConnectorStatus enum — DOWN value
# ---------------------------------------------------------------------------

class TestConnectorStatusDown:

    def test_down_value(self):
        assert ConnectorStatus.DOWN == "down"

    def test_all_values_present(self):
        values = {s.value for s in ConnectorStatus}
        assert values == {"healthy", "degraded", "down", "disabled"}


# ---------------------------------------------------------------------------
# ConnectorHealthTracker — latency & error windowed metrics
# ---------------------------------------------------------------------------

class TestTrackerLatencyMetrics:

    def test_record_success_with_latency(self):
        t = ConnectorHealthTracker()
        t.record_success("openai", latency_ms=150.0)
        s = t.get_status("openai")
        assert s["last_success"] is not None
        assert len(s["latency_samples"]) == 1
        assert s["latency_samples"][0][1] == 150.0

    def test_record_failure_with_latency(self):
        t = ConnectorHealthTracker()
        t.record_failure("openai", "timeout", latency_ms=3000.0)
        s = t.get_status("openai")
        assert len(s["latency_samples"]) == 1
        assert len(s["error_samples"]) == 1

    def test_latency_samples_accumulate(self):
        t = ConnectorHealthTracker()
        for i in range(1, 6):
            t.record_success("openai", latency_ms=float(i * 100))
        s = t.get_status("openai")
        assert len(s["latency_samples"]) == 5

    def test_zero_latency_not_recorded(self):
        t = ConnectorHealthTracker()
        t.record_success("openai", latency_ms=0.0)
        s = t.get_status("openai")
        assert len(s["latency_samples"]) == 0

    def test_prune_window_removes_old_samples(self):
        t = ConnectorHealthTracker()
        now = time.time()
        # Manually inject old samples
        s = t._ensure("openai")
        s["latency_samples"] = [
            (now - 600, 100.0),  # 10 min ago — outside window
            (now - 200, 200.0),  # 3 min ago — inside window
        ]
        s["error_samples"] = [now - 600, now - 100]
        t._prune_window(s, now)
        assert len(s["latency_samples"]) == 1
        assert s["latency_samples"][0][1] == 200.0
        assert len(s["error_samples"]) == 1


# ---------------------------------------------------------------------------
# Dashboard status derivation
# ---------------------------------------------------------------------------

class TestDashboardStatus:

    def test_healthy_when_no_errors_low_latency(self):
        t = ConnectorHealthTracker()
        t.record_success("openai", latency_ms=100.0)
        ds = t.get_dashboard_status("openai")
        assert ds["dashboard_status"] == ConnectorStatus.HEALTHY
        assert ds["avg_latency_ms"] == 100.0
        assert ds["error_count_5m"] == 0

    def test_degraded_when_errors_below_5(self):
        t = ConnectorHealthTracker(disable_threshold=10)
        t.record_failure("openai", "err", latency_ms=100.0)
        ds = t.get_dashboard_status("openai")
        assert ds["dashboard_status"] == ConnectorStatus.DEGRADED
        assert ds["error_count_5m"] == 1

    def test_degraded_when_latency_above_500(self):
        t = ConnectorHealthTracker()
        t.record_success("openai", latency_ms=700.0)
        ds = t.get_dashboard_status("openai")
        assert ds["dashboard_status"] == ConnectorStatus.DEGRADED
        assert ds["avg_latency_ms"] == 700.0

    def test_down_when_5_or_more_errors(self):
        t = ConnectorHealthTracker(disable_threshold=10)
        for i in range(5):
            t.record_failure("openai", f"err{i}", latency_ms=100.0)
        ds = t.get_dashboard_status("openai")
        assert ds["dashboard_status"] == ConnectorStatus.DOWN
        assert ds["error_count_5m"] == 5

    def test_down_when_latency_above_2000(self):
        t = ConnectorHealthTracker()
        t.record_success("openai", latency_ms=2500.0)
        ds = t.get_dashboard_status("openai")
        assert ds["dashboard_status"] == ConnectorStatus.DOWN

    def test_down_when_disabled(self):
        t = ConnectorHealthTracker(disable_threshold=2)
        t.record_failure("openai", "a", latency_ms=100.0)
        t.record_failure("openai", "b", latency_ms=100.0)
        assert t.is_disabled("openai")
        ds = t.get_dashboard_status("openai")
        assert ds["dashboard_status"] == ConnectorStatus.DOWN

    def test_last_success_field(self):
        t = ConnectorHealthTracker()
        t.record_success("openai", latency_ms=50.0)
        ds = t.get_dashboard_status("openai")
        assert ds["last_success"] is not None
        assert isinstance(ds["last_success"], float)

    def test_sample_count_5m(self):
        t = ConnectorHealthTracker()
        t.record_success("openai", latency_ms=50.0)
        t.record_success("openai", latency_ms=100.0)
        ds = t.get_dashboard_status("openai")
        assert ds["sample_count_5m"] == 2

    def test_avg_latency_zero_when_no_samples(self):
        t = ConnectorHealthTracker()
        t.record_success("openai")  # no latency_ms
        ds = t.get_dashboard_status("openai")
        assert ds["avg_latency_ms"] == 0.0

    def test_recovery_from_degraded_to_healthy(self):
        t = ConnectorHealthTracker(disable_threshold=10)
        # Record one failure to go degraded
        t.record_failure("openai", "err", latency_ms=100.0)
        ds = t.get_dashboard_status("openai")
        assert ds["dashboard_status"] == ConnectorStatus.DEGRADED
        # Record success to recover — but error is still in window
        t.record_success("openai", latency_ms=50.0)
        ds = t.get_dashboard_status("openai")
        # Still degraded because error_count_5m >= 1
        assert ds["dashboard_status"] == ConnectorStatus.DEGRADED


# ---------------------------------------------------------------------------
# all_dashboard_statuses
# ---------------------------------------------------------------------------

class TestAllDashboardStatuses:

    def test_returns_all_tracked(self):
        t = ConnectorHealthTracker()
        t.record_success("openai", latency_ms=100.0)
        t.record_failure("anthropic", "err", latency_ms=200.0)
        all_ds = t.all_dashboard_statuses()
        assert "openai" in all_ds
        assert "anthropic" in all_ds
        assert all_ds["openai"]["dashboard_status"] == ConnectorStatus.HEALTHY
        assert all_ds["anthropic"]["dashboard_status"] == ConnectorStatus.DEGRADED

    def test_empty_when_no_connectors(self):
        t = ConnectorHealthTracker()
        assert t.all_dashboard_statuses() == {}


# ---------------------------------------------------------------------------
# probe_connector — latency & HTTP ping
# ---------------------------------------------------------------------------

class TestProbeConnectorEnhanced:

    @pytest.mark.asyncio
    async def test_probe_returns_latency_ms(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            result = await probe_connector("openai")
            assert "latency_ms" in result
            assert isinstance(result["latency_ms"], float)
            assert result["latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_probe_unknown_returns_zero_latency(self):
        result = await probe_connector("nonexistent")
        assert result["latency_ms"] == 0.0

    @pytest.mark.asyncio
    async def test_probe_records_latency_in_tracker(self):
        connector_health.reset()
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            await probe_connector("openai")
        s = connector_health.get_status("openai")
        assert s["total_probes"] >= 1


# ---------------------------------------------------------------------------
# probe_all_connectors — caching
# ---------------------------------------------------------------------------

class TestProbeAllCaching:

    @pytest.mark.asyncio
    async def test_caches_results(self):
        r1 = await probe_all_connectors()
        assert r1 is not None
        assert _health_cache["results"] is not None
        # Second call should return cached
        r2 = await probe_all_connectors()
        assert r2 is r1  # same object (cached)

    @pytest.mark.asyncio
    async def test_cache_expires(self):
        r1 = await probe_all_connectors()
        # Artificially expire cache
        _health_cache["timestamp"] = time.time() - HEALTH_CACHE_TTL_SECONDS - 1
        r2 = await probe_all_connectors()
        # Should be a new list (not same object)
        assert r2 is not r1

    @pytest.mark.asyncio
    async def test_cache_ttl_constant(self):
        assert HEALTH_CACHE_TTL_SECONDS == 30

    @pytest.mark.asyncio
    async def test_probe_timeout_constant(self):
        assert HEALTH_PROBE_TIMEOUT_SECONDS == 5

    @pytest.mark.asyncio
    async def test_window_constant(self):
        assert HEALTH_WINDOW_SECONDS == 300


# ---------------------------------------------------------------------------
# GET /connectors/health endpoint — enhanced fields
# ---------------------------------------------------------------------------

class TestConnectorsHealthDashboard:

    def test_returns_200(self, client):
        resp = client.get("/api/v1/connectors/health")
        assert resp.status_code == 200

    def test_summary_has_down(self, client):
        data = client.get("/api/v1/connectors/health").json()
        summary = data["summary"]
        assert "healthy" in summary
        assert "degraded" in summary
        assert "down" in summary
        assert "disabled" in summary

    def test_connector_has_dashboard_status(self, client):
        data = client.get("/api/v1/connectors/health").json()
        for c in data["connectors"]:
            assert "dashboard_status" in c
            assert c["dashboard_status"] in ("healthy", "degraded", "down")

    def test_connector_has_latency(self, client):
        data = client.get("/api/v1/connectors/health").json()
        for c in data["connectors"]:
            assert "latency_ms" in c
            assert "avg_latency_ms" in c

    def test_connector_has_error_count(self, client):
        data = client.get("/api/v1/connectors/health").json()
        for c in data["connectors"]:
            assert "error_count_5m" in c
            assert isinstance(c["error_count_5m"], int)

    def test_connector_has_last_success(self, client):
        data = client.get("/api/v1/connectors/health").json()
        for c in data["connectors"]:
            assert "last_success" in c

    def test_summary_counts_match_total(self, client):
        data = client.get("/api/v1/connectors/health").json()
        s = data["summary"]
        # dashboard_status counts should match total
        assert s["healthy"] + s["degraded"] + s["down"] == data["total"]


# ---------------------------------------------------------------------------
# GET /api/v1/health — aggregate status from connectors
# ---------------------------------------------------------------------------

class TestHealthAggregateStatus:

    def test_healthy_when_no_connectors_tracked(self, client):
        connector_health.reset()
        resp = client.get("/api/v1/health")
        data = resp.json()
        assert data["status"] == "healthy"

    def test_healthy_when_all_healthy(self, client):
        connector_health.reset()
        connector_health.record_success("openai", latency_ms=50.0)
        connector_health.record_success("anthropic", latency_ms=60.0)
        resp = client.get("/api/v1/health")
        data = resp.json()
        assert data["status"] == "healthy"

    def test_degraded_when_one_degraded(self, client):
        connector_health.reset()
        connector_health.record_success("openai", latency_ms=50.0)
        # Record a failure — error_count_5m >= 1 → degraded
        connector_health.record_failure("anthropic", "err", latency_ms=100.0)
        resp = client.get("/api/v1/health")
        data = resp.json()
        assert data["status"] == "degraded"

    def test_down_when_one_down(self, client):
        connector_health.reset()
        connector_health.record_success("openai", latency_ms=50.0)
        # 5 errors → down
        for i in range(5):
            connector_health.record_failure("anthropic", f"err{i}", latency_ms=100.0)
        resp = client.get("/api/v1/health")
        data = resp.json()
        assert data["status"] == "down"

    def test_still_has_version_and_uptime(self, client):
        resp = client.get("/api/v1/health")
        data = resp.json()
        assert "version" in data
        assert "uptime" in data
        assert "active_connectors" in data
        assert "service" in data


# ---------------------------------------------------------------------------
# Rate limit exemption
# ---------------------------------------------------------------------------

class TestRateLimitExemption:

    def test_connectors_health_exempt(self):
        from apps.orchestrator.middleware.rate_limiter import EXEMPT_PATHS
        assert "/api/v1/connectors/health" in EXEMPT_PATHS

    def test_health_endpoint_exempt(self):
        from apps.orchestrator.middleware.rate_limiter import EXEMPT_PATHS
        assert "/api/v1/health" in EXEMPT_PATHS

    def test_health_detailed_exempt(self):
        from apps.orchestrator.middleware.rate_limiter import EXEMPT_PATHS
        assert "/api/v1/health/detailed" in EXEMPT_PATHS


# ---------------------------------------------------------------------------
# Status transition scenarios
# ---------------------------------------------------------------------------

class TestStatusTransitions:

    def test_healthy_to_degraded_via_error(self):
        t = ConnectorHealthTracker(disable_threshold=10)
        t.record_success("x", latency_ms=50.0)
        assert t.get_dashboard_status("x")["dashboard_status"] == ConnectorStatus.HEALTHY
        t.record_failure("x", "err", latency_ms=50.0)
        assert t.get_dashboard_status("x")["dashboard_status"] == ConnectorStatus.DEGRADED

    def test_healthy_to_degraded_via_high_latency(self):
        t = ConnectorHealthTracker()
        t.record_success("x", latency_ms=600.0)
        assert t.get_dashboard_status("x")["dashboard_status"] == ConnectorStatus.DEGRADED

    def test_degraded_to_down_via_many_errors(self):
        t = ConnectorHealthTracker(disable_threshold=20)
        for i in range(5):
            t.record_failure("x", f"err{i}", latency_ms=100.0)
        assert t.get_dashboard_status("x")["dashboard_status"] == ConnectorStatus.DOWN

    def test_down_recovery_after_window_expires(self):
        t = ConnectorHealthTracker(disable_threshold=20)
        now = time.time()
        # Inject old errors (outside window)
        s = t._ensure("x")
        s["error_samples"] = [now - 400 for _ in range(5)]
        s["latency_samples"] = [(now, 100.0)]
        s["consecutive_failures"] = 0
        s["status"] = ConnectorStatus.HEALTHY
        ds = t.get_dashboard_status("x")
        # Errors should be pruned — connector should be healthy
        assert ds["dashboard_status"] == ConnectorStatus.HEALTHY
        assert ds["error_count_5m"] == 0

    def test_probe_endpoint_includes_dashboard_status(self, client):
        resp = client.post("/api/v1/connectors/openai/probe")
        data = resp.json()
        assert "dashboard_status" in data
        assert "avg_latency_ms" in data
        assert "error_count_5m" in data
        assert "last_success" in data
