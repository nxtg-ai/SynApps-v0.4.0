"""
Tests for DIRECTIVE-NXTG-20260223-06 — Health Dashboard Endpoint + Metrics Collection.

Covers:
- _MetricsRingBuffer: push, query, overflow, capacity, clear
- _MetricsCollector: windowed stats, per-connector stats, snapshot
- /health endpoint: status, uptime, version, active_connectors
- /metrics endpoint: request counts, windows, connector_stats
"""

import time
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from apps.orchestrator.main import (
    _MetricsRingBuffer,
    _MetricsCollector,
    app,
)


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Ring Buffer unit tests
# ---------------------------------------------------------------------------

class TestMetricsRingBuffer:

    def test_push_and_query_all(self):
        buf = _MetricsRingBuffer(capacity=5)
        now = time.time()
        for i in range(3):
            buf.push(float(i), ts=now)
        assert len(buf) == 3
        assert buf.all_values() == [0.0, 1.0, 2.0]

    def test_query_within_window(self):
        buf = _MetricsRingBuffer(capacity=10)
        now = time.time()
        buf.push(1.0, ts=now - 7200)  # 2h ago — outside 1h window
        buf.push(2.0, ts=now - 1800)  # 30min ago — inside 1h window
        buf.push(3.0, ts=now)          # now — inside
        values = buf.query(3600)
        assert values == [2.0, 3.0]

    def test_overflow_wraps_around(self):
        buf = _MetricsRingBuffer(capacity=3)
        now = time.time()
        for i in range(5):
            buf.push(float(i), ts=now)
        assert len(buf) == 3
        # Oldest entries (0, 1) are overwritten; newest are 2, 3, 4
        assert buf.all_values() == [2.0, 3.0, 4.0]

    def test_overflow_preserves_newest(self):
        buf = _MetricsRingBuffer(capacity=4)
        now = time.time()
        for i in range(10):
            buf.push(float(i), ts=now)
        assert len(buf) == 4
        assert buf.all_values() == [6.0, 7.0, 8.0, 9.0]

    def test_clear(self):
        buf = _MetricsRingBuffer(capacity=5)
        buf.push(1.0)
        buf.push(2.0)
        buf.clear()
        assert len(buf) == 0
        assert buf.all_values() == []

    def test_capacity_property(self):
        buf = _MetricsRingBuffer(capacity=42)
        assert buf.capacity == 42

    def test_empty_query(self):
        buf = _MetricsRingBuffer(capacity=5)
        assert buf.query(3600) == []

    def test_query_excludes_old_entries(self):
        buf = _MetricsRingBuffer(capacity=100)
        now = time.time()
        for i in range(50):
            buf.push(1.0, ts=now - 90000)  # 25h ago
        for i in range(10):
            buf.push(2.0, ts=now)  # now
        one_hour = buf.query(3600)
        assert len(one_hour) == 10
        day = buf.query(86400)
        assert len(day) == 10

    def test_exact_capacity_fill(self):
        buf = _MetricsRingBuffer(capacity=5)
        for i in range(5):
            buf.push(float(i))
        assert len(buf) == 5
        assert buf.all_values() == [0.0, 1.0, 2.0, 3.0, 4.0]


# ---------------------------------------------------------------------------
# Metrics Collector unit tests
# ---------------------------------------------------------------------------

class TestMetricsCollector:

    def test_record_request_increments_total(self):
        m = _MetricsCollector(ring_capacity=100)
        m.record_request(10.0, 200, "/test")
        m.record_request(20.0, 200, "/test")
        snap = m.snapshot()
        assert snap["requests"]["total"] == 2
        assert snap["requests"]["errors"] == 0

    def test_record_request_counts_errors(self):
        m = _MetricsCollector(ring_capacity=100)
        m.record_request(10.0, 200, "/ok")
        m.record_request(5.0, 500, "/fail")
        m.record_request(3.0, 404, "/missing")
        snap = m.snapshot()
        assert snap["requests"]["total"] == 3
        assert snap["requests"]["errors"] == 2
        assert snap["requests"]["error_rate_pct"] == pytest.approx(66.67, abs=0.1)

    def test_avg_response_ms(self):
        m = _MetricsCollector(ring_capacity=100)
        m.record_request(10.0, 200, "/a")
        m.record_request(30.0, 200, "/b")
        snap = m.snapshot()
        assert snap["requests"]["avg_response_ms"] == 20.0

    def test_windowed_stats_in_snapshot(self):
        m = _MetricsCollector(ring_capacity=100)
        for _ in range(5):
            m.record_request(10.0, 200, "/x")
        snap = m.snapshot()
        assert "last_1h" in snap["requests"]
        assert "last_24h" in snap["requests"]
        assert snap["requests"]["last_1h"]["count"] == 5
        assert snap["requests"]["last_24h"]["count"] == 5

    def test_windowed_errors_in_snapshot(self):
        m = _MetricsCollector(ring_capacity=100)
        m.record_request(10.0, 500, "/err")
        m.record_request(10.0, 200, "/ok")
        snap = m.snapshot()
        assert snap["requests"]["errors_last_1h"] == 1
        assert snap["requests"]["errors_last_24h"] == 1

    def test_record_provider_call_with_duration(self):
        m = _MetricsCollector(ring_capacity=100)
        m.record_provider_call("openai", 150.0)
        m.record_provider_call("openai", 250.0)
        m.record_provider_call("anthropic", 100.0)
        snap = m.snapshot()
        assert snap["provider_usage"]["openai"] == 2
        assert snap["provider_usage"]["anthropic"] == 1
        assert "connector_stats" in snap
        assert snap["connector_stats"]["openai"]["total_calls"] == 2
        assert snap["connector_stats"]["openai"]["last_1h"]["count"] == 2

    def test_connector_stats_per_provider(self):
        m = _MetricsCollector(ring_capacity=100)
        m.record_provider_call("openai", 100.0)
        m.record_provider_call("google", 200.0)
        snap = m.snapshot()
        assert "openai" in snap["connector_stats"]
        assert "google" in snap["connector_stats"]
        assert snap["connector_stats"]["openai"]["last_1h"]["avg_ms"] == 100.0
        assert snap["connector_stats"]["google"]["last_1h"]["avg_ms"] == 200.0

    def test_percentiles(self):
        m = _MetricsCollector(ring_capacity=200)
        # Record 100 requests with durations 1..100
        for i in range(1, 101):
            m.record_request(float(i), 200, "/x")
        snap = m.snapshot()
        stats_1h = snap["requests"]["last_1h"]
        assert stats_1h["count"] == 100
        assert stats_1h["p50_ms"] == 50.0 or stats_1h["p50_ms"] == 51.0
        assert stats_1h["p95_ms"] >= 95.0
        assert stats_1h["p99_ms"] >= 99.0

    def test_template_runs(self):
        m = _MetricsCollector(ring_capacity=100)
        m.record_template_run("2brain")
        m.record_template_run("2brain")
        m.record_template_run("content-engine")
        snap = m.snapshot()
        assert snap["template_runs"]["2brain"] == 2
        assert snap["template_runs"]["content-engine"] == 1
        assert snap["last_template_run_at"] is not None

    def test_reset(self):
        m = _MetricsCollector(ring_capacity=100)
        m.record_request(10.0, 200, "/x")
        m.record_provider_call("openai", 50.0)
        m.record_template_run("t1")
        m.reset()
        snap = m.snapshot()
        assert snap["requests"]["total"] == 0
        assert snap["provider_usage"] == {}
        assert snap["connector_stats"] == {}
        assert snap["template_runs"] == {}

    def test_ring_buffer_overflow_in_collector(self):
        """When ring buffer overflows, only recent data remains."""
        m = _MetricsCollector(ring_capacity=5)
        now = time.time()
        for i in range(10):
            m.record_request(float(i), 200, "/x")
        snap = m.snapshot()
        # All 10 counted in total, but ring only has last 5
        assert snap["requests"]["total"] == 10
        assert snap["requests"]["last_1h"]["count"] == 5


# ---------------------------------------------------------------------------
# Health endpoint tests
# ---------------------------------------------------------------------------

class TestHealthEndpoint:

    def test_health_returns_healthy(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "uptime" in data

    def test_health_has_active_connectors(self, client):
        resp = client.get("/api/v1/health")
        data = resp.json()
        assert "active_connectors" in data
        assert isinstance(data["active_connectors"], int)
        assert data["active_connectors"] >= 0

    def test_health_root(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "active_connectors" in data

    def test_health_unversioned(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "active_connectors" in data


# ---------------------------------------------------------------------------
# Health detailed endpoint tests
# ---------------------------------------------------------------------------

class TestHealthDetailedEndpoint:

    def test_detailed_health_structure(self, client):
        resp = client.get("/api/v1/health/detailed")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("ok", "degraded", "down")
        assert "uptime_seconds" in data
        assert "database" in data
        assert "providers" in data

    def test_detailed_health_providers_list(self, client):
        resp = client.get("/api/v1/health/detailed")
        data = resp.json()
        providers = data["providers"]
        assert isinstance(providers, list)
        for p in providers:
            assert "name" in p
            assert "connected" in p


# ---------------------------------------------------------------------------
# Metrics endpoint tests
# ---------------------------------------------------------------------------

class TestMetricsEndpoint:

    def test_metrics_returns_requests_block(self, client):
        resp = client.get("/api/v1/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "requests" in data
        req = data["requests"]
        assert "total" in req
        assert "errors" in req
        assert "error_rate_pct" in req
        assert "avg_response_ms" in req

    def test_metrics_has_windowed_stats(self, client):
        resp = client.get("/api/v1/metrics")
        data = resp.json()
        req = data["requests"]
        assert "last_1h" in req
        assert "last_24h" in req
        assert "errors_last_1h" in req
        assert "errors_last_24h" in req

    def test_metrics_has_connector_stats(self, client):
        resp = client.get("/api/v1/metrics")
        data = resp.json()
        assert "connector_stats" in data
        assert isinstance(data["connector_stats"], dict)

    def test_metrics_has_template_runs(self, client):
        resp = client.get("/api/v1/metrics")
        data = resp.json()
        assert "template_runs" in data
        assert "last_template_run_at" in data

    def test_metrics_1h_window_has_percentiles(self, client):
        # Make a few requests first to populate metrics
        for _ in range(3):
            client.get("/api/v1/health")
        resp = client.get("/api/v1/metrics")
        data = resp.json()
        stats_1h = data["requests"]["last_1h"]
        assert stats_1h["count"] > 0
        assert "avg_ms" in stats_1h
        assert "p50_ms" in stats_1h
        assert "p95_ms" in stats_1h
        assert "p99_ms" in stats_1h
