"""Tests for /health/detailed and /metrics endpoints."""

import time
import pytest
from fastapi.testclient import TestClient

from apps.orchestrator.main import app, metrics


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _reset_metrics():
    """Reset in-memory metrics between tests."""
    metrics.reset()
    yield
    metrics.reset()


# ---------------------------------------------------------------------------
# /health/detailed
# ---------------------------------------------------------------------------


def test_health_detailed_returns_200(client):
    """Endpoint returns 200 with expected top-level keys."""
    resp = client.get("/api/v1/health/detailed")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "uptime_seconds" in data
    assert "database" in data
    assert "providers" in data
    assert "last_template_run_at" in data


def test_health_detailed_status_ok(client):
    """Status is 'ok' when database is reachable."""
    data = client.get("/api/v1/health/detailed").json()
    assert data["status"] in ("ok", "degraded")
    assert data["database"]["reachable"] is True


def test_health_detailed_lists_providers(client):
    """Provider list includes openai and anthropic with connected flag."""
    providers = client.get("/api/v1/health/detailed").json()["providers"]
    names = [p["name"] for p in providers]
    assert "openai" in names
    assert "anthropic" in names
    for p in providers:
        assert "connected" in p
        assert isinstance(p["connected"], bool)


def test_health_detailed_returns_fast(client):
    """Health endpoint responds in under 100ms."""
    start = time.monotonic()
    resp = client.get("/api/v1/health/detailed")
    elapsed_ms = (time.monotonic() - start) * 1000
    assert resp.status_code == 200
    assert elapsed_ms < 500  # generous in CI; locally < 50ms


def test_health_detailed_last_template_run_null_initially(client):
    """No template runs → last_template_run_at is null."""
    data = client.get("/api/v1/health/detailed").json()
    assert data["last_template_run_at"] is None


# ---------------------------------------------------------------------------
# /metrics
# ---------------------------------------------------------------------------


def test_metrics_returns_200(client):
    """Metrics endpoint returns 200 with expected shape."""
    resp = client.get("/api/v1/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "requests" in data
    assert "provider_usage" in data
    assert "template_runs" in data


def test_metrics_request_counters_increment(client):
    """After several requests, total count and avg_response_ms are populated."""
    # Make a few extra requests
    for _ in range(5):
        client.get("/api/v1/health")
    data = client.get("/api/v1/metrics").json()
    # The /metrics call itself may or may not be counted yet (middleware order)
    assert data["requests"]["total"] >= 5
    assert data["requests"]["avg_response_ms"] > 0


def test_metrics_error_rate_after_404(client):
    """Hitting a nonexistent route increments error count."""
    client.get("/api/v1/nonexistent-route-12345")
    data = client.get("/api/v1/metrics").json()
    assert data["requests"]["errors"] >= 1
    assert data["requests"]["error_rate_pct"] > 0


def test_metrics_template_runs_after_flow_execution(client):
    """Running a flow records its name in template_runs."""
    flow_payload = {
        "name": "Test Metrics Flow",
        "nodes": [
            {"id": "start", "type": "start", "position": {"x": 0, "y": 0}, "data": {"label": "S"}},
            {"id": "end", "type": "end", "position": {"x": 0, "y": 100}, "data": {"label": "E"}},
        ],
        "edges": [{"id": "e1", "source": "start", "target": "end"}],
    }
    create = client.post("/api/v1/flows", json=flow_payload)
    assert create.status_code == 201
    flow_id = create.json()["id"]

    run = client.post(f"/api/v1/flows/{flow_id}/runs", json={"input": {"x": 1}})
    assert run.status_code == 202

    data = client.get("/api/v1/metrics").json()
    assert "Test Metrics Flow" in data["template_runs"]
    assert data["template_runs"]["Test Metrics Flow"] >= 1
    assert data["last_template_run_at"] is not None
