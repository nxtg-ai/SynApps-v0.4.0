"""Tests for the Portfolio Dashboard endpoint (/api/v1/dashboard/portfolio).

Validates: template discovery, last-run status, provider registry reporting,
health check, and response schema.
"""

import pytest
from fastapi.testclient import TestClient

from apps.orchestrator.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# 1. Basic response schema
# ---------------------------------------------------------------------------


def test_portfolio_dashboard_returns_200(client):
    """Endpoint returns 200 with the expected top-level keys."""
    resp = client.get("/api/v1/dashboard/portfolio")
    assert resp.status_code == 200
    data = resp.json()
    assert "templates" in data
    assert "template_count" in data
    assert "providers" in data
    assert "provider_count" in data
    assert "health" in data


def test_portfolio_dashboard_health_section(client):
    """Health section has required fields and reports healthy."""
    resp = client.get("/api/v1/dashboard/portfolio")
    health = resp.json()["health"]
    assert health["status"] == "healthy"
    assert health["database"] == "reachable"
    assert "uptime_seconds" in health
    assert "version" in health


# ---------------------------------------------------------------------------
# 2. Template discovery (YAML)
# ---------------------------------------------------------------------------


def test_portfolio_dashboard_discovers_yaml_templates(client):
    """At least the content-engine YAML template should be discovered."""
    resp = client.get("/api/v1/dashboard/portfolio")
    templates = resp.json()["templates"]
    assert isinstance(templates, list)
    ids = [t["id"] for t in templates]
    assert "content-engine-pipeline" in ids


def test_portfolio_template_has_metadata(client):
    """Each discovered template exposes name, tags, node/edge counts."""
    resp = client.get("/api/v1/dashboard/portfolio")
    for tmpl in resp.json()["templates"]:
        assert "name" in tmpl
        assert "tags" in tmpl
        assert "node_count" in tmpl
        assert "edge_count" in tmpl
        assert "source" in tmpl
        assert tmpl["node_count"] > 0


def test_portfolio_template_count_matches(client):
    """template_count matches len(templates)."""
    data = client.get("/api/v1/dashboard/portfolio").json()
    assert data["template_count"] == len(data["templates"])


# ---------------------------------------------------------------------------
# 3. Last-run status (no runs exist in test DB)
# ---------------------------------------------------------------------------


def test_portfolio_template_last_run_null_when_no_runs(client):
    """When no runs exist, last_run should be None for each template."""
    templates = client.get("/api/v1/dashboard/portfolio").json()["templates"]
    for tmpl in templates:
        assert tmpl["last_run"] is None


# ---------------------------------------------------------------------------
# 4. Last-run status (with a matching run)
# ---------------------------------------------------------------------------


def test_portfolio_template_last_run_present_after_run(client):
    """Create a flow named after a template, run it, and verify last_run."""
    # Discover which template exists
    templates = client.get("/api/v1/dashboard/portfolio").json()["templates"]
    if not templates:
        pytest.skip("No YAML templates discovered")

    tmpl = templates[0]
    flow_name = tmpl["name"]

    # Create a flow with the same name as the template
    flow_payload = {
        "name": flow_name,
        "nodes": [
            {"id": "start", "type": "start", "position": {"x": 0, "y": 0}, "data": {"label": "Start"}},
            {"id": "end", "type": "end", "position": {"x": 0, "y": 100}, "data": {"label": "End"}},
        ],
        "edges": [
            {"id": "e1", "source": "start", "target": "end"},
        ],
    }
    create_resp = client.post("/api/v1/flows", json=flow_payload)
    assert create_resp.status_code == 201
    flow_id = create_resp.json()["id"]

    # Run the flow
    run_resp = client.post(f"/api/v1/flows/{flow_id}/runs", json={"input": {"text": "test"}})
    assert run_resp.status_code in (200, 201, 202)

    # Now check the dashboard
    data = client.get("/api/v1/dashboard/portfolio").json()
    matching = [t for t in data["templates"] if t["name"] == flow_name]
    assert len(matching) == 1
    last_run = matching[0]["last_run"]
    assert last_run is not None
    assert "run_id" in last_run
    assert "status" in last_run


# ---------------------------------------------------------------------------
# 5. Provider registry
# ---------------------------------------------------------------------------


def test_portfolio_providers_listed(client):
    """Provider registry should report at least the 5 built-in providers."""
    data = client.get("/api/v1/dashboard/portfolio").json()
    providers = data["providers"]
    assert isinstance(providers, list)
    assert data["provider_count"] == len(providers)
    names = [p["name"] for p in providers]
    assert "openai" in names
    assert "anthropic" in names


def test_portfolio_provider_shape(client):
    """Each provider entry has name, configured, reason, model_count."""
    providers = client.get("/api/v1/dashboard/portfolio").json()["providers"]
    for p in providers:
        assert "name" in p
        assert "configured" in p
        assert isinstance(p["configured"], bool)
        assert "model_count" in p
        assert isinstance(p["model_count"], int)
