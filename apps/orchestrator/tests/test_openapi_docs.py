"""Tests for OpenAPI spec and docs endpoints."""

import pytest
from fastapi.testclient import TestClient

from apps.orchestrator.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_swagger_ui_accessible(client):
    """Swagger UI at /api/v1/docs returns 200."""
    resp = client.get("/api/v1/docs")
    assert resp.status_code == 200
    assert "swagger" in resp.text.lower() or "openapi" in resp.text.lower()


def test_redoc_accessible(client):
    """ReDoc at /api/v1/redoc returns 200."""
    resp = client.get("/api/v1/redoc")
    assert resp.status_code == 200


def test_openapi_json_accessible(client):
    """OpenAPI JSON spec at /api/v1/openapi.json returns valid JSON."""
    resp = client.get("/api/v1/openapi.json")
    assert resp.status_code == 200
    spec = resp.json()
    assert spec["openapi"].startswith("3.")
    assert spec["info"]["title"] == "SynApps Orchestrator"


def test_openapi_has_tags(client):
    """OpenAPI spec includes all expected tag groups."""
    spec = client.get("/api/v1/openapi.json").json()
    tag_names = [t["name"] for t in spec.get("tags", [])]
    for expected in ["Auth", "Flows", "Runs", "Providers", "Dashboard", "Health"]:
        assert expected in tag_names, f"Missing tag: {expected}"


def test_openapi_has_description(client):
    """OpenAPI spec includes a non-empty API description."""
    spec = client.get("/api/v1/openapi.json").json()
    assert len(spec["info"]["description"]) > 20


def test_openapi_paths_cover_core_endpoints(client):
    """OpenAPI spec includes paths for core API endpoints."""
    spec = client.get("/api/v1/openapi.json").json()
    paths = list(spec["paths"].keys())
    for expected_path in [
        "/api/v1/auth/register",
        "/api/v1/auth/login",
        "/api/v1/flows",
        "/api/v1/runs",
        "/api/v1/llm/providers",
        "/api/v1/dashboard/portfolio",
        "/api/v1/health",
    ]:
        assert expected_path in paths, f"Missing path: {expected_path}"
