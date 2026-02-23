"""Tests for Template Marketplace — import, export, versioning (DIRECTIVE-23-01)."""

import pytest
from fastapi.testclient import TestClient

from apps.orchestrator.main import app, template_registry, _load_yaml_template


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _reset_templates():
    template_registry.reset()
    yield
    template_registry.reset()


SAMPLE_TEMPLATE = {
    "id": "test-pipeline",
    "name": "Test Pipeline",
    "description": "A test workflow template",
    "tags": ["test", "pipeline"],
    "nodes": [
        {"id": "start", "type": "start", "position": {"x": 0, "y": 0}, "data": {}},
        {"id": "end", "type": "end", "position": {"x": 0, "y": 100}, "data": {}},
    ],
    "edges": [
        {"id": "e1", "source": "start", "target": "end"},
    ],
}


# ---------------------------------------------------------------------------
# TemplateRegistry — unit tests
# ---------------------------------------------------------------------------


def test_registry_import():
    """import_template() stores a template at version 1."""
    entry = template_registry.import_template(SAMPLE_TEMPLATE)
    assert entry["id"] == "test-pipeline"
    assert entry["version"] == 1
    assert entry["name"] == "Test Pipeline"
    assert len(entry["nodes"]) == 2


def test_registry_import_auto_id():
    """import_template() auto-generates ID when not provided."""
    data = {**SAMPLE_TEMPLATE, "id": None}
    entry = template_registry.import_template(data)
    assert entry["id"]  # non-empty
    assert entry["version"] == 1


def test_registry_versioning():
    """Importing same ID creates incremental versions."""
    template_registry.import_template(SAMPLE_TEMPLATE)
    v2_data = {**SAMPLE_TEMPLATE, "name": "Test Pipeline v2"}
    entry = template_registry.import_template(v2_data)
    assert entry["version"] == 2
    assert entry["name"] == "Test Pipeline v2"


def test_registry_get_latest():
    """get() returns latest version by default."""
    template_registry.import_template(SAMPLE_TEMPLATE)
    template_registry.import_template({**SAMPLE_TEMPLATE, "name": "Updated"})
    latest = template_registry.get("test-pipeline")
    assert latest["version"] == 2
    assert latest["name"] == "Updated"


def test_registry_get_specific_version():
    """get(version=1) returns that specific version."""
    template_registry.import_template(SAMPLE_TEMPLATE)
    template_registry.import_template({**SAMPLE_TEMPLATE, "name": "V2"})
    v1 = template_registry.get("test-pipeline", version=1)
    assert v1["version"] == 1
    assert v1["name"] == "Test Pipeline"


def test_registry_get_nonexistent():
    """get() returns None for unknown template."""
    assert template_registry.get("nonexistent") is None


def test_registry_get_invalid_version():
    """get(version=99) returns None for out-of-range version."""
    template_registry.import_template(SAMPLE_TEMPLATE)
    assert template_registry.get("test-pipeline", version=99) is None


def test_registry_list_templates():
    """list_templates() returns latest version of each template."""
    template_registry.import_template(SAMPLE_TEMPLATE)
    template_registry.import_template({"id": "other", "name": "Other", "nodes": [], "edges": []})
    templates = template_registry.list_templates()
    assert len(templates) == 2
    # Each entry has total_versions
    for t in templates:
        assert "total_versions" in t


def test_registry_list_versions():
    """list_versions() returns all versions of a template."""
    template_registry.import_template(SAMPLE_TEMPLATE)
    template_registry.import_template({**SAMPLE_TEMPLATE, "name": "V2"})
    template_registry.import_template({**SAMPLE_TEMPLATE, "name": "V3"})
    versions = template_registry.list_versions("test-pipeline")
    assert len(versions) == 3
    assert versions[0]["version"] == 1
    assert versions[2]["version"] == 3


def test_registry_list_versions_nonexistent():
    """list_versions() returns empty for unknown template."""
    assert template_registry.list_versions("nonexistent") == []


def test_registry_delete():
    """delete() removes template and all versions."""
    template_registry.import_template(SAMPLE_TEMPLATE)
    assert template_registry.delete("test-pipeline") is True
    assert template_registry.get("test-pipeline") is None


def test_registry_delete_nonexistent():
    """delete() returns False for unknown template."""
    assert template_registry.delete("fake") is False


def test_registry_reset():
    """reset() clears all templates."""
    template_registry.import_template(SAMPLE_TEMPLATE)
    template_registry.reset()
    assert template_registry.list_templates() == []


# ---------------------------------------------------------------------------
# POST /api/v1/templates/import
# ---------------------------------------------------------------------------


def test_import_endpoint_returns_201(client):
    """POST /templates/import creates template and returns 201."""
    resp = client.post("/api/v1/templates/import", json=SAMPLE_TEMPLATE)
    assert resp.status_code == 201
    data = resp.json()
    assert data["id"] == "test-pipeline"
    assert data["version"] == 1
    assert data["name"] == "Test Pipeline"
    assert data["description"] == "A test workflow template"
    assert data["tags"] == ["test", "pipeline"]


def test_import_endpoint_versioning(client):
    """POST /templates/import twice creates v1 and v2."""
    client.post("/api/v1/templates/import", json=SAMPLE_TEMPLATE)
    resp = client.post("/api/v1/templates/import", json=SAMPLE_TEMPLATE)
    assert resp.status_code == 201
    assert resp.json()["version"] == 2


def test_import_endpoint_auto_id(client):
    """POST /templates/import without id auto-generates one."""
    data = {k: v for k, v in SAMPLE_TEMPLATE.items() if k != "id"}
    resp = client.post("/api/v1/templates/import", json=data)
    assert resp.status_code == 201
    assert resp.json()["id"]  # non-empty auto-generated


def test_import_endpoint_validation(client):
    """POST /templates/import with empty name returns 422."""
    resp = client.post("/api/v1/templates/import", json={"name": "", "nodes": [], "edges": []})
    assert resp.status_code == 422


def test_import_endpoint_has_metadata(client):
    """POST /templates/import stores metadata field."""
    data = {**SAMPLE_TEMPLATE, "metadata": {"author": "test", "category": "research"}}
    resp = client.post("/api/v1/templates/import", json=data)
    assert resp.status_code == 201
    assert resp.json()["metadata"] == {"author": "test", "category": "research"}


# ---------------------------------------------------------------------------
# GET /api/v1/templates/{id}/export
# ---------------------------------------------------------------------------


def test_export_endpoint_returns_200(client):
    """GET /templates/{id}/export returns the template as portable JSON."""
    client.post("/api/v1/templates/import", json=SAMPLE_TEMPLATE)
    resp = client.get("/api/v1/templates/test-pipeline/export")
    assert resp.status_code == 200
    data = resp.json()
    assert data["synapps_export_version"] == "1.0.0"
    assert data["name"] == "Test Pipeline"
    assert "exported_at" in data
    assert len(data["nodes"]) == 2
    assert len(data["edges"]) == 1


def test_export_endpoint_content_disposition(client):
    """GET /templates/{id}/export sets Content-Disposition header."""
    client.post("/api/v1/templates/import", json=SAMPLE_TEMPLATE)
    resp = client.get("/api/v1/templates/test-pipeline/export")
    assert "content-disposition" in resp.headers
    assert "synapps-template.json" in resp.headers["content-disposition"]


def test_export_specific_version(client):
    """GET /templates/{id}/export?version=1 exports that version."""
    client.post("/api/v1/templates/import", json=SAMPLE_TEMPLATE)
    client.post("/api/v1/templates/import", json={**SAMPLE_TEMPLATE, "name": "V2"})
    resp = client.get("/api/v1/templates/test-pipeline/export?version=1")
    assert resp.status_code == 200
    assert resp.json()["name"] == "Test Pipeline"
    assert resp.json()["version"] == 1


def test_export_latest_by_default(client):
    """GET /templates/{id}/export returns latest version by default."""
    client.post("/api/v1/templates/import", json=SAMPLE_TEMPLATE)
    client.post("/api/v1/templates/import", json={**SAMPLE_TEMPLATE, "name": "Latest"})
    resp = client.get("/api/v1/templates/test-pipeline/export")
    assert resp.json()["name"] == "Latest"
    assert resp.json()["version"] == 2


def test_export_not_found(client):
    """GET /templates/{id}/export returns 404 for unknown template."""
    resp = client.get("/api/v1/templates/nonexistent/export")
    assert resp.status_code == 404


def test_export_yaml_fallback(client):
    """GET /templates/{id}/export falls back to YAML templates on disk."""
    yaml_data = _load_yaml_template("content-engine-pipeline")
    if yaml_data is None:
        pytest.skip("content_engine.yaml not found in templates/")
    resp = client.get("/api/v1/templates/content-engine-pipeline/export")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "Content Engine Pipeline"
    assert data["synapps_export_version"] == "1.0.0"


# ---------------------------------------------------------------------------
# GET /api/v1/templates/{id}/versions
# ---------------------------------------------------------------------------


def test_versions_endpoint(client):
    """GET /templates/{id}/versions lists all versions."""
    client.post("/api/v1/templates/import", json=SAMPLE_TEMPLATE)
    client.post("/api/v1/templates/import", json={**SAMPLE_TEMPLATE, "name": "V2"})
    client.post("/api/v1/templates/import", json={**SAMPLE_TEMPLATE, "name": "V3"})
    resp = client.get("/api/v1/templates/test-pipeline/versions")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 3
    assert data["template_id"] == "test-pipeline"
    assert len(data["versions"]) == 3
    assert data["versions"][0]["version"] == 1
    assert data["versions"][2]["name"] == "V3"


def test_versions_not_found(client):
    """GET /templates/{id}/versions returns 404 for unknown template."""
    resp = client.get("/api/v1/templates/nonexistent/versions")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /api/v1/templates
# ---------------------------------------------------------------------------


def test_list_templates_endpoint(client):
    """GET /templates lists all imported templates."""
    client.post("/api/v1/templates/import", json=SAMPLE_TEMPLATE)
    client.post("/api/v1/templates/import", json={
        "id": "other-pipeline",
        "name": "Other Pipeline",
        "nodes": [],
        "edges": [],
    })
    resp = client.get("/api/v1/templates")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    assert len(data["templates"]) == 2


def test_list_templates_empty(client):
    """GET /templates with no imports returns empty list."""
    resp = client.get("/api/v1/templates")
    assert resp.status_code == 200
    assert resp.json()["total"] == 0


def test_list_templates_shows_total_versions(client):
    """GET /templates includes total_versions for each template."""
    client.post("/api/v1/templates/import", json=SAMPLE_TEMPLATE)
    client.post("/api/v1/templates/import", json=SAMPLE_TEMPLATE)  # v2
    resp = client.get("/api/v1/templates")
    entry = resp.json()["templates"][0]
    assert entry["total_versions"] == 2


# ---------------------------------------------------------------------------
# Roundtrip: import → export → re-import
# ---------------------------------------------------------------------------


def test_import_export_roundtrip(client):
    """Exported template can be re-imported to create a new version."""
    client.post("/api/v1/templates/import", json=SAMPLE_TEMPLATE)
    exported = client.get("/api/v1/templates/test-pipeline/export").json()

    # Re-import the exported data
    resp = client.post("/api/v1/templates/import", json=exported)
    assert resp.status_code == 201
    assert resp.json()["version"] == 2
    assert resp.json()["name"] == "Test Pipeline"
