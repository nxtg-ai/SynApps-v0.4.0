"""Tests for Template Marketplace — import, export, versioning, publish, instantiate."""

import pytest
from fastapi.testclient import TestClient

from apps.orchestrator.main import (
    app, template_registry, _load_yaml_template, MARKETPLACE_CATEGORIES,
)


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


# ---------------------------------------------------------------------------
# GET /api/v1/templates — category filtering
# ---------------------------------------------------------------------------

NOTIFICATION_TEMPLATE = {
    "id": "notify-tmpl",
    "name": "RSS to Slack",
    "description": "Send RSS items to Slack",
    "nodes": [{"id": "n1", "type": "http", "position": {"x": 0, "y": 0}, "data": {}}],
    "edges": [],
    "metadata": {"category": "notification", "author": "alice"},
}

DEVOPS_TEMPLATE = {
    "id": "devops-tmpl",
    "name": "GitHub PR to Discord",
    "description": "Post PR events to Discord",
    "nodes": [{"id": "n1", "type": "http", "position": {"x": 0, "y": 0}, "data": {}}],
    "edges": [],
    "metadata": {"category": "devops", "author": "bob"},
}

MONITORING_TEMPLATE = {
    "id": "monitoring-tmpl",
    "name": "Uptime Monitor",
    "description": "Check endpoint health periodically",
    "nodes": [{"id": "n1", "type": "http", "position": {"x": 0, "y": 0}, "data": {}}],
    "edges": [],
    "metadata": {"category": "monitoring", "author": "carol"},
}


def test_list_templates_filter_by_category(client):
    """GET /templates?category=notification returns only matching templates."""
    client.post("/api/v1/templates/import", json=NOTIFICATION_TEMPLATE)
    client.post("/api/v1/templates/import", json=DEVOPS_TEMPLATE)
    client.post("/api/v1/templates/import", json=MONITORING_TEMPLATE)

    resp = client.get("/api/v1/templates?category=notification")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert data["templates"][0]["name"] == "RSS to Slack"


def test_list_templates_filter_devops(client):
    """GET /templates?category=devops returns only devops templates."""
    client.post("/api/v1/templates/import", json=NOTIFICATION_TEMPLATE)
    client.post("/api/v1/templates/import", json=DEVOPS_TEMPLATE)

    resp = client.get("/api/v1/templates?category=devops")
    assert resp.status_code == 200
    assert resp.json()["total"] == 1
    assert resp.json()["templates"][0]["name"] == "GitHub PR to Discord"


def test_list_templates_filter_no_match(client):
    """GET /templates?category=content returns empty if none match."""
    client.post("/api/v1/templates/import", json=NOTIFICATION_TEMPLATE)
    resp = client.get("/api/v1/templates?category=content")
    assert resp.status_code == 200
    assert resp.json()["total"] == 0


def test_list_templates_filter_case_insensitive(client):
    """Category filter is case-insensitive."""
    client.post("/api/v1/templates/import", json=NOTIFICATION_TEMPLATE)
    resp = client.get("/api/v1/templates?category=NOTIFICATION")
    assert resp.status_code == 200
    assert resp.json()["total"] == 1


def test_list_templates_no_filter_returns_all(client):
    """GET /templates without category returns all templates."""
    client.post("/api/v1/templates/import", json=NOTIFICATION_TEMPLATE)
    client.post("/api/v1/templates/import", json=DEVOPS_TEMPLATE)
    client.post("/api/v1/templates/import", json=MONITORING_TEMPLATE)
    resp = client.get("/api/v1/templates")
    assert resp.status_code == 200
    assert resp.json()["total"] == 3


# ---------------------------------------------------------------------------
# POST /api/v1/templates — publish from existing flow
# ---------------------------------------------------------------------------


def _create_flow(client, name="My Workflow", nodes=None, edges=None):
    """Helper: create a flow and return its ID."""
    flow_data = {
        "name": name,
        "nodes": nodes or [
            {"id": "s1", "type": "start", "position": {"x": 0, "y": 0}, "data": {}},
            {"id": "llm1", "type": "llm", "position": {"x": 200, "y": 0}, "data": {"provider": "openai"}},
            {"id": "e1", "type": "end", "position": {"x": 400, "y": 0}, "data": {}},
        ],
        "edges": edges or [
            {"id": "edge1", "source": "s1", "target": "llm1"},
            {"id": "edge2", "source": "llm1", "target": "e1"},
        ],
    }
    resp = client.post("/api/v1/flows", json=flow_data)
    assert resp.status_code == 201
    return resp.json()["id"]


def test_publish_template_from_flow(client):
    """POST /templates publishes a flow as a marketplace template."""
    flow_id = _create_flow(client)
    resp = client.post("/api/v1/templates", json={
        "flow_id": flow_id,
        "name": "RSS to Slack",
        "description": "Send RSS feed items to a Slack channel",
        "category": "notification",
        "author": "alice",
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "RSS to Slack"
    assert data["version"] == 1
    assert data["metadata"]["category"] == "notification"
    assert data["metadata"]["author"] == "alice"
    assert data["metadata"]["source_flow_id"] == flow_id
    assert len(data["nodes"]) == 3
    assert len(data["edges"]) == 2


def test_publish_template_appears_in_list(client):
    """Published template is discoverable via GET /templates."""
    flow_id = _create_flow(client)
    client.post("/api/v1/templates", json={
        "flow_id": flow_id,
        "name": "Data Sync Pipeline",
        "category": "data-sync",
        "author": "bob",
    })
    resp = client.get("/api/v1/templates?category=data-sync")
    assert resp.status_code == 200
    assert resp.json()["total"] == 1
    assert resp.json()["templates"][0]["name"] == "Data Sync Pipeline"


def test_publish_template_flow_not_found(client):
    """POST /templates with nonexistent flow_id returns 404."""
    resp = client.post("/api/v1/templates", json={
        "flow_id": "nonexistent-flow-id",
        "name": "Phantom Template",
        "category": "notification",
    })
    assert resp.status_code == 404
    assert "not found" in resp.json()["error"]["message"].lower()


def test_publish_template_invalid_category(client):
    """POST /templates with invalid category returns 422."""
    flow_id = _create_flow(client)
    resp = client.post("/api/v1/templates", json={
        "flow_id": flow_id,
        "name": "Bad Category Template",
        "category": "invalid-category",
    })
    assert resp.status_code == 422


def test_publish_template_all_valid_categories(client):
    """POST /templates accepts all defined marketplace categories."""
    flow_id = _create_flow(client)
    for cat in sorted(MARKETPLACE_CATEGORIES):
        resp = client.post("/api/v1/templates", json={
            "flow_id": flow_id,
            "name": f"Template for {cat}",
            "category": cat,
            "author": "test",
        })
        assert resp.status_code == 201, f"Category '{cat}' should be valid"
        assert resp.json()["metadata"]["category"] == cat


def test_publish_template_category_case_insensitive(client):
    """POST /templates normalizes category to lowercase."""
    flow_id = _create_flow(client)
    resp = client.post("/api/v1/templates", json={
        "flow_id": flow_id,
        "name": "Mixed Case",
        "category": "Notification",
    })
    assert resp.status_code == 201
    assert resp.json()["metadata"]["category"] == "notification"


def test_publish_template_default_author(client):
    """POST /templates defaults author to 'anonymous'."""
    flow_id = _create_flow(client)
    resp = client.post("/api/v1/templates", json={
        "flow_id": flow_id,
        "name": "Anonymous Template",
        "category": "content",
    })
    assert resp.status_code == 201
    assert resp.json()["metadata"]["author"] == "anonymous"


def test_publish_template_with_explicit_version(client):
    """POST /templates with explicit semver version stores it."""
    flow_id = _create_flow(client)
    resp = client.post("/api/v1/templates", json={
        "flow_id": flow_id,
        "name": "Versioned Template",
        "category": "monitoring",
        "version": "2.0.0",
    })
    assert resp.status_code == 201
    assert resp.json()["semver"] == "2.0.0"


def test_publish_template_missing_name(client):
    """POST /templates without name returns 422."""
    flow_id = _create_flow(client)
    resp = client.post("/api/v1/templates", json={
        "flow_id": flow_id,
        "category": "devops",
    })
    assert resp.status_code == 422


def test_publish_template_missing_category(client):
    """POST /templates without category returns 422."""
    flow_id = _create_flow(client)
    resp = client.post("/api/v1/templates", json={
        "flow_id": flow_id,
        "name": "Missing Category",
    })
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /api/v1/templates/{id}/instantiate — create flow from template
# ---------------------------------------------------------------------------


def test_instantiate_template_creates_flow(client):
    """POST /templates/{id}/instantiate creates a new flow from the template."""
    flow_id = _create_flow(client)
    pub_resp = client.post("/api/v1/templates", json={
        "flow_id": flow_id,
        "name": "Instantiable Template",
        "category": "notification",
    })
    template_id = pub_resp.json()["id"]

    resp = client.post(f"/api/v1/templates/{template_id}/instantiate", json={})
    assert resp.status_code == 201
    data = resp.json()
    assert data["message"] == "Flow created from template"
    assert data["template_id"] == template_id
    assert "flow_id" in data

    # Verify the flow was actually created
    flow_resp = client.get(f"/api/v1/flows/{data['flow_id']}")
    assert flow_resp.status_code == 200
    assert flow_resp.json()["name"] == "Instantiable Template"


def test_instantiate_template_custom_name(client):
    """POST /templates/{id}/instantiate uses custom flow_name if given."""
    flow_id = _create_flow(client)
    pub = client.post("/api/v1/templates", json={
        "flow_id": flow_id,
        "name": "Template Name",
        "category": "devops",
    }).json()

    resp = client.post(f"/api/v1/templates/{pub['id']}/instantiate", json={
        "flow_name": "My Custom Flow"
    })
    assert resp.status_code == 201
    flow_resp = client.get(f"/api/v1/flows/{resp.json()['flow_id']}")
    assert flow_resp.json()["name"] == "My Custom Flow"


def test_instantiate_template_remaps_node_ids(client):
    """Instantiation re-maps all node IDs to avoid collisions."""
    flow_id = _create_flow(client)
    pub = client.post("/api/v1/templates", json={
        "flow_id": flow_id,
        "name": "ID Remap Test",
        "category": "data-sync",
    }).json()

    resp = client.post(f"/api/v1/templates/{pub['id']}/instantiate", json={})
    new_flow_id = resp.json()["flow_id"]
    flow = client.get(f"/api/v1/flows/{new_flow_id}").json()

    # Original nodes had IDs: s1, llm1, e1 — new ones should be different
    new_node_ids = {n["id"] for n in flow["nodes"]}
    assert "s1" not in new_node_ids
    assert "llm1" not in new_node_ids
    assert "e1" not in new_node_ids
    assert len(new_node_ids) == 3


def test_instantiate_template_remaps_edge_references(client):
    """Instantiation updates edge source/target to new node IDs."""
    flow_id = _create_flow(client)
    pub = client.post("/api/v1/templates", json={
        "flow_id": flow_id,
        "name": "Edge Remap Test",
        "category": "monitoring",
    }).json()

    resp = client.post(f"/api/v1/templates/{pub['id']}/instantiate", json={})
    flow = client.get(f"/api/v1/flows/{resp.json()['flow_id']}").json()

    node_ids = {n["id"] for n in flow["nodes"]}
    for edge in flow["edges"]:
        assert edge["source"] in node_ids, f"Edge source {edge['source']} not in node IDs"
        assert edge["target"] in node_ids, f"Edge target {edge['target']} not in node IDs"


def test_instantiate_template_with_connector_overrides(client):
    """Instantiation merges connector_overrides into node data."""
    flow_id = _create_flow(client, nodes=[
        {"id": "http1", "type": "http", "position": {"x": 0, "y": 0},
         "data": {"url": "https://placeholder.example.com", "method": "GET"}},
    ], edges=[])
    pub = client.post("/api/v1/templates", json={
        "flow_id": flow_id,
        "name": "Override Test",
        "category": "notification",
    }).json()

    resp = client.post(f"/api/v1/templates/{pub['id']}/instantiate", json={
        "connector_overrides": {
            "http1": {"url": "https://my-real-api.example.com", "headers": {"Authorization": "Bearer xyz"}}
        }
    })
    assert resp.status_code == 201
    flow = client.get(f"/api/v1/flows/{resp.json()['flow_id']}").json()
    node_data = flow["nodes"][0]["data"]
    assert node_data["url"] == "https://my-real-api.example.com"
    assert node_data["headers"] == {"Authorization": "Bearer xyz"}
    # Original field preserved via merge
    assert node_data["method"] == "GET"


def test_instantiate_template_not_found(client):
    """POST /templates/{id}/instantiate returns 404 for unknown template."""
    resp = client.post("/api/v1/templates/nonexistent/instantiate", json={})
    assert resp.status_code == 404
    assert "not found" in resp.json()["error"]["message"].lower()


def test_instantiate_template_multiple_times(client):
    """Instantiating same template twice creates distinct flows."""
    flow_id = _create_flow(client)
    pub = client.post("/api/v1/templates", json={
        "flow_id": flow_id,
        "name": "Multi-Instantiate",
        "category": "content",
    }).json()

    resp1 = client.post(f"/api/v1/templates/{pub['id']}/instantiate", json={})
    resp2 = client.post(f"/api/v1/templates/{pub['id']}/instantiate", json={})
    assert resp1.json()["flow_id"] != resp2.json()["flow_id"]


def test_instantiate_returns_template_version(client):
    """Instantiate response includes the template version that was used."""
    flow_id = _create_flow(client)
    pub = client.post("/api/v1/templates", json={
        "flow_id": flow_id,
        "name": "Version Track",
        "category": "devops",
    }).json()

    resp = client.post(f"/api/v1/templates/{pub['id']}/instantiate", json={})
    assert resp.json()["template_version"] == 1


# ---------------------------------------------------------------------------
# End-to-end: publish → list → instantiate → run flow
# ---------------------------------------------------------------------------


def test_full_marketplace_roundtrip(client):
    """Publish a flow as template, find it by category, instantiate it."""
    # 1) Create a source flow
    flow_id = _create_flow(client, name="Source Workflow")

    # 2) Publish as template
    pub_resp = client.post("/api/v1/templates", json={
        "flow_id": flow_id,
        "name": "RSS to Slack",
        "description": "Send RSS items to Slack",
        "category": "notification",
        "author": "marketplace-test",
    })
    assert pub_resp.status_code == 201
    template_id = pub_resp.json()["id"]

    # 3) Discover via category filter
    list_resp = client.get("/api/v1/templates?category=notification")
    assert list_resp.status_code == 200
    found = [t for t in list_resp.json()["templates"] if t["id"] == template_id]
    assert len(found) == 1
    assert found[0]["metadata"]["author"] == "marketplace-test"

    # 4) Instantiate the template into a new flow
    inst_resp = client.post(f"/api/v1/templates/{template_id}/instantiate", json={
        "flow_name": "My RSS to Slack"
    })
    assert inst_resp.status_code == 201
    new_flow_id = inst_resp.json()["flow_id"]

    # 5) Verify the new flow exists and has correct structure
    flow = client.get(f"/api/v1/flows/{new_flow_id}").json()
    assert flow["name"] == "My RSS to Slack"
    assert len(flow["nodes"]) == 3
    assert len(flow["edges"]) == 2
