"""Tests for template validation + POST /api/v1/templates/validate."""

import pytest
from fastapi.testclient import TestClient

from apps.orchestrator.main import app, validate_template


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Helper: minimal valid template
# ---------------------------------------------------------------------------

def _valid_template(**overrides):
    base = {
        "name": "Test Template",
        "nodes": [
            {"id": "start", "type": "start", "position": {"x": 0, "y": 0}, "data": {"label": "S"}},
            {"id": "end", "type": "end", "position": {"x": 0, "y": 100}, "data": {"label": "E"}},
        ],
        "edges": [
            {"id": "e1", "source": "start", "target": "end"},
        ],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# validate_template() — unit tests
# ---------------------------------------------------------------------------


def test_valid_template_passes():
    """A well-formed template passes validation."""
    result = validate_template(_valid_template())
    assert result["valid"] is True
    assert result["errors"] == []
    assert result["summary"]["node_count"] == 2
    assert result["summary"]["edge_count"] == 1
    assert result["summary"]["has_start"] is True
    assert result["summary"]["has_end"] is True


def test_missing_name():
    """Template without a name is invalid."""
    result = validate_template(_valid_template(name=""))
    assert result["valid"] is False
    assert any("name" in e.lower() for e in result["errors"])


def test_missing_nodes():
    """Template without nodes list is invalid."""
    result = validate_template({"name": "X"})
    assert result["valid"] is False
    assert any("nodes" in e.lower() for e in result["errors"])


def test_missing_start_node():
    """Template without a start node is invalid."""
    result = validate_template({
        "name": "No Start",
        "nodes": [
            {"id": "end", "type": "end", "position": {"x": 0, "y": 0}, "data": {}},
        ],
        "edges": [],
    })
    assert result["valid"] is False
    assert any("start" in e.lower() for e in result["errors"])


def test_missing_end_node():
    """Template without an end node is invalid."""
    result = validate_template({
        "name": "No End",
        "nodes": [
            {"id": "start", "type": "start", "position": {"x": 0, "y": 0}, "data": {}},
        ],
        "edges": [],
    })
    assert result["valid"] is False
    assert any("end" in e.lower() for e in result["errors"])


def test_duplicate_node_ids():
    """Duplicate node IDs are rejected."""
    result = validate_template({
        "name": "Dupes",
        "nodes": [
            {"id": "start", "type": "start", "position": {"x": 0, "y": 0}, "data": {}},
            {"id": "start", "type": "end", "position": {"x": 0, "y": 100}, "data": {}},
        ],
        "edges": [],
    })
    assert result["valid"] is False
    assert any("duplicate" in e.lower() for e in result["errors"])


def test_edge_references_unknown_source():
    """Edge referencing a non-existent source is invalid."""
    result = validate_template({
        "name": "Bad Edge",
        "nodes": [
            {"id": "start", "type": "start", "position": {"x": 0, "y": 0}, "data": {}},
            {"id": "end", "type": "end", "position": {"x": 0, "y": 100}, "data": {}},
        ],
        "edges": [
            {"id": "e1", "source": "ghost", "target": "end"},
        ],
    })
    assert result["valid"] is False
    assert any("unknown node" in e.lower() for e in result["errors"])


def test_edge_references_unknown_target():
    """Edge referencing a non-existent target is invalid."""
    result = validate_template({
        "name": "Bad Edge Target",
        "nodes": [
            {"id": "start", "type": "start", "position": {"x": 0, "y": 0}, "data": {}},
            {"id": "end", "type": "end", "position": {"x": 0, "y": 100}, "data": {}},
        ],
        "edges": [
            {"id": "e1", "source": "start", "target": "phantom"},
        ],
    })
    assert result["valid"] is False
    assert any("unknown node" in e.lower() for e in result["errors"])


def test_self_loop_detected():
    """An edge with source == target is invalid."""
    result = validate_template({
        "name": "Self Loop",
        "nodes": [
            {"id": "start", "type": "start", "position": {"x": 0, "y": 0}, "data": {}},
            {"id": "end", "type": "end", "position": {"x": 0, "y": 100}, "data": {}},
        ],
        "edges": [
            {"id": "e1", "source": "start", "target": "start"},
            {"id": "e2", "source": "start", "target": "end"},
        ],
    })
    assert result["valid"] is False
    assert any("self-loop" in e.lower() for e in result["errors"])


def test_circular_dependency_detected():
    """A cycle in the edge graph is detected and reported."""
    result = validate_template({
        "name": "Circular",
        "nodes": [
            {"id": "start", "type": "start", "position": {"x": 0, "y": 0}, "data": {}},
            {"id": "a", "type": "llm", "position": {"x": 0, "y": 50}, "data": {}},
            {"id": "b", "type": "code", "position": {"x": 0, "y": 100}, "data": {}},
            {"id": "end", "type": "end", "position": {"x": 0, "y": 200}, "data": {}},
        ],
        "edges": [
            {"id": "e1", "source": "start", "target": "a"},
            {"id": "e2", "source": "a", "target": "b"},
            {"id": "e3", "source": "b", "target": "a"},  # cycle: a -> b -> a
            {"id": "e4", "source": "b", "target": "end"},
        ],
    })
    assert result["valid"] is False
    assert any("circular" in e.lower() for e in result["errors"])


def test_unknown_node_type_warns():
    """An unknown node type produces a warning (not an error)."""
    result = validate_template({
        "name": "Unknown Type",
        "nodes": [
            {"id": "start", "type": "start", "position": {"x": 0, "y": 0}, "data": {}},
            {"id": "x", "type": "banana_splitter", "position": {"x": 0, "y": 50}, "data": {}},
            {"id": "end", "type": "end", "position": {"x": 0, "y": 100}, "data": {}},
        ],
        "edges": [
            {"id": "e1", "source": "start", "target": "x"},
            {"id": "e2", "source": "x", "target": "end"},
        ],
    })
    assert result["valid"] is True  # warnings, not errors
    assert any("banana_splitter" in w for w in result["warnings"])


def test_node_types_in_summary():
    """Summary includes sorted list of node types."""
    result = validate_template(_valid_template(
        nodes=[
            {"id": "start", "type": "start", "position": {"x": 0, "y": 0}, "data": {}},
            {"id": "llm1", "type": "llm", "position": {"x": 0, "y": 50}, "data": {}},
            {"id": "code1", "type": "code", "position": {"x": 0, "y": 100}, "data": {}},
            {"id": "end", "type": "end", "position": {"x": 0, "y": 150}, "data": {}},
        ],
        edges=[
            {"id": "e1", "source": "start", "target": "llm1"},
            {"id": "e2", "source": "llm1", "target": "code1"},
            {"id": "e3", "source": "code1", "target": "end"},
        ],
    ))
    assert result["valid"] is True
    assert result["summary"]["node_types"] == ["code", "end", "llm", "start"]


def test_content_engine_template_validates():
    """The real content_engine.yaml template passes validation."""
    import yaml
    import pathlib
    template_path = pathlib.Path(__file__).resolve().parent.parent.parent.parent / "templates" / "content_engine.yaml"
    if not template_path.exists():
        pytest.skip("content_engine.yaml not found")
    with open(template_path) as f:
        data = yaml.safe_load(f)
    result = validate_template(data)
    assert result["valid"] is True, f"Errors: {result['errors']}"
    assert result["summary"]["node_count"] == 6
    assert result["summary"]["has_start"] is True
    assert result["summary"]["has_end"] is True


# ---------------------------------------------------------------------------
# POST /api/v1/templates/validate — endpoint tests
# ---------------------------------------------------------------------------


def test_validate_endpoint_valid_template(client):
    """POST /api/v1/templates/validate returns valid=true for good template."""
    resp = client.post("/api/v1/templates/validate", json=_valid_template())
    assert resp.status_code == 200
    data = resp.json()
    assert data["valid"] is True
    assert data["errors"] == []


def test_validate_endpoint_invalid_template(client):
    """POST /api/v1/templates/validate returns valid=false with errors."""
    resp = client.post("/api/v1/templates/validate", json={
        "name": "",
        "nodes": [],
        "edges": [],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["valid"] is False
    assert len(data["errors"]) > 0


def test_validate_endpoint_circular_deps(client):
    """POST /api/v1/templates/validate catches circular dependencies."""
    resp = client.post("/api/v1/templates/validate", json={
        "name": "Cycle",
        "nodes": [
            {"id": "start", "type": "start", "position": {"x": 0, "y": 0}, "data": {}},
            {"id": "a", "type": "llm", "position": {"x": 0, "y": 50}, "data": {}},
            {"id": "b", "type": "code", "position": {"x": 0, "y": 100}, "data": {}},
            {"id": "end", "type": "end", "position": {"x": 0, "y": 200}, "data": {}},
        ],
        "edges": [
            {"id": "e1", "source": "start", "target": "a"},
            {"id": "e2", "source": "a", "target": "b"},
            {"id": "e3", "source": "b", "target": "a"},
        ],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["valid"] is False
    assert any("circular" in e.lower() for e in data["errors"])


def test_validate_endpoint_returns_summary(client):
    """POST /api/v1/templates/validate includes summary in response."""
    resp = client.post("/api/v1/templates/validate", json=_valid_template())
    assert resp.status_code == 200
    summary = resp.json()["summary"]
    assert summary["node_count"] == 2
    assert summary["edge_count"] == 1
    assert summary["has_start"] is True
    assert summary["has_end"] is True


def test_validate_endpoint_returns_warnings(client):
    """POST /api/v1/templates/validate returns warnings for unknown types."""
    resp = client.post("/api/v1/templates/validate", json={
        "name": "Warn",
        "nodes": [
            {"id": "start", "type": "start", "position": {"x": 0, "y": 0}, "data": {}},
            {"id": "x", "type": "alien_type", "position": {"x": 0, "y": 50}, "data": {}},
            {"id": "end", "type": "end", "position": {"x": 0, "y": 100}, "data": {}},
        ],
        "edges": [],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["valid"] is True
    assert len(data["warnings"]) > 0
