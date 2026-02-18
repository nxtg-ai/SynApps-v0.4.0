"""
Basic tests for the SynApps Orchestrator
"""
import pytest
from fastapi.testclient import TestClient

from apps.orchestrator.main import app, Flow, AppletMessage


@pytest.fixture
def client():
    """Create a TestClient that triggers lifespan events."""
    with TestClient(app) as c:
        yield c

def test_health_check(client):
    """Test the health check endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "SynApps Orchestrator API"
    assert "version" in data


def test_versioned_health_check(client):
    """Test the versioned health check endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "SynApps Orchestrator API"
    assert "version" in data

def test_list_applets(client):
    """Test listing applets returns paginated response."""
    response = client.get("/api/v1/applets")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert "total" in data
    assert "page" in data
    assert "page_size" in data
    assert "total_pages" in data
    assert isinstance(data["items"], list)

def test_list_applets_pagination(client):
    """Test applet listing with custom pagination parameters."""
    response = client.get("/api/v1/applets?page=1&page_size=2")
    assert response.status_code == 200
    data = response.json()
    assert data["page"] == 1
    assert data["page_size"] == 2

def test_create_flow(client):
    """Test creating a flow."""
    flow = {
        "id": "test-flow",
        "name": "Test Flow",
        "nodes": [
            {
                "id": "start",
                "type": "start",
                "position": {
                    "x": 250,
                    "y": 25
                },
                "data": {
                    "label": "Start"
                }
            },
            {
                "id": "end",
                "type": "end",
                "position": {
                    "x": 250,
                    "y": 125
                },
                "data": {
                    "label": "End"
                }
            }
        ],
        "edges": [
            {
                "id": "start-end",
                "source": "start",
                "target": "end",
                "animated": False
            }
        ]
    }

    response = client.post("/api/v1/flows", json=flow)
    assert response.status_code == 201
    assert response.json()["id"] == "test-flow"

def test_create_flow_validation_error(client):
    """Test that invalid flow data returns structured validation error."""
    invalid_flow = {
        "name": "",  # Empty name should fail validation
        "nodes": [],
        "edges": []
    }

    response = client.post("/api/v1/flows", json=invalid_flow)
    assert response.status_code == 422
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "VALIDATION_ERROR"
    assert "details" in data["error"]

def test_list_flows(client):
    """Test listing flows returns paginated response."""
    response = client.get("/api/v1/flows")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert "total" in data
    assert isinstance(data["items"], list)

def test_get_flow(client):
    """Test getting a flow."""
    # First create a flow
    flow = {
        "id": "test-flow-2",
        "name": "Test Flow 2",
        "nodes": [],
        "edges": []
    }

    client.post("/api/v1/flows", json=flow)

    # Then get it
    response = client.get("/api/v1/flows/test-flow-2")
    assert response.status_code == 200
    assert response.json()["id"] == "test-flow-2"

def test_get_flow_not_found(client):
    """Test getting a non-existent flow returns consistent error format."""
    response = client.get("/api/v1/flows/non-existent")
    assert response.status_code == 404
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "NOT_FOUND"
    assert data["error"]["status"] == 404
    assert data["error"]["message"] == "Flow not found"

def test_delete_flow(client):
    """Test deleting a flow."""
    # First create a flow
    flow = {
        "id": "test-flow-3",
        "name": "Test Flow 3",
        "nodes": [],
        "edges": []
    }

    client.post("/api/v1/flows", json=flow)

    # Then delete it
    response = client.delete("/api/v1/flows/test-flow-3")
    assert response.status_code == 200
    assert response.json()["message"] == "Flow deleted"

    # Verify it's gone (consistent error format)
    response = client.get("/api/v1/flows/test-flow-3")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"

def test_list_runs_paginated(client):
    """Test listing runs returns paginated response."""
    response = client.get("/api/v1/runs")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert "total" in data
    assert "page" in data
    assert "page_size" in data

def test_consistent_error_format_404(client):
    """Test that 404 errors use the consistent error format."""
    response = client.get("/api/v1/runs/non-existent")
    assert response.status_code == 404
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "NOT_FOUND"
    assert data["error"]["status"] == 404

def test_consistent_error_format_501(client):
    """Test that 501 errors use the consistent error format."""
    response = client.post("/api/v1/ai/suggest", json={"prompt": "test suggestion"})
    assert response.status_code == 501
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "NOT_IMPLEMENTED"
    assert data["error"]["status"] == 501


def test_ai_suggest_validation_error(client):
    """Test that ai/suggest validates request body with Pydantic v2."""
    # Missing required 'prompt' field
    response = client.post("/api/v1/ai/suggest", json={})
    assert response.status_code == 422
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "VALIDATION_ERROR"
    assert "details" in data["error"]

    # Empty prompt should fail min_length=1
    response = client.post("/api/v1/ai/suggest", json={"prompt": ""})
    assert response.status_code == 422
