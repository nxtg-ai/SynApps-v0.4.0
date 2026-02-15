"""
Basic tests for the SynApps Orchestrator
"""
import pytest
from fastapi.testclient import TestClient

from main import app, Flow, AppletMessage


@pytest.fixture
def client():
    """Create a TestClient that triggers lifespan events."""
    with TestClient(app) as c:
        yield c

def test_read_root(client):
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "SynApps Orchestrator API" in response.json()["message"]

def test_list_applets(client):
    """Test listing applets."""
    response = client.get("/applets")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

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
    
    response = client.post("/flows", json=flow)
    assert response.status_code == 200
    assert response.json()["id"] == "test-flow"

def test_list_flows(client):
    """Test listing flows."""
    response = client.get("/flows")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_get_flow(client):
    """Test getting a flow."""
    # First create a flow
    flow = {
        "id": "test-flow-2",
        "name": "Test Flow 2",
        "nodes": [],
        "edges": []
    }
    
    client.post("/flows", json=flow)
    
    # Then get it
    response = client.get("/flows/test-flow-2")
    assert response.status_code == 200
    assert response.json()["id"] == "test-flow-2"

def test_delete_flow(client):
    """Test deleting a flow."""
    # First create a flow
    flow = {
        "id": "test-flow-3",
        "name": "Test Flow 3",
        "nodes": [],
        "edges": []
    }
    
    client.post("/flows", json=flow)
    
    # Then delete it
    response = client.delete("/flows/test-flow-3")
    assert response.status_code == 200
    assert response.json()["message"] == "Flow deleted"
    
    # Verify it's gone
    response = client.get("/flows/test-flow-3")
    assert response.status_code == 404
