import pytest
import os
import asyncio
from unittest.mock import patch

import tempfile

# Use a temporary file for the test database
db_fd, db_path = tempfile.mkstemp()
os.close(db_fd)
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{db_path}"

from apps.orchestrator.db import init_db, close_db_connections
from apps.orchestrator.repositories import FlowRepository, WorkflowRunRepository
from apps.orchestrator.models import Flow, FlowNode, FlowEdge, WorkflowRun

import pytest_asyncio

@pytest_asyncio.fixture(scope="function")
async def db():
    await init_db()
    yield
    await close_db_connections()

@pytest.mark.asyncio
async def test_flow_repository_save_and_get(db):
    flow_data = {
        "id": "test-flow",
        "name": "Test Flow",
        "nodes": [
            {"id": "node1", "type": "writer", "position": {"x": 10, "y": 20}, "data": {"p": 1}}
        ],
        "edges": [
            {"id": "edge1", "source": "start", "target": "node1"}
        ]
    }
    
    # Save
    saved_flow = await FlowRepository.save(flow_data)
    assert saved_flow["id"] == "test-flow"
    assert len(saved_flow["nodes"]) == 1
    assert saved_flow["nodes"][0]["id"] == "node1"
    
    # Get by ID
    retrieved_flow = await FlowRepository.get_by_id("test-flow")
    assert retrieved_flow is not None
    assert retrieved_flow["name"] == "Test Flow"
    
    # Get all
    all_flows = await FlowRepository.get_all()
    assert len(all_flows) >= 1
    assert any(f["id"] == "test-flow" for f in all_flows)

@pytest.mark.asyncio
async def test_flow_repository_update(db):
    flow_data = {"id": "update-flow", "name": "Initial Name"}
    await FlowRepository.save(flow_data)
    
    updated_data = {"id": "update-flow", "name": "Updated Name"}
    await FlowRepository.save(updated_data)
    
    retrieved = await FlowRepository.get_by_id("update-flow")
    assert retrieved["name"] == "Updated Name"

@pytest.mark.asyncio
async def test_flow_repository_delete(db):
    flow_id = "delete-flow"
    await FlowRepository.save({"id": flow_id, "name": "Delete Me"})
    
    success = await FlowRepository.delete(flow_id)
    assert success is True
    
    retrieved = await FlowRepository.get_by_id(flow_id)
    assert retrieved is None
    
    # Delete non-existent
    success = await FlowRepository.delete("not-there")
    assert success is False

@pytest.mark.asyncio
async def test_workflow_run_repository(db):
    run_data = {
        "run_id": "test-run",
        "flow_id": "test-flow",
        "status": "running",
        "progress": 50,
        "total_steps": 100,
        "input_data": {"query": "hello"}
    }
    
    # Save new
    saved_run = await WorkflowRunRepository.save(run_data)
    assert saved_run["run_id"] == "test-run"
    assert saved_run["status"] == "running"
    
    # Update
    update_data = {
        "run_id": "test-run",
        "status": "success",
        "progress": 100
    }
    updated_run = await WorkflowRunRepository.save(update_data)
    assert updated_run["status"] == "success"
    assert updated_run["progress"] == 100
    
    # Get by ID
    retrieved = await WorkflowRunRepository.get_by_run_id("test-run")
    assert retrieved["status"] == "success"
    
    # Get all
    all_runs = await WorkflowRunRepository.get_all()
    assert len(all_runs) >= 1
