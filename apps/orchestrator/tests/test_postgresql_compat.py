"""
Tests for PostgreSQL compatibility and schema enhancements.
"""
import pytest
import time
from apps.orchestrator.models import WorkflowRun
from apps.orchestrator.repositories import WorkflowRunRepository
from apps.orchestrator.db import init_db

@pytest.mark.asyncio
async def test_workflow_run_completed_applets_persistence():
    """Test that completed_applets is correctly persisted and retrieved."""
    await init_db()
    
    repo = WorkflowRunRepository()
    run_id = "test-compat-run"
    
    # Create a run with completed_applets
    run_data = {
        "run_id": run_id,
        "flow_id": "test-flow",
        "status": "running",
        "completed_applets": ["node1", "node2"],
        "start_time": time.time()
    }
    
    await repo.save(run_data)
    
    # Retrieve the run
    retrieved_run = await repo.get_by_run_id(run_id)
    
    assert retrieved_run is not None
    assert retrieved_run["run_id"] == run_id
    assert retrieved_run["completed_applets"] == ["node1", "node2"]
    
    # Update completed_applets
    run_data["completed_applets"] = ["node1", "node2", "node3"]
    await repo.save(run_data)
    
    # Retrieve again
    retrieved_run = await repo.get_by_run_id(run_id)
    assert retrieved_run["completed_applets"] == ["node1", "node2", "node3"]

@pytest.mark.asyncio
async def test_workflow_run_input_data_persistence():
    """Test that input_data is correctly persisted and retrieved."""
    await init_db()
    
    repo = WorkflowRunRepository()
    run_id = "test-input-run"
    
    input_data = {"key1": "value1", "key2": 42}
    
    run_data = {
        "run_id": run_id,
        "flow_id": "test-flow",
        "status": "running",
        "input_data": input_data,
        "start_time": time.time()
    }
    
    await repo.save(run_data)
    
    retrieved_run = await repo.get_by_run_id(run_id)
    assert retrieved_run["input_data"] == input_data
