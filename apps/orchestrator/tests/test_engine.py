import pytest
import os
import asyncio
import time
from unittest.mock import patch, MagicMock, AsyncMock
import tempfile

# Setup test DB
db_fd, db_path = tempfile.mkstemp()
os.close(db_fd)
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{db_path}"

from apps.orchestrator.main import Orchestrator, Flow, AppletMessage, BaseApplet
from apps.orchestrator.db import init_db, close_db_connections
from apps.orchestrator.repositories import FlowRepository, WorkflowRunRepository
import pytest_asyncio

@pytest_asyncio.fixture(scope="function")
async def db():
    await init_db()
    yield
    await close_db_connections()

@pytest.mark.asyncio
async def test_load_applet():
    # Test loading a built-in applet
    applet = await Orchestrator.load_applet("writer")
    assert isinstance(applet, BaseApplet)
    assert applet.__class__.__name__ == "WriterApplet"
    
    # Test loading non-existent applet
    with pytest.raises(ValueError):
        await Orchestrator.load_applet("non_existent")

@pytest.mark.asyncio
async def test_execute_flow_basic(db):
    flow_data = {
        "id": "test-flow-engine",
        "name": "Test Flow Engine",
        "nodes": [
            {"id": "start-node", "type": "start", "position": {"x": 0, "y": 0}},
            {"id": "end-node", "type": "end", "position": {"x": 100, "y": 0}}
        ],
        "edges": [
            {"id": "e1", "source": "start-node", "target": "end-node"}
        ]
    }
    
    # Save flow first as execute_flow might need it if it was fetching from DB 
    # (actually execute_flow takes a Flow object, but get_by_id is used in run_flow)
    
    flow_obj = Flow(**flow_data)
    
    # Mock broadcast_status to avoid websocket issues
    with patch("apps.orchestrator.main.broadcast_status", new_callable=AsyncMock) as mock_broadcast:
        run_id = await Orchestrator.execute_flow(flow_data, {"input": "data"})
        assert run_id is not None
        
        # Wait a bit for background task
        for _ in range(10):
            await asyncio.sleep(0.1)
            run = await WorkflowRunRepository.get_by_run_id(run_id)
            if run and run["status"] in ["success", "error"]:
                break
        
        assert run["status"] == "success"
        assert run["total_steps"] == 2

@pytest.mark.asyncio
async def test_execute_flow_with_applet(db):
    flow_data = {
        "id": "applet-flow",
        "name": "Applet Flow",
        "nodes": [
            {"id": "start", "type": "start", "position": {"x": 0, "y": 0}},
            {"id": "writer-node", "type": "writer", "position": {"x": 100, "y": 0}},
            {"id": "end", "type": "end", "position": {"x": 200, "y": 0}}
        ],
        "edges": [
            {"id": "e1", "source": "start", "target": "writer-node"},
            {"id": "e2", "source": "writer-node", "target": "end"}
        ]
    }
    
    # Mock WriterApplet.on_message
    mock_response = AppletMessage(content="Mocked Output", context={}, metadata={})
    
    with patch("apps.applets.writer.applet.WriterApplet.on_message", new_callable=AsyncMock) as mock_on_message:
        mock_on_message.return_value = mock_response
        
        with patch("apps.orchestrator.main.broadcast_status", new_callable=AsyncMock):
            run_id = await Orchestrator.execute_flow(flow_data, {"prompt": "hello"})
            
            # Wait for completion
            for _ in range(20):
                await asyncio.sleep(0.1)
                run = await WorkflowRunRepository.get_by_run_id(run_id)
                if run and run["status"] in ["success", "error"]:
                    break
            
            assert run["status"] == "success"
            assert "writer-node" in run["results"]
            assert run["results"]["writer-node"]["output"] == "Mocked Output"

@pytest.mark.asyncio
async def test_execute_flow_error(db):
    flow_data = {
        "id": "error-flow",
        "name": "Error Flow",
        "nodes": [
            {"id": "start", "type": "start", "position": {"x": 0, "y": 0}},
            {"id": "fail-node", "type": "writer", "position": {"x": 100, "y": 0}}
        ],
        "edges": [
            {"id": "e1", "source": "start", "target": "fail-node"}
        ]
    }
    
    with patch("apps.applets.writer.applet.WriterApplet.on_message", side_effect=Exception("Applet failed")):
        with patch("apps.orchestrator.main.broadcast_status", new_callable=AsyncMock):
            run_id = await Orchestrator.execute_flow(flow_data, {})
            
            for _ in range(10):
                await asyncio.sleep(0.1)
                run = await WorkflowRunRepository.get_by_run_id(run_id)
                if run and run["status"] == "error":
                    break
            
            assert run["status"] == "error"
            assert "Applet failed" in run["error"]
