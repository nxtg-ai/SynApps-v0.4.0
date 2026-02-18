
import pytest
import asyncio
import pytest_asyncio
from unittest.mock import patch, AsyncMock
from apps.orchestrator.main import Orchestrator, AppletMessage
from apps.orchestrator.db import init_db, close_db_connections
from apps.orchestrator.repositories import WorkflowRunRepository

@pytest_asyncio.fixture(scope="function")
async def db():
    await init_db()
    yield
    await close_db_connections()

@pytest.mark.asyncio
async def test_node_retry_logic(db):
    flow_data = {
        "id": "retry-flow",
        "name": "Retry Flow",
        "nodes": [
            {"id": "start", "type": "start", "position": {"x": 0, "y": 0}},
            {
                "id": "failing-node", 
                "type": "writer", 
                "position": {"x": 100, "y": 0},
                "data": {
                    "retry_config": {
                        "max_retries": 2,
                        "delay": 0.1,
                        "backoff": 1.0
                    }
                }
            },
            {"id": "end", "type": "end", "position": {"x": 200, "y": 0}}
        ],
        "edges": [
            {"id": "e1", "source": "start", "target": "failing-node"},
            {"id": "e2", "source": "failing-node", "target": "end"}
        ]
    }

    # Mock WriterApplet to fail twice then succeed
    mock_responses = [
        Exception("Attempt 1 Failure"),
        Exception("Attempt 2 Failure"),
        AppletMessage(content="Success after retries", context={}, metadata={})
    ]
    
    call_count = 0
    async def side_effect(*args, **kwargs):
        nonlocal call_count
        res = mock_responses[call_count]
        call_count += 1
        if isinstance(res, Exception):
            raise res
        return res

    with patch("apps.applets.writer.applet.WriterApplet.on_message", side_effect=side_effect):
        with patch("apps.orchestrator.main.broadcast_status", new_callable=AsyncMock):
            run_id = await Orchestrator.execute_flow(flow_data, {"prompt": "test"})
            
            # Wait for completion
            for _ in range(50):
                await asyncio.sleep(0.1)
                run = await WorkflowRunRepository.get_by_run_id(run_id)
                if run and run["status"] in ["success", "error"]:
                    break
            
            assert run["status"] == "success"
            assert call_count == 3  # Initial + 2 retries
            assert run["results"]["failing-node"]["output"] == "Success after retries"

@pytest.mark.asyncio
async def test_node_timeout_logic(db):
    flow_data = {
        "id": "timeout-flow",
        "name": "Timeout Flow",
        "nodes": [
            {"id": "start", "type": "start", "position": {"x": 0, "y": 0}},
            {
                "id": "slow-node", 
                "type": "writer", 
                "position": {"x": 100, "y": 0},
                "data": {
                    "timeout_seconds": 0.2
                }
            },
            {"id": "end", "type": "end", "position": {"x": 200, "y": 0}}
        ],
        "edges": [
            {"id": "e1", "source": "start", "target": "slow-node"},
            {"id": "e2", "source": "slow-node", "target": "end"}
        ]
    }

    async def slow_side_effect(*args, **kwargs):
        await asyncio.sleep(1.0)
        return AppletMessage(content="Too slow", context={}, metadata={})

    with patch("apps.applets.writer.applet.WriterApplet.on_message", side_effect=slow_side_effect):
        with patch("apps.orchestrator.main.broadcast_status", new_callable=AsyncMock):
            run_id = await Orchestrator.execute_flow(flow_data, {"prompt": "test"})
            
            # Wait for completion
            for _ in range(50):
                await asyncio.sleep(0.1)
                run = await WorkflowRunRepository.get_by_run_id(run_id)
                if run and run["status"] in ["success", "error"]:
                    break
            
            assert run["status"] == "error"
            assert "timed out" in run["error"]

@pytest.mark.asyncio
async def test_node_fallback_logic(db):
    flow_data = {
        "id": "fallback-flow",
        "name": "Fallback Flow",
        "nodes": [
            {"id": "start", "type": "start", "position": {"x": 0, "y": 0}},
            {
                "id": "failing-node", 
                "type": "writer", 
                "position": {"x": 100, "y": 0},
                "data": {
                    "fallback_node_id": "fallback-node"
                }
            },
            {"id": "fallback-node", "type": "writer", "position": {"x": 100, "y": 100}},
            {"id": "end", "type": "end", "position": {"x": 200, "y": 0}}
        ],
        "edges": [
            {"id": "e1", "source": "start", "target": "failing-node"},
            {"id": "e2", "source": "failing-node", "target": "end"}
        ]
    }

    async def failing_side_effect(message):
        node_id = message.metadata.get("node_id")
        if node_id == "failing-node":
            raise Exception("Primary failed")
        return AppletMessage(content=f"Result from {node_id}", context={}, metadata={})

    with patch("apps.applets.writer.applet.WriterApplet.on_message", side_effect=failing_side_effect):
        with patch("apps.orchestrator.main.broadcast_status", new_callable=AsyncMock):
            run_id = await Orchestrator.execute_flow(flow_data, {"prompt": "test"})
            
            # Wait for completion
            for _ in range(50):
                await asyncio.sleep(0.1)
                run = await WorkflowRunRepository.get_by_run_id(run_id)
                if run and run["status"] in ["success", "error"]:
                    break
            
            assert run["status"] == "success"
            assert "fallback-node" in run["completed_applets"]
            assert run["results"]["failing-node"]["status"] == "fallback"
            assert run["results"]["fallback-node"]["output"] == "Result from fallback-node"
