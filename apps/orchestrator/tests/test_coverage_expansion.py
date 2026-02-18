import pytest
import asyncio
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock
import json

from apps.orchestrator.main import app, broadcast_status, connected_clients, Orchestrator, FlowRepository, init_db


@pytest.fixture(autouse=True)
async def setup_db():
    await init_db()
    yield


@pytest.mark.asyncio
async def test_websocket_and_broadcast():
    # Test websocket connection with structured protocol
    client = TestClient(app)
    with client.websocket_connect("/api/v1/ws") as websocket:
        # Receive the auto-auth result message (sent on connect when no WS_AUTH_TOKEN is set)
        auth_msg = websocket.receive_json()
        assert auth_msg["type"] == "auth.result"
        assert auth_msg["data"]["authenticated"] is True
        assert "id" in auth_msg
        assert "timestamp" in auth_msg

        assert len(connected_clients) > 0

        # Test broadcast_status sends structured message
        status_data = {"run_id": "test-run", "status": "running", "completed_applets": []}
        await broadcast_status(status_data)

        data = websocket.receive_json()
        assert data["type"] == "workflow.status"
        assert data["data"]["run_id"] == "test-run"
        assert "id" in data
        assert "timestamp" in data


@pytest.mark.asyncio
async def test_websocket_ping_pong():
    """Test WebSocket ping/pong structured messaging."""
    client = TestClient(app)
    with client.websocket_connect("/api/v1/ws") as websocket:
        # Receive auth result
        auth_msg = websocket.receive_json()
        assert auth_msg["type"] == "auth.result"

        # Send ping
        websocket.send_json({"type": "ping"})

        # Receive pong
        pong_msg = websocket.receive_json()
        assert pong_msg["type"] == "pong"
        assert "id" in pong_msg
        assert "timestamp" in pong_msg


@pytest.mark.asyncio
async def test_websocket_invalid_json():
    """Test WebSocket handles invalid JSON gracefully."""
    client = TestClient(app)
    with client.websocket_connect("/api/v1/ws") as websocket:
        # Receive auth result
        auth_msg = websocket.receive_json()
        assert auth_msg["type"] == "auth.result"

        # Send invalid JSON
        websocket.send_text("not json")

        # Receive error
        error_msg = websocket.receive_json()
        assert error_msg["type"] == "error"
        assert error_msg["data"]["code"] == "INVALID_MESSAGE"


@pytest.mark.asyncio
async def test_websocket_unknown_message_type():
    """Test WebSocket responds to unknown message types."""
    client = TestClient(app)
    with client.websocket_connect("/api/v1/ws") as websocket:
        # Receive auth result
        auth_msg = websocket.receive_json()
        assert auth_msg["type"] == "auth.result"

        # Send unknown type
        websocket.send_json({"type": "unknown_type"})

        # Receive error
        error_msg = websocket.receive_json()
        assert error_msg["type"] == "error"
        assert error_msg["data"]["code"] == "UNKNOWN_MESSAGE_TYPE"


@pytest.mark.asyncio
async def test_get_flow_404():
    client = TestClient(app)
    response = client.get("/api/v1/flows/non-existent-flow")
    assert response.status_code == 404
    data = response.json()
    assert data["error"]["code"] == "NOT_FOUND"
    assert data["error"]["message"] == "Flow not found"

@pytest.mark.asyncio
async def test_delete_flow_404():
    client = TestClient(app)
    response = client.delete("/api/v1/flows/non-existent-flow")
    assert response.status_code == 404

@pytest.mark.asyncio
async def test_get_run_404():
    client = TestClient(app)
    response = client.get("/api/v1/runs/non-existent-run")
    assert response.status_code == 404

@pytest.mark.asyncio
async def test_ai_suggest_501():
    client = TestClient(app)
    response = client.post("/api/v1/ai/suggest", json={"prompt": "test"})
    assert response.status_code == 501
    data = response.json()
    assert data["error"]["code"] == "NOT_IMPLEMENTED"

@pytest.mark.asyncio
async def test_create_flow_auto_id():
    client = TestClient(app)
    flow_data = {
        "name": "Auto ID Flow",
        "nodes": [],
        "edges": []
    }
    response = client.post("/api/v1/flows", json=flow_data)
    assert response.status_code == 201
    assert "id" in response.json()
    generated_id = response.json()["id"]

    # Verify it can be retrieved
    response = client.get(f"/api/v1/flows/{generated_id}")
    assert response.status_code == 200
    assert response.json()["name"] == "Auto ID Flow"

@pytest.mark.asyncio
async def test_execute_flow_with_parsed_input():
    # Test the path where start node has parsedInputData
    flow_data = {
        "id": "parsed-input-flow",
        "name": "Parsed Input Flow",
        "nodes": [
            {
                "id": "start",
                "type": "start",
                "data": {"parsedInputData": {"key": "value"}},
                "position": {"x": 0, "y": 0}
            },
            {"id": "end", "type": "end", "position": {"x": 100, "y": 0}}
        ],
        "edges": [
            {"id": "e1", "source": "start", "target": "end"}
        ]
    }

    with patch("apps.orchestrator.main.broadcast_status", new_callable=AsyncMock):
        run_id = await Orchestrator.execute_flow(flow_data, {})

        # Wait for completion
        from apps.orchestrator.repositories import WorkflowRunRepository
        for _ in range(10):
            await asyncio.sleep(0.1)
            run = await WorkflowRunRepository.get_by_run_id(run_id)
            if run and run["status"] == "success":
                break

        assert run["status"] == "success"
        assert run["input_data"] == {"key": "value"}

@pytest.mark.asyncio
async def test_execute_flow_with_artist_config():
    # Test the path where artist node has config
    flow_data = {
        "id": "artist-config-flow",
        "name": "Artist Config Flow",
        "nodes": [
            {"id": "start", "type": "start", "position": {"x": 0, "y": 0}},
            {
                "id": "artist-node",
                "type": "artist",
                "data": {"systemPrompt": "draw a cat", "generator": "dall-e-3"},
                "position": {"x": 100, "y": 0}
            },
            {"id": "end", "type": "end", "position": {"x": 200, "y": 0}}
        ],
        "edges": [
            {"id": "e1", "source": "start", "target": "artist-node"},
            {"id": "e2", "source": "artist-node", "target": "end"}
        ]
    }

    # Mock ArtistApplet.on_message
    from apps.applets.artist.applet import ArtistApplet
    mock_response = MagicMock()
    mock_response.content = "Image URL"
    mock_response.context = {}

    with patch("apps.applets.artist.applet.ArtistApplet.on_message", new_callable=AsyncMock) as mock_on_message:
        mock_on_message.return_value = mock_response

        with patch("apps.orchestrator.main.broadcast_status", new_callable=AsyncMock):
            run_id = await Orchestrator.execute_flow(flow_data, {"prompt": "hello"})

            # Wait for completion
            from apps.orchestrator.repositories import WorkflowRunRepository
            for _ in range(20):
                await asyncio.sleep(0.1)
                run = await WorkflowRunRepository.get_by_run_id(run_id)
                if run and run["status"] == "success":
                    break

            assert run["status"] == "success"
            # Verify mock was called with correct metadata
            args, kwargs = mock_on_message.call_args
            message = args[0]
            assert message.metadata["system_prompt"] == "draw a cat"
            assert message.metadata["generator"] == "dall-e-3"

@pytest.mark.asyncio
async def test_broadcast_status_error_handling():
    mock_ws = MagicMock()
    mock_ws.send_json = AsyncMock(side_effect=Exception("Send failed"))
    connected_clients.append(mock_ws)

    try:
        await broadcast_status({"status": "test"})
        # Should catch exception and log it, not crash
    finally:
        if mock_ws in connected_clients:
            connected_clients.remove(mock_ws)

@pytest.mark.asyncio
async def test_list_applets_discovery():
    client = TestClient(app)
    # Mocking os.path.exists and os.listdir to simulate discovering applets
    with patch("os.path.exists", return_value=True):
        with patch("os.listdir", return_value=["writer", "memory", "new_applet"]):
            with patch("apps.orchestrator.main.Orchestrator.load_applet", new_callable=AsyncMock) as mock_load:
                mock_applet = MagicMock()
                mock_applet.get_metadata.return_value = {"name": "NewApplet", "description": "Desc", "version": "0.1.0", "capabilities": []}
                mock_load.return_value = mock_applet

                response = client.get("/api/v1/applets")
                assert response.status_code == 200
                data = response.json()
                # Paginated response
                assert "items" in data
                applets = data["items"]
                assert any(a["type"] == "new_applet" for a in applets)

@pytest.mark.asyncio
async def test_orchestrator_execute_flow_full():
    flow = {
        "id": "full-flow",
        "nodes": [
            {"id": "s1", "type": "start", "data": {"parsedInputData": {"init": "val"}}, "position": {"x":0, "y":0}},
            {"id": "w1", "type": "writer", "data": {"systemPrompt": "be helpful"}, "position": {"x":100, "y":0}},
            {"id": "a1", "type": "artist", "data": {"generator": "openai"}, "position": {"x":200, "y":0}},
            {"id": "e1", "type": "end", "position": {"x":300, "y":0}}
        ],
        "edges": [
            {"id": "e1", "source": "s1", "target": "w1"},
            {"id": "e2", "source": "w1", "target": "a1"},
            {"id": "e3", "source": "a1", "target": "e1"}
        ]
    }

    mock_run = {"run_id": "test-run", "status": "running", "progress": 0, "nodes": flow["nodes"]}

    from apps.orchestrator.repositories import WorkflowRunRepository
    with patch("apps.orchestrator.repositories.WorkflowRunRepository.save", new_callable=AsyncMock) as mock_save:
        with patch("apps.orchestrator.repositories.WorkflowRunRepository.get_by_run_id", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_run

            # Mock applets
            mock_writer = MagicMock()
            mock_writer.on_message = AsyncMock(return_value=MagicMock(content="written", context={}))

            mock_artist = MagicMock()
            mock_artist.on_message = AsyncMock(return_value=MagicMock(content="drawn", context={}))

            async def mock_load_applet(applet_type):
                if applet_type == "writer": return mock_writer
                if applet_type == "artist": return mock_artist
                raise ValueError("Unknown")

            with patch("apps.orchestrator.main.Orchestrator.load_applet", side_effect=mock_load_applet):
                mock_broadcast = AsyncMock()
                repo = WorkflowRunRepository()

                await Orchestrator._execute_flow_async("test-run", flow, {"input": "initial"}, repo, mock_broadcast)

                # Check results
                last_status = mock_save.call_args[0][0]
                assert last_status["status"] == "success"
                assert "w1" in last_status["results"]
                assert "a1" in last_status["results"]

@pytest.mark.asyncio
async def test_orchestrator_execute_flow_multiple_starts():
    flow = {
        "id": "multi-start",
        "nodes": [
            {"id": "s1", "type": "start", "position": {"x":0, "y":0}},
            {"id": "s2", "type": "start", "position": {"x":0, "y":100}},
            {"id": "e1", "type": "end", "position": {"x":100, "y":50}}
        ],
        "edges": [
            {"id": "e1", "source": "s1", "target": "e1"},
            {"id": "e2", "source": "s2", "target": "e1"}
        ]
    }

    mock_run = {"run_id": "test-run", "status": "running", "progress": 0}

    from apps.orchestrator.repositories import WorkflowRunRepository
    with patch("apps.orchestrator.repositories.WorkflowRunRepository.save", new_callable=AsyncMock) as mock_save:
        with patch("apps.orchestrator.repositories.WorkflowRunRepository.get_by_run_id", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_run

            mock_broadcast = AsyncMock()
            repo = WorkflowRunRepository()

            await Orchestrator._execute_flow_async("test-run", flow, {}, repo, mock_broadcast)

            assert mock_save.call_args[0][0]["status"] == "success"

@pytest.mark.asyncio
async def test_orchestrator_execute_flow_generic_error():
    flow = {
        "id": "error-flow",
        "nodes": [{"id": "s1", "type": "start", "position": {"x":0, "y":0}}],
        "edges": []
    }

    from apps.orchestrator.repositories import WorkflowRunRepository
    # Force a generic exception
    with patch("apps.orchestrator.repositories.WorkflowRunRepository.get_by_run_id", side_effect=Exception("DB Error")):
        repo = WorkflowRunRepository()
        mock_broadcast = AsyncMock()

        await Orchestrator._execute_flow_async("test-run", flow, {}, repo, mock_broadcast)
        # Should catch exception and broadcast error
        assert mock_broadcast.call_count >= 1
        assert mock_broadcast.call_args[0][0]["status"] == "error"
        assert "DB Error" in mock_broadcast.call_args[0][0]["error"]
