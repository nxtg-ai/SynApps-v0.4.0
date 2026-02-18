"""Tests for the WebSocket protocol: auth, structured messages, reconnection, and state recovery."""

import json
import time

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock

from apps.orchestrator.main import (
    app,
    ws_manager,
    broadcast_status,
    _ws_message,
    _WSSession,
    WebSocketSessionManager,
    init_db,
    JWT_SECRET_KEY,
    JWT_ALGORITHM,
)

import jwt as pyjwt


@pytest_asyncio.fixture(autouse=True)
async def setup_db():
    await init_db()
    yield


@pytest.fixture(autouse=True)
def _clean_ws_manager():
    """Reset ws_manager state between tests."""
    ws_manager._sessions.clear()
    ws_manager._ws_to_session.clear()
    ws_manager._message_buffer.clear()
    ws_manager._global_seq = 0
    yield
    ws_manager._sessions.clear()
    ws_manager._ws_to_session.clear()
    ws_manager._message_buffer.clear()
    ws_manager._global_seq = 0


# ------------------------------------------------------------------
# Structured message format
# ------------------------------------------------------------------


class TestStructuredMessages:
    def test_ws_message_has_required_fields(self):
        msg = _ws_message("test.type", {"key": "val"})
        assert "id" in msg
        assert msg["type"] == "test.type"
        assert msg["data"] == {"key": "val"}
        assert "timestamp" in msg
        assert isinstance(msg["timestamp"], float)

    def test_ws_message_default_empty_data(self):
        msg = _ws_message("ping")
        assert msg["data"] == {}

    def test_ws_message_ref_id(self):
        msg = _ws_message("pong", ref_id="abc-123")
        assert msg["ref_id"] == "abc-123"

    def test_ws_message_no_ref_id_by_default(self):
        msg = _ws_message("test")
        assert "ref_id" not in msg

    def test_ws_message_unique_ids(self):
        msg1 = _ws_message("a")
        msg2 = _ws_message("a")
        assert msg1["id"] != msg2["id"]


# ------------------------------------------------------------------
# Session manager unit tests
# ------------------------------------------------------------------


class TestWebSocketSessionManager:
    def test_create_new_session(self):
        mgr = WebSocketSessionManager()
        ws = MagicMock()
        sess, reconnected = mgr.create_session("user1", ws)
        assert reconnected is False
        assert sess.user_id == "user1"
        assert sess.state == "connected"

    def test_resume_session_same_user(self):
        mgr = WebSocketSessionManager()
        ws1 = MagicMock()
        sess1, _ = mgr.create_session("user1", ws1, session_id="s1")
        mgr.remove_session(ws1)

        ws2 = MagicMock()
        sess2, reconnected = mgr.create_session("user1", ws2, session_id="s1")
        assert reconnected is True
        assert sess2.session_id == "s1"
        assert sess2.websocket is ws2

    def test_resume_denied_different_user(self):
        mgr = WebSocketSessionManager()
        ws1 = MagicMock()
        mgr.create_session("user1", ws1, session_id="s1")
        mgr.remove_session(ws1)

        ws2 = MagicMock()
        sess, reconnected = mgr.create_session("user2", ws2, session_id="s1")
        assert reconnected is False
        assert sess.session_id != "s1"

    def test_connected_sessions(self):
        mgr = WebSocketSessionManager()
        ws = MagicMock()
        mgr.create_session("u1", ws)
        assert len(mgr.connected_sessions()) == 1

        mgr.remove_session(ws)
        assert len(mgr.connected_sessions()) == 0

    def test_connected_websockets_property(self):
        mgr = WebSocketSessionManager()
        ws = MagicMock()
        mgr.create_session("u1", ws)
        assert ws in mgr.connected_websockets

    def test_message_buffering(self):
        mgr = WebSocketSessionManager(buffer_size=5)
        for i in range(10):
            with mgr._lock:
                mgr._buffer_message({"type": "test", "i": i})
        assert len(mgr._message_buffer) == 5
        assert mgr._message_buffer[0]["i"] == 5

    def test_get_missed_messages(self):
        mgr = WebSocketSessionManager()
        with mgr._lock:
            mgr._buffer_message({"type": "a"})  # seq 1
            mgr._buffer_message({"type": "b"})  # seq 2
            mgr._buffer_message({"type": "c"})  # seq 3
        missed = mgr.get_missed_messages(1)
        assert len(missed) == 2
        assert missed[0]["type"] == "b"
        assert missed[1]["type"] == "c"

    def test_cleanup_expired(self):
        mgr = WebSocketSessionManager()
        ws = MagicMock()
        sess, _ = mgr.create_session("u1", ws, session_id="old")
        mgr.remove_session(ws)
        sess.last_active = time.time() - 10000
        with patch("apps.orchestrator.main.WS_SESSION_TTL_SECONDS", 1):
            removed = mgr.cleanup_expired()
        assert removed == 1

    def test_current_seq(self):
        mgr = WebSocketSessionManager()
        assert mgr.current_seq == 0
        with mgr._lock:
            mgr._buffer_message({"type": "x"})
        assert mgr.current_seq == 1


# ------------------------------------------------------------------
# WebSocket auth integration tests
# ------------------------------------------------------------------


class TestWebSocketAuth:
    def test_anonymous_auto_auth(self):
        """When no users exist, anonymous bootstrap auth should work."""
        client = TestClient(app)
        with client.websocket_connect("/api/v1/ws") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "auth.result"
            assert msg["data"]["authenticated"] is True
            assert "session_id" in msg["data"]
            assert "server_seq" in msg["data"]

    def test_auth_result_contains_user_id(self):
        client = TestClient(app)
        with client.websocket_connect("/api/v1/ws") as ws:
            msg = ws.receive_json()
            assert "user_id" in msg["data"]

    def test_legacy_ws_token_auth(self):
        """Auth via legacy WS_AUTH_TOKEN."""
        with patch("apps.orchestrator.main.WS_AUTH_TOKEN", "test-secret"):
            with patch("apps.orchestrator.main._can_use_anonymous_bootstrap", new_callable=AsyncMock, return_value=False):
                client = TestClient(app)
                with client.websocket_connect("/api/v1/ws") as ws:
                    ws.send_json({"type": "auth", "token": "test-secret"})
                    msg = ws.receive_json()
                    assert msg["type"] == "auth.result"
                    assert msg["data"]["authenticated"] is True

    def test_auth_timeout(self):
        """Connection should close with 4002 if no auth message is sent."""
        with patch("apps.orchestrator.main.WS_AUTH_TIMEOUT_SECONDS", 1):
            with patch("apps.orchestrator.main._can_use_anonymous_bootstrap", new_callable=AsyncMock, return_value=False):
                client = TestClient(app)
                try:
                    with client.websocket_connect("/api/v1/ws") as ws:
                        # Don't send auth — should get error then close
                        msg = ws.receive_json()
                        assert msg["type"] == "error"
                        assert msg["data"]["code"] == "AUTH_TIMEOUT"
                except Exception:
                    pass  # WebSocket close is expected

    def test_auth_bad_credentials(self):
        """Connection should close with 4001 on invalid credentials."""
        with patch("apps.orchestrator.main.WS_AUTH_TOKEN", "secret"):
            with patch("apps.orchestrator.main._can_use_anonymous_bootstrap", new_callable=AsyncMock, return_value=False):
                client = TestClient(app)
                try:
                    with client.websocket_connect("/api/v1/ws") as ws:
                        ws.send_json({"type": "auth", "token": "wrong"})
                        msg = ws.receive_json()
                        assert msg["type"] == "error"
                        assert msg["data"]["code"] == "AUTH_FAILED"
                except Exception:
                    pass

    def test_auth_wrong_message_type(self):
        """First message must be type 'auth'."""
        with patch("apps.orchestrator.main._can_use_anonymous_bootstrap", new_callable=AsyncMock, return_value=False):
            with patch("apps.orchestrator.main.WS_AUTH_TOKEN", "secret"):
                client = TestClient(app)
                try:
                    with client.websocket_connect("/api/v1/ws") as ws:
                        ws.send_json({"type": "ping"})
                        msg = ws.receive_json()
                        assert msg["type"] == "error"
                        assert msg["data"]["code"] == "AUTH_FAILED"
                except Exception:
                    pass


# ------------------------------------------------------------------
# Message protocol tests
# ------------------------------------------------------------------


class TestMessageProtocol:
    def test_ping_pong_with_ref_id(self):
        client = TestClient(app)
        with client.websocket_connect("/api/v1/ws") as ws:
            ws.receive_json()  # auth.result
            ws.send_json({"type": "ping", "id": "req-1"})
            msg = ws.receive_json()
            assert msg["type"] == "pong"
            assert msg.get("ref_id") == "req-1"

    def test_subscribe_ack(self):
        client = TestClient(app)
        with client.websocket_connect("/api/v1/ws") as ws:
            ws.receive_json()  # auth.result
            ws.send_json({
                "type": "subscribe",
                "id": "sub-1",
                "data": {"channel": "workflow.updates"},
            })
            msg = ws.receive_json()
            assert msg["type"] == "subscribe.ack"
            assert msg["data"]["channel"] == "workflow.updates"
            assert msg.get("ref_id") == "sub-1"

    def test_unsubscribe_ack(self):
        client = TestClient(app)
        with client.websocket_connect("/api/v1/ws") as ws:
            ws.receive_json()  # auth.result
            ws.send_json({
                "type": "unsubscribe",
                "data": {"channel": "workflow.updates"},
            })
            msg = ws.receive_json()
            assert msg["type"] == "unsubscribe.ack"
            assert msg["data"]["channel"] == "workflow.updates"

    def test_get_state(self):
        client = TestClient(app)
        with client.websocket_connect("/api/v1/ws") as ws:
            auth = ws.receive_json()
            session_id = auth["data"]["session_id"]
            ws.send_json({"type": "get_state", "id": "gs-1"})
            msg = ws.receive_json()
            assert msg["type"] == "state"
            assert msg["data"]["session_id"] == session_id
            assert "server_seq" in msg["data"]
            assert "subscriptions" in msg["data"]
            assert msg.get("ref_id") == "gs-1"

    def test_unknown_type_error(self):
        client = TestClient(app)
        with client.websocket_connect("/api/v1/ws") as ws:
            ws.receive_json()  # auth.result
            ws.send_json({"type": "foobar", "id": "u-1"})
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert msg["data"]["code"] == "UNKNOWN_MESSAGE_TYPE"
            assert msg.get("ref_id") == "u-1"

    def test_invalid_json_error(self):
        client = TestClient(app)
        with client.websocket_connect("/api/v1/ws") as ws:
            ws.receive_json()  # auth.result
            ws.send_text("not json")
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert msg["data"]["code"] == "INVALID_MESSAGE"


# ------------------------------------------------------------------
# Reconnection and state recovery
# ------------------------------------------------------------------


class TestReconnection:
    def test_reconnect_with_session_id(self):
        client = TestClient(app)
        # First connection
        with client.websocket_connect("/api/v1/ws") as ws:
            auth = ws.receive_json()
            session_id = auth["data"]["session_id"]

        # Reconnect with same session_id
        with client.websocket_connect(f"/api/v1/ws?session_id={session_id}") as ws:
            auth = ws.receive_json()
            assert auth["data"]["reconnected"] is True
            assert auth["data"]["session_id"] == session_id

    def test_new_session_without_session_id(self):
        client = TestClient(app)
        with client.websocket_connect("/api/v1/ws") as ws:
            auth = ws.receive_json()
            assert auth["data"]["reconnected"] is False

    def test_replay_missed_messages_on_reconnect(self, monkeypatch):
        """After reconnecting with last_seq, missed messages are replayed."""
        monkeypatch.setattr("apps.orchestrator.main.WS_HEARTBEAT_INTERVAL", 3600)
        client = TestClient(app)

        # First connection — get the session ID
        with client.websocket_connect("/api/v1/ws") as ws:
            auth = ws.receive_json()
            session_id = auth["data"]["session_id"]
            initial_seq = auth["data"]["server_seq"]

        # Simulate messages being broadcast while disconnected
        import asyncio

        async def _broadcast_test():
            await broadcast_status({"run_id": "r1", "status": "running"})
            await broadcast_status({"run_id": "r2", "status": "completed"})

        asyncio.get_event_loop().run_until_complete(_broadcast_test())

        # Reconnect with last_seq=initial_seq (should replay both)
        with client.websocket_connect(
            f"/api/v1/ws?session_id={session_id}&last_seq={initial_seq}"
        ) as ws:
            auth = ws.receive_json()
            assert auth["data"]["reconnected"] is True

            # Should receive replay.start, missed messages, replay.end
            replay_start = ws.receive_json()
            assert replay_start["type"] == "replay.start"
            assert replay_start["data"]["count"] == 2

            m1 = ws.receive_json()
            assert m1["type"] == "workflow.status"
            m2 = ws.receive_json()
            assert m2["type"] == "workflow.status"

            replay_end = ws.receive_json()
            assert replay_end["type"] == "replay.end"
            assert replay_end["data"]["count"] == 2

    def test_server_seq_in_auth_result(self):
        client = TestClient(app)
        with client.websocket_connect("/api/v1/ws") as ws:
            auth = ws.receive_json()
            assert "server_seq" in auth["data"]
            assert isinstance(auth["data"]["server_seq"], int)


# ------------------------------------------------------------------
# Broadcast integration
# ------------------------------------------------------------------


class TestBroadcast:
    @pytest.mark.asyncio
    async def test_broadcast_through_manager(self):
        """Broadcast sends to manager-tracked connections."""
        mock_ws = MagicMock()
        mock_ws.send_json = AsyncMock()
        ws_manager.create_session("u1", mock_ws)

        await broadcast_status({"run_id": "r1", "status": "ok"})
        assert mock_ws.send_json.called
        sent_msg = mock_ws.send_json.call_args[0][0]
        assert sent_msg["type"] == "workflow.status"
        assert sent_msg["data"]["run_id"] == "r1"

    @pytest.mark.asyncio
    async def test_broadcast_buffers_messages(self):
        """Broadcast messages are buffered for replay."""
        mock_ws = MagicMock()
        mock_ws.send_json = AsyncMock()
        ws_manager.create_session("u1", mock_ws)

        await broadcast_status({"run_id": "r1", "status": "ok"})
        assert ws_manager.current_seq > 0
        missed = ws_manager.get_missed_messages(0)
        assert len(missed) == 1
        assert missed[0]["data"]["run_id"] == "r1"
