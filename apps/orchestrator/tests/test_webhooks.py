"""Tests for webhook registration, event emission, HMAC signing, and delivery."""

import asyncio
import hashlib
import hmac
import json
import time
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from apps.orchestrator.main import (
    app,
    emit_event,
    webhook_registry,
    _sign_payload,
    _deliver_webhook,
    WEBHOOK_EVENTS,
)


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _reset_webhooks():
    """Clean webhook registry between tests."""
    webhook_registry.reset()
    yield
    webhook_registry.reset()


# ---------------------------------------------------------------------------
# WebhookRegistry — unit tests
# ---------------------------------------------------------------------------


def test_register_webhook():
    """Registering a webhook returns its ID and fields (no secret)."""
    hook = webhook_registry.register(
        url="https://example.com/hook",
        events=["template_completed"],
        secret="my-secret",
    )
    assert "id" in hook
    assert hook["url"] == "https://example.com/hook"
    assert hook["events"] == ["template_completed"]
    assert "secret" not in hook  # never leaked


def test_list_webhooks():
    """list_hooks returns all registered webhooks."""
    webhook_registry.register("https://a.com", ["template_started"])
    webhook_registry.register("https://b.com", ["template_failed"])
    hooks = webhook_registry.list_hooks()
    assert len(hooks) == 2
    urls = {h["url"] for h in hooks}
    assert "https://a.com" in urls
    assert "https://b.com" in urls


def test_delete_webhook():
    """Deleting a webhook removes it from the registry."""
    hook = webhook_registry.register("https://del.com", ["template_started"])
    assert webhook_registry.delete(hook["id"]) is True
    assert webhook_registry.list_hooks() == []


def test_delete_nonexistent_webhook():
    """Deleting a non-existent webhook returns False."""
    assert webhook_registry.delete("fake-id") is False


def test_hooks_for_event():
    """hooks_for_event returns only hooks subscribed to that event."""
    webhook_registry.register("https://a.com", ["template_started", "template_completed"])
    webhook_registry.register("https://b.com", ["template_failed"])
    started = webhook_registry.hooks_for_event("template_started")
    assert len(started) == 1
    assert started[0]["url"] == "https://a.com"


def test_record_delivery():
    """record_delivery updates counters."""
    hook = webhook_registry.register("https://x.com", ["step_completed"])
    hook_data = webhook_registry.get(hook["id"])
    assert hook_data["delivery_count"] == 0
    webhook_registry.record_delivery(hook["id"], success=True)
    hook_data = webhook_registry.get(hook["id"])
    assert hook_data["delivery_count"] == 1
    assert hook_data["failure_count"] == 0
    webhook_registry.record_delivery(hook["id"], success=False)
    hook_data = webhook_registry.get(hook["id"])
    assert hook_data["delivery_count"] == 2
    assert hook_data["failure_count"] == 1


# ---------------------------------------------------------------------------
# HMAC signing
# ---------------------------------------------------------------------------


def test_sign_payload():
    """_sign_payload produces a valid HMAC-SHA256 hex digest."""
    payload = b'{"event": "test"}'
    secret = "test-secret"
    sig = _sign_payload(payload, secret)
    expected = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    assert sig == expected


def test_sign_payload_different_secrets():
    """Different secrets produce different signatures."""
    payload = b'{"event": "test"}'
    sig1 = _sign_payload(payload, "secret-a")
    sig2 = _sign_payload(payload, "secret-b")
    assert sig1 != sig2


# ---------------------------------------------------------------------------
# Webhook delivery with retry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deliver_webhook_success():
    """Successful delivery returns True and records success."""
    hook_data = webhook_registry.register("https://mock.com/ok", ["template_started"])
    hook = webhook_registry.get(hook_data["id"])
    mock_resp = AsyncMock()
    mock_resp.status_code = 200
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await _deliver_webhook(hook, {"event": "template_started"})
    assert result is True
    updated = webhook_registry.get(hook_data["id"])
    assert updated["delivery_count"] == 1
    assert updated["failure_count"] == 0


@pytest.mark.asyncio
async def test_deliver_webhook_with_hmac_header():
    """Delivery with secret includes X-Webhook-Signature header."""
    hook_data = webhook_registry.register("https://mock.com/signed", ["step_completed"], secret="s3cret")
    hook = webhook_registry.get(hook_data["id"])
    mock_resp = AsyncMock()
    mock_resp.status_code = 200
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    with patch("httpx.AsyncClient", return_value=mock_client):
        await _deliver_webhook(hook, {"event": "step_completed"})
    call_kwargs = mock_client.post.call_args
    headers = call_kwargs.kwargs.get("headers", {})
    assert "X-Webhook-Signature" in headers
    assert headers["X-Webhook-Signature"].startswith("sha256=")


@pytest.mark.asyncio
async def test_deliver_webhook_retry_on_failure():
    """Delivery retries on failure and records failure after exhausting attempts."""
    hook_data = webhook_registry.register("https://mock.com/fail", ["template_failed"])
    hook = webhook_registry.get(hook_data["id"])
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=Exception("connection refused"))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    with patch("httpx.AsyncClient", return_value=mock_client):
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await _deliver_webhook(hook, {"event": "template_failed"})
    assert result is False
    assert mock_client.post.call_count == 3  # 3 retry attempts
    updated = webhook_registry.get(hook_data["id"])
    assert updated["failure_count"] == 1


# ---------------------------------------------------------------------------
# emit_event
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_emit_event_no_hooks():
    """emit_event with no registered hooks does nothing."""
    # Should not raise
    await emit_event("template_started", {"run_id": "123"})


@pytest.mark.asyncio
async def test_emit_event_triggers_delivery():
    """emit_event creates delivery tasks for matching hooks."""
    webhook_registry.register("https://mock.com/hook", ["template_completed"])
    with patch("apps.orchestrator.main._deliver_webhook", new_callable=AsyncMock) as mock_deliver:
        mock_deliver.return_value = True
        await emit_event("template_completed", {"run_id": "r1"})
        # Give the task a chance to run
        await asyncio.sleep(0.05)
    assert mock_deliver.call_count == 1
    payload = mock_deliver.call_args[0][1]
    assert payload["event"] == "template_completed"
    assert payload["data"]["run_id"] == "r1"
    assert "timestamp" in payload


# ---------------------------------------------------------------------------
# API Endpoints: POST/GET/DELETE /webhooks
# ---------------------------------------------------------------------------


def test_register_webhook_endpoint(client):
    """POST /api/v1/webhooks creates a webhook."""
    resp = client.post("/api/v1/webhooks", json={
        "url": "https://example.com/hook",
        "events": ["template_started", "template_completed"],
    })
    assert resp.status_code == 201
    data = resp.json()
    assert "id" in data
    assert data["url"] == "https://example.com/hook"
    assert "secret" not in data


def test_register_webhook_with_secret(client):
    """POST /api/v1/webhooks with secret creates hook (secret not returned)."""
    resp = client.post("/api/v1/webhooks", json={
        "url": "https://example.com/signed",
        "events": ["step_failed"],
        "secret": "my-hmac-secret",
    })
    assert resp.status_code == 201
    assert "secret" not in resp.json()


def test_register_webhook_invalid_event(client):
    """POST /api/v1/webhooks rejects invalid event names."""
    resp = client.post("/api/v1/webhooks", json={
        "url": "https://example.com/bad",
        "events": ["invalid_event"],
    })
    assert resp.status_code == 422


def test_list_webhooks_endpoint(client):
    """GET /api/v1/webhooks returns all hooks."""
    client.post("/api/v1/webhooks", json={
        "url": "https://a.com/h1", "events": ["template_started"],
    })
    client.post("/api/v1/webhooks", json={
        "url": "https://b.com/h2", "events": ["template_failed"],
    })
    resp = client.get("/api/v1/webhooks")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    assert len(data["webhooks"]) == 2


def test_delete_webhook_endpoint(client):
    """DELETE /api/v1/webhooks/{id} removes a webhook."""
    create = client.post("/api/v1/webhooks", json={
        "url": "https://d.com", "events": ["step_completed"],
    })
    hook_id = create.json()["id"]
    resp = client.delete(f"/api/v1/webhooks/{hook_id}")
    assert resp.status_code == 200
    # Verify it's gone
    listing = client.get("/api/v1/webhooks").json()
    assert listing["total"] == 0


def test_delete_webhook_not_found(client):
    """DELETE /api/v1/webhooks/nonexistent returns 404."""
    resp = client.delete("/api/v1/webhooks/nonexistent-id")
    assert resp.status_code == 404


def test_webhook_events_constant():
    """WEBHOOK_EVENTS contains all expected event types."""
    expected = {"template_started", "template_completed", "template_failed", "step_completed", "step_failed"}
    assert WEBHOOK_EVENTS == expected
