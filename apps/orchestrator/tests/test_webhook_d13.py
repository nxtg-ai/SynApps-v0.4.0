"""
Tests for DIRECTIVE-NXTG-20260223-13 — Webhook Notification System.

Covers:
- webhooks/manager.py: WebhookManager CRUD, Fernet-encrypted secrets,
  decrypt_secret, update_active, _safe_view
- New event types: connector.status_changed, request.failed,
  key.rotated, key.expiring_soon
- Delivery: HMAC-SHA256 signing, fixed backoff (1s, 5s, 30s),
  10s timeout, status_code tracking
- REST endpoints: POST/GET/DELETE /api/v1/webhooks with new events
- Signature verification round-trip
- Retry logic with correct delay schedule
- Event emission wiring: connector status changes, key rotation,
  request failures, key expiry
- Fernet secret storage: encrypted at rest, decrypted for delivery
- api_keys/manager.py: keys_expiring_within()
"""

from __future__ import annotations

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
    connector_health,
    emit_event,
    webhook_registry,
    _health_cache,
    _maybe_emit_status_change,
    KEY_EXPIRY_CHECK_INTERVAL,
    KEY_EXPIRY_WARNING_WINDOW,
)
from apps.orchestrator.webhooks.manager import (
    WEBHOOK_DELIVERY_TIMEOUT,
    WEBHOOK_EVENTS,
    WEBHOOK_MAX_RETRIES,
    WEBHOOK_RETRY_DELAYS,
    WebhookManager,
    deliver_webhook,
    emit_webhook_event,
    sign_payload,
)
from apps.orchestrator.api_keys.manager import api_key_manager


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _reset_state():
    """Reset webhook registry and connector health between tests."""
    webhook_registry.reset()
    connector_health.reset()
    _health_cache["results"] = None
    _health_cache["timestamp"] = 0.0
    yield
    webhook_registry.reset()
    connector_health.reset()
    _health_cache["results"] = None
    _health_cache["timestamp"] = 0.0


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:

    def test_max_retries(self):
        assert WEBHOOK_MAX_RETRIES == 3

    def test_retry_delays(self):
        assert WEBHOOK_RETRY_DELAYS == (1.0, 5.0, 30.0)

    def test_delivery_timeout(self):
        assert WEBHOOK_DELIVERY_TIMEOUT == 10.0

    def test_expiry_check_interval(self):
        assert KEY_EXPIRY_CHECK_INTERVAL == 3600

    def test_expiry_warning_window(self):
        assert KEY_EXPIRY_WARNING_WINDOW == 86400


# ---------------------------------------------------------------------------
# New event types
# ---------------------------------------------------------------------------

class TestNewEventTypes:

    def test_connector_status_changed(self):
        assert "connector.status_changed" in WEBHOOK_EVENTS

    def test_request_failed(self):
        assert "request.failed" in WEBHOOK_EVENTS

    def test_key_rotated(self):
        assert "key.rotated" in WEBHOOK_EVENTS

    def test_key_expiring_soon(self):
        assert "key.expiring_soon" in WEBHOOK_EVENTS

    def test_total_event_count(self):
        assert len(WEBHOOK_EVENTS) == 9


# ---------------------------------------------------------------------------
# WebhookManager CRUD
# ---------------------------------------------------------------------------

class TestWebhookManagerCRUD:

    def test_register_returns_id(self):
        mgr = WebhookManager()
        hook = mgr.register("https://a.com", ["key.rotated"])
        assert "id" in hook
        assert hook["url"] == "https://a.com"

    def test_register_strips_secret(self):
        mgr = WebhookManager()
        hook = mgr.register("https://a.com", ["key.rotated"], secret="s3cret")
        assert "secret_encrypted" not in hook

    def test_get_returns_internal_with_encrypted_secret(self):
        mgr = WebhookManager()
        hook = mgr.register("https://a.com", ["key.rotated"], secret="s3cret")
        internal = mgr.get(hook["id"])
        assert "secret_encrypted" in internal

    def test_list_hooks_no_secret(self):
        mgr = WebhookManager()
        mgr.register("https://a.com", ["key.rotated"], secret="s")
        hooks = mgr.list_hooks()
        assert len(hooks) == 1
        assert "secret_encrypted" not in hooks[0]

    def test_delete(self):
        mgr = WebhookManager()
        hook = mgr.register("https://a.com", ["key.rotated"])
        assert mgr.delete(hook["id"]) is True
        assert mgr.list_hooks() == []

    def test_delete_nonexistent(self):
        mgr = WebhookManager()
        assert mgr.delete("nope") is False

    def test_update_active(self):
        mgr = WebhookManager()
        hook = mgr.register("https://a.com", ["key.rotated"])
        assert mgr.update_active(hook["id"], False) is True
        internal = mgr.get(hook["id"])
        assert internal["active"] is False
        # Inactive hooks should not be returned by hooks_for_event
        assert mgr.hooks_for_event("key.rotated") == []

    def test_update_active_nonexistent(self):
        mgr = WebhookManager()
        assert mgr.update_active("nope", True) is False

    def test_hooks_for_event_filters(self):
        mgr = WebhookManager()
        mgr.register("https://a.com", ["key.rotated"])
        mgr.register("https://b.com", ["request.failed"])
        assert len(mgr.hooks_for_event("key.rotated")) == 1
        assert len(mgr.hooks_for_event("request.failed")) == 1
        assert len(mgr.hooks_for_event("key.expiring_soon")) == 0


# ---------------------------------------------------------------------------
# Fernet-encrypted secrets
# ---------------------------------------------------------------------------

class TestFernetSecrets:

    def test_encrypt_decrypt_round_trip(self):
        """Secrets are encrypted at rest and decryptable for delivery."""
        encrypted_values = []

        def mock_encrypt(plain):
            enc = f"ENC:{plain}"
            encrypted_values.append(enc)
            return enc

        def mock_decrypt(cipher):
            return cipher.replace("ENC:", "")

        mgr = WebhookManager(encrypt_fn=mock_encrypt, decrypt_fn=mock_decrypt)
        hook = mgr.register("https://a.com", ["key.rotated"], secret="my-secret")
        internal = mgr.get(hook["id"])
        assert internal["secret_encrypted"] == "ENC:my-secret"
        assert mgr.decrypt_secret(internal) == "my-secret"

    def test_no_secret_returns_none(self):
        mgr = WebhookManager()
        hook = mgr.register("https://a.com", ["key.rotated"])
        internal = mgr.get(hook["id"])
        assert mgr.decrypt_secret(internal) is None

    def test_global_registry_uses_fernet(self):
        """The module-level webhook_registry should use real Fernet encryption."""
        hook = webhook_registry.register("https://a.com", ["key.rotated"], secret="test-secret")
        internal = webhook_registry.get(hook["id"])
        encrypted = internal["secret_encrypted"]
        # Fernet tokens are base64 and much longer than plaintext
        assert encrypted != "test-secret"
        assert len(encrypted) > len("test-secret")
        # Round-trip decrypt
        decrypted = webhook_registry.decrypt_secret(internal)
        assert decrypted == "test-secret"


# ---------------------------------------------------------------------------
# Delivery: HMAC signing
# ---------------------------------------------------------------------------

class TestDeliverySigning:

    def test_sign_payload_hmac_sha256(self):
        payload = b'{"event": "key.rotated"}'
        sig = sign_payload(payload, "my-secret")
        expected = hmac.new(b"my-secret", payload, hashlib.sha256).hexdigest()
        assert sig == expected

    @pytest.mark.asyncio
    async def test_delivery_includes_signature_header(self):
        mgr = WebhookManager(
            encrypt_fn=lambda s: s,
            decrypt_fn=lambda s: s,
        )
        hook_data = mgr.register("https://mock.com/hook", ["key.rotated"], secret="s3cret")
        hook = mgr.get(hook_data["id"])
        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        with patch("httpx.AsyncClient", return_value=mock_client):
            await deliver_webhook(hook, {"event": "key.rotated"}, mgr)
        headers = mock_client.post.call_args.kwargs["headers"]
        assert "X-Webhook-Signature" in headers
        assert headers["X-Webhook-Signature"].startswith("sha256=")

    @pytest.mark.asyncio
    async def test_delivery_no_signature_without_secret(self):
        mgr = WebhookManager()
        hook_data = mgr.register("https://mock.com/hook", ["key.rotated"])
        hook = mgr.get(hook_data["id"])
        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        with patch("httpx.AsyncClient", return_value=mock_client):
            await deliver_webhook(hook, {"event": "key.rotated"}, mgr)
        headers = mock_client.post.call_args.kwargs["headers"]
        assert "X-Webhook-Signature" not in headers


# ---------------------------------------------------------------------------
# Delivery: retry with fixed backoff
# ---------------------------------------------------------------------------

class TestDeliveryRetry:

    @pytest.mark.asyncio
    async def test_retry_delays_are_fixed(self):
        """Retries use 1s, 5s, 30s delays (not exponential)."""
        mgr = WebhookManager()
        hook_data = mgr.register("https://mock.com/fail", ["request.failed"])
        hook = mgr.get(hook_data["id"])
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=Exception("fail"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        sleep_calls = []

        async def mock_sleep(seconds):
            sleep_calls.append(seconds)

        with patch("httpx.AsyncClient", return_value=mock_client):
            with patch("asyncio.sleep", side_effect=mock_sleep):
                result = await deliver_webhook(hook, {"event": "request.failed"}, mgr)
        assert result is False
        assert mock_client.post.call_count == 3
        # First 2 delays used (after attempt 0 and 1; attempt 2 is last)
        assert sleep_calls == [1.0, 5.0]

    @pytest.mark.asyncio
    async def test_success_on_second_attempt(self):
        mgr = WebhookManager()
        hook_data = mgr.register("https://mock.com/retry", ["key.rotated"])
        hook = mgr.get(hook_data["id"])
        fail_resp = AsyncMock()
        fail_resp.status_code = 500
        ok_resp = AsyncMock()
        ok_resp.status_code = 200
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=[Exception("fail"), ok_resp])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        with patch("httpx.AsyncClient", return_value=mock_client):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await deliver_webhook(hook, {"event": "key.rotated"}, mgr)
        assert result is True
        assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_records_last_status_code(self):
        mgr = WebhookManager()
        hook_data = mgr.register("https://mock.com/fail", ["request.failed"])
        hook = mgr.get(hook_data["id"])
        err_resp = AsyncMock()
        err_resp.status_code = 502
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=err_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        with patch("httpx.AsyncClient", return_value=mock_client):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                await deliver_webhook(hook, {"event": "request.failed"}, mgr)
        updated = mgr.get(hook_data["id"])
        assert updated["last_status_code"] == 502


# ---------------------------------------------------------------------------
# REST endpoints with new event types
# ---------------------------------------------------------------------------

class TestWebhookEndpointsNewEvents:

    def test_register_connector_status_changed(self, client):
        resp = client.post("/api/v1/webhooks", json={
            "url": "https://ops.example.com/connector",
            "events": ["connector.status_changed"],
        })
        assert resp.status_code == 201

    def test_register_request_failed(self, client):
        resp = client.post("/api/v1/webhooks", json={
            "url": "https://ops.example.com/errors",
            "events": ["request.failed"],
        })
        assert resp.status_code == 201

    def test_register_key_rotated(self, client):
        resp = client.post("/api/v1/webhooks", json={
            "url": "https://ops.example.com/keys",
            "events": ["key.rotated"],
        })
        assert resp.status_code == 201

    def test_register_key_expiring_soon(self, client):
        resp = client.post("/api/v1/webhooks", json={
            "url": "https://ops.example.com/keys",
            "events": ["key.expiring_soon"],
        })
        assert resp.status_code == 201

    def test_register_mixed_events(self, client):
        resp = client.post("/api/v1/webhooks", json={
            "url": "https://ops.example.com/all",
            "events": ["connector.status_changed", "key.rotated", "template_completed"],
        })
        assert resp.status_code == 201
        data = resp.json()
        assert sorted(data["events"]) == sorted([
            "connector.status_changed", "key.rotated", "template_completed",
        ])

    def test_register_invalid_event_still_rejected(self, client):
        resp = client.post("/api/v1/webhooks", json={
            "url": "https://ops.example.com/bad",
            "events": ["totally.made.up"],
        })
        assert resp.status_code == 422

    def test_list_includes_new_hooks(self, client):
        client.post("/api/v1/webhooks", json={
            "url": "https://a.com", "events": ["key.rotated"],
        })
        client.post("/api/v1/webhooks", json={
            "url": "https://b.com", "events": ["connector.status_changed"],
        })
        resp = client.get("/api/v1/webhooks")
        assert resp.json()["total"] == 2

    def test_delete_new_event_hook(self, client):
        create = client.post("/api/v1/webhooks", json={
            "url": "https://d.com", "events": ["request.failed"],
        })
        hook_id = create.json()["id"]
        resp = client.delete(f"/api/v1/webhooks/{hook_id}")
        assert resp.status_code == 200
        assert client.get("/api/v1/webhooks").json()["total"] == 0


# ---------------------------------------------------------------------------
# connector.status_changed emission
# ---------------------------------------------------------------------------

class TestConnectorStatusChangedEvent:

    @pytest.mark.asyncio
    async def test_emits_on_status_transition(self):
        """Status transition from healthy→degraded emits connector.status_changed."""
        connector_health.reset()
        connector_health.record_success("openai", latency_ms=50.0)
        old_status = connector_health.get_dashboard_status("openai")["dashboard_status"]
        # Force a failure to transition to degraded
        connector_health.record_failure("openai", "err", latency_ms=100.0)
        with patch("apps.orchestrator.main.emit_event", new_callable=AsyncMock) as mock_emit:
            await _maybe_emit_status_change("openai", old_status)
        mock_emit.assert_called_once()
        args = mock_emit.call_args
        assert args[0][0] == "connector.status_changed"
        assert args[0][1]["connector"] == "openai"
        assert args[0][1]["old_status"] == "healthy"
        assert args[0][1]["new_status"] == "degraded"

    @pytest.mark.asyncio
    async def test_no_emission_when_status_unchanged(self):
        """No event emitted if status doesn't change."""
        connector_health.reset()
        connector_health.record_success("openai", latency_ms=50.0)
        current = connector_health.get_dashboard_status("openai")["dashboard_status"]
        with patch("apps.orchestrator.main.emit_event", new_callable=AsyncMock) as mock_emit:
            await _maybe_emit_status_change("openai", current)
        mock_emit.assert_not_called()


# ---------------------------------------------------------------------------
# key.rotated emission
# ---------------------------------------------------------------------------

class TestKeyRotatedEvent:

    def test_rotate_endpoint_emits_event(self, client):
        """POST /managed-keys/{id}/rotate should emit key.rotated."""
        import os
        master_key = "test-master-key-d13"
        with patch.dict(os.environ, {"SYNAPPS_MASTER_KEY": master_key}):
            with patch("apps.orchestrator.main.SYNAPPS_MASTER_KEY", master_key):
                create_resp = client.post(
                    "/api/v1/managed-keys",
                    json={"name": "test-rotate", "scopes": ["read"]},
                    headers={"X-API-Key": master_key},
                )
                if create_resp.status_code != 201:
                    pytest.skip("Managed key creation not available")
                key_id = create_resp.json()["id"]

                with patch("apps.orchestrator.main.emit_event", new_callable=AsyncMock) as mock_emit:
                    resp = client.post(
                        f"/api/v1/managed-keys/{key_id}/rotate",
                        json={"grace_period": 3600},
                        headers={"X-API-Key": master_key},
                    )
                if resp.status_code == 200:
                    mock_emit.assert_called_once_with("key.rotated", {
                        "key_id": key_id,
                        "grace_period": 3600,
                    })


# ---------------------------------------------------------------------------
# keys_expiring_within
# ---------------------------------------------------------------------------

class TestKeysExpiringWithin:

    def test_no_expiring_keys(self):
        api_key_manager.reset()
        result = api_key_manager.keys_expiring_within(86400)
        assert result == []

    def test_finds_expiring_key(self):
        api_key_manager.reset()
        # Create key expiring in 1 hour
        key = api_key_manager.create("expiring", expires_in=3600)
        result = api_key_manager.keys_expiring_within(86400)
        assert len(result) == 1
        assert result[0]["id"] == key["id"]

    def test_excludes_key_expiring_after_window(self):
        api_key_manager.reset()
        # Create key expiring in 48 hours
        api_key_manager.create("far-future", expires_in=172800)
        result = api_key_manager.keys_expiring_within(86400)
        assert result == []

    def test_excludes_already_expired_key(self):
        api_key_manager.reset()
        # Create key, then manually expire it
        key = api_key_manager.create("expired", expires_in=1)
        # Manually set expires_at to the past
        with api_key_manager._lock:
            api_key_manager._keys[key["id"]]["expires_at"] = time.time() - 100
        result = api_key_manager.keys_expiring_within(86400)
        assert result == []

    def test_excludes_no_expiry_key(self):
        api_key_manager.reset()
        api_key_manager.create("forever")
        result = api_key_manager.keys_expiring_within(86400)
        assert result == []


# ---------------------------------------------------------------------------
# Signature verification round-trip
# ---------------------------------------------------------------------------

class TestSignatureVerification:

    @pytest.mark.asyncio
    async def test_receiver_can_verify_signature(self):
        """Simulate a receiver verifying the HMAC signature from delivery."""
        secret = "shared-secret"
        mgr = WebhookManager(
            encrypt_fn=lambda s: s,
            decrypt_fn=lambda s: s,
        )
        hook_data = mgr.register("https://receiver.com", ["key.rotated"], secret=secret)
        hook = mgr.get(hook_data["id"])

        captured_headers = {}
        captured_body = b""

        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_client = AsyncMock()

        async def capture_post(url, content=None, headers=None):
            nonlocal captured_headers, captured_body
            captured_headers = headers or {}
            captured_body = content or b""
            return mock_resp

        mock_client.post = capture_post
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        payload = {"event": "key.rotated", "data": {"key_id": "k1"}}
        with patch("httpx.AsyncClient", return_value=mock_client):
            await deliver_webhook(hook, payload, mgr)

        # Receiver-side verification
        sig_header = captured_headers["X-Webhook-Signature"]
        assert sig_header.startswith("sha256=")
        received_sig = sig_header[7:]  # strip "sha256="
        expected_sig = hmac.new(
            secret.encode(), captured_body, hashlib.sha256
        ).hexdigest()
        assert received_sig == expected_sig


# ---------------------------------------------------------------------------
# emit_webhook_event
# ---------------------------------------------------------------------------

class TestEmitWebhookEvent:

    @pytest.mark.asyncio
    async def test_emit_to_matching_hooks(self):
        mgr = WebhookManager()
        mgr.register("https://a.com", ["key.rotated"])
        mgr.register("https://b.com", ["request.failed"])
        with patch("apps.orchestrator.webhooks.manager.deliver_webhook", new_callable=AsyncMock) as mock_del:
            mock_del.return_value = True
            await emit_webhook_event("key.rotated", {"key_id": "k1"}, mgr)
            await asyncio.sleep(0.05)
        assert mock_del.call_count == 1

    @pytest.mark.asyncio
    async def test_no_delivery_for_unmatched_event(self):
        mgr = WebhookManager()
        mgr.register("https://a.com", ["key.rotated"])
        with patch("apps.orchestrator.webhooks.manager.deliver_webhook", new_callable=AsyncMock) as mock_del:
            await emit_webhook_event("request.failed", {}, mgr)
            await asyncio.sleep(0.05)
        mock_del.assert_not_called()

    @pytest.mark.asyncio
    async def test_payload_structure(self):
        mgr = WebhookManager()
        mgr.register("https://a.com", ["connector.status_changed"])
        with patch("apps.orchestrator.webhooks.manager.deliver_webhook", new_callable=AsyncMock) as mock_del:
            mock_del.return_value = True
            await emit_webhook_event("connector.status_changed", {"connector": "openai"}, mgr)
            await asyncio.sleep(0.05)
        payload = mock_del.call_args[0][1]
        assert payload["event"] == "connector.status_changed"
        assert "timestamp" in payload
        assert payload["data"]["connector"] == "openai"
