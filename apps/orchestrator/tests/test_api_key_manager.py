"""Tests for API Key Manager — CRUD, Fernet encryption, rotation, scopes (DIRECTIVE-11)."""

import time

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from apps.orchestrator.api_keys.manager import (
    APIKeyManager,
    _encrypt,
    _decrypt,
    _hash_key,
    api_key_manager,
    VALID_SCOPES,
    DEFAULT_GRACE_PERIOD,
)
from apps.orchestrator.main import app


MASTER_KEY = "test-master-key-secret"


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _reset_manager():
    api_key_manager.reset()
    yield
    api_key_manager.reset()


# ---------------------------------------------------------------------------
# Fernet encryption helpers
# ---------------------------------------------------------------------------


class TestEncryption:
    def test_encrypt_decrypt_roundtrip(self):
        plaintext = "sk-abc123secret"
        ct = _encrypt(plaintext)
        assert ct != plaintext
        assert _decrypt(ct) == plaintext

    def test_decrypt_invalid_returns_none(self):
        assert _decrypt("not-valid-ciphertext") is None

    def test_hash_key_deterministic(self):
        assert _hash_key("abc") == _hash_key("abc")

    def test_hash_key_differs_for_different_input(self):
        assert _hash_key("abc") != _hash_key("xyz")


# ---------------------------------------------------------------------------
# APIKeyManager — Create
# ---------------------------------------------------------------------------


class TestCreate:
    def test_create_returns_api_key(self):
        result = api_key_manager.create("test-key")
        assert result["api_key"].startswith("sk-")
        assert result["name"] == "test-key"
        assert result["is_active"] is True

    def test_create_default_scopes(self):
        result = api_key_manager.create("default")
        assert result["scopes"] == ["read", "write"]

    def test_create_custom_scopes(self):
        result = api_key_manager.create("admin", scopes=["admin", "read"])
        assert "admin" in result["scopes"]
        assert "read" in result["scopes"]

    def test_create_invalid_scopes_raises(self):
        with pytest.raises(ValueError, match="Invalid scopes"):
            api_key_manager.create("bad", scopes=["invalid"])

    def test_create_with_expiry(self):
        result = api_key_manager.create("expiring", expires_in=3600)
        assert result["expires_at"] is not None
        assert result["expires_at"] > time.time()

    def test_create_no_expiry(self):
        result = api_key_manager.create("permanent")
        assert result["expires_at"] is None

    def test_create_with_rate_limit(self):
        result = api_key_manager.create("limited", rate_limit=100)
        assert result["rate_limit"] == 100

    def test_key_prefix_stored(self):
        result = api_key_manager.create("prefix-test")
        assert len(result["key_prefix"]) == 12
        assert result["api_key"].startswith(result["key_prefix"])

    def test_encrypted_key_not_in_response(self):
        result = api_key_manager.create("no-cipher")
        assert "encrypted_key" not in result
        assert "key_hash" not in result

    def test_usage_count_starts_at_zero(self):
        result = api_key_manager.create("usage")
        assert result["usage_count"] == 0


# ---------------------------------------------------------------------------
# APIKeyManager — Read
# ---------------------------------------------------------------------------


class TestRead:
    def test_get_by_id(self):
        created = api_key_manager.create("findme")
        found = api_key_manager.get(created["id"])
        assert found is not None
        assert found["name"] == "findme"

    def test_get_nonexistent(self):
        assert api_key_manager.get("nonexistent") is None

    def test_list_keys(self):
        api_key_manager.create("a")
        api_key_manager.create("b")
        keys = api_key_manager.list_keys()
        assert len(keys) == 2

    def test_list_keys_excludes_inactive_by_default(self):
        created = api_key_manager.create("revocable")
        api_key_manager.revoke(created["id"])
        keys = api_key_manager.list_keys()
        assert len(keys) == 0

    def test_list_keys_includes_inactive(self):
        created = api_key_manager.create("revocable")
        api_key_manager.revoke(created["id"])
        keys = api_key_manager.list_keys(include_inactive=True)
        assert len(keys) == 1


# ---------------------------------------------------------------------------
# APIKeyManager — Validate
# ---------------------------------------------------------------------------


class TestValidate:
    def test_validate_active_key(self):
        created = api_key_manager.create("valid")
        result = api_key_manager.validate(created["api_key"])
        assert result is not None
        assert result["name"] == "valid"

    def test_validate_increments_usage(self):
        created = api_key_manager.create("counter")
        api_key_manager.validate(created["api_key"])
        api_key_manager.validate(created["api_key"])
        record = api_key_manager.get(created["id"])
        assert record["usage_count"] == 2

    def test_validate_updates_last_used(self):
        created = api_key_manager.create("timestamp")
        api_key_manager.validate(created["api_key"])
        record = api_key_manager.get(created["id"])
        assert record["last_used_at"] is not None

    def test_validate_nonexistent_key(self):
        assert api_key_manager.validate("sk-nonexistent") is None

    def test_validate_revoked_key(self):
        created = api_key_manager.create("revoked")
        api_key_manager.revoke(created["id"])
        assert api_key_manager.validate(created["api_key"]) is None

    def test_validate_expired_key(self):
        mgr = APIKeyManager()
        created = mgr.create("expired", expires_in=1)
        # Manually set expires_at to the past
        with mgr._lock:
            mgr._keys[created["id"]]["expires_at"] = time.time() - 10
        assert mgr.validate(created["api_key"]) is None


# ---------------------------------------------------------------------------
# APIKeyManager — Scope checking
# ---------------------------------------------------------------------------


class TestScopes:
    def test_check_scope_valid(self):
        created = api_key_manager.create("scoped", scopes=["read", "write"])
        assert api_key_manager.check_scope(created["api_key"], "read") is True
        assert api_key_manager.check_scope(created["api_key"], "write") is True

    def test_check_scope_missing(self):
        created = api_key_manager.create("readonly", scopes=["read"])
        assert api_key_manager.check_scope(created["api_key"], "write") is False
        assert api_key_manager.check_scope(created["api_key"], "admin") is False

    def test_check_scope_invalid_key(self):
        assert api_key_manager.check_scope("sk-fake", "read") is False


# ---------------------------------------------------------------------------
# APIKeyManager — Rotation
# ---------------------------------------------------------------------------


class TestRotation:
    def test_rotate_returns_new_key(self):
        old = api_key_manager.create("rotatable")
        new = api_key_manager.rotate(old["id"])
        assert new is not None
        assert new["api_key"] != old["api_key"]
        assert new["rotated_from"] == old["id"]

    def test_old_key_valid_during_grace(self):
        old = api_key_manager.create("grace")
        api_key_manager.rotate(old["id"], grace_period=3600)
        # Old key should still work
        result = api_key_manager.validate(old["api_key"])
        assert result is not None

    def test_old_key_invalid_after_grace(self):
        mgr = APIKeyManager()
        old = mgr.create("grace-expired")
        mgr.rotate(old["id"], grace_period=1)
        # Manually expire the grace period
        with mgr._lock:
            mgr._keys[old["id"]]["grace_deadline"] = time.time() - 10
        assert mgr.validate(old["api_key"]) is None

    def test_new_key_works_immediately(self):
        old = api_key_manager.create("rotate-new")
        new = api_key_manager.rotate(old["id"])
        result = api_key_manager.validate(new["api_key"])
        assert result is not None
        assert result["name"] == "rotate-new"

    def test_rotate_preserves_scopes(self):
        old = api_key_manager.create("scoped-rotate", scopes=["admin"])
        new = api_key_manager.rotate(old["id"])
        assert new["scopes"] == ["admin"]

    def test_rotate_preserves_rate_limit(self):
        old = api_key_manager.create("rl-rotate", rate_limit=42)
        new = api_key_manager.rotate(old["id"])
        assert new["rate_limit"] == 42

    def test_rotate_nonexistent(self):
        assert api_key_manager.rotate("nonexistent") is None

    def test_rotate_revoked_key(self):
        old = api_key_manager.create("revoked-rotate")
        api_key_manager.revoke(old["id"])
        assert api_key_manager.rotate(old["id"]) is None

    def test_default_grace_period(self):
        assert DEFAULT_GRACE_PERIOD == 86400  # 24 hours

    def test_zero_grace_period_immediate_revoke(self):
        mgr = APIKeyManager()
        old = mgr.create("zero-grace")
        mgr.rotate(old["id"], grace_period=0)
        # Grace deadline is now, old key should still technically be valid
        # until the next validate call after the deadline
        with mgr._lock:
            mgr._keys[old["id"]]["grace_deadline"] = time.time() - 1
        assert mgr.validate(old["api_key"]) is None


# ---------------------------------------------------------------------------
# APIKeyManager — Revoke / Delete
# ---------------------------------------------------------------------------


class TestRevokeDelete:
    def test_revoke_deactivates(self):
        created = api_key_manager.create("revokable")
        assert api_key_manager.revoke(created["id"]) is True
        record = api_key_manager.get(created["id"])
        assert record["is_active"] is False

    def test_revoke_nonexistent(self):
        assert api_key_manager.revoke("fake") is False

    def test_delete_removes(self):
        created = api_key_manager.create("deletable")
        assert api_key_manager.delete(created["id"]) is True
        assert api_key_manager.get(created["id"]) is None

    def test_delete_nonexistent(self):
        assert api_key_manager.delete("fake") is False


# ---------------------------------------------------------------------------
# APIKeyManager — Decrypt (admin)
# ---------------------------------------------------------------------------


class TestDecrypt:
    def test_decrypt_key(self):
        created = api_key_manager.create("decrypt-test")
        plain = api_key_manager.decrypt_key(created["id"])
        assert plain == created["api_key"]

    def test_decrypt_nonexistent(self):
        assert api_key_manager.decrypt_key("fake") is None


# ---------------------------------------------------------------------------
# REST API Endpoints — POST /managed-keys
# ---------------------------------------------------------------------------


class TestCreateEndpoint:
    def test_create_201(self, client):
        with patch("apps.orchestrator.main.SYNAPPS_MASTER_KEY", MASTER_KEY):
            resp = client.post(
                "/api/v1/managed-keys",
                json={"name": "test-key", "scopes": ["read"]},
                headers={"X-API-Key": MASTER_KEY},
            )
        assert resp.status_code == 201
        assert resp.json()["api_key"].startswith("sk-")
        assert resp.json()["scopes"] == ["read"]

    def test_create_requires_master_key(self, client):
        resp = client.post("/api/v1/managed-keys", json={"name": "bad"})
        assert resp.status_code in (401, 403, 503)

    def test_create_with_expiry(self, client):
        with patch("apps.orchestrator.main.SYNAPPS_MASTER_KEY", MASTER_KEY):
            resp = client.post(
                "/api/v1/managed-keys",
                json={"name": "expiring", "expires_in": 3600},
                headers={"X-API-Key": MASTER_KEY},
            )
        assert resp.status_code == 201
        assert resp.json()["expires_at"] is not None

    def test_create_invalid_scopes_422(self, client):
        with patch("apps.orchestrator.main.SYNAPPS_MASTER_KEY", MASTER_KEY):
            resp = client.post(
                "/api/v1/managed-keys",
                json={"name": "bad", "scopes": ["invalid"]},
                headers={"X-API-Key": MASTER_KEY},
            )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# REST API Endpoints — GET /managed-keys
# ---------------------------------------------------------------------------


class TestListEndpoint:
    def test_list_empty(self, client):
        with patch("apps.orchestrator.main.SYNAPPS_MASTER_KEY", MASTER_KEY):
            resp = client.get("/api/v1/managed-keys", headers={"X-API-Key": MASTER_KEY})
        assert resp.status_code == 200
        assert resp.json()["total"] == 0

    def test_list_returns_keys(self, client):
        api_key_manager.create("a")
        api_key_manager.create("b")
        with patch("apps.orchestrator.main.SYNAPPS_MASTER_KEY", MASTER_KEY):
            resp = client.get("/api/v1/managed-keys", headers={"X-API-Key": MASTER_KEY})
        assert resp.json()["total"] == 2


# ---------------------------------------------------------------------------
# REST API Endpoints — GET /managed-keys/{id}
# ---------------------------------------------------------------------------


class TestGetEndpoint:
    def test_get_by_id(self, client):
        created = api_key_manager.create("findme")
        with patch("apps.orchestrator.main.SYNAPPS_MASTER_KEY", MASTER_KEY):
            resp = client.get(
                f"/api/v1/managed-keys/{created['id']}",
                headers={"X-API-Key": MASTER_KEY},
            )
        assert resp.status_code == 200
        assert resp.json()["name"] == "findme"

    def test_get_not_found(self, client):
        with patch("apps.orchestrator.main.SYNAPPS_MASTER_KEY", MASTER_KEY):
            resp = client.get(
                "/api/v1/managed-keys/nonexistent",
                headers={"X-API-Key": MASTER_KEY},
            )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# REST API Endpoints — POST /managed-keys/{id}/rotate
# ---------------------------------------------------------------------------


class TestRotateEndpoint:
    def test_rotate_success(self, client):
        created = api_key_manager.create("rotatable")
        with patch("apps.orchestrator.main.SYNAPPS_MASTER_KEY", MASTER_KEY):
            resp = client.post(
                f"/api/v1/managed-keys/{created['id']}/rotate",
                json={"grace_period": 3600},
                headers={"X-API-Key": MASTER_KEY},
            )
        assert resp.status_code == 200
        assert resp.json()["api_key"].startswith("sk-")
        assert resp.json()["rotated_from"] == created["id"]

    def test_rotate_not_found(self, client):
        with patch("apps.orchestrator.main.SYNAPPS_MASTER_KEY", MASTER_KEY):
            resp = client.post(
                "/api/v1/managed-keys/nonexistent/rotate",
                json={"grace_period": 3600},
                headers={"X-API-Key": MASTER_KEY},
            )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# REST API Endpoints — POST /managed-keys/{id}/revoke, DELETE
# ---------------------------------------------------------------------------


class TestRevokeDeleteEndpoints:
    def test_revoke_success(self, client):
        created = api_key_manager.create("revokable")
        with patch("apps.orchestrator.main.SYNAPPS_MASTER_KEY", MASTER_KEY):
            resp = client.post(
                f"/api/v1/managed-keys/{created['id']}/revoke",
                headers={"X-API-Key": MASTER_KEY},
            )
        assert resp.status_code == 200

    def test_delete_success(self, client):
        created = api_key_manager.create("deletable")
        with patch("apps.orchestrator.main.SYNAPPS_MASTER_KEY", MASTER_KEY):
            resp = client.delete(
                f"/api/v1/managed-keys/{created['id']}",
                headers={"X-API-Key": MASTER_KEY},
            )
        assert resp.status_code == 200

    def test_delete_not_found(self, client):
        with patch("apps.orchestrator.main.SYNAPPS_MASTER_KEY", MASTER_KEY):
            resp = client.delete(
                "/api/v1/managed-keys/nonexistent",
                headers={"X-API-Key": MASTER_KEY},
            )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Auth integration — managed keys authenticate requests
# ---------------------------------------------------------------------------


class TestAuthIntegration:
    def test_managed_key_authenticates(self, client):
        created = api_key_manager.create("auth-test", scopes=["read", "write"])
        resp = client.get(
            "/api/v1/flows",
            headers={"X-API-Key": created["api_key"]},
        )
        assert resp.status_code == 200

    def test_revoked_managed_key_rejected(self, client):
        created = api_key_manager.create("revoked-auth")
        api_key_manager.revoke(created["id"])
        resp = client.get(
            "/api/v1/flows",
            headers={"X-API-Key": created["api_key"]},
        )
        # Should fall through to other auth methods and fail
        assert resp.status_code in (401, 403)
