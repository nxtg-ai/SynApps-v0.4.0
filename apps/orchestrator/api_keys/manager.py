"""
API Key Manager — CRUD, Fernet encryption at rest, rotation with grace period.

Keys are stored in-memory with the raw key material encrypted using Fernet
symmetric encryption.  The encryption key is sourced from the
``SYNAPPS_KEY_ENCRYPTION_KEY`` environment variable (base64-url-safe 32 bytes).
If not set, a random key is generated at startup (keys will not survive restart).

Usage::

    from apps.orchestrator.api_keys.manager import api_key_manager

    key = api_key_manager.create("2brain-prod", scopes=["read", "write"], expires_in=3600)
    validated = api_key_manager.validate("sk-abc123...")
    rotated = api_key_manager.rotate(key["id"], grace_period=86400)
"""

import base64
import hashlib
import os
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Set

from cryptography.fernet import Fernet, InvalidToken

# ---------------------------------------------------------------------------
# Encryption setup
# ---------------------------------------------------------------------------

_ENV_KEY = os.environ.get("SYNAPPS_KEY_ENCRYPTION_KEY", "")

if _ENV_KEY:
    FERNET_KEY = _ENV_KEY.encode() if isinstance(_ENV_KEY, str) else _ENV_KEY
else:
    FERNET_KEY = Fernet.generate_key()

_fernet = Fernet(FERNET_KEY)

VALID_SCOPES: Set[str] = {"read", "write", "admin"}

# Default grace period for rotation (24 hours in seconds)
DEFAULT_GRACE_PERIOD = 24 * 60 * 60


def _encrypt(plaintext: str) -> str:
    """Encrypt a string and return base64-encoded ciphertext."""
    return _fernet.encrypt(plaintext.encode()).decode()


def _decrypt(ciphertext: str) -> Optional[str]:
    """Decrypt base64-encoded ciphertext, returning None on failure."""
    try:
        return _fernet.decrypt(ciphertext.encode()).decode()
    except (InvalidToken, Exception):
        return None


def _hash_key(plain_key: str) -> str:
    """SHA-256 hash of a plain key for fast lookup."""
    return hashlib.sha256(plain_key.encode()).hexdigest()


# ---------------------------------------------------------------------------
# API Key Manager
# ---------------------------------------------------------------------------


class APIKeyManager:
    """In-memory API key store with Fernet encryption, scoped permissions,
    and key rotation with configurable grace period.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # key_id -> key record
        self._keys: Dict[str, Dict[str, Any]] = {}
        # hash(plain_key) -> key_id  (fast lookup index)
        self._hash_index: Dict[str, str] = {}

    # ---- Create ----

    def create(
        self,
        name: str,
        scopes: Optional[List[str]] = None,
        expires_in: Optional[int] = None,
        rate_limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Create a new API key.

        Args:
            name: Human-readable label for the key.
            scopes: Permission scopes (subset of ``read``, ``write``, ``admin``).
                    Defaults to ``["read", "write"]``.
            expires_in: Seconds until expiry.  ``None`` = no expiry.
            rate_limit: Per-key rate limit override (requests/min).

        Returns:
            Key record **including the plain API key** (only returned once).

        Raises:
            ValueError: If scopes are invalid.
        """
        resolved_scopes = sorted(set(scopes or ["read", "write"]))
        invalid = [s for s in resolved_scopes if s not in VALID_SCOPES]
        if invalid:
            raise ValueError(f"Invalid scopes: {invalid}. Valid: {sorted(VALID_SCOPES)}")

        key_id = str(uuid.uuid4())
        plain_key = f"sk-{uuid.uuid4().hex}"
        key_hash = _hash_key(plain_key)
        encrypted_key = _encrypt(plain_key)

        now = time.time()
        entry: Dict[str, Any] = {
            "id": key_id,
            "name": name,
            "key_prefix": plain_key[:12],
            "key_hash": key_hash,
            "encrypted_key": encrypted_key,
            "scopes": resolved_scopes,
            "rate_limit": rate_limit,
            "is_active": True,
            "created_at": now,
            "expires_at": (now + expires_in) if expires_in else None,
            "last_used_at": None,
            "usage_count": 0,
            "rotated_from": None,
            "grace_deadline": None,
        }

        with self._lock:
            self._keys[key_id] = entry
            self._hash_index[key_hash] = key_id

        return {**self._safe_record(entry), "api_key": plain_key}

    # ---- Read ----

    def get(self, key_id: str) -> Optional[Dict[str, Any]]:
        """Get a key record by ID (without plain key)."""
        with self._lock:
            entry = self._keys.get(key_id)
            return self._safe_record(entry) if entry else None

    def list_keys(self, include_inactive: bool = False) -> List[Dict[str, Any]]:
        """List all keys (without plain keys)."""
        with self._lock:
            result = []
            for entry in self._keys.values():
                if not include_inactive and not entry["is_active"]:
                    continue
                result.append(self._safe_record(entry))
            return sorted(result, key=lambda k: k.get("created_at", 0), reverse=True)

    # ---- Update ----

    def revoke(self, key_id: str) -> bool:
        """Deactivate a key immediately."""
        with self._lock:
            entry = self._keys.get(key_id)
            if not entry:
                return False
            entry["is_active"] = False
            # Remove from hash index
            self._hash_index.pop(entry["key_hash"], None)
            return True

    def delete(self, key_id: str) -> bool:
        """Permanently remove a key."""
        with self._lock:
            entry = self._keys.pop(key_id, None)
            if not entry:
                return False
            self._hash_index.pop(entry["key_hash"], None)
            return True

    # ---- Rotate ----

    def rotate(
        self, key_id: str, grace_period: int = DEFAULT_GRACE_PERIOD
    ) -> Optional[Dict[str, Any]]:
        """Rotate a key: generate a new key, keep old valid during grace period.

        Args:
            key_id: ID of the key to rotate.
            grace_period: Seconds the old key remains valid after rotation (default 24h).

        Returns:
            New key record with ``api_key`` field, or ``None`` if key_id not found.
        """
        with self._lock:
            old = self._keys.get(key_id)
            if not old or not old["is_active"]:
                return None

            now = time.time()

            # Mark old key with grace deadline
            old["grace_deadline"] = now + grace_period
            old["is_active"] = True  # still active during grace

            # Generate new key
            new_id = str(uuid.uuid4())
            new_plain = f"sk-{uuid.uuid4().hex}"
            new_hash = _hash_key(new_plain)
            new_encrypted = _encrypt(new_plain)

            new_entry: Dict[str, Any] = {
                "id": new_id,
                "name": old["name"],
                "key_prefix": new_plain[:12],
                "key_hash": new_hash,
                "encrypted_key": new_encrypted,
                "scopes": list(old["scopes"]),
                "rate_limit": old["rate_limit"],
                "is_active": True,
                "created_at": now,
                "expires_at": old.get("expires_at"),
                "last_used_at": None,
                "usage_count": 0,
                "rotated_from": key_id,
                "grace_deadline": None,
            }

            self._keys[new_id] = new_entry
            self._hash_index[new_hash] = new_id

        return {**self._safe_record(new_entry), "api_key": new_plain}

    # ---- Validate ----

    def validate(self, plain_key: str) -> Optional[Dict[str, Any]]:
        """Validate a plain API key.  Returns the key record if valid, else None.

        Checks: hash lookup → active flag → expiry → grace period.
        Increments usage stats on success.
        """
        key_hash = _hash_key(plain_key)
        now = time.time()

        with self._lock:
            key_id = self._hash_index.get(key_hash)
            if not key_id:
                # Could be an old key in grace period — scan
                for entry in self._keys.values():
                    decrypted = _decrypt(entry["encrypted_key"])
                    if decrypted == plain_key:
                        key_id = entry["id"]
                        break
                if not key_id:
                    return None

            entry = self._keys.get(key_id)
            if not entry:
                return None

            # Check active
            if not entry["is_active"]:
                return None

            # Check expiry
            if entry["expires_at"] and now > entry["expires_at"]:
                entry["is_active"] = False
                self._hash_index.pop(entry["key_hash"], None)
                return None

            # Check grace period (old rotated key)
            if entry["grace_deadline"] and now > entry["grace_deadline"]:
                entry["is_active"] = False
                self._hash_index.pop(entry["key_hash"], None)
                return None

            # Success — update stats
            entry["last_used_at"] = now
            entry["usage_count"] += 1

            return self._safe_record(entry)

    def check_scope(self, plain_key: str, required_scope: str) -> bool:
        """Validate key and check it has the required scope."""
        record = self.validate(plain_key)
        if not record:
            return False
        return required_scope in record.get("scopes", [])

    # ---- Helpers ----

    def _safe_record(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Return a record without sensitive fields."""
        return {
            k: v for k, v in entry.items()
            if k not in ("encrypted_key", "key_hash")
        }

    def reset(self) -> None:
        """Clear all keys (for testing)."""
        with self._lock:
            self._keys.clear()
            self._hash_index.clear()

    def decrypt_key(self, key_id: str) -> Optional[str]:
        """Decrypt and return the plain key for a given ID (admin use only)."""
        with self._lock:
            entry = self._keys.get(key_id)
            if not entry:
                return None
            return _decrypt(entry["encrypted_key"])

    def keys_expiring_within(self, seconds: int) -> List[Dict[str, Any]]:
        """Return active keys that expire within *seconds* from now."""
        now = time.time()
        cutoff = now + seconds
        with self._lock:
            return [
                self._safe_record(e)
                for e in self._keys.values()
                if e["is_active"] and e.get("expires_at") and now < e["expires_at"] <= cutoff
            ]


# Module-level singleton
api_key_manager = APIKeyManager()
