"""Webhook registration CRUD, delivery with HMAC-SHA256 signing, and retry logic.

Secrets are stored encrypted via Fernet (shared with the API key manager).
Delivery uses async HTTP POST with a fixed backoff schedule: 1 s, 5 s, 30 s.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Event types
# -----------------------------------------------------------------------

WEBHOOK_EVENTS = frozenset({
    # Original workflow lifecycle events
    "template_started",
    "template_completed",
    "template_failed",
    "step_completed",
    "step_failed",
    # Directive-13: operational events
    "connector.status_changed",
    "request.failed",
    "key.rotated",
    "key.expiring_soon",
})

# -----------------------------------------------------------------------
# Delivery constants
# -----------------------------------------------------------------------

WEBHOOK_MAX_RETRIES = 3
WEBHOOK_RETRY_DELAYS = (1.0, 5.0, 30.0)  # fixed backoff per attempt
WEBHOOK_DELIVERY_TIMEOUT = 10.0            # seconds per attempt


# -----------------------------------------------------------------------
# HMAC signing
# -----------------------------------------------------------------------

def sign_payload(payload_bytes: bytes, secret: str) -> str:
    """Compute HMAC-SHA256 hex digest for *payload_bytes* using *secret*."""
    return hmac.new(secret.encode(), payload_bytes, hashlib.sha256).hexdigest()


# -----------------------------------------------------------------------
# WebhookManager
# -----------------------------------------------------------------------

class WebhookManager:
    """In-memory webhook registry with Fernet-encrypted secret storage.

    Parameters
    ----------
    encrypt_fn:
        Callable that encrypts a plaintext string and returns ciphertext.
    decrypt_fn:
        Callable that decrypts ciphertext and returns plaintext (or None).
    """

    def __init__(
        self,
        encrypt_fn: Optional[Callable[[str], str]] = None,
        decrypt_fn: Optional[Callable[[str], Optional[str]]] = None,
    ) -> None:
        self._lock = threading.Lock()
        self._hooks: Dict[str, Dict[str, Any]] = {}
        self._encrypt = encrypt_fn or (lambda s: s)
        self._decrypt = decrypt_fn or (lambda s: s)

    # -- CRUD -------------------------------------------------------------

    def register(
        self,
        url: str,
        events: List[str],
        secret: Optional[str] = None,
        active: bool = True,
    ) -> Dict[str, Any]:
        """Register a new webhook. Returns the hook dict (secret omitted)."""
        hook_id = str(uuid.uuid4())
        encrypted_secret: Optional[str] = None
        if secret:
            encrypted_secret = self._encrypt(secret)
        hook = {
            "id": hook_id,
            "url": url,
            "events": sorted(set(events)),
            "secret_encrypted": encrypted_secret,
            "active": active,
            "created_at": time.time(),
            "delivery_count": 0,
            "failure_count": 0,
            "last_delivery_at": None,
            "last_status_code": None,
        }
        with self._lock:
            self._hooks[hook_id] = hook
        return self._safe_view(hook)

    def get(self, hook_id: str) -> Optional[Dict[str, Any]]:
        """Get a hook by ID (internal view with encrypted secret)."""
        with self._lock:
            h = self._hooks.get(hook_id)
            return dict(h) if h else None

    def list_hooks(self) -> List[Dict[str, Any]]:
        """List all hooks (secrets omitted)."""
        with self._lock:
            return [self._safe_view(h) for h in self._hooks.values()]

    def delete(self, hook_id: str) -> bool:
        """Delete a hook. Returns True if it existed."""
        with self._lock:
            return self._hooks.pop(hook_id, None) is not None

    def update_active(self, hook_id: str, active: bool) -> bool:
        """Activate or deactivate a hook. Returns True if found."""
        with self._lock:
            h = self._hooks.get(hook_id)
            if h is None:
                return False
            h["active"] = active
            return True

    # -- Query ------------------------------------------------------------

    def hooks_for_event(self, event: str) -> List[Dict[str, Any]]:
        """Return internal copies of all active hooks subscribed to *event*."""
        with self._lock:
            return [
                dict(h)
                for h in self._hooks.values()
                if h["active"] and event in h["events"]
            ]

    # -- Delivery bookkeeping --------------------------------------------

    def record_delivery(
        self,
        hook_id: str,
        success: bool,
        status_code: Optional[int] = None,
    ) -> None:
        with self._lock:
            h = self._hooks.get(hook_id)
            if h:
                h["delivery_count"] += 1
                h["last_delivery_at"] = time.time()
                h["last_status_code"] = status_code
                if not success:
                    h["failure_count"] += 1

    # -- Secret helpers ---------------------------------------------------

    def decrypt_secret(self, hook: Dict[str, Any]) -> Optional[str]:
        """Decrypt the secret for a hook dict."""
        enc = hook.get("secret_encrypted")
        if enc is None:
            return None
        return self._decrypt(enc)

    # -- Lifecycle --------------------------------------------------------

    def reset(self) -> None:
        with self._lock:
            self._hooks.clear()

    # -- Internal ---------------------------------------------------------

    @staticmethod
    def _safe_view(h: Dict[str, Any]) -> Dict[str, Any]:
        """Return a copy with the encrypted secret stripped."""
        return {k: v for k, v in h.items() if k != "secret_encrypted"}


# -----------------------------------------------------------------------
# Delivery helpers (standalone async functions)
# -----------------------------------------------------------------------

async def deliver_webhook(
    hook: Dict[str, Any],
    payload: Dict[str, Any],
    manager: WebhookManager,
) -> bool:
    """Deliver *payload* to *hook* with HMAC signing and fixed-schedule retries.

    Retries up to ``WEBHOOK_MAX_RETRIES`` times with delays from
    ``WEBHOOK_RETRY_DELAYS`` (1 s, 5 s, 30 s).  Returns True on success.
    """
    payload_bytes = json.dumps(payload, default=str).encode()
    headers: Dict[str, str] = {"Content-Type": "application/json"}

    secret = manager.decrypt_secret(hook)
    if secret:
        sig = sign_payload(payload_bytes, secret)
        headers["X-Webhook-Signature"] = f"sha256={sig}"

    last_status: Optional[int] = None
    for attempt in range(WEBHOOK_MAX_RETRIES):
        try:
            async with httpx.AsyncClient(timeout=WEBHOOK_DELIVERY_TIMEOUT) as client:
                resp = await client.post(hook["url"], content=payload_bytes, headers=headers)
                last_status = resp.status_code
                if resp.status_code < 400:
                    manager.record_delivery(hook["id"], success=True, status_code=resp.status_code)
                    return True
        except Exception:
            pass
        if attempt < WEBHOOK_MAX_RETRIES - 1:
            await asyncio.sleep(WEBHOOK_RETRY_DELAYS[attempt])

    manager.record_delivery(hook["id"], success=False, status_code=last_status)
    return False


async def emit_webhook_event(
    event: str,
    data: Dict[str, Any],
    manager: WebhookManager,
) -> None:
    """Fire-and-forget delivery to all hooks registered for *event*."""
    hooks = manager.hooks_for_event(event)
    if not hooks:
        return
    payload = {"event": event, "timestamp": time.time(), "data": data}
    for hook in hooks:
        asyncio.create_task(deliver_webhook(hook, payload, manager))
