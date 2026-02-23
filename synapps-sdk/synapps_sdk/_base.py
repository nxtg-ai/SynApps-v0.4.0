"""Shared constants and helpers used by both sync and async clients."""

from __future__ import annotations

from typing import Any, Dict, Optional

DEFAULT_BASE_URL = "http://localhost:8000/api/v1"
DEFAULT_TIMEOUT = 30.0


def _build_headers(api_key: Optional[str]) -> Dict[str, str]:
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    return headers


def _extract_error(status_code: int, body: Any) -> str:
    """Pull the detail string from a FastAPI error response."""
    if isinstance(body, dict):
        return body.get("detail", str(body))
    return str(body)
