"""
Rate Limiter middleware for SynApps Orchestrator.

Implements a sliding-window rate limiter that enforces per-user request limits
on all API endpoints. Limits are configurable via environment variables and
can vary by user tier.
"""

import os
import time
import threading
from collections import defaultdict, deque
from typing import Callable, Deque, Dict, Optional, Tuple

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

# ---------------------------------------------------------------------------
# Default rate limits (requests per window)
# Override via environment variables.
# ---------------------------------------------------------------------------
RATE_LIMIT_WINDOW_SECONDS = int(os.environ.get("RATE_LIMIT_WINDOW_SECONDS", "60"))
RATE_LIMIT_FREE = int(os.environ.get("RATE_LIMIT_FREE", "60"))
RATE_LIMIT_PRO = int(os.environ.get("RATE_LIMIT_PRO", "200"))
RATE_LIMIT_ENTERPRISE = int(os.environ.get("RATE_LIMIT_ENTERPRISE", "1000"))
RATE_LIMIT_ANONYMOUS = int(os.environ.get("RATE_LIMIT_ANONYMOUS", "30"))

# Paths that are exempt from rate limiting (health checks, docs)
EXEMPT_PATHS = frozenset({
    "/",
    "/api/v1/health",
    "/api/v1/docs",
    "/api/v1/redoc",
    "/api/v1/openapi.json",
})

_TIER_LIMITS: Dict[str, int] = {
    "free": RATE_LIMIT_FREE,
    "pro": RATE_LIMIT_PRO,
    "enterprise": RATE_LIMIT_ENTERPRISE,
    "anonymous": RATE_LIMIT_ANONYMOUS,
}


def _get_limit_for_tier(tier: str) -> int:
    return _TIER_LIMITS.get(tier, RATE_LIMIT_ANONYMOUS)


class _SlidingWindowCounter:
    """Thread-safe sliding-window counter for rate limiting."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # key -> deque of timestamps
        self._windows: Dict[str, Deque[float]] = defaultdict(deque)

    def check_and_record(
        self, key: str, limit: int, window: int
    ) -> Tuple[bool, int, int]:
        """Check whether request is allowed and record it.

        Returns (allowed, remaining, retry_after_seconds).
        """
        now = time.monotonic()
        cutoff = now - window

        with self._lock:
            q = self._windows[key]
            # Evict expired entries
            while q and q[0] <= cutoff:
                q.popleft()

            if len(q) >= limit:
                # Determine when the oldest entry expires
                retry_after = int(q[0] - cutoff) + 1
                return False, 0, max(retry_after, 1)

            q.append(now)
            remaining = limit - len(q)
            return True, remaining, 0

    def cleanup_stale(self, window: int) -> None:
        """Remove keys whose entire window has expired (housekeeping)."""
        now = time.monotonic()
        cutoff = now - window
        with self._lock:
            stale_keys = [
                k for k, q in self._windows.items() if not q or q[-1] <= cutoff
            ]
            for k in stale_keys:
                del self._windows[k]


# Module-level counter shared by all requests
_counter = _SlidingWindowCounter()


def _identify_client(request: Request) -> Tuple[str, str]:
    """Extract a rate-limit key and tier from the request.

    The key is built from the authenticated user ID when available,
    falling back to IP address for unauthenticated requests.
    """
    # Check for user info set by auth middleware/dependency
    user = getattr(request.state, "user", None)
    if user and isinstance(user, dict):
        user_id = user.get("id", "")
        if user_id and user_id != "anonymous":
            tier = user.get("tier", "free")
            return f"user:{user_id}", tier

    # Fall back to IP-based limiting
    client_host = request.client.host if request.client else "unknown"
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        client_host = forwarded.split(",")[0].strip()
    return f"ip:{client_host}", "anonymous"


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """Sliding-window rate limiter applied to all API endpoints."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for exempt paths and WebSocket upgrades
        if request.url.path in EXEMPT_PATHS:
            return await call_next(request)
        if request.headers.get("upgrade", "").lower() == "websocket":
            return await call_next(request)

        key, tier = _identify_client(request)
        limit = _get_limit_for_tier(tier)
        window = RATE_LIMIT_WINDOW_SECONDS

        allowed, remaining, retry_after = _counter.check_and_record(key, limit, window)

        if not allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "code": "RATE_LIMIT_EXCEEDED",
                        "status": 429,
                        "message": f"Rate limit exceeded. Try again in {retry_after} seconds.",
                    }
                },
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(time.time()) + retry_after),
                },
            )

        response = await call_next(request)

        # Attach rate-limit info headers to successful responses
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(
            int(time.time()) + window
        )

        return response


def add_rate_limiter(app) -> None:
    """Register the rate limiter middleware on a FastAPI application."""
    app.add_middleware(RateLimiterMiddleware)
