"""
Rate Limiter middleware for SynApps Orchestrator.

Implements both a sliding-window counter and a token bucket rate limiter.
The token bucket provides smooth rate limiting with burst allowance, while
the sliding window enforces hard per-user caps.

Both per-API-key and global limits are enforced. Limits are configurable
via environment variables and can vary by user tier.
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

# Token bucket defaults
TOKEN_BUCKET_RATE = int(os.environ.get("TOKEN_BUCKET_RATE", "60"))  # tokens/min per key
TOKEN_BUCKET_BURST = int(os.environ.get("TOKEN_BUCKET_BURST", "10"))  # extra burst tokens
TOKEN_BUCKET_GLOBAL_RATE = int(os.environ.get("TOKEN_BUCKET_GLOBAL_RATE", "300"))  # tokens/min global
TOKEN_BUCKET_GLOBAL_BURST = int(os.environ.get("TOKEN_BUCKET_GLOBAL_BURST", "50"))  # global burst

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


# ---------------------------------------------------------------------------
# Token Bucket algorithm
# ---------------------------------------------------------------------------


class TokenBucket:
    """Classic token bucket: tokens refill at a steady rate up to a maximum.

    ``rate`` is tokens added per second.  ``burst`` is the maximum token
    count (capacity).  A request consumes one token; if the bucket is empty
    the request is rejected.
    """

    def __init__(self, rate: float, burst: int) -> None:
        self.rate = rate  # tokens per second
        self.burst = burst  # max tokens (capacity)
        self._tokens = float(burst)  # start full
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
        self._last_refill = now

    def consume(self, tokens: int = 1) -> Tuple[bool, float, float]:
        """Try to consume ``tokens``.

        Returns ``(allowed, remaining, retry_after_seconds)``.
        ``retry_after_seconds`` is 0 when allowed, otherwise the estimated
        wait until enough tokens are available.
        """
        with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True, self._tokens, 0.0
            # How long until we have enough?
            deficit = tokens - self._tokens
            retry_after = deficit / self.rate if self.rate > 0 else 60.0
            return False, 0.0, retry_after

    @property
    def tokens(self) -> float:
        with self._lock:
            self._refill()
            return self._tokens

    def reset(self) -> None:
        with self._lock:
            self._tokens = float(self.burst)
            self._last_refill = time.monotonic()


class TokenBucketRegistry:
    """Manages per-key token buckets and a single global bucket."""

    def __init__(
        self,
        default_rate: float = TOKEN_BUCKET_RATE / 60.0,  # convert per-min to per-sec
        default_burst: int = TOKEN_BUCKET_BURST,
        global_rate: float = TOKEN_BUCKET_GLOBAL_RATE / 60.0,
        global_burst: int = TOKEN_BUCKET_GLOBAL_BURST,
    ) -> None:
        self._lock = threading.Lock()
        self.default_rate = default_rate
        self.default_burst = default_burst
        self._buckets: Dict[str, TokenBucket] = {}
        self.global_bucket = TokenBucket(rate=global_rate, burst=global_burst)

    def get_or_create(self, key: str, rate: Optional[float] = None, burst: Optional[int] = None) -> TokenBucket:
        with self._lock:
            if key not in self._buckets:
                self._buckets[key] = TokenBucket(
                    rate=rate if rate is not None else self.default_rate,
                    burst=burst if burst is not None else self.default_burst + int((rate or self.default_rate) * 60),
                )
            return self._buckets[key]

    def consume(self, key: str, rate: Optional[float] = None, burst: Optional[int] = None) -> Tuple[bool, float, float, str]:
        """Consume a token from both the per-key and global buckets.

        Returns ``(allowed, remaining, retry_after, limited_by)`` where
        ``limited_by`` is ``"key"`` or ``"global"`` when blocked, ``""`` when allowed.
        """
        # Check global first
        g_allowed, g_remaining, g_retry = self.global_bucket.consume()
        if not g_allowed:
            return False, 0.0, g_retry, "global"

        bucket = self.get_or_create(key, rate=rate, burst=burst)
        k_allowed, k_remaining, k_retry = bucket.consume()
        if not k_allowed:
            # Refund global token since per-key was rejected
            with self.global_bucket._lock:
                self.global_bucket._tokens = min(
                    self.global_bucket.burst, self.global_bucket._tokens + 1
                )
            return False, 0.0, k_retry, "key"

        return True, k_remaining, 0.0, ""

    def reset(self) -> None:
        with self._lock:
            self._buckets.clear()
        self.global_bucket.reset()


# Module-level token bucket registry
_token_buckets = TokenBucketRegistry()


def _identify_client(request: Request) -> Tuple[str, str, Optional[int]]:
    """Extract a rate-limit key, tier, and optional per-key limit from the request.

    The key is built from the authenticated user ID when available,
    falling back to IP address for unauthenticated requests.

    Returns (key, tier, custom_limit) where custom_limit is None to use
    tier default or an integer for per-key override.
    """
    # Check for user info set by auth middleware/dependency
    user = getattr(request.state, "user", None)
    if user and isinstance(user, dict):
        user_id = user.get("id", "")
        if user_id and user_id != "anonymous":
            tier = user.get("tier", "free")
            custom_limit = user.get("rate_limit")  # per-key override
            return f"user:{user_id}", tier, custom_limit

    # Fall back to IP-based limiting
    client_host = request.client.host if request.client else "unknown"
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        client_host = forwarded.split(",")[0].strip()
    return f"ip:{client_host}", "anonymous", None


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """Rate limiter combining sliding-window counters and token bucket.

    Requests must pass **both** the sliding-window check (hard cap) and
    the token-bucket check (smooth rate + burst) to be allowed.  The
    token bucket also enforces a global limit across all keys.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for exempt paths and WebSocket upgrades
        if request.url.path in EXEMPT_PATHS:
            return await call_next(request)
        if request.headers.get("upgrade", "").lower() == "websocket":
            return await call_next(request)

        key, tier, custom_limit = _identify_client(request)
        limit = custom_limit if custom_limit is not None else _get_limit_for_tier(tier)
        window = RATE_LIMIT_WINDOW_SECONDS

        # --- Sliding window check (hard cap) ---
        sw_allowed, sw_remaining, sw_retry = _counter.check_and_record(key, limit, window)

        if not sw_allowed:
            return _rate_limit_response(limit, sw_retry)

        # --- Token bucket check (smooth rate + burst + global) ---
        per_key_rate = (custom_limit / 60.0) if custom_limit else None
        tb_allowed, _tb_remaining, tb_retry, limited_by = _token_buckets.consume(
            key, rate=per_key_rate
        )

        if not tb_allowed:
            retry = max(int(tb_retry) + 1, 1)
            scope = "Global rate" if limited_by == "global" else "Rate"
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "code": "RATE_LIMIT_EXCEEDED",
                        "status": 429,
                        "message": f"{scope} limit exceeded. Try again in {retry} seconds.",
                    }
                },
                headers={
                    "Retry-After": str(retry),
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(time.time()) + retry),
                    "X-RateLimit-Scope": limited_by,
                },
            )

        response = await call_next(request)

        # Attach rate-limit info headers to successful responses
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(sw_remaining)
        response.headers["X-RateLimit-Reset"] = str(
            int(time.time()) + window
        )

        return response


def _rate_limit_response(limit: int, retry_after: int) -> JSONResponse:
    """Build a 429 JSONResponse for sliding-window rejections."""
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


def add_rate_limiter(app) -> None:
    """Register the rate limiter middleware on a FastAPI application."""
    app.add_middleware(RateLimiterMiddleware)


def get_token_bucket_registry() -> TokenBucketRegistry:
    """Return the module-level token bucket registry (for testing/introspection)."""
    return _token_buckets
