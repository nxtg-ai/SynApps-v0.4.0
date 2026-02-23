"""
SynApps Orchestrator - Core Module

This is the lightweight microkernel that routes messages between applets in a
defined sequence. The orchestrator's job is purely to pass messages and data
between applets.
"""
import asyncio
from abc import ABC, abstractmethod
import base64
import hashlib
import hmac
import importlib
import json
import logging
import math
import os
import re
import secrets
import shutil
import sqlite3
import sys
import time
import tempfile
import uuid
from enum import Enum
import threading
from typing import Any, AsyncIterator, Dict, List, Optional, Type
from pathlib import Path

# Load environment variables: .env takes priority, falls back to .env.development
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent.parent
env_path = project_root / ".env"
if not env_path.exists():
    env_path = project_root / ".env.development"
load_dotenv(dotenv_path=env_path)

from fastapi import (
    Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect,
    BackgroundTasks, APIRouter, Query, Request, Header,
)
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import httpx
import jwt
from pydantic import BaseModel, ConfigDict, Field, field_validator
from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy import select
from starlette.exceptions import HTTPException as StarletteHTTPException

# Import database modules
from contextlib import asynccontextmanager
from apps.orchestrator.db import init_db, close_db_connections, get_db_session
from apps.orchestrator.repositories import FlowRepository, WorkflowRunRepository
from apps.orchestrator.models import (
    APIKeyCreateRequestModel,
    APIKeyCreateResponseModel,
    APIKeyResponseModel,
    AuthLoginRequestModel,
    AuthRefreshRequestModel,
    AuthRegisterRequestModel,
    AuthTokenResponseModel,
    CodeNodeConfigModel,
    FlowModel,
    FlowNodeModel,
    FlowEdgeModel,
    IfElseNodeConfigModel,
    MergeNodeConfigModel,
    ForEachNodeConfigModel,
    WorkflowRunStatusModel,
    ImageGenNodeConfigModel,
    ImageGenRequestModel,
    ImageGenResponseModel,
    ImageModelInfoModel,
    ImageProviderInfoModel,
    HTTPRequestNodeConfigModel,
    LLMMessageModel,
    LLMModelInfoModel,
    MemoryNodeConfigModel,
    MemorySearchResultModel,
    TransformNodeConfigModel,
    LLMNodeConfigModel,
    LLMProviderInfoModel,
    LLMRequestModel,
    LLMResponseModel,
    LLMStreamChunkModel,
    LLMUsageModel,
    RefreshToken as AuthRefreshToken,
    SUPPORTED_MEMORY_BACKENDS,
    SUPPORTED_IMAGE_PROVIDERS,
    SUPPORTED_LLM_PROVIDERS,
    User as AuthUser,
    UserAPIKey as AuthUserAPIKey,
    UserProfileModel,
)

try:
    import resource
except Exception:  # pragma: no cover - non-Unix fallback
    resource = None  # type: ignore[assignment]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("orchestrator")

# ============================================================
# Constants
# ============================================================

API_VERSION = "1.0.0"
APP_START_TIME = time.time()
WS_AUTH_TOKEN = os.environ.get("WS_AUTH_TOKEN")
JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "synapps-dev-jwt-secret-change-me")
JWT_ALGORITHM = os.environ.get("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("JWT_ACCESS_EXPIRE_MINUTES", "15"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.environ.get("JWT_REFRESH_EXPIRE_DAYS", "14"))
PASSWORD_HASH_ITERATIONS = int(os.environ.get("PASSWORD_HASH_ITERATIONS", "390000"))
API_KEY_VALUE_PREFIX = os.environ.get("API_KEY_PREFIX", "synapps")
API_KEY_LOOKUP_PREFIX_LEN = int(os.environ.get("API_KEY_LOOKUP_PREFIX_LEN", "18"))
ALLOW_ANONYMOUS_WHEN_NO_USERS = os.environ.get(
    "ALLOW_ANONYMOUS_WHEN_NO_USERS",
    "true",
).strip().lower() in {"1", "true", "yes"}
LEGACY_WRITER_NODE_TYPE = "writer"
LEGACY_ARTIST_NODE_TYPE = "artist"
LEGACY_MEMORY_NODE_TYPE = "memory"
LLM_NODE_TYPE = "llm"
IMAGE_NODE_TYPE = "image"
MEMORY_NODE_TYPE = "memory"
HTTP_REQUEST_NODE_TYPE = "http_request"
CODE_NODE_TYPE = "code"
TRANSFORM_NODE_TYPE = "transform"
IF_ELSE_NODE_TYPE = "if_else"
MERGE_NODE_TYPE = "merge"
FOR_EACH_NODE_TYPE = "for_each"
ENGINE_MAX_CONCURRENCY = int(os.environ.get("ENGINE_MAX_CONCURRENCY", "10"))
TRACE_RESULTS_KEY = "__trace__"
TRACE_SCHEMA_VERSION = 1
MAX_DIFF_CHANGES = 250
# ============================================================
# In-Memory Metrics Collector
# ============================================================

class _MetricsCollector:
    """Thread-safe in-memory request/response metrics.

    Tracks request counts, error counts, response times, and provider usage.
    No external dependencies — designed for /metrics endpoint consumption.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.total_requests: int = 0
        self.total_errors: int = 0
        self._response_times: List[float] = []  # recent response times (capped)
        self._provider_usage: Dict[str, int] = {}
        self._template_runs: Dict[str, int] = {}
        self._last_template_run_time: Optional[float] = None
        self._max_samples: int = 1000

    def record_request(self, duration_ms: float, status_code: int, path: str) -> None:
        with self._lock:
            self.total_requests += 1
            if status_code >= 400:
                self.total_errors += 1
            if len(self._response_times) >= self._max_samples:
                self._response_times = self._response_times[self._max_samples // 2:]
            self._response_times.append(duration_ms)

    def record_provider_call(self, provider: str) -> None:
        with self._lock:
            self._provider_usage[provider] = self._provider_usage.get(provider, 0) + 1

    def record_template_run(self, template_name: str) -> None:
        with self._lock:
            self._template_runs[template_name] = self._template_runs.get(template_name, 0) + 1
            self._last_template_run_time = time.time()

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            avg_ms = (
                sum(self._response_times) / len(self._response_times)
                if self._response_times
                else 0.0
            )
            error_rate = (
                (self.total_errors / self.total_requests * 100)
                if self.total_requests > 0
                else 0.0
            )
            return {
                "requests": {
                    "total": self.total_requests,
                    "errors": self.total_errors,
                    "error_rate_pct": round(error_rate, 2),
                    "avg_response_ms": round(avg_ms, 2),
                },
                "provider_usage": dict(self._provider_usage),
                "template_runs": dict(self._template_runs),
                "last_template_run_at": self._last_template_run_time,
            }

    def reset(self) -> None:
        with self._lock:
            self.total_requests = 0
            self.total_errors = 0
            self._response_times.clear()
            self._provider_usage.clear()
            self._template_runs.clear()
            self._last_template_run_time = None


metrics = _MetricsCollector()

# ============================================================
# Webhook / Event System
# ============================================================

WEBHOOK_EVENTS = frozenset({
    "template_started",
    "template_completed",
    "template_failed",
    "step_completed",
    "step_failed",
})

WEBHOOK_MAX_RETRIES = 3
WEBHOOK_RETRY_BASE_SECONDS = 1.0  # exponential backoff base


class WebhookRegistry:
    """In-memory webhook registration and delivery."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._hooks: Dict[str, Dict[str, Any]] = {}  # id -> hook data

    def register(self, url: str, events: List[str], secret: Optional[str] = None) -> Dict[str, Any]:
        hook_id = str(uuid.uuid4())
        hook = {
            "id": hook_id,
            "url": url,
            "events": sorted(set(events)),
            "secret": secret,
            "active": True,
            "created_at": time.time(),
            "delivery_count": 0,
            "failure_count": 0,
        }
        with self._lock:
            self._hooks[hook_id] = hook
        return {k: v for k, v in hook.items() if k != "secret"}

    def list_hooks(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                {k: v for k, v in h.items() if k != "secret"}
                for h in self._hooks.values()
            ]

    def get(self, hook_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            h = self._hooks.get(hook_id)
            return dict(h) if h else None

    def delete(self, hook_id: str) -> bool:
        with self._lock:
            return self._hooks.pop(hook_id, None) is not None

    def hooks_for_event(self, event: str) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                dict(h)
                for h in self._hooks.values()
                if h["active"] and event in h["events"]
            ]

    def record_delivery(self, hook_id: str, success: bool) -> None:
        with self._lock:
            h = self._hooks.get(hook_id)
            if h:
                h["delivery_count"] += 1
                if not success:
                    h["failure_count"] += 1

    def reset(self) -> None:
        with self._lock:
            self._hooks.clear()


webhook_registry = WebhookRegistry()


def _sign_payload(payload_bytes: bytes, secret: str) -> str:
    return hmac.new(secret.encode(), payload_bytes, hashlib.sha256).hexdigest()


async def _deliver_webhook(hook: Dict[str, Any], payload: Dict[str, Any]) -> bool:
    """Deliver a webhook with retry + exponential backoff. Returns True on success."""
    import httpx

    payload_bytes = json.dumps(payload, default=str).encode()
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if hook.get("secret"):
        headers["X-Webhook-Signature"] = f"sha256={_sign_payload(payload_bytes, hook['secret'])}"

    for attempt in range(WEBHOOK_MAX_RETRIES):
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(hook["url"], content=payload_bytes, headers=headers)
                if resp.status_code < 400:
                    webhook_registry.record_delivery(hook["id"], success=True)
                    return True
        except Exception:
            pass
        if attempt < WEBHOOK_MAX_RETRIES - 1:
            await asyncio.sleep(WEBHOOK_RETRY_BASE_SECONDS * (2 ** attempt))

    webhook_registry.record_delivery(hook["id"], success=False)
    return False


async def emit_event(event: str, data: Dict[str, Any]) -> None:
    """Fire-and-forget delivery to all webhooks registered for *event*."""
    hooks = webhook_registry.hooks_for_event(event)
    if not hooks:
        return
    payload = {"event": event, "timestamp": time.time(), "data": data}
    for hook in hooks:
        asyncio.create_task(_deliver_webhook(hook, payload))

# ============================================================
# Async Task Queue
# ============================================================

TASK_STATUSES = ("pending", "running", "completed", "failed")


class TaskQueue:
    """In-memory async task tracker for background workflow execution."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tasks: Dict[str, Dict[str, Any]] = {}

    def create(self, template_id: str, flow_name: str) -> str:
        task_id = str(uuid.uuid4())
        with self._lock:
            self._tasks[task_id] = {
                "task_id": task_id,
                "template_id": template_id,
                "flow_name": flow_name,
                "status": "pending",
                "progress_pct": 0,
                "run_id": None,
                "result": None,
                "error": None,
                "created_at": time.time(),
                "started_at": None,
                "completed_at": None,
            }
        return task_id

    def get(self, task_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            t = self._tasks.get(task_id)
            return dict(t) if t else None

    def list_tasks(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock:
            tasks = list(self._tasks.values())
        if status:
            tasks = [t for t in tasks if t["status"] == status]
        tasks.sort(key=lambda t: t["created_at"], reverse=True)
        return [dict(t) for t in tasks]

    def update(self, task_id: str, **fields: Any) -> None:
        with self._lock:
            t = self._tasks.get(task_id)
            if t:
                t.update(fields)

    def reset(self) -> None:
        with self._lock:
            self._tasks.clear()


task_queue = TaskQueue()


# ============================================================
# Admin API Key Registry (master-key-protected)
# ============================================================

SYNAPPS_MASTER_KEY = os.environ.get("SYNAPPS_MASTER_KEY", "")

ADMIN_KEY_SCOPES = frozenset({"read", "write", "admin"})


class AdminKeyRegistry:
    """In-memory admin API key store, protected by master key."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._keys: Dict[str, Dict[str, Any]] = {}  # id -> key data

    def create(self, name: str, scopes: Optional[List[str]] = None) -> Dict[str, Any]:
        key_id = str(uuid.uuid4())
        plain_key = f"sk-{uuid.uuid4().hex}"
        key_prefix = plain_key[:12]
        entry = {
            "id": key_id,
            "name": name,
            "key_prefix": key_prefix,
            "scopes": sorted(set(scopes or ["read", "write"])),
            "is_active": True,
            "created_at": time.time(),
            "last_used_at": None,
        }
        with self._lock:
            self._keys[key_id] = {**entry, "_plain_key": plain_key}
        return {**entry, "api_key": plain_key}

    def get(self, key_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            k = self._keys.get(key_id)
            if not k:
                return None
            return {kk: vv for kk, vv in k.items() if kk != "_plain_key"}

    def list_keys(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                {kk: vv for kk, vv in k.items() if kk != "_plain_key"}
                for k in self._keys.values()
            ]

    def revoke(self, key_id: str) -> bool:
        with self._lock:
            k = self._keys.get(key_id)
            if not k:
                return False
            k["is_active"] = False
            return True

    def delete(self, key_id: str) -> bool:
        with self._lock:
            return self._keys.pop(key_id, None) is not None

    def validate_key(self, plain_key: str) -> Optional[Dict[str, Any]]:
        """Validate a plain API key and return its data if active."""
        with self._lock:
            for k in self._keys.values():
                if k.get("_plain_key") == plain_key and k.get("is_active"):
                    k["last_used_at"] = time.time()
                    return {kk: vv for kk, vv in k.items() if kk != "_plain_key"}
        return None

    def reset(self) -> None:
        with self._lock:
            self._keys.clear()


admin_key_registry = AdminKeyRegistry()


def require_master_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
) -> str:
    """Dependency that requires the SYNAPPS_MASTER_KEY for admin operations."""
    master = SYNAPPS_MASTER_KEY
    if not master:
        raise HTTPException(
            status_code=503,
            detail="Admin API not configured — SYNAPPS_MASTER_KEY environment variable not set",
        )

    provided = None
    if x_api_key:
        provided = x_api_key.strip()
    elif authorization:
        auth_text = authorization.strip()
        if auth_text.lower().startswith("bearer "):
            provided = auth_text[7:].strip()

    if not provided or not hmac.compare_digest(provided, master):
        raise HTTPException(status_code=403, detail="Invalid or missing master key")

    return provided


DEFAULT_MEMORY_BACKEND = os.environ.get("MEMORY_BACKEND", "sqlite_fts").strip().lower()
DEFAULT_MEMORY_NAMESPACE = os.environ.get("MEMORY_NAMESPACE", "default").strip() or "default"
DEFAULT_MEMORY_SQLITE_PATH = str(
    Path(os.environ.get("MEMORY_SQLITE_PATH", project_root / "synapps_memory.db")).expanduser()
)
DEFAULT_MEMORY_CHROMA_PATH = str(
    Path(os.environ.get("MEMORY_CHROMA_PATH", project_root / ".chroma")).expanduser()
)
DEFAULT_MEMORY_COLLECTION = os.environ.get("MEMORY_COLLECTION", "synapps_memory").strip() or "synapps_memory"
LEGACY_WRITER_LLM_PRESET: Dict[str, Any] = {
    "label": "Writer",
    "provider": "openai",
    "model": "gpt-4o",
    "temperature": 0.7,
    "max_tokens": 1000,
}
LEGACY_ARTIST_IMAGE_PRESET: Dict[str, Any] = {
    "label": "Image Gen",
    "provider": "stability",
    "model": "stable-diffusion-xl-1024-v1-0",
    "size": "1024x1024",
    "style": "photorealistic",
    "quality": "standard",
    "n": 1,
    "response_format": "b64_json",
}
LEGACY_MEMORY_BACKEND_ALIASES: Dict[str, str] = {
    "sqlite": "sqlite_fts",
    "sqlite_fts": "sqlite_fts",
    "sqlite-fts": "sqlite_fts",
    "sqlitefts": "sqlite_fts",
    "fts": "sqlite_fts",
    "chroma": "chroma",
    "chromadb": "chroma",
    "chroma_db": "chroma",
}
_TRUE_BRANCH_HINTS = {
    "true",
    "then",
    "yes",
    "pass",
    "match",
    "matched",
    "on_true",
    "if_true",
}
_FALSE_BRANCH_HINTS = {
    "false",
    "else",
    "no",
    "fail",
    "nomatch",
    "not_match",
    "on_false",
    "if_false",
}

# ============================================================
# Application Setup
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for database initialization and cleanup."""
    logger.info("Initializing database...")
    await init_db()
    logger.info("Database initialization complete")
    yield
    logger.info("Closing database connections...")
    await close_db_connections()
    logger.info("Database connections closed")

app = FastAPI(
    title="SynApps Orchestrator",
    description=(
        "Visual AI workflow builder API. Connect specialized AI agent nodes, "
        "execute workflows in real-time, and manage LLM provider integrations. "
        "Authenticate via JWT bearer tokens or API keys."
    ),
    version=API_VERSION,
    lifespan=lifespan,
    openapi_url="/api/v1/openapi.json",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
    openapi_tags=[
        {"name": "Auth", "description": "Registration, login, token refresh, API key management."},
        {"name": "Flows", "description": "Create, list, update, delete, import, and export workflows."},
        {"name": "Runs", "description": "Execute workflows and inspect run results, traces, and diffs."},
        {"name": "Providers", "description": "LLM and image generation provider registries."},
        {"name": "Applets", "description": "Node type catalog (registered applet metadata)."},
        {"name": "Dashboard", "description": "Portfolio health, template status, provider overview."},
        {"name": "Health", "description": "Service health checks."},
    ],
)

# ============================================================
# CORS Configuration (environment-aware)
# ============================================================
_is_production = os.environ.get("PRODUCTION", "false").strip().lower() in {"1", "true", "yes"}


def _is_secure_request(request: Request) -> bool:
    """Return True when the incoming request is HTTPS (direct or proxy-terminated)."""
    if request.url.scheme.lower() == "https":
        return True

    # Common reverse-proxy hint (e.g. nginx, ALB, ingress).
    forwarded_proto = request.headers.get("x-forwarded-proto", "")
    if forwarded_proto:
        proto = forwarded_proto.split(",", 1)[0].strip().lower()
        if proto == "https":
            return True

    # RFC 7239 Forwarded header support.
    forwarded = request.headers.get("forwarded", "")
    if forwarded:
        match = re.search(r"(?:^|[;,\s])proto=(https)(?:[;,\s]|$)", forwarded, flags=re.IGNORECASE)
        if match:
            return True

    return False


if _is_production:
    @app.middleware("http")
    async def enforce_https_in_production(request: Request, call_next):
        """Fail closed in production when traffic is not served over HTTPS."""
        if not _is_secure_request(request):
            return _error_response(
                426,
                "HTTPS_REQUIRED",
                "HTTPS is required in production.",
            )

        response = await call_next(request)
        response.headers.setdefault(
            "Strict-Transport-Security",
            "max-age=31536000; includeSubDomains; preload",
        )
        return response

_cors_raw = os.environ.get("BACKEND_CORS_ORIGINS", "")
_cors_origins = [o.strip() for o in _cors_raw.split(",") if o.strip()]

if _is_production and not _cors_origins:
    logger.error(
        "BACKEND_CORS_ORIGINS is required in production. "
        "Set it to a comma-separated list of allowed origins."
    )
    # Fail-closed: no origins allowed if unset in production
    _cors_origins = []
elif not _cors_origins:
    logger.warning("No CORS origins specified, allowing localhost origins in development mode")
    _cors_origins = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]

# In production, never allow wildcard origins or credentials-with-wildcard
if _is_production and "*" in _cors_origins:
    logger.error("Wildcard CORS origin '*' is not allowed in production, removing it")
    _cors_origins = [o for o in _cors_origins if o != "*"]

_cors_methods = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]
_cors_headers = [
    "Authorization",
    "Content-Type",
    "X-API-Key",
    "X-Requested-With",
    "Accept",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=_cors_methods,
    allow_headers=_cors_headers,
    expose_headers=[
        "X-RateLimit-Limit",
        "X-RateLimit-Remaining",
        "X-RateLimit-Reset",
        "Retry-After",
    ],
    max_age=600 if _is_production else 0,
)

# ============================================================
# Rate Limiting
# ============================================================
from apps.orchestrator.middleware.rate_limiter import add_rate_limiter  # noqa: E402

add_rate_limiter(app)


async def _resolve_rate_limit_user(request: Request) -> Optional[Dict[str, Any]]:
    """Best-effort auth parsing for per-user rate-limit keys."""
    x_api_key = request.headers.get("X-API-Key")
    authorization = request.headers.get("Authorization")

    try:
        if x_api_key and x_api_key.strip():
            principal = await _authenticate_user_by_api_key(x_api_key.strip())
            principal.setdefault("tier", "free")
            return principal

        if authorization:
            auth_text = authorization.strip()
            if auth_text.lower().startswith("bearer "):
                principal = await _authenticate_user_by_jwt(auth_text[7:].strip())
                principal.setdefault("tier", "free")
                return principal
            if auth_text.lower().startswith("apikey "):
                principal = await _authenticate_user_by_api_key(auth_text[7:].strip())
                principal.setdefault("tier", "free")
                return principal
    except HTTPException:
        # Invalid credentials are handled by endpoint auth dependencies.
        return None
    except Exception:
        return None

    return None


def _anonymous_rate_limit_principal(request: Request) -> Dict[str, Any]:
    """Build a stable anonymous principal from the direct socket client."""
    client_host = request.client.host if request.client else "unknown"
    return {
        "id": f"anonymous:{client_host}",
        "tier": "anonymous",
    }


@app.middleware("http")
async def attach_rate_limit_identity(request: Request, call_next):
    """Attach authenticated principal to request state for per-user rate limiting."""
    principal = await _resolve_rate_limit_user(request)
    if principal is None:
        principal = _anonymous_rate_limit_principal(request)
    request.state.user = principal
    return await call_next(request)


@app.middleware("http")
async def collect_metrics(request: Request, call_next):
    """Record request count, duration, and status for /metrics."""
    start = time.monotonic()
    response = await call_next(request)
    duration_ms = (time.monotonic() - start) * 1000
    metrics.record_request(duration_ms, response.status_code, request.url.path)
    return response


# ============================================================
# Error Handling - Consistent Error Format
# ============================================================

_HTTP_ERROR_CODES = {
    400: "BAD_REQUEST",
    401: "UNAUTHORIZED",
    403: "FORBIDDEN",
    404: "NOT_FOUND",
    409: "CONFLICT",
    422: "VALIDATION_ERROR",
    429: "RATE_LIMIT_EXCEEDED",
    500: "INTERNAL_SERVER_ERROR",
    501: "NOT_IMPLEMENTED",
    503: "SERVICE_UNAVAILABLE",
}


def _error_response(
    status: int,
    code: str,
    message: str,
    details: Optional[List[Dict[str, Any]]] = None,
) -> JSONResponse:
    """Create a standardized error JSONResponse."""
    body: Dict[str, Any] = {
        "error": {
            "code": code,
            "status": status,
            "message": message,
        }
    }
    if details is not None:
        body["error"]["details"] = details
    return JSONResponse(status_code=status, content=body)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    code = _HTTP_ERROR_CODES.get(exc.status_code, "ERROR")
    return _error_response(exc.status_code, code, str(exc.detail))


@app.exception_handler(StarletteHTTPException)
async def starlette_http_exception_handler(request: Request, exc: StarletteHTTPException):
    code = _HTTP_ERROR_CODES.get(exc.status_code, "ERROR")
    detail = exc.detail if isinstance(exc.detail, str) else "Request error"
    return _error_response(exc.status_code, code, detail)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    details = []
    for err in exc.errors():
        field = ".".join(str(loc) for loc in err.get("loc", []))
        details.append({
            "field": field,
            "message": err.get("msg", ""),
            "type": err.get("type", ""),
        })
    return _error_response(422, "VALIDATION_ERROR", "Request validation failed", details)


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return _error_response(500, "INTERNAL_SERVER_ERROR", "An unexpected error occurred")


# ============================================================
# Pagination
# ============================================================

def paginate(items: list, page: int, page_size: int) -> dict:
    """Apply offset-based pagination to a list of items."""
    total = len(items)
    total_pages = math.ceil(total / page_size) if page_size > 0 else 0
    start = (page - 1) * page_size
    end = start + page_size
    return {
        "items": items[start:end],
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
    }


# ============================================================
# Request Validation Models (Pydantic v2 strict)
# ============================================================

class StrictRequestModel(BaseModel):
    """Base model for API request payloads that rejects unknown fields."""

    model_config = ConfigDict(extra="forbid")


class FlowNodeRequest(StrictRequestModel):
    """Strictly validated flow node for API requests."""
    id: str = Field(..., min_length=1, max_length=200)
    type: str = Field(..., min_length=1, max_length=100)
    position: Dict[str, float]
    data: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("position")
    @classmethod
    def validate_position(cls, v):
        if "x" not in v or "y" not in v:
            raise ValueError("Position must contain 'x' and 'y' keys")
        return v


class FlowEdgeRequest(StrictRequestModel):
    """Strictly validated flow edge for API requests."""
    id: str = Field(..., min_length=1, max_length=200)
    source: str = Field(..., min_length=1)
    target: str = Field(..., min_length=1)
    animated: bool = False


class CreateFlowRequest(StrictRequestModel):
    """Strictly validated flow creation/update request."""
    id: Optional[str] = Field(None, max_length=200)
    name: str = Field(..., min_length=1, max_length=200)
    nodes: List[FlowNodeRequest] = Field(default_factory=list)
    edges: List[FlowEdgeRequest] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def name_not_blank(cls, v):
        if not v.strip():
            raise ValueError("Flow name cannot be blank")
        return v.strip()

    @field_validator("id")
    @classmethod
    def id_not_blank(cls, v):
        if v is not None and v.strip() == "":
            return None  # Treat blank as None so a UUID is auto-generated
        return v


class RunFlowRequest(StrictRequestModel):
    """Strictly validated request body for running a flow."""
    input: Dict[str, Any] = Field(default_factory=dict, description="Input data for the workflow")


class RerunFlowRequest(StrictRequestModel):
    """Request body for re-running a previous flow execution with input overrides."""
    input: Dict[str, Any] = Field(
        default_factory=dict,
        description="Input overrides for the re-run",
    )
    merge_with_original_input: bool = Field(
        default=True,
        description="When true, merge overrides on top of the source run input",
    )


class AISuggestRequest(StrictRequestModel):
    """Strictly validated request body for AI suggestions."""
    prompt: str = Field(..., min_length=1, max_length=5000, description="The prompt for AI suggestion")
    context: Optional[str] = Field(None, max_length=10000, description="Optional context for the suggestion")


class AuthRegisterRequestStrict(AuthRegisterRequestModel):
    """Strict request model that rejects unknown registration fields."""

    model_config = ConfigDict(extra="forbid")


class AuthLoginRequestStrict(AuthLoginRequestModel):
    """Strict request model that rejects unknown login fields."""

    model_config = ConfigDict(extra="forbid")


class AuthRefreshRequestStrict(AuthRefreshRequestModel):
    """Strict request model that rejects unknown refresh-token fields."""

    model_config = ConfigDict(extra="forbid")


class APIKeyCreateRequestStrict(APIKeyCreateRequestModel):
    """Strict request model that rejects unknown API-key creation fields."""

    model_config = ConfigDict(extra="forbid")


def _trace_value(value: Any, depth: int = 0) -> Any:
    """Convert arbitrary values into a JSON-serializable structure."""
    if depth >= 8:
        return str(value)

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {str(k): _trace_value(v, depth + 1) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_trace_value(v, depth + 1) for v in list(value)]

    if isinstance(value, BaseModel):
        return _trace_value(value.model_dump(), depth + 1)

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _trace_value(model_dump(), depth + 1)
        except Exception:
            return str(value)

    return str(value)


def _new_execution_trace(
    run_id: str,
    flow_id: Optional[str],
    input_data: Dict[str, Any],
    start_time: float,
) -> Dict[str, Any]:
    """Create a baseline execution trace document for a run."""
    return {
        "version": TRACE_SCHEMA_VERSION,
        "run_id": run_id,
        "flow_id": flow_id,
        "status": "running",
        "input": _trace_value(input_data),
        "started_at": float(start_time),
        "ended_at": None,
        "duration_ms": None,
        "nodes": [],
        "errors": [],
    }


def _finalize_execution_trace(trace: Dict[str, Any], status: str, end_time: float) -> None:
    """Finalize aggregate timing/status fields in a trace object."""
    trace["status"] = status
    trace["ended_at"] = float(end_time)
    started_at = trace.get("started_at")
    if isinstance(started_at, (int, float)):
        trace["duration_ms"] = max(0.0, (float(end_time) - float(started_at)) * 1000.0)


def _extract_trace_from_run(run: Dict[str, Any]) -> Dict[str, Any]:
    """Return a normalized execution trace for any run, including legacy runs."""
    run_id = str(run.get("run_id", ""))
    flow_id = run.get("flow_id")
    start_time = run.get("start_time")
    if not isinstance(start_time, (int, float)):
        start_time = time.time()

    input_data = run.get("input_data")
    if not isinstance(input_data, dict):
        input_data = {}

    results = run.get("results")
    if not isinstance(results, dict):
        results = {}

    stored_trace = results.get(TRACE_RESULTS_KEY)
    if isinstance(stored_trace, dict):
        trace = _trace_value(stored_trace)
        if not isinstance(trace, dict):
            trace = _new_execution_trace(run_id, flow_id, input_data, float(start_time))
    else:
        trace = _new_execution_trace(run_id, flow_id, input_data, float(start_time))
        nodes: List[Dict[str, Any]] = []
        for node_id, node_result in results.items():
            if node_id == TRACE_RESULTS_KEY:
                continue
            if isinstance(node_result, dict):
                error_payload = node_result.get("error")
                node_errors = []
                if error_payload is not None:
                    node_errors.append(_trace_value(error_payload))
                nodes.append(
                    {
                        "node_id": str(node_id),
                        "node_type": node_result.get("type"),
                        "status": node_result.get("status", "success"),
                        "input": _trace_value(node_result.get("input")),
                        "output": _trace_value(node_result.get("output")),
                        "attempts": node_result.get("attempts", 1),
                        "errors": node_errors,
                        "started_at": node_result.get("started_at"),
                        "ended_at": node_result.get("ended_at"),
                        "duration_ms": node_result.get("duration_ms"),
                    }
                )
            else:
                nodes.append(
                    {
                        "node_id": str(node_id),
                        "node_type": None,
                        "status": "success",
                        "input": None,
                        "output": _trace_value(node_result),
                        "attempts": 1,
                        "errors": [],
                    }
                )
        trace["nodes"] = nodes

    trace["run_id"] = run_id
    trace["flow_id"] = flow_id
    trace["status"] = str(run.get("status", trace.get("status", "unknown")))

    trace_start = trace.get("started_at")
    if not isinstance(trace_start, (int, float)):
        trace_start = float(start_time)
        trace["started_at"] = trace_start

    end_time = run.get("end_time")
    if isinstance(end_time, (int, float)):
        trace["ended_at"] = float(end_time)
        trace["duration_ms"] = max(0.0, (float(end_time) - float(trace_start)) * 1000.0)
    elif trace.get("ended_at") is None and trace.get("status") in {"success", "error"}:
        trace["ended_at"] = float(trace_start)
        trace["duration_ms"] = 0.0

    trace_input = trace.get("input")
    if not isinstance(trace_input, dict):
        trace["input"] = _trace_value(input_data)

    if not isinstance(trace.get("nodes"), list):
        trace["nodes"] = []
    if not isinstance(trace.get("errors"), list):
        trace["errors"] = []

    return trace


def _flatten_for_diff(value: Any, path: str, out: Dict[str, Any]) -> None:
    """Flatten nested structures into a path/value map for deterministic diffing."""
    if isinstance(value, dict):
        if not value:
            out[path] = {}
            return
        for key in sorted(value.keys(), key=lambda k: str(k)):
            child_path = f"{path}.{key}"
            _flatten_for_diff(value[key], child_path, out)
        return

    if isinstance(value, list):
        if not value:
            out[path] = []
            return
        for index, item in enumerate(value):
            _flatten_for_diff(item, f"{path}[{index}]", out)
        return

    out[path] = value


def _build_json_diff(left: Any, right: Any, max_changes: int = MAX_DIFF_CHANGES) -> Dict[str, Any]:
    """Build a bounded structural diff between two JSON-like values."""
    left_normalized = _trace_value(left)
    right_normalized = _trace_value(right)

    left_flat: Dict[str, Any] = {}
    right_flat: Dict[str, Any] = {}
    _flatten_for_diff(left_normalized, "$", left_flat)
    _flatten_for_diff(right_normalized, "$", right_flat)

    all_paths = sorted(set(left_flat.keys()) | set(right_flat.keys()))
    changes: List[Dict[str, Any]] = []
    total_changes = 0

    for path in all_paths:
        in_left = path in left_flat
        in_right = path in right_flat
        if in_left and in_right and left_flat[path] == right_flat[path]:
            continue

        total_changes += 1
        if len(changes) >= max_changes:
            continue

        if in_left and not in_right:
            change_type = "removed"
        elif in_right and not in_left:
            change_type = "added"
        else:
            change_type = "modified"

        changes.append(
            {
                "path": path,
                "type": change_type,
                "before": left_flat.get(path),
                "after": right_flat.get(path),
            }
        )

    return {
        "changed": total_changes > 0,
        "change_count": total_changes,
        "truncated": total_changes > max_changes,
        "changes": changes,
    }


def _node_result_index(run: Dict[str, Any]) -> Dict[str, Any]:
    """Return run result payload keyed by node ID, excluding trace metadata."""
    results = run.get("results")
    if not isinstance(results, dict):
        return {}
    return {
        str(node_id): _trace_value(result)
        for node_id, result in results.items()
        if node_id != TRACE_RESULTS_KEY
    }


def _build_run_diff(base_run: Dict[str, Any], compare_run: Dict[str, Any]) -> Dict[str, Any]:
    """Compute an execution diff between two runs."""
    base_trace = _extract_trace_from_run(base_run)
    compare_trace = _extract_trace_from_run(compare_run)

    base_nodes = {
        str(item.get("node_id")): item
        for item in base_trace.get("nodes", [])
        if isinstance(item, dict) and item.get("node_id") is not None
    }
    compare_nodes = {
        str(item.get("node_id")): item
        for item in compare_trace.get("nodes", [])
        if isinstance(item, dict) and item.get("node_id") is not None
    }

    node_diffs: List[Dict[str, Any]] = []
    for node_id in sorted(set(base_nodes.keys()) | set(compare_nodes.keys())):
        left = base_nodes.get(node_id)
        right = compare_nodes.get(node_id)
        if left is None:
            node_diffs.append({"node_id": node_id, "type": "added", "after": _trace_value(right)})
            continue
        if right is None:
            node_diffs.append({"node_id": node_id, "type": "removed", "before": _trace_value(left)})
            continue

        left_duration = left.get("duration_ms")
        right_duration = right.get("duration_ms")
        duration_delta = None
        if isinstance(left_duration, (int, float)) and isinstance(right_duration, (int, float)):
            duration_delta = float(right_duration) - float(left_duration)

        status_before = left.get("status")
        status_after = right.get("status")
        attempts_before = left.get("attempts")
        attempts_after = right.get("attempts")

        node_changed = left != right
        if not node_changed:
            continue

        node_diffs.append(
            {
                "node_id": node_id,
                "type": "modified",
                "status": {
                    "before": status_before,
                    "after": status_after,
                    "changed": status_before != status_after,
                },
                "attempts": {
                    "before": attempts_before,
                    "after": attempts_after,
                    "changed": attempts_before != attempts_after,
                },
                "duration_ms": {
                    "before": left_duration,
                    "after": right_duration,
                    "delta_ms": duration_delta,
                },
                "input_changed": left.get("input") != right.get("input"),
                "output_changed": left.get("output") != right.get("output"),
                "errors_changed": left.get("errors") != right.get("errors"),
            }
        )

    base_duration = base_trace.get("duration_ms")
    compare_duration = compare_trace.get("duration_ms")
    duration_delta_ms = None
    if isinstance(base_duration, (int, float)) and isinstance(compare_duration, (int, float)):
        duration_delta_ms = float(compare_duration) - float(base_duration)

    return {
        "base_run_id": base_run.get("run_id"),
        "compare_run_id": compare_run.get("run_id"),
        "flow_id": base_run.get("flow_id") or compare_run.get("flow_id"),
        "summary": {
            "base_status": base_run.get("status"),
            "compare_status": compare_run.get("status"),
            "status_changed": base_run.get("status") != compare_run.get("status"),
            "base_node_count": len(base_nodes),
            "compare_node_count": len(compare_nodes),
            "changed_node_count": len(node_diffs),
        },
        "timing": {
            "base_duration_ms": base_duration,
            "compare_duration_ms": compare_duration,
            "duration_delta_ms": duration_delta_ms,
        },
        "input_diff": _build_json_diff(base_trace.get("input", {}), compare_trace.get("input", {})),
        "output_diff": _build_json_diff(_node_result_index(base_run), _node_result_index(compare_run)),
        "trace_diff": _build_json_diff(base_trace, compare_trace),
        "node_diffs": node_diffs,
        "base_trace": base_trace,
        "compare_trace": compare_trace,
    }


# ============================================================
# Authentication Utilities
# ============================================================

def _utc_now() -> float:
    return time.time()


def _hash_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _derive_fernet_key() -> bytes:
    configured = os.environ.get("FERNET_KEY", "").strip()
    if configured:
        try:
            return configured.encode("utf-8")
        except Exception:
            logger.warning("Invalid FERNET_KEY value; falling back to derived key")
    digest = hashlib.sha256(JWT_SECRET_KEY.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)


FERNET_CIPHER = Fernet(_derive_fernet_key())


def _encrypt_api_key(plain_value: str) -> str:
    return FERNET_CIPHER.encrypt(plain_value.encode("utf-8")).decode("utf-8")


def _decrypt_api_key(encrypted_value: str) -> Optional[str]:
    try:
        return FERNET_CIPHER.decrypt(encrypted_value.encode("utf-8")).decode("utf-8")
    except (InvalidToken, ValueError, TypeError):
        return None


def _hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PASSWORD_HASH_ITERATIONS,
    )
    salt_text = base64.urlsafe_b64encode(salt).decode("utf-8")
    hash_text = base64.urlsafe_b64encode(digest).decode("utf-8")
    return f"pbkdf2_sha256${PASSWORD_HASH_ITERATIONS}${salt_text}${hash_text}"


def _verify_password(password: str, password_hash: str) -> bool:
    try:
        scheme, raw_iterations, salt_text, hash_text = password_hash.split("$", 3)
        if scheme != "pbkdf2_sha256":
            return False
        iterations = int(raw_iterations)
        salt = base64.urlsafe_b64decode(salt_text.encode("utf-8"))
        expected = base64.urlsafe_b64decode(hash_text.encode("utf-8"))
    except Exception:
        return False

    candidate = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        iterations,
    )
    return hmac.compare_digest(candidate, expected)


def _create_access_token(user: AuthUser) -> tuple[str, int]:
    now = int(_utc_now())
    expiry = now + ACCESS_TOKEN_EXPIRE_MINUTES * 60
    payload = {
        "sub": user.id,
        "email": user.email,
        "token_type": "access",
        "iat": now,
        "exp": expiry,
    }
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return token, expiry - now


def _create_refresh_token(user: AuthUser) -> tuple[str, float, int]:
    now = int(_utc_now())
    expiry = now + REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60
    payload = {
        "sub": user.id,
        "email": user.email,
        "token_type": "refresh",
        "jti": str(uuid.uuid4()),
        "iat": now,
        "exp": expiry,
    }
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return token, float(expiry), expiry - now


def _decode_token(token: str, expected_type: str) -> Dict[str, Any]:
    try:
        payload = jwt.decode(
            token,
            JWT_SECRET_KEY,
            algorithms=[JWT_ALGORITHM],
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

    token_type = payload.get("token_type")
    if token_type != expected_type:
        raise HTTPException(status_code=401, detail="Invalid token type")
    return payload


def _issue_api_tokens(user: AuthUser) -> tuple[AuthTokenResponseModel, str, float]:
    access_token, access_expires_in = _create_access_token(user)
    refresh_token, refresh_expires_at, refresh_expires_in = _create_refresh_token(user)
    response_payload = AuthTokenResponseModel(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        access_expires_in=access_expires_in,
        refresh_expires_in=refresh_expires_in,
    )
    return response_payload, refresh_token, refresh_expires_at


def _normalize_key_header_value(raw_value: str) -> str:
    return raw_value.strip()


def _api_key_lookup_prefix(api_key_value: str) -> str:
    return api_key_value[:API_KEY_LOOKUP_PREFIX_LEN]


def _user_to_principal(user: AuthUser) -> Dict[str, Any]:
    return {
        "id": user.id,
        "email": user.email,
        "is_active": user.is_active,
        "created_at": user.created_at,
    }


async def _store_refresh_token(
    user_id: str,
    refresh_token: str,
    expires_at: float,
) -> None:
    token_hash = _hash_sha256(refresh_token)
    async with get_db_session() as session:
        session.add(
            AuthRefreshToken(
                id=str(uuid.uuid4()),
                user_id=user_id,
                token_hash=token_hash,
                expires_at=expires_at,
                revoked=False,
                created_at=_utc_now(),
                last_used_at=None,
            )
        )


async def _authenticate_user_by_jwt(access_token: str) -> Dict[str, Any]:
    payload = _decode_token(access_token, expected_type="access")
    user_id = payload.get("sub")
    if not isinstance(user_id, str) or not user_id:
        raise HTTPException(status_code=401, detail="Invalid access token subject")

    async with get_db_session() as session:
        result = await session.execute(
            select(AuthUser).where(AuthUser.id == user_id)
        )
        user = result.scalars().first()
        if not user or not user.is_active:
            raise HTTPException(status_code=401, detail="User is not active")
        return _user_to_principal(user)


async def _authenticate_user_by_api_key(api_key_value: str) -> Dict[str, Any]:
    normalized_key = _normalize_key_header_value(api_key_value)
    if not normalized_key:
        raise HTTPException(status_code=401, detail="Invalid API key")

    lookup_prefix = _api_key_lookup_prefix(normalized_key)

    async with get_db_session() as session:
        query = select(AuthUserAPIKey).where(
            AuthUserAPIKey.is_active == True,  # noqa: E712 - SQLAlchemy boolean comparison
            AuthUserAPIKey.key_prefix == lookup_prefix,
        )
        result = await session.execute(query)
        candidates = result.scalars().all()

        for credential in candidates:
            plain_key = _decrypt_api_key(credential.encrypted_key)
            if plain_key is None:
                continue
            if not hmac.compare_digest(plain_key, normalized_key):
                continue

            user_result = await session.execute(
                select(AuthUser).where(AuthUser.id == credential.user_id)
            )
            user = user_result.scalars().first()
            if not user or not user.is_active:
                break

            credential.last_used_at = _utc_now()
            return _user_to_principal(user)

    raise HTTPException(status_code=401, detail="Invalid API key")


async def _can_use_anonymous_bootstrap() -> bool:
    if not ALLOW_ANONYMOUS_WHEN_NO_USERS:
        return False
    try:
        async with get_db_session() as session:
            result = await session.execute(select(AuthUser.id).limit(1))
            first_user_id = result.scalar_one_or_none()
            return first_user_id is None
    except Exception:
        # Allow bootstrap traffic before auth tables are initialized.
        return True


async def get_authenticated_user(
    authorization: Optional[str] = Header(None, alias="Authorization"),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> Dict[str, Any]:
    if x_api_key:
        return await _authenticate_user_by_api_key(x_api_key)

    if authorization:
        auth_text = authorization.strip()
        if auth_text.lower().startswith("bearer "):
            return await _authenticate_user_by_jwt(auth_text[7:].strip())
        if auth_text.lower().startswith("apikey "):
            return await _authenticate_user_by_api_key(auth_text[7:].strip())

    if await _can_use_anonymous_bootstrap():
        return {
            "id": "anonymous",
            "email": "anonymous@local",
            "is_active": True,
            "created_at": _utc_now(),
        }

    raise HTTPException(status_code=401, detail="Authentication required")


# ============================================================
# Internal Models
# ============================================================

class AppletStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"

class NodeErrorCode(str, Enum):
    TIMEOUT = "TIMEOUT"
    RETRY_EXHAUSTED = "RETRY_EXHAUSTED"
    EXECUTION_ERROR = "EXECUTION_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    NOT_FOUND = "NOT_FOUND"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"

class NodeError(Exception):
    """Structured error for node execution."""
    def __init__(
        self,
        code: NodeErrorCode,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        node_id: Optional[str] = None,
    ):
        self.code = code
        self.message = message
        self.details = details or {}
        self.node_id = node_id
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
            "node_id": self.node_id,
        }

class FlowNode(BaseModel):
    id: str
    type: str
    position: Dict[str, int]
    data: Dict[str, Any] = Field(default_factory=dict)

class FlowEdge(BaseModel):
    id: str
    source: str
    target: str
    animated: bool = False

class Flow(BaseModel):
    id: str
    name: str
    nodes: List[FlowNode]
    edges: List[FlowEdge]

class AppletMessage(BaseModel):
    content: Any
    context: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class WorkflowRunStatus(BaseModel):
    run_id: str
    flow_id: str
    status: str
    current_applet: Optional[str] = None
    progress: int = 0
    total_steps: int = 0
    start_time: float = 0
    end_time: Optional[float] = None
    results: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    error_details: Dict[str, Any] = Field(default_factory=dict)
    completed_applets: List[str] = Field(default_factory=list)


# ============================================================
# WebSocket Protocol – structured messages, auth, reconnection
# ============================================================

WS_AUTH_TIMEOUT_SECONDS = int(os.environ.get("WS_AUTH_TIMEOUT_SECONDS", "10"))
WS_HEARTBEAT_INTERVAL = int(os.environ.get("WS_HEARTBEAT_INTERVAL", "30"))
WS_MESSAGE_BUFFER_SIZE = int(os.environ.get("WS_MESSAGE_BUFFER_SIZE", "200"))
WS_SESSION_TTL_SECONDS = int(os.environ.get("WS_SESSION_TTL_SECONDS", "300"))

applet_registry: Dict[str, Type['BaseApplet']] = {}


def _ws_message(
    msg_type: str,
    data: Optional[dict] = None,
    *,
    ref_id: Optional[str] = None,
) -> dict:
    """Create a structured WebSocket message.

    Fields:
        id        – unique message identifier (UUIDv4)
        type      – dot-namespaced message type
        data      – payload dict
        timestamp – seconds since epoch (float)
        ref_id    – optional correlation ID linking a response to a request
    """
    msg: dict = {
        "id": str(uuid.uuid4()),
        "type": msg_type,
        "data": data or {},
        "timestamp": time.time(),
    }
    if ref_id:
        msg["ref_id"] = ref_id
    return msg


class _WSSession:
    """State for a single WebSocket session."""

    __slots__ = (
        "session_id",
        "user_id",
        "websocket",
        "subscriptions",
        "connected_at",
        "last_active",
        "state",
        "_message_seq",
    )

    def __init__(
        self,
        session_id: str,
        user_id: str,
        websocket: Optional[WebSocket] = None,
    ) -> None:
        self.session_id = session_id
        self.user_id = user_id
        self.websocket = websocket
        self.subscriptions: set = set()
        self.connected_at = time.time()
        self.last_active = time.time()
        self.state = "connected"
        self._message_seq = 0

    def next_seq(self) -> int:
        self._message_seq += 1
        return self._message_seq


class WebSocketSessionManager:
    """Manages connected WebSocket clients, sessions, and message replay."""

    def __init__(self, buffer_size: int = WS_MESSAGE_BUFFER_SIZE) -> None:
        self._sessions: Dict[str, _WSSession] = {}
        self._ws_to_session: Dict[int, str] = {}  # id(websocket) -> session_id
        self._message_buffer: List[dict] = []
        self._buffer_size = buffer_size
        self._lock = threading.Lock()
        self._global_seq = 0

    # ------ session management ------

    def create_session(
        self,
        user_id: str,
        websocket: WebSocket,
        session_id: Optional[str] = None,
    ) -> tuple:
        """Create or resume a session.  Returns (session, reconnected)."""
        reconnected = False
        with self._lock:
            if session_id and session_id in self._sessions:
                sess = self._sessions[session_id]
                if sess.user_id == user_id:
                    sess.websocket = websocket
                    sess.state = "connected"
                    sess.last_active = time.time()
                    reconnected = True
                else:
                    session_id = None

            if not reconnected:
                session_id = session_id or str(uuid.uuid4())
                sess = _WSSession(session_id, user_id, websocket)
                self._sessions[session_id] = sess

            self._ws_to_session[id(websocket)] = sess.session_id
        return sess, reconnected

    def remove_session(self, websocket: WebSocket) -> Optional[_WSSession]:
        """Mark session as disconnected and unlink the websocket."""
        with self._lock:
            ws_id = id(websocket)
            sid = self._ws_to_session.pop(ws_id, None)
            if sid and sid in self._sessions:
                sess = self._sessions[sid]
                sess.websocket = None
                sess.state = "disconnected"
                sess.last_active = time.time()
                return sess
        return None

    def get_session_by_ws(self, websocket: WebSocket) -> Optional[_WSSession]:
        with self._lock:
            sid = self._ws_to_session.get(id(websocket))
            return self._sessions.get(sid) if sid else None

    def connected_sessions(self) -> List[_WSSession]:
        """Return sessions that currently have a live websocket."""
        with self._lock:
            return [
                s for s in self._sessions.values()
                if s.websocket and s.state == "connected"
            ]

    @property
    def connected_websockets(self) -> List[WebSocket]:
        """List of live websockets (backward-compatible helper)."""
        return [s.websocket for s in self.connected_sessions() if s.websocket]

    def cleanup_expired(self) -> int:
        """Purge sessions disconnected longer than TTL."""
        cutoff = time.time() - WS_SESSION_TTL_SECONDS
        removed = 0
        with self._lock:
            expired = [
                sid
                for sid, s in self._sessions.items()
                if s.state == "disconnected" and s.last_active < cutoff
            ]
            for sid in expired:
                del self._sessions[sid]
                removed += 1
        return removed

    # ------ message buffering for replay ------

    def _buffer_message(self, message: dict) -> None:
        self._global_seq += 1
        message["_seq"] = self._global_seq
        self._message_buffer.append(message)
        if len(self._message_buffer) > self._buffer_size:
            self._message_buffer = self._message_buffer[-self._buffer_size:]

    def get_missed_messages(self, last_seq: int) -> List[dict]:
        """Return messages with _seq > last_seq for reconnection replay."""
        with self._lock:
            return [m for m in self._message_buffer if m.get("_seq", 0) > last_seq]

    @property
    def current_seq(self) -> int:
        with self._lock:
            return self._global_seq

    # ------ broadcast helpers ------

    async def broadcast(self, message: dict) -> None:
        """Send a message to all connected clients and buffer it for replay."""
        with self._lock:
            self._buffer_message(message)
            sessions = [
                s for s in self._sessions.values()
                if s.websocket and s.state == "connected"
            ]

        disconnected: List[WebSocket] = []
        for sess in sessions:
            try:
                if sess.websocket:
                    await sess.websocket.send_json(message)
            except Exception as e:
                logger.error(f"Failed to send to session {sess.session_id}: {e}")
                if sess.websocket:
                    disconnected.append(sess.websocket)

        for ws in disconnected:
            self.remove_session(ws)


# Module-level manager instance
ws_manager = WebSocketSessionManager()

# Backward-compatible accessor (returns a fresh snapshot each call)
connected_clients: List[WebSocket] = []  # legacy shim – use ws_manager directly


async def broadcast_status(status: Dict[str, Any]):
    """Broadcast workflow status to all connected clients using structured messages."""
    broadcast_data = status.copy()
    if "completed_applets" not in broadcast_data:
        broadcast_data["completed_applets"] = []

    message = _ws_message("workflow.status", broadcast_data)

    # Always route through session manager so messages are buffered for replay.
    await ws_manager.broadcast(message)

    # Legacy path: bare websockets appended to connected_clients directly (tests)
    if connected_clients:
        managed_ws_ids = {id(ws) for ws in ws_manager.connected_websockets}
        disconnected = []
        for client in connected_clients:
            # Avoid double-sending to sockets already tracked by ws_manager.
            if id(client) in managed_ws_ids:
                continue
            try:
                await client.send_json(message)
            except Exception as e:
                logger.error(f"Failed to send to legacy client: {e}")
                disconnected.append(client)
        for client in disconnected:
            if client in connected_clients:
                connected_clients.remove(client)


# ============================================================
# Base Applet
# ============================================================

class BaseApplet:
    """Base class that all applets must implement."""

    @classmethod
    def get_metadata(cls) -> Dict[str, Any]:
        """Return applet metadata."""
        return {
            "name": cls.__name__,
            "description": cls.__doc__ or "No description provided",
            "version": getattr(cls, "VERSION", "0.1.0"),
            "capabilities": getattr(cls, "CAPABILITIES", []),
        }

    async def on_message(self, message: AppletMessage) -> AppletMessage:
        """Process an incoming message and return a response."""
        raise NotImplementedError("Applets must implement on_message")


# ============================================================
# LLM Providers / Adapters
# ============================================================

def _safe_json_loads(payload: str) -> Optional[Dict[str, Any]]:
    """Parse JSON string safely."""
    try:
        data = json.loads(payload)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        return None
    return None


async def _iter_sse_data_lines(response: httpx.Response) -> AsyncIterator[str]:
    """Yield SSE data payload lines from an HTTP response stream."""
    async for line in response.aiter_lines():
        if not line:
            continue
        if line.startswith("data:"):
            yield line[5:].strip()


def _as_text(content: Any) -> str:
    """Normalize arbitrary content payloads into prompt text."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        for key in ("prompt", "text", "input", "content", "message"):
            value = content.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def _as_serialized_text(content: Any) -> str:
    """Serialize arbitrary content into a durable text form."""
    if isinstance(content, str):
        return content
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def _parse_json_or_default(raw: Optional[str], default: Any) -> Any:
    """Parse JSON content and return a fallback value if parsing fails."""
    if not raw:
        return default
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return default


def _normalize_memory_tags(raw_tags: Any) -> List[str]:
    """Normalize memory tag payloads into a de-duplicated list of strings."""
    if raw_tags is None:
        return []
    if isinstance(raw_tags, str):
        candidates = [raw_tags]
    elif isinstance(raw_tags, list):
        candidates = [item for item in raw_tags if isinstance(item, (str, int, float))]
    else:
        return []

    tags: List[str] = []
    for item in candidates:
        cleaned = str(item).strip()
        if cleaned and cleaned not in tags:
            tags.append(cleaned)
    return tags


def _fts_terms(text: str) -> List[str]:
    """Build safe FTS terms from arbitrary free text."""
    cleaned = "".join(char if char.isalnum() else " " for char in text.lower())
    return [term for term in cleaned.split() if term]


_TEMPLATE_PATTERN = re.compile(r"\{\{\s*([a-zA-Z0-9_.-]+)\s*\}\}")


def _resolve_template_path(data: Any, path: str) -> tuple[Any, bool]:
    """Resolve a dotted path from nested dict/list payloads."""
    current = data
    for segment in path.split("."):
        if isinstance(current, dict):
            if segment not in current:
                return None, False
            current = current[segment]
            continue
        if isinstance(current, list):
            if not segment.isdigit():
                return None, False
            index = int(segment)
            if index < 0 or index >= len(current):
                return None, False
            current = current[index]
            continue
        return None, False
    return current, True


def _render_template_string(template: str, data: Dict[str, Any]) -> Any:
    """Render {{path}} tokens in a string template."""
    matches = list(_TEMPLATE_PATTERN.finditer(template))
    if not matches:
        return template

    if len(matches) == 1 and matches[0].span() == (0, len(template)):
        value, found = _resolve_template_path(data, matches[0].group(1))
        return value if found else template

    def replace_token(match: re.Match[str]) -> str:
        path = match.group(1)
        value, found = _resolve_template_path(data, path)
        if not found or value is None:
            return ""
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    return _TEMPLATE_PATTERN.sub(replace_token, template)


def _render_template_payload(template: Any, data: Dict[str, Any]) -> Any:
    """Render templates recursively for strings/dicts/lists."""
    if isinstance(template, str):
        return _render_template_string(template, data)
    if isinstance(template, list):
        return [_render_template_payload(item, data) for item in template]
    if isinstance(template, dict):
        return {key: _render_template_payload(value, data) for key, value in template.items()}
    return template


_JSON_PATH_SEGMENT_PATTERN = re.compile(
    r"\.([a-zA-Z0-9_\-]+)|\[(\d+)\]|\['([^']+)'\]|\[\"([^\"]+)\"\]"
)


def _parse_json_path(path: str) -> Optional[List[Any]]:
    """Parse a restricted JSON path expression into key/index segments."""
    normalized = path.strip() or "$"
    if not normalized.startswith("$"):
        normalized = f"${normalized if normalized.startswith('.') else f'.{normalized}'}"

    if normalized == "$":
        return []

    segments: List[Any] = []
    index = 1
    while index < len(normalized):
        match = _JSON_PATH_SEGMENT_PATTERN.match(normalized, index)
        if not match:
            return None
        key = match.group(1) or match.group(3) or match.group(4)
        if key is not None:
            segments.append(key)
        else:
            raw_index = match.group(2)
            if raw_index is None:
                return None
            segments.append(int(raw_index))
        index = match.end()
    return segments


def _resolve_json_path(data: Any, path: str) -> tuple[Any, bool]:
    """Resolve a restricted JSON path against nested dictionaries/lists."""
    segments = _parse_json_path(path)
    if segments is None:
        return None, False

    current = data
    for segment in segments:
        if isinstance(segment, int):
            if not isinstance(current, list):
                return None, False
            if segment < 0 or segment >= len(current):
                return None, False
            current = current[segment]
            continue
        if not isinstance(current, dict):
            return None, False
        if segment not in current:
            return None, False
        current = current[segment]
    return current, True


def _safe_tmp_dir(path_value: str) -> str:
    """Normalize and enforce that a working directory stays under /tmp."""
    tmp_root = Path("/tmp").resolve()
    candidate = Path(path_value or "/tmp").expanduser()
    try:
        resolved = candidate.resolve(strict=False)
    except Exception:
        resolved = tmp_root
    if resolved == tmp_root or tmp_root in resolved.parents:
        return str(resolved)
    return str(tmp_root)


def _sandbox_preexec_fn(
    cpu_time_seconds: int,
    memory_limit_mb: int,
    max_output_bytes: int,
):
    """Build a pre-exec hook that applies OS-level resource limits."""
    if resource is None:
        return None

    memory_bytes = int(memory_limit_mb * 1024 * 1024)
    max_file_size = max(1024, int(max_output_bytes * 2))

    def _preexec() -> None:
        try:
            os.setsid()
        except Exception:
            pass

        try:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_time_seconds, cpu_time_seconds))
        except Exception:
            pass
        try:
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        except Exception:
            pass
        try:
            resource.setrlimit(resource.RLIMIT_FSIZE, (max_file_size, max_file_size))
        except Exception:
            pass
        try:
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        except Exception:
            pass
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        except Exception:
            pass
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (1, 1)) # Limit to a single process
        except Exception:
            pass
    return _preexec


async def _read_stream_limited(
    stream: Optional[asyncio.StreamReader],
    max_bytes: int,
) -> tuple[bytes, bool]:
    """Read an async stream and cap captured bytes."""
    if stream is None:
        return b"", False

    chunks: List[bytes] = []
    total = 0
    truncated = False

    while True:
        chunk = await stream.read(8192)
        if not chunk:
            break
        if total < max_bytes:
            remaining = max_bytes - total
            chunks.append(chunk[:remaining])
            if len(chunk) > remaining:
                truncated = True
        else:
            truncated = True
        total += len(chunk)

    return b"".join(chunks), truncated


def _extract_sandbox_result(stdout_text: str) -> tuple[str, Optional[Dict[str, Any]]]:
    """Extract structured wrapper result markers from stdout."""
    start_marker = "__SYNAPPS_RESULT_START__"
    end_marker = "__SYNAPPS_RESULT_END__"

    start_index = stdout_text.rfind(start_marker)
    if start_index < 0:
        return stdout_text, None

    end_index = stdout_text.find(end_marker, start_index)
    if end_index < 0:
        return stdout_text, None

    payload_start = start_index + len(start_marker)
    payload_text = stdout_text[payload_start:end_index].strip()

    cleaned_stdout = (stdout_text[:start_index] + stdout_text[end_index + len(end_marker):]).strip()
    parsed = _safe_json_loads(payload_text)
    return cleaned_stdout, parsed


PYTHON_CODE_WRAPPER = r"""
import os
import sys
import json
import builtins
import pathlib
import traceback

_clean_env = {
    "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
    "HOME": "/tmp",
    "TMPDIR": "/tmp",
    "TMP": "/tmp",
    "TEMP": "/tmp",
    "PYTHONIOENCODING": "utf-8",
    "PYTHONUNBUFFERED": "1",
}
os.environ.clear()
os.environ.update(_clean_env)


ALLOWED_ROOT = pathlib.Path("/tmp").resolve()

def _resolve_path(raw_path):
    path_obj = pathlib.Path(raw_path)
    if not path_obj.is_absolute():
        path_obj = pathlib.Path(os.getcwd()) / path_obj
    return path_obj.resolve(strict=False)

def _assert_tmp_path(raw_path):
    resolved = _resolve_path(raw_path)
    if resolved == ALLOWED_ROOT or ALLOWED_ROOT in resolved.parents:
        return str(resolved)
    raise PermissionError(f"Filesystem access is restricted to /tmp: {raw_path}")

def _wrap_path_func(module_obj, func_name, indices):
    original = getattr(module_obj, func_name, None)
    if not callable(original):
        return

    def wrapped(*args, **kwargs):
        mutable = list(args)
        for index in indices:
            if index < len(mutable):
                mutable[index] = _assert_tmp_path(mutable[index])
        return original(*mutable, **kwargs)

    setattr(module_obj, func_name, wrapped)

_original_open = builtins.open
def _safe_open(path, *args, **kwargs):
    return _original_open(_assert_tmp_path(path), *args, **kwargs)
builtins.open = _safe_open

_wrap_path_func(os, "open", [0])
_wrap_path_func(os, "listdir", [0])
_wrap_path_func(os, "scandir", [0])
_wrap_path_func(os, "mkdir", [0])
_wrap_path_func(os, "makedirs", [0])
_wrap_path_func(os, "remove", [0])
_wrap_path_func(os, "unlink", [0])
_wrap_path_func(os, "rmdir", [0])
_wrap_path_func(os, "rename", [0, 1])
_wrap_path_func(os, "replace", [0, 1])

for blocked_name in ("system", "popen", "fork", "forkpty"):
    if hasattr(os, blocked_name):
        setattr(os, blocked_name, lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("Process spawning is blocked")))

blocked_modules = {"subprocess", "socket", "ctypes", "multiprocessing", "pathlib"}
_original_import = builtins.__import__
def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    module_root = name.split(".", 1)[0]
    if module_root in blocked_modules:
        raise ImportError(f"Import '{module_root}' is blocked in code sandbox")
    return _original_import(name, globals, locals, fromlist, level)
builtins.__import__ = _safe_import

payload_raw = sys.stdin.read()
try:
    payload = json.loads(payload_raw) if payload_raw.strip() else {}
except Exception:
    payload = {}

user_code = str(payload.get("code", ""))
globals_scope = {
    "__name__": "__main__",
    "data": payload.get("data"),
    "context": payload.get("context", {}),
    "metadata": payload.get("metadata", {}),
    "result": None,
}

wrapper_result = {"ok": True, "result": None}
try:
    exec(compile(user_code, "<user_code.py>", "exec"), globals_scope, globals_scope)
    wrapper_result["result"] = globals_scope.get("result")
except Exception as exc:
    wrapper_result = {
        "ok": False,
        "error": {
            "type": exc.__class__.__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(limit=20),
        },
    }

print("__SYNAPPS_RESULT_START__")
print(json.dumps(wrapper_result, ensure_ascii=False))
print("__SYNAPPS_RESULT_END__")
"""


JAVASCRIPT_CODE_WRAPPER = r"""
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const Module = require('module');

// --- ENVIRONMENT VARIABLE SANITIZATION START ---
const cleanEnv = {
  PATH: process.env.PATH || '/usr/local/bin:/usr/bin:/bin',
  HOME: '/tmp',
  TMPDIR: '/tmp',
  TMP: '/tmp',
  TEMP: '/tmp',
};
// Clear existing process.env and set only allowed variables
for (const key in process.env) {
  delete process.env[key];
}
Object.assign(process.env, cleanEnv);
// --- ENVIRONMENT VARIABLE SANITIZATION END ---


const ALLOWED_ROOT = path.resolve('/tmp');

function resolvePath(rawPath) {
  const inputPath = String(rawPath);
  if (path.isAbsolute(inputPath)) {
    return path.resolve(inputPath);
  }
  return path.resolve(process.cwd(), inputPath);
}

function assertTmpPath(rawPath) {
  const resolved = resolvePath(rawPath);
  if (resolved === ALLOWED_ROOT || resolved.startsWith(ALLOWED_ROOT + path.sep)) {
    return resolved;
  }
  throw new Error(`Filesystem access is restricted to /tmp: ${rawPath}`);
}

function wrapFs(target) {
  const singlePathFns = [
    'readFileSync', 'writeFileSync', 'appendFileSync', 'openSync', 'readdirSync',
    'statSync', 'lstatSync', 'unlinkSync', 'rmSync', 'mkdirSync', 'mkdtempSync',
    'accessSync', 'chmodSync', 'realpathSync', 'readlinkSync'
  ];
  for (const fn of singlePathFns) {
    if (typeof target[fn] !== 'function') continue;
    const original = target[fn].bind(target);
    target[fn] = function(p, ...args) {
      if (typeof p === 'number') {
        return original(p, ...args);
      }
      return original(assertTmpPath(p), ...args);
    };
  }

  for (const fn of ['renameSync', 'copyFileSync', 'linkSync', 'symlinkSync']) {
    if (typeof target[fn] !== 'function') continue;
    const original = target[fn].bind(target);
    target[fn] = function(src, dst, ...args) {
      return original(assertTmpPath(src), assertTmpPath(dst), ...args);
    };
  }
}

function wrapFsPromises(promisesApi) {
  if (!promisesApi) return;
  const singlePathFns = ['readFile', 'writeFile', 'appendFile', 'open', 'readdir', 'stat', 'lstat', 'unlink', 'rm', 'mkdir', 'realpath', 'readlink'];
  for (const fn of singlePathFns) {
    if (typeof promisesApi[fn] !== 'function') continue;
    const original = promisesApi[fn].bind(promisesApi);
    promisesApi[fn] = async function(p, ...args) {
      if (typeof p === 'number') {
        return original(p, ...args);
      }
      return original(assertTmpPath(p), ...args);
    };
  }
  for (const fn of ['rename', 'copyFile', 'link', 'symlink']) {
    if (typeof promisesApi[fn] !== 'function') continue;
    const original = promisesApi[fn].bind(promisesApi);
    promisesApi[fn] = async function(src, dst, ...args) {
      return original(assertTmpPath(src), assertTmpPath(dst), ...args);
    };
  }
}

wrapFs(fs);
wrapFsPromises(fs.promises);

const blockedModules = new Set(['child_process', 'worker_threads', 'cluster', 'net', 'dgram']);
const originalLoad = Module._load;
Module._load = function(request, parent, isMain) {
  const normalized = request.startsWith('node:') ? request.slice(5) : request;
  if (blockedModules.has(normalized)) {
    throw new Error(`Import '${normalized}' is blocked in code sandbox`);
  }
  if (normalized === 'fs') return fs;
  if (normalized === 'fs/promises') return fs.promises;
  return originalLoad.apply(this, arguments);
};

let payload = {};
try {
  const raw = fs.readFileSync(0, 'utf8');
  payload = raw.trim() ? JSON.parse(raw) : {};
} catch (err) {
  payload = {};
}

const sandbox = {
  data: payload.data,
  context: payload.context || {},
  metadata: payload.metadata || {},
  result: null,
  console,
  require,
  Buffer,
  setTimeout,
  clearTimeout,
};

const execTimeoutMs = Math.max(1, Number(payload.exec_timeout_ms || 1000));
let wrapperResult = { ok: true, result: null };
try {
  const context = vm.createContext(sandbox, {
    codeGeneration: { strings: false, wasm: false },
  });
  const script = new vm.Script(String(payload.code || ''), { filename: '<user_code.js>' });
  script.runInContext(context, { timeout: execTimeoutMs });
  wrapperResult.result = sandbox.result;
} catch (err) {
  wrapperResult = {
    ok: false,
    error: {
      type: err && err.name ? err.name : 'Error',
      message: err && err.message ? err.message : String(err),
      stack: err && err.stack ? String(err.stack) : '',
    },
  };
}

console.log('__SYNAPPS_RESULT_START__');
console.log(JSON.stringify(wrapperResult));
console.log('__SYNAPPS_RESULT_END__');
"""


class MemoryStoreBackend(ABC):
    """Abstract persistence backend for memory node operations."""

    backend_name: str

    @abstractmethod
    def upsert(
        self,
        key: str,
        namespace: str,
        content: str,
        payload: Any,
        metadata: Dict[str, Any],
    ) -> None:
        """Persist or replace one memory record."""

    @abstractmethod
    def get(self, key: str, namespace: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Fetch one memory record by key."""

    @abstractmethod
    def search(
        self,
        namespace: str,
        query: str,
        tags: List[str],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Search memory records in a namespace."""

    @abstractmethod
    def delete(self, key: str, namespace: Optional[str] = None) -> bool:
        """Delete one memory record."""

    @abstractmethod
    def clear(self, namespace: str) -> int:
        """Delete all records in a namespace."""


class SQLiteFTSMemoryStoreBackend(MemoryStoreBackend):
    """SQLite-backed persistent store with FTS5 search and LIKE fallback."""

    backend_name = "sqlite_fts"

    def __init__(self, db_path: str):
        self.db_path = str(Path(db_path).expanduser())
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._schema_lock = threading.Lock()
        self._initialized = False
        self._fts_enabled = True
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _ensure_schema(self) -> None:
        with self._schema_lock:
            if self._initialized:
                return
            with self._connect() as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memories (
                        id TEXT PRIMARY KEY,
                        namespace TEXT NOT NULL,
                        content TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        metadata_json TEXT,
                        created_at REAL NOT NULL,
                        flow_id TEXT,
                        node_id TEXT
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_memories_namespace_created_at
                    ON memories(namespace, created_at DESC)
                    """
                )
                try:
                    conn.execute(
                        """
                        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
                        USING fts5(
                            memory_id UNINDEXED,
                            namespace UNINDEXED,
                            content,
                            payload,
                            tags
                        )
                        """
                    )
                    self._fts_enabled = True
                except sqlite3.OperationalError:
                    logger.warning(
                        "SQLite FTS5 is unavailable for '%s'; using LIKE fallback.",
                        self.db_path,
                    )
                    self._fts_enabled = False
                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS memories_fts (
                            memory_id TEXT PRIMARY KEY,
                            namespace TEXT NOT NULL,
                            content TEXT NOT NULL,
                            payload TEXT NOT NULL,
                            tags TEXT NOT NULL
                        )
                        """
                    )
                    conn.execute(
                        """
                        CREATE INDEX IF NOT EXISTS idx_memories_fts_namespace
                        ON memories_fts(namespace)
                        """
                    )
                conn.commit()
            self._initialized = True

    def _row_to_result(self, row: sqlite3.Row, score: float) -> Dict[str, Any]:
        payload = _parse_json_or_default(row["payload_json"], row["content"])
        metadata = _parse_json_or_default(row["metadata_json"], {})
        if not isinstance(metadata, dict):
            metadata = {}
        metadata.setdefault("created_at", row["created_at"])
        return MemorySearchResultModel(
            key=row["id"],
            data=payload,
            score=score,
            metadata=metadata,
        ).model_dump()

    def upsert(
        self,
        key: str,
        namespace: str,
        content: str,
        payload: Any,
        metadata: Dict[str, Any],
    ) -> None:
        self._ensure_schema()
        now = float(metadata.get("timestamp", time.time()))
        payload_json = json.dumps(payload, ensure_ascii=False)
        metadata_json = json.dumps(metadata, ensure_ascii=False)
        tags_text = " ".join(_normalize_memory_tags(metadata.get("tags", [])))
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memories (
                    id, namespace, content, payload_json, metadata_json, created_at, flow_id, node_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    namespace = excluded.namespace,
                    content = excluded.content,
                    payload_json = excluded.payload_json,
                    metadata_json = excluded.metadata_json,
                    created_at = excluded.created_at,
                    flow_id = excluded.flow_id,
                    node_id = excluded.node_id
                """,
                (
                    key,
                    namespace,
                    content,
                    payload_json,
                    metadata_json,
                    now,
                    metadata.get("flow_id"),
                    metadata.get("node_id"),
                ),
            )
            if self._fts_enabled:
                conn.execute("DELETE FROM memories_fts WHERE memory_id = ?", (key,))
                conn.execute(
                    """
                    INSERT INTO memories_fts(memory_id, namespace, content, payload, tags)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (key, namespace, content, _as_serialized_text(payload), tags_text),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO memories_fts(memory_id, namespace, content, payload, tags)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(memory_id) DO UPDATE SET
                        namespace = excluded.namespace,
                        content = excluded.content,
                        payload = excluded.payload,
                        tags = excluded.tags
                    """,
                    (key, namespace, content, _as_serialized_text(payload), tags_text),
                )
            conn.commit()

    def get(self, key: str, namespace: Optional[str] = None) -> Optional[Dict[str, Any]]:
        self._ensure_schema()
        with self._connect() as conn:
            if namespace:
                row = conn.execute(
                    """
                    SELECT id, content, payload_json, metadata_json, created_at
                    FROM memories
                    WHERE id = ? AND namespace = ?
                    LIMIT 1
                    """,
                    (key, namespace),
                ).fetchone()
            else:
                row = conn.execute(
                    """
                    SELECT id, content, payload_json, metadata_json, created_at
                    FROM memories
                    WHERE id = ?
                    LIMIT 1
                    """,
                    (key,),
                ).fetchone()
        if not row:
            return None
        return self._row_to_result(row, score=1.0)

    def search(
        self,
        namespace: str,
        query: str,
        tags: List[str],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        self._ensure_schema()
        normalized_tags = _normalize_memory_tags(tags)
        query_text = (query or "").strip()

        rows: List[sqlite3.Row] = []
        if self._fts_enabled:
            terms = _fts_terms(query_text or " ".join(normalized_tags))
            if terms:
                match_query = " OR ".join(f'"{term}"' for term in terms)
                with self._connect() as conn:
                    try:
                        rows = conn.execute(
                            """
                            SELECT
                                m.id,
                                m.content,
                                m.payload_json,
                                m.metadata_json,
                                m.created_at,
                                bm25(memories_fts) AS rank
                            FROM memories_fts
                            JOIN memories m ON memories_fts.memory_id = m.id
                            WHERE m.namespace = ? AND memories_fts MATCH ?
                            ORDER BY rank
                            LIMIT ?
                            """,
                            (namespace, match_query, top_k),
                        ).fetchall()
                    except sqlite3.OperationalError as exc:
                        logger.debug("SQLite FTS query failed, using LIKE fallback: %s", exc)
                        rows = []

        if not rows:
            sql = (
                "SELECT id, content, payload_json, metadata_json, created_at "
                "FROM memories WHERE namespace = ?"
            )
            params: List[Any] = [namespace]
            search_text = query_text or " ".join(normalized_tags)
            if search_text:
                sql += " AND (content LIKE ? OR payload_json LIKE ? OR metadata_json LIKE ?)"
                like_value = f"%{search_text}%"
                params.extend([like_value, like_value, like_value])
            for tag in normalized_tags:
                sql += " AND metadata_json LIKE ?"
                params.append(f"%{tag}%")
            sql += " ORDER BY created_at DESC LIMIT ?"
            params.append(top_k)
            with self._connect() as conn:
                rows = conn.execute(sql, tuple(params)).fetchall()

        results: List[Dict[str, Any]] = []
        for row in rows:
            rank = row["rank"] if "rank" in row.keys() else None
            score = 0.8
            if rank is not None:
                try:
                    score = 1.0 / (1.0 + abs(float(rank)))
                except (TypeError, ValueError):
                    score = 0.8
            results.append(self._row_to_result(row, score=score))

        if normalized_tags:
            filtered_results: List[Dict[str, Any]] = []
            for result in results:
                metadata = result.get("metadata", {})
                memory_tags = _normalize_memory_tags(metadata.get("tags", []))
                if any(tag in memory_tags for tag in normalized_tags):
                    filtered_results.append(result)
            results = filtered_results

        return results[:top_k]

    def delete(self, key: str, namespace: Optional[str] = None) -> bool:
        self._ensure_schema()
        with self._connect() as conn:
            if namespace:
                existing = conn.execute(
                    "SELECT 1 FROM memories WHERE id = ? AND namespace = ?",
                    (key, namespace),
                ).fetchone()
                if not existing:
                    return False
                conn.execute("DELETE FROM memories WHERE id = ? AND namespace = ?", (key, namespace))
            else:
                existing = conn.execute("SELECT 1 FROM memories WHERE id = ?", (key,)).fetchone()
                if not existing:
                    return False
                conn.execute("DELETE FROM memories WHERE id = ?", (key,))
            conn.execute("DELETE FROM memories_fts WHERE memory_id = ?", (key,))
            conn.commit()
        return True

    def clear(self, namespace: str) -> int:
        self._ensure_schema()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id FROM memories WHERE namespace = ?",
                (namespace,),
            ).fetchall()
            memory_ids = [row["id"] for row in rows]
            conn.execute("DELETE FROM memories WHERE namespace = ?", (namespace,))
            if self._fts_enabled:
                for memory_id in memory_ids:
                    conn.execute("DELETE FROM memories_fts WHERE memory_id = ?", (memory_id,))
            else:
                conn.execute("DELETE FROM memories_fts WHERE namespace = ?", (namespace,))
            conn.commit()
        return len(memory_ids)


class ChromaMemoryStoreBackend(MemoryStoreBackend):
    """ChromaDB-backed persistent vector store."""

    backend_name = "chroma"

    def __init__(self, persist_path: str, collection_name: str):
        self.persist_path = str(Path(persist_path).expanduser())
        Path(self.persist_path).mkdir(parents=True, exist_ok=True)
        try:
            import chromadb  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("chromadb package is not installed") from exc

        self._client = chromadb.PersistentClient(path=self.persist_path)
        self._collection = self._client.get_or_create_collection(name=collection_name)

    def _entry_to_result(
        self,
        memory_id: str,
        document: str,
        metadata: Optional[Dict[str, Any]],
        score: float,
    ) -> Dict[str, Any]:
        raw_metadata = metadata or {}
        payload = _parse_json_or_default(raw_metadata.get("payload_json"), document)
        stored_metadata = _parse_json_or_default(raw_metadata.get("metadata_json"), {})
        if not isinstance(stored_metadata, dict):
            stored_metadata = {}
        if "created_at" in raw_metadata:
            stored_metadata.setdefault("created_at", raw_metadata["created_at"])
        return MemorySearchResultModel(
            key=memory_id,
            data=payload,
            score=score,
            metadata=stored_metadata,
        ).model_dump()

    def upsert(
        self,
        key: str,
        namespace: str,
        content: str,
        payload: Any,
        metadata: Dict[str, Any],
    ) -> None:
        now = float(metadata.get("timestamp", time.time()))
        safe_metadata: Dict[str, Any] = {
            "namespace": namespace,
            "created_at": now,
            "payload_json": json.dumps(payload, ensure_ascii=False),
            "metadata_json": json.dumps(metadata, ensure_ascii=False),
            "tags_text": " ".join(_normalize_memory_tags(metadata.get("tags", []))),
        }
        if metadata.get("flow_id") is not None:
            safe_metadata["flow_id"] = str(metadata["flow_id"])
        if metadata.get("node_id") is not None:
            safe_metadata["node_id"] = str(metadata["node_id"])
        self._collection.upsert(ids=[key], documents=[content], metadatas=[safe_metadata])

    def get(self, key: str, namespace: Optional[str] = None) -> Optional[Dict[str, Any]]:
        payload = self._collection.get(ids=[key], include=["documents", "metadatas"])
        ids = payload.get("ids") or []
        if not ids:
            return None
        documents = payload.get("documents") or []
        metadatas = payload.get("metadatas") or []
        for index, memory_id in enumerate(ids):
            metadata = metadatas[index] if index < len(metadatas) else {}
            if namespace and str((metadata or {}).get("namespace", "")) != namespace:
                continue
            document = documents[index] if index < len(documents) else ""
            return self._entry_to_result(memory_id, document, metadata, score=1.0)
        return None

    def search(
        self,
        namespace: str,
        query: str,
        tags: List[str],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        normalized_tags = _normalize_memory_tags(tags)
        query_text = (query or "").strip() or " ".join(normalized_tags)
        results: List[Dict[str, Any]] = []

        if query_text:
            payload = self._collection.query(
                query_texts=[query_text],
                n_results=top_k,
                where={"namespace": namespace},
                include=["documents", "metadatas", "distances"],
            )
            ids = (payload.get("ids") or [[]])[0]
            documents = (payload.get("documents") or [[]])[0]
            metadatas = (payload.get("metadatas") or [[]])[0]
            distances = (payload.get("distances") or [[]])[0]
            for index, memory_id in enumerate(ids):
                document = documents[index] if index < len(documents) else ""
                metadata = metadatas[index] if index < len(metadatas) else {}
                distance = distances[index] if index < len(distances) else 0.0
                try:
                    score = 1.0 / (1.0 + max(float(distance), 0.0))
                except (TypeError, ValueError):
                    score = 0.8
                results.append(self._entry_to_result(memory_id, document, metadata, score=score))
        else:
            payload = self._collection.get(
                where={"namespace": namespace},
                limit=top_k,
                include=["documents", "metadatas"],
            )
            ids = payload.get("ids") or []
            documents = payload.get("documents") or []
            metadatas = payload.get("metadatas") or []
            for index, memory_id in enumerate(ids):
                document = documents[index] if index < len(documents) else ""
                metadata = metadatas[index] if index < len(metadatas) else {}
                results.append(self._entry_to_result(memory_id, document, metadata, score=0.7))

        if normalized_tags:
            filtered: List[Dict[str, Any]] = []
            for result in results:
                memory_tags = _normalize_memory_tags(result.get("metadata", {}).get("tags", []))
                if any(tag in memory_tags for tag in normalized_tags):
                    filtered.append(result)
            results = filtered

        return results[:top_k]

    def delete(self, key: str, namespace: Optional[str] = None) -> bool:
        record = self.get(key, namespace=namespace)
        if not record:
            return False
        self._collection.delete(ids=[key])
        return True

    def clear(self, namespace: str) -> int:
        payload = self._collection.get(where={"namespace": namespace}, include=["metadatas"])
        ids = payload.get("ids") or []
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)


class MemoryStoreFactory:
    """Factory and cache for memory backends."""

    _stores: Dict[str, MemoryStoreBackend] = {}
    _lock = threading.Lock()

    @classmethod
    def get_store(cls, config: MemoryNodeConfigModel) -> MemoryStoreBackend:
        backend = config.backend if config.backend in SUPPORTED_MEMORY_BACKENDS else DEFAULT_MEMORY_BACKEND

        if backend == "chroma":
            chroma_path = config.persist_path or DEFAULT_MEMORY_CHROMA_PATH
            key = f"chroma::{chroma_path}::{config.collection}"
            with cls._lock:
                cached = cls._stores.get(key)
                if cached:
                    return cached
                try:
                    store = ChromaMemoryStoreBackend(chroma_path, config.collection)
                    cls._stores[key] = store
                    return store
                except Exception as exc:
                    logger.warning(
                        "Chroma backend unavailable, falling back to sqlite_fts: %s",
                        exc,
                    )

        if config.persist_path:
            expanded_path = Path(config.persist_path).expanduser()
            if str(expanded_path).lower().endswith((".db", ".sqlite", ".sqlite3")):
                sqlite_path = str(expanded_path)
            else:
                sqlite_path = str(expanded_path / "memory.sqlite3")
        else:
            sqlite_path = DEFAULT_MEMORY_SQLITE_PATH
        key = f"sqlite_fts::{sqlite_path}"
        with cls._lock:
            cached = cls._stores.get(key)
            if cached:
                return cached
            store = SQLiteFTSMemoryStoreBackend(str(sqlite_path))
            cls._stores[key] = store
            return store


class MemoryNodeApplet(BaseApplet):
    """Persistent memory node with SQLite FTS and ChromaDB backends."""

    VERSION = "1.0.0"
    CAPABILITIES = [
        "persistent-memory",
        "vector-search",
        "tag-retrieval",
        "memory-management",
    ]

    async def on_message(self, message: AppletMessage) -> AppletMessage:
        try:
            config = self._resolve_config(message)
            operation = self._resolve_operation(message, config)
            namespace = self._resolve_namespace(message, config)
            store = MemoryStoreFactory.get_store(config)
        except Exception as exc:
            return AppletMessage(
                content={"error": f"Invalid memory configuration: {exc}"},
                context=message.context,
                metadata={"applet": MEMORY_NODE_TYPE, "status": "error"},
            )

        try:
            if operation == "store":
                return await self._handle_store(message, config, namespace, store)
            if operation == "retrieve":
                return await self._handle_retrieve(message, config, namespace, store)
            if operation == "delete":
                return await self._handle_delete(message, config, namespace, store)
            if operation == "clear":
                return await self._handle_clear(message, namespace, store)
        except Exception as exc:
            logger.error("Memory node operation failed: %s", exc, exc_info=True)
            return AppletMessage(
                content={"error": f"Memory operation failed: {exc}"},
                context=message.context,
                metadata={
                    "applet": MEMORY_NODE_TYPE,
                    "status": "error",
                    "operation": operation,
                    "backend": store.backend_name,
                    "namespace": namespace,
                },
            )

        return AppletMessage(
            content={"error": f"Unsupported memory operation: {operation}"},
            context=message.context,
            metadata={"applet": MEMORY_NODE_TYPE, "status": "error"},
        )

    def _resolve_config(self, message: AppletMessage) -> MemoryNodeConfigModel:
        node_data = message.metadata.get("node_data", {})
        if not isinstance(node_data, dict):
            node_data = {}

        context_config = message.context.get("memory_config", {})
        if not isinstance(context_config, dict):
            context_config = {}

        metadata_config = message.metadata.get("memory_config", {})
        if not isinstance(metadata_config, dict):
            metadata_config = {}

        merged = {**context_config, **metadata_config, **node_data}
        backend = str(merged.get("backend", DEFAULT_MEMORY_BACKEND)).strip().lower()
        if backend not in SUPPORTED_MEMORY_BACKENDS:
            backend = "sqlite_fts"

        payload = {
            "label": merged.get("label", "Memory"),
            "operation": merged.get("operation", "store"),
            "backend": backend,
            "namespace": merged.get("namespace", DEFAULT_MEMORY_NAMESPACE),
            "key": merged.get("key"),
            "query": merged.get("query"),
            "tags": merged.get("tags", []),
            "top_k": merged.get("top_k", merged.get("topK", 5)),
            "persist_path": merged.get("persist_path", merged.get("persistPath")),
            "collection": merged.get("collection", DEFAULT_MEMORY_COLLECTION),
            "include_metadata": merged.get("include_metadata", merged.get("includeMetadata", False)),
            "extra": merged.get("extra", {}),
        }
        return MemoryNodeConfigModel.model_validate(payload)

    def _resolve_operation(self, message: AppletMessage, config: MemoryNodeConfigModel) -> str:
        operation = config.operation
        if isinstance(message.content, dict) and "operation" in message.content:
            raw_operation = message.content.get("operation")
            if raw_operation is not None:
                operation = str(raw_operation).strip().lower()
        if operation not in {"store", "retrieve", "delete", "clear"}:
            raise ValueError("operation must be one of: store, retrieve, delete, clear")
        return operation

    def _resolve_namespace(self, message: AppletMessage, config: MemoryNodeConfigModel) -> str:
        namespace = config.namespace
        if isinstance(message.context.get("memory_namespace"), str):
            raw = message.context["memory_namespace"].strip()
            if raw:
                namespace = raw
        if isinstance(message.content, dict) and isinstance(message.content.get("namespace"), str):
            raw = message.content["namespace"].strip()
            if raw:
                namespace = raw
        return namespace

    def _resolve_key(
        self,
        message: AppletMessage,
        config: MemoryNodeConfigModel,
        default_generate: bool = False,
    ) -> Optional[str]:
        key: Optional[str] = config.key
        if isinstance(message.context.get("memory_key"), str):
            raw_context_key = message.context["memory_key"].strip()
            if raw_context_key:
                key = raw_context_key
        if isinstance(message.content, dict) and isinstance(message.content.get("key"), str):
            raw_content_key = message.content["key"].strip()
            if raw_content_key:
                key = raw_content_key
        if not key and default_generate:
            key = str(uuid.uuid4())
        return key

    def _resolve_query(self, message: AppletMessage, config: MemoryNodeConfigModel) -> str:
        query = config.query or ""
        if isinstance(message.content, str):
            query = message.content
        elif isinstance(message.content, dict):
            for key in ("query", "text", "content"):
                candidate = message.content.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    query = candidate
                    break
        return query.strip()

    def _resolve_tags(self, message: AppletMessage, config: MemoryNodeConfigModel) -> List[str]:
        tags = _normalize_memory_tags(config.tags)
        tags.extend(_normalize_memory_tags(message.context.get("memory_tags")))
        if isinstance(message.content, dict):
            tags.extend(_normalize_memory_tags(message.content.get("tags")))
        deduped: List[str] = []
        for tag in tags:
            if tag not in deduped:
                deduped.append(tag)
        return deduped

    def _extract_store_payload(self, message: AppletMessage) -> Any:
        content = message.content
        if isinstance(content, dict):
            if "data" in content:
                return content["data"]
            ignored_keys = {
                "operation",
                "key",
                "tags",
                "query",
                "namespace",
                "backend",
                "top_k",
                "topK",
                "persist_path",
                "persistPath",
                "collection",
                "include_metadata",
                "includeMetadata",
            }
            payload = {k: v for k, v in content.items() if k not in ignored_keys}
            return payload if payload else content
        return {"value": content}

    async def _handle_store(
        self,
        message: AppletMessage,
        config: MemoryNodeConfigModel,
        namespace: str,
        store: MemoryStoreBackend,
    ) -> AppletMessage:
        payload = self._extract_store_payload(message)
        key = self._resolve_key(message, config, default_generate=True)
        if not key:
            raise ValueError("key could not be resolved for store operation")

        tags = self._resolve_tags(message, config)
        metadata: Dict[str, Any] = {
            "timestamp": message.context.get("timestamp", time.time()),
            "run_id": message.context.get("run_id", message.metadata.get("run_id")),
            "flow_id": message.context.get("flow_id", message.metadata.get("flow_id")),
            "node_id": message.metadata.get("node_id"),
            "tags": tags,
        }
        await asyncio.to_thread(
            store.upsert,
            key,
            namespace,
            _as_serialized_text(payload),
            payload,
            metadata,
        )

        output_context = {
            **message.context,
            "memory_key": key,
            "memory_retrieved": False,
            "memory_backend": store.backend_name,
            "memory_namespace": namespace,
        }
        return AppletMessage(
            content={
                "key": key,
                "status": "stored",
                "backend": store.backend_name,
                "namespace": namespace,
            },
            context=output_context,
            metadata={
                "applet": MEMORY_NODE_TYPE,
                "operation": "store",
                "backend": store.backend_name,
                "namespace": namespace,
            },
        )

    async def _handle_retrieve(
        self,
        message: AppletMessage,
        config: MemoryNodeConfigModel,
        namespace: str,
        store: MemoryStoreBackend,
    ) -> AppletMessage:
        key = self._resolve_key(message, config, default_generate=False)
        if key:
            record = await asyncio.to_thread(store.get, key, namespace)
            if record:
                content: Any = record["data"]
                if config.include_metadata:
                    content = {
                        "key": key,
                        "data": record["data"],
                        "metadata": record["metadata"],
                        "score": record["score"],
                    }
                return AppletMessage(
                    content=content,
                    context={
                        **message.context,
                        "memory_key": key,
                        "memory_retrieved": True,
                        "memory_backend": store.backend_name,
                        "memory_namespace": namespace,
                    },
                    metadata={
                        "applet": MEMORY_NODE_TYPE,
                        "operation": "retrieve",
                        "key": key,
                        "backend": store.backend_name,
                        "namespace": namespace,
                    },
                )

        tags = self._resolve_tags(message, config)
        query = self._resolve_query(message, config)
        results = await asyncio.to_thread(store.search, namespace, query, tags, config.top_k)
        if results:
            response_content: Dict[str, Any] = {
                "memories": {item["key"]: item["data"] for item in results},
                "status": "retrieved",
            }
            if config.include_metadata:
                response_content["results"] = results
                response_content["count"] = len(results)
                if query:
                    response_content["query"] = query
                if tags:
                    response_content["tags"] = tags
            return AppletMessage(
                content=response_content,
                context={
                    **message.context,
                    "memory_retrieved": True,
                    "memory_backend": store.backend_name,
                    "memory_namespace": namespace,
                },
                metadata={
                    "applet": MEMORY_NODE_TYPE,
                    "operation": "retrieve",
                    "backend": store.backend_name,
                    "namespace": namespace,
                },
            )

        return AppletMessage(
            content={"status": "not_found"},
            context={
                **message.context,
                "memory_retrieved": False,
                "memory_backend": store.backend_name,
                "memory_namespace": namespace,
            },
            metadata={
                "applet": MEMORY_NODE_TYPE,
                "operation": "retrieve",
                "backend": store.backend_name,
                "namespace": namespace,
                "status": "not_found",
            },
        )

    async def _handle_delete(
        self,
        message: AppletMessage,
        config: MemoryNodeConfigModel,
        namespace: str,
        store: MemoryStoreBackend,
    ) -> AppletMessage:
        key = self._resolve_key(message, config, default_generate=False)
        if not key:
            return AppletMessage(
                content={"status": "not_found", "key": None},
                context=message.context,
                metadata={
                    "applet": MEMORY_NODE_TYPE,
                    "operation": "delete",
                    "backend": store.backend_name,
                    "namespace": namespace,
                    "status": "not_found",
                },
            )
        deleted = await asyncio.to_thread(store.delete, key, namespace)
        return AppletMessage(
            content={"status": "deleted" if deleted else "not_found", "key": key},
            context=message.context,
            metadata={
                "applet": MEMORY_NODE_TYPE,
                "operation": "delete",
                "backend": store.backend_name,
                "namespace": namespace,
                "status": "success" if deleted else "not_found",
            },
        )

    async def _handle_clear(
        self,
        message: AppletMessage,
        namespace: str,
        store: MemoryStoreBackend,
    ) -> AppletMessage:
        deleted_count = await asyncio.to_thread(store.clear, namespace)
        return AppletMessage(
            content={"status": "cleared", "count": deleted_count, "namespace": namespace},
            context=message.context,
            metadata={
                "applet": MEMORY_NODE_TYPE,
                "operation": "clear",
                "backend": store.backend_name,
                "namespace": namespace,
                "status": "success",
            },
        )


class LLMProviderAdapter(ABC):
    """Common interface for all LLM provider adapters."""

    name: str

    def __init__(self, config: LLMNodeConfigModel):
        self.config = config

    @abstractmethod
    async def complete(self, request: LLMRequestModel) -> LLMResponseModel:
        """Run a non-streaming completion call."""

    @abstractmethod
    async def stream(self, request: LLMRequestModel) -> AsyncIterator[LLMStreamChunkModel]:
        """Run a streaming completion call."""

    @abstractmethod
    def get_models(self) -> List[LLMModelInfoModel]:
        """List known models for this provider."""

    @abstractmethod
    def validate_config(self) -> tuple[bool, str]:
        """Validate provider configuration."""

    def default_model(self) -> str:
        models = self.get_models()
        return models[0].id if models else ""


class OpenAIProviderAdapter(LLMProviderAdapter):
    """OpenAI chat completion adapter."""

    name = "openai"

    def __init__(self, config: LLMNodeConfigModel):
        super().__init__(config)
        self.api_key = config.api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = (config.base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")

    def validate_config(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, "OPENAI_API_KEY not set"
        return True, ""

    def get_models(self) -> List[LLMModelInfoModel]:
        return [
            LLMModelInfoModel(id="gpt-4o", name="GPT-4o", provider=self.name, context_window=128000, supports_vision=True, max_output_tokens=16384),
            LLMModelInfoModel(id="gpt-4o-mini", name="GPT-4o Mini", provider=self.name, context_window=128000, supports_vision=True, max_output_tokens=16384),
            LLMModelInfoModel(id="gpt-4.1", name="GPT-4.1", provider=self.name, context_window=1047576, supports_vision=True, max_output_tokens=32768),
            LLMModelInfoModel(id="o3-mini", name="o3-mini", provider=self.name, context_window=200000, max_output_tokens=100000),
        ]

    def _build_payload(self, request: LLMRequestModel, stream: bool) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": request.model,
            "messages": [m.model_dump() for m in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p,
            "stream": stream,
        }
        if request.stop_sequences:
            payload["stop"] = request.stop_sequences
        if stream:
            payload["stream_options"] = {"include_usage": True}
        if request.structured_output and "response_format" not in request.extra:
            if request.json_schema:
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "structured_response",
                        "schema": request.json_schema,
                    },
                }
            else:
                payload["response_format"] = {"type": "json_object"}
        payload.update(request.extra)
        return payload

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self.config.headers,
        }

    async def complete(self, request: LLMRequestModel) -> LLMResponseModel:
        payload = self._build_payload(request, stream=False)
        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
            )
            if response.status_code >= 400:
                detail = response.text or f"HTTP {response.status_code}"
                raise RuntimeError(f"OpenAI request failed: {detail}")
            response.raise_for_status()
            data = response.json()

        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage = data.get("usage") or {}
        return LLMResponseModel(
            content=message.get("content", ""),
            model=data.get("model", request.model),
            provider=self.name,
            usage=LLMUsageModel(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            ),
            finish_reason=choice.get("finish_reason", "stop"),
            raw=data,
        )

    async def stream(self, request: LLMRequestModel) -> AsyncIterator[LLMStreamChunkModel]:
        payload = self._build_payload(request, stream=True)
        usage = LLMUsageModel()
        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
            ) as response:
                response.raise_for_status()
                async for data_line in _iter_sse_data_lines(response):
                    if data_line == "[DONE]":
                        break
                    chunk = _safe_json_loads(data_line)
                    if not chunk:
                        continue
                    raw_usage = chunk.get("usage") or {}
                    if raw_usage:
                        usage = LLMUsageModel(
                            prompt_tokens=raw_usage.get("prompt_tokens", 0),
                            completion_tokens=raw_usage.get("completion_tokens", 0),
                            total_tokens=raw_usage.get("total_tokens", 0),
                        )
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    text_delta = delta.get("content", "")
                    if text_delta:
                        yield LLMStreamChunkModel(content=text_delta)
        yield LLMStreamChunkModel(done=True, usage=usage)


class CustomProviderAdapter(OpenAIProviderAdapter):
    """Adapter for OpenAI-compatible custom endpoints."""

    name = "custom"

    def __init__(self, config: LLMNodeConfigModel):
        super().__init__(config)
        self.api_key = config.api_key or os.environ.get("CUSTOM_LLM_API_KEY")
        self.base_url = (config.base_url or os.environ.get("CUSTOM_LLM_BASE_URL", "")).rstrip("/")

    def validate_config(self) -> tuple[bool, str]:
        if not self.base_url:
            return False, "base_url is required for custom provider"
        return True, ""

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json", **self.config.headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def get_models(self) -> List[LLMModelInfoModel]:
        model_id = self.config.model or "custom-model"
        return [LLMModelInfoModel(id=model_id, name=model_id, provider=self.name)]


class AnthropicProviderAdapter(LLMProviderAdapter):
    """Anthropic Messages API adapter."""

    name = "anthropic"

    def __init__(self, config: LLMNodeConfigModel):
        super().__init__(config)
        self.api_key = config.api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.base_url = (config.base_url or os.environ.get("ANTHROPIC_BASE_URL") or "https://api.anthropic.com").rstrip("/")

    def validate_config(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, "ANTHROPIC_API_KEY not set"
        return True, ""

    def get_models(self) -> List[LLMModelInfoModel]:
        return [
            LLMModelInfoModel(id="claude-sonnet-4-6", name="Claude Sonnet 4.6", provider=self.name, context_window=200000, supports_vision=True, max_output_tokens=16000),
            LLMModelInfoModel(id="claude-haiku-4-5-20251001", name="Claude Haiku 4.5", provider=self.name, context_window=200000, supports_vision=True, max_output_tokens=8192),
            LLMModelInfoModel(id="claude-opus-4-6", name="Claude Opus 4.6", provider=self.name, context_window=200000, supports_vision=True, max_output_tokens=32000),
        ]

    def _headers(self) -> Dict[str, str]:
        return {
            "x-api-key": self.api_key or "",
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
            **self.config.headers,
        }

    def _build_payload(self, request: LLMRequestModel, stream: bool) -> Dict[str, Any]:
        system_messages = [m.content for m in request.messages if m.role == "system"]
        normal_messages = [
            {"role": m.role, "content": m.content}
            for m in request.messages
            if m.role != "system"
        ]
        payload: Dict[str, Any] = {
            "model": request.model,
            "messages": normal_messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stream": stream,
        }
        if request.stop_sequences:
            payload["stop_sequences"] = request.stop_sequences
        if system_messages:
            payload["system"] = "\n".join(system_messages).strip()
        payload.update(request.extra)
        return payload

    async def complete(self, request: LLMRequestModel) -> LLMResponseModel:
        payload = self._build_payload(request, stream=False)
        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            response = await client.post(
                f"{self.base_url}/v1/messages",
                headers=self._headers(),
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        text_content = ""
        for block in data.get("content", []):
            if isinstance(block, dict) and block.get("type") == "text":
                text_content += block.get("text", "")

        usage = data.get("usage") or {}
        prompt_tokens = usage.get("input_tokens", 0)
        completion_tokens = usage.get("output_tokens", 0)
        return LLMResponseModel(
            content=text_content,
            model=data.get("model", request.model),
            provider=self.name,
            usage=LLMUsageModel(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            finish_reason=data.get("stop_reason", "end_turn"),
            raw=data,
        )

    async def stream(self, request: LLMRequestModel) -> AsyncIterator[LLMStreamChunkModel]:
        payload = self._build_payload(request, stream=True)
        usage = LLMUsageModel()
        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/v1/messages",
                headers=self._headers(),
                json=payload,
            ) as response:
                response.raise_for_status()
                async for data_line in _iter_sse_data_lines(response):
                    if data_line == "[DONE]":
                        break
                    chunk = _safe_json_loads(data_line)
                    if not chunk:
                        continue

                    chunk_type = chunk.get("type", "")
                    if chunk_type == "content_block_delta":
                        delta = chunk.get("delta") or {}
                        text_delta = delta.get("text", "")
                        if text_delta:
                            yield LLMStreamChunkModel(content=text_delta)
                    elif chunk_type == "message_delta":
                        raw_usage = chunk.get("usage") or {}
                        if raw_usage:
                            prompt_tokens = raw_usage.get("input_tokens", usage.prompt_tokens)
                            completion_tokens = raw_usage.get("output_tokens", usage.completion_tokens)
                            usage = LLMUsageModel(
                                prompt_tokens=prompt_tokens,
                                completion_tokens=completion_tokens,
                                total_tokens=prompt_tokens + completion_tokens,
                            )
        yield LLMStreamChunkModel(done=True, usage=usage)


class GoogleProviderAdapter(LLMProviderAdapter):
    """Google Gemini REST adapter."""

    name = "google"

    def __init__(self, config: LLMNodeConfigModel):
        super().__init__(config)
        self.api_key = config.api_key or os.environ.get("GOOGLE_API_KEY")
        self.base_url = (config.base_url or os.environ.get("GOOGLE_BASE_URL") or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")

    def validate_config(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, "GOOGLE_API_KEY not set"
        return True, ""

    def get_models(self) -> List[LLMModelInfoModel]:
        return [
            LLMModelInfoModel(id="gemini-2.5-flash", name="Gemini 2.5 Flash", provider=self.name, context_window=1048576, supports_vision=True, max_output_tokens=65536),
            LLMModelInfoModel(id="gemini-2.5-pro", name="Gemini 2.5 Pro", provider=self.name, context_window=1048576, supports_vision=True, max_output_tokens=65536),
        ]

    def _build_payload(self, request: LLMRequestModel) -> Dict[str, Any]:
        contents: List[Dict[str, Any]] = []
        system_messages = []
        for message in request.messages:
            if message.role == "system":
                system_messages.append(message.content)
                continue
            role = "model" if message.role == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": message.content}]})

        generation_config: Dict[str, Any] = {
            "temperature": request.temperature,
            "topP": request.top_p,
            "maxOutputTokens": request.max_tokens,
        }
        if request.stop_sequences:
            generation_config["stopSequences"] = request.stop_sequences
        if request.structured_output:
            generation_config["responseMimeType"] = "application/json"
            if request.json_schema:
                generation_config["responseSchema"] = request.json_schema

        payload: Dict[str, Any] = {
            "contents": contents or [{"role": "user", "parts": [{"text": ""}]}],
            "generationConfig": generation_config,
            **request.extra,
        }
        if system_messages:
            payload["systemInstruction"] = {"parts": [{"text": "\n".join(system_messages).strip()}]}
        return payload

    async def complete(self, request: LLMRequestModel) -> LLMResponseModel:
        payload = self._build_payload(request)
        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            response = await client.post(
                f"{self.base_url}/models/{request.model}:generateContent",
                params={"key": self.api_key},
                headers={"Content-Type": "application/json", **self.config.headers},
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        candidates = data.get("candidates") or [{}]
        candidate = candidates[0]
        content_parts = (candidate.get("content") or {}).get("parts") or []
        text_content = "".join(part.get("text", "") for part in content_parts if isinstance(part, dict))

        metadata = data.get("usageMetadata") or {}
        usage = LLMUsageModel(
            prompt_tokens=metadata.get("promptTokenCount", 0),
            completion_tokens=metadata.get("candidatesTokenCount", 0),
            total_tokens=metadata.get("totalTokenCount", 0),
        )
        finish_reason = candidate.get("finishReason", "STOP")
        return LLMResponseModel(
            content=text_content,
            model=request.model,
            provider=self.name,
            usage=usage,
            finish_reason=finish_reason,
            raw=data,
        )

    async def stream(self, request: LLMRequestModel) -> AsyncIterator[LLMStreamChunkModel]:
        payload = self._build_payload(request)
        usage = LLMUsageModel()
        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/models/{request.model}:streamGenerateContent",
                params={"alt": "sse", "key": self.api_key},
                headers={"Content-Type": "application/json", **self.config.headers},
                json=payload,
            ) as response:
                response.raise_for_status()
                async for data_line in _iter_sse_data_lines(response):
                    chunk = _safe_json_loads(data_line)
                    if not chunk:
                        continue
                    candidates = chunk.get("candidates") or []
                    if candidates:
                        content = (candidates[0].get("content") or {}).get("parts") or []
                        text_delta = "".join(part.get("text", "") for part in content if isinstance(part, dict))
                        if text_delta:
                            yield LLMStreamChunkModel(content=text_delta)
                    metadata = chunk.get("usageMetadata") or {}
                    if metadata:
                        usage = LLMUsageModel(
                            prompt_tokens=metadata.get("promptTokenCount", usage.prompt_tokens),
                            completion_tokens=metadata.get("candidatesTokenCount", usage.completion_tokens),
                            total_tokens=metadata.get("totalTokenCount", usage.total_tokens),
                        )
        yield LLMStreamChunkModel(done=True, usage=usage)


class OllamaProviderAdapter(LLMProviderAdapter):
    """Ollama local model adapter."""

    name = "ollama"

    def __init__(self, config: LLMNodeConfigModel):
        super().__init__(config)
        self.base_url = (config.base_url or os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")

    def validate_config(self) -> tuple[bool, str]:
        if not self.base_url:
            return False, "base_url is required for ollama"
        return True, ""

    def get_models(self) -> List[LLMModelInfoModel]:
        return [
            LLMModelInfoModel(id="llama3.1", name="Llama 3.1", provider=self.name, context_window=131072, max_output_tokens=4096),
            LLMModelInfoModel(id="mistral", name="Mistral", provider=self.name, context_window=32768, max_output_tokens=4096),
            LLMModelInfoModel(id="codellama", name="Code Llama", provider=self.name, context_window=16384, max_output_tokens=4096),
        ]

    def _build_payload(self, request: LLMRequestModel, stream: bool) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": request.model,
            "messages": [m.model_dump() for m in request.messages],
            "stream": stream,
            "options": {
                "temperature": request.temperature,
                "top_p": request.top_p,
                "num_predict": request.max_tokens,
            },
        }
        if request.stop_sequences:
            payload["options"]["stop"] = request.stop_sequences
        if request.structured_output:
            payload["format"] = request.json_schema or "json"
        payload.update(request.extra)
        return payload

    async def complete(self, request: LLMRequestModel) -> LLMResponseModel:
        payload = self._build_payload(request, stream=False)
        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                headers={"Content-Type": "application/json", **self.config.headers},
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        usage = LLMUsageModel(
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
        )
        return LLMResponseModel(
            content=((data.get("message") or {}).get("content", "")),
            model=data.get("model", request.model),
            provider=self.name,
            usage=usage,
            finish_reason="stop" if data.get("done", True) else "incomplete",
            raw=data,
        )

    async def stream(self, request: LLMRequestModel) -> AsyncIterator[LLMStreamChunkModel]:
        payload = self._build_payload(request, stream=True)
        usage = LLMUsageModel()
        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                headers={"Content-Type": "application/json", **self.config.headers},
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    chunk = _safe_json_loads(line)
                    if not chunk:
                        continue
                    text_delta = ((chunk.get("message") or {}).get("content", ""))
                    if text_delta:
                        yield LLMStreamChunkModel(content=text_delta)
                    if chunk.get("done", False):
                        usage = LLMUsageModel(
                            prompt_tokens=chunk.get("prompt_eval_count", 0),
                            completion_tokens=chunk.get("eval_count", 0),
                            total_tokens=chunk.get("prompt_eval_count", 0) + chunk.get("eval_count", 0),
                        )
        yield LLMStreamChunkModel(done=True, usage=usage)


class LLMProviderRegistry:
    """Runtime registry for provider adapters."""

    _providers: Dict[str, Type[LLMProviderAdapter]] = {
        "openai": OpenAIProviderAdapter,
        "anthropic": AnthropicProviderAdapter,
        "google": GoogleProviderAdapter,
        "ollama": OllamaProviderAdapter,
        "custom": CustomProviderAdapter,
    }

    @classmethod
    def get(cls, name: str, config: LLMNodeConfigModel) -> LLMProviderAdapter:
        provider_name = name.lower().strip()
        provider_cls = cls._providers.get(provider_name)
        if not provider_cls:
            raise ValueError(f"Unknown provider '{name}'. Available: {list(cls._providers.keys())}")
        return provider_cls(config)

    @classmethod
    def list_providers(cls) -> List[LLMProviderInfoModel]:
        providers: List[LLMProviderInfoModel] = []
        for name in SUPPORTED_LLM_PROVIDERS:
            provider_cls = cls._providers.get(name)
            if provider_cls is None:
                continue
            default_cfg = LLMNodeConfigModel(provider=name)
            provider = provider_cls(default_cfg)
            is_valid, reason = provider.validate_config()
            providers.append(
                LLMProviderInfoModel(
                    name=name,
                    configured=is_valid,
                    reason="" if is_valid else reason,
                    models=provider.get_models(),
                )
            )
        return providers


class LLMNodeApplet(BaseApplet):
    """Universal LLM node with provider adapter routing."""

    VERSION = "1.0.0"
    CAPABILITIES = [
        "text-generation",
        "multi-provider",
        "streaming",
        "structured-output",
    ]

    async def on_message(self, message: AppletMessage) -> AppletMessage:
        config = self._resolve_config(message)
        provider = LLMProviderRegistry.get(config.provider, config)
        metrics.record_provider_call(config.provider)
        is_valid, reason = provider.validate_config()
        if not is_valid:
            return AppletMessage(
                content=f"Provider '{config.provider}' is not configured: {reason}",
                context=message.context,
                metadata={"applet": "llm", "status": "error"},
            )

        request = self._build_request(message, config, provider)

        if config.stream:
            response = await self._run_streaming(provider, request, message, config)
        else:
            response = await provider.complete(request)

        parsed_content: Any = response.content
        parse_error = None
        if config.structured_output:
            try:
                parsed_content = json.loads(response.content)
            except json.JSONDecodeError:
                parse_error = "Provider response was not valid JSON"

        output_context = {**message.context}
        output_context["last_llm_response"] = response.model_dump()
        output_context["llm_provider"] = response.provider
        output_context["llm_model"] = response.model

        metadata: Dict[str, Any] = {
            "applet": "llm",
            "provider": response.provider,
            "model": response.model,
            "usage": response.usage.model_dump(),
            "finish_reason": response.finish_reason,
            "status": "success",
        }
        if config.structured_output:
            metadata["structured_output"] = parse_error is None
            if parse_error:
                metadata["structured_output_error"] = parse_error

        return AppletMessage(content=parsed_content, context=output_context, metadata=metadata)

    async def _run_streaming(
        self,
        provider: LLMProviderAdapter,
        request: LLMRequestModel,
        message: AppletMessage,
        config: LLMNodeConfigModel,
    ) -> LLMResponseModel:
        full_text = ""
        usage = LLMUsageModel()
        node_id = message.metadata.get("node_id")
        run_id = message.metadata.get("run_id")

        async for chunk in provider.stream(request):
            if chunk.content:
                full_text += chunk.content
                await self._broadcast_stream(
                    node_id=node_id,
                    run_id=run_id,
                    provider_name=provider.name,
                    model=request.model,
                    chunk=chunk.content,
                    done=False,
                )
            if chunk.usage:
                usage = chunk.usage

        await self._broadcast_stream(
            node_id=node_id,
            run_id=run_id,
            provider_name=provider.name,
            model=request.model,
            chunk="",
            done=True,
        )
        return LLMResponseModel(
            content=full_text,
            model=request.model,
            provider=provider.name,
            usage=usage,
            finish_reason="stop",
            raw={},
        )

    async def _broadcast_stream(
        self,
        node_id: Optional[str],
        run_id: Optional[str],
        provider_name: str,
        model: str,
        chunk: str,
        done: bool,
    ) -> None:
        if not ws_manager.connected_websockets:
            return
        message = _ws_message(
            "llm.stream",
            {
                "node_id": node_id,
                "run_id": run_id,
                "provider": provider_name,
                "model": model,
                "chunk": chunk,
                "done": done,
            },
        )
        await ws_manager.broadcast(message)

    def _resolve_config(self, message: AppletMessage) -> LLMNodeConfigModel:
        node_data = message.metadata.get("node_data", {})
        if not isinstance(node_data, dict):
            node_data = {}

        context_config = message.context.get("llm_config", {})
        if not isinstance(context_config, dict):
            context_config = {}

        metadata_config = message.metadata.get("llm_config", {})
        if not isinstance(metadata_config, dict):
            metadata_config = {}

        merged = {**context_config, **metadata_config, **node_data}

        config_payload = {
            "label": merged.get("label", "LLM"),
            "provider": merged.get("provider", "openai"),
            "model": merged.get("model"),
            "system_prompt": merged.get("system_prompt", merged.get("systemPrompt", message.metadata.get("system_prompt", ""))),
            "temperature": merged.get("temperature", 0.7),
            "max_tokens": merged.get("max_tokens", merged.get("maxTokens", 1024)),
            "top_p": merged.get("top_p", merged.get("topP", 1.0)),
            "stop_sequences": merged.get("stop_sequences", merged.get("stopSequences", [])),
            "stream": merged.get("stream", False),
            "structured_output": merged.get("structured_output", merged.get("structuredOutput", False)),
            "json_schema": merged.get("json_schema", merged.get("jsonSchema")),
            "api_key": merged.get("api_key", merged.get("apiKey")),
            "base_url": merged.get("base_url", merged.get("baseUrl")),
            "timeout_seconds": merged.get("timeout_seconds", merged.get("timeoutSeconds", 120.0)),
            "headers": merged.get("headers", {}),
            "extra": merged.get("extra", {}),
        }
        if isinstance(config_payload["stop_sequences"], str):
            config_payload["stop_sequences"] = [config_payload["stop_sequences"]]
        return LLMNodeConfigModel.model_validate(config_payload)

    def _build_request(
        self,
        message: AppletMessage,
        config: LLMNodeConfigModel,
        provider: LLMProviderAdapter,
    ) -> LLMRequestModel:
        messages: List[LLMMessageModel] = []

        raw_history = message.context.get("messages", [])
        if isinstance(raw_history, list):
            for item in raw_history:
                if not isinstance(item, dict):
                    continue
                role = item.get("role")
                content = item.get("content")
                if isinstance(role, str) and isinstance(content, str):
                    try:
                        messages.append(LLMMessageModel(role=role, content=content))
                    except Exception:
                        continue

        if config.system_prompt:
            messages.insert(0, LLMMessageModel(role="system", content=config.system_prompt))

        messages.append(LLMMessageModel(role="user", content=_as_text(message.content)))

        model_id = config.model or provider.default_model()

        return LLMRequestModel(
            messages=messages,
            model=model_id,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
            stop_sequences=config.stop_sequences,
            stream=config.stream,
            structured_output=config.structured_output,
            json_schema=config.json_schema,
            extra=dict(config.extra),
        )


def _parse_image_size(size: str) -> tuple[int, int]:
    """Parse an image size string like '1024x1024' into integer width/height."""
    if not isinstance(size, str):
        return 1024, 1024
    raw = size.strip().lower()
    if "x" not in raw:
        return 1024, 1024
    width_raw, height_raw = raw.split("x", 1)
    try:
        width = int(width_raw)
        height = int(height_raw)
    except ValueError:
        return 1024, 1024
    if width <= 0 or height <= 0:
        return 1024, 1024
    return width, height


def _extract_openai_style_images(payload: Dict[str, Any]) -> List[str]:
    """Extract image payload values from OpenAI-compatible image responses."""
    images: List[str] = []
    for item in payload.get("data") or []:
        if not isinstance(item, dict):
            continue
        b64_value = item.get("b64_json")
        if isinstance(b64_value, str) and b64_value:
            images.append(b64_value)
            continue
        url_value = item.get("url")
        if isinstance(url_value, str) and url_value:
            images.append(url_value)
    return images


class ImageProviderAdapter(ABC):
    """Common interface for all image provider adapters."""

    name: str

    def __init__(self, config: ImageGenNodeConfigModel):
        self.config = config

    @abstractmethod
    async def generate(self, request: ImageGenRequestModel) -> ImageGenResponseModel:
        """Generate one or more images from the request prompt."""

    @abstractmethod
    def get_models(self) -> List[ImageModelInfoModel]:
        """List known models for this provider."""

    @abstractmethod
    def validate_config(self) -> tuple[bool, str]:
        """Validate provider configuration."""

    def default_model(self) -> str:
        models = self.get_models()
        return models[0].id if models else ""


class OpenAIImageProviderAdapter(ImageProviderAdapter):
    """OpenAI image generation adapter (DALL-E 3)."""

    name = "openai"

    def __init__(self, config: ImageGenNodeConfigModel):
        super().__init__(config)
        self.api_key = config.api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = (config.base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")

    def validate_config(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, "OPENAI_API_KEY not set"
        return True, ""

    def get_models(self) -> List[ImageModelInfoModel]:
        return [
            ImageModelInfoModel(
                id="dall-e-3",
                name="DALL-E 3",
                provider=self.name,
                supports_base64=True,
                supports_url=True,
                max_images=1,
            ),
        ]

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self.config.headers,
        }

    def _build_payload(self, request: ImageGenRequestModel) -> Dict[str, Any]:
        prompt_text = request.prompt
        if request.style:
            prompt_text = f"{prompt_text}, {request.style} style"

        payload: Dict[str, Any] = {
            "model": request.model,
            "prompt": prompt_text,
            "n": 1 if request.model == "dall-e-3" else request.n,
            "size": request.size,
            "quality": request.quality,
            "response_format": request.response_format,
        }
        payload.update(request.extra)
        return payload

    async def generate(self, request: ImageGenRequestModel) -> ImageGenResponseModel:
        payload = self._build_payload(request)
        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            response = await client.post(
                f"{self.base_url}/images/generations",
                headers=self._headers(),
                json=payload,
            )
            if response.status_code >= 400:
                detail = response.text or f"HTTP {response.status_code}"
                raise RuntimeError(f"OpenAI image request failed: {detail}")
            response.raise_for_status()
            data = response.json()

        images = _extract_openai_style_images(data)
        revised_prompt = None
        entries = data.get("data") or []
        if entries and isinstance(entries[0], dict):
            raw_revised = entries[0].get("revised_prompt")
            if isinstance(raw_revised, str):
                revised_prompt = raw_revised

        return ImageGenResponseModel(
            images=images,
            model=data.get("model", request.model),
            provider=self.name,
            revised_prompt=revised_prompt,
            raw=data if isinstance(data, dict) else {},
        )


class StabilityImageProviderAdapter(ImageProviderAdapter):
    """Stability AI image generation adapter."""

    name = "stability"

    def __init__(self, config: ImageGenNodeConfigModel):
        super().__init__(config)
        self.api_key = config.api_key or os.environ.get("STABILITY_API_KEY")
        self.base_url = (config.base_url or os.environ.get("STABILITY_BASE_URL") or "https://api.stability.ai").rstrip("/")

    def validate_config(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, "STABILITY_API_KEY not set"
        return True, ""

    def get_models(self) -> List[ImageModelInfoModel]:
        return [
            ImageModelInfoModel(
                id="stable-diffusion-xl-1024-v1-0",
                name="Stable Diffusion XL",
                provider=self.name,
                supports_base64=True,
                supports_url=False,
                max_images=4,
            ),
            ImageModelInfoModel(
                id="stable-diffusion-3",
                name="Stable Diffusion 3",
                provider=self.name,
                supports_base64=True,
                supports_url=False,
                max_images=4,
            ),
        ]

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            **self.config.headers,
        }

    def _build_payload(self, request: ImageGenRequestModel) -> Dict[str, Any]:
        width, height = _parse_image_size(request.size)
        prompt_text = request.prompt
        if request.style:
            prompt_text = f"{prompt_text}, {request.style} style"

        text_prompts: List[Dict[str, Any]] = [{"text": prompt_text, "weight": 1.0}]
        if request.negative_prompt:
            text_prompts.append({"text": request.negative_prompt, "weight": -1.0})

        payload: Dict[str, Any] = {
            "text_prompts": text_prompts,
            "cfg_scale": float(request.extra.get("cfg_scale", 7)),
            "height": height,
            "width": width,
            "samples": request.n,
            "steps": int(request.extra.get("steps", 30)),
        }
        payload.update({k: v for k, v in request.extra.items() if k not in {"cfg_scale", "steps"}})
        return payload

    async def generate(self, request: ImageGenRequestModel) -> ImageGenResponseModel:
        payload = self._build_payload(request)
        model_id = request.model
        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            response = await client.post(
                f"{self.base_url}/v1/generation/{model_id}/text-to-image",
                headers=self._headers(),
                json=payload,
            )
            if response.status_code >= 400:
                detail = response.text or f"HTTP {response.status_code}"
                raise RuntimeError(f"Stability image request failed: {detail}")
            response.raise_for_status()
            data = response.json()

        images = []
        for artifact in data.get("artifacts") or []:
            if isinstance(artifact, dict):
                raw_b64 = artifact.get("base64")
                if isinstance(raw_b64, str) and raw_b64:
                    images.append(raw_b64)

        return ImageGenResponseModel(
            images=images,
            model=model_id,
            provider=self.name,
            raw=data if isinstance(data, dict) else {},
        )


class FluxImageProviderAdapter(ImageProviderAdapter):
    """Flux image generation adapter using an OpenAI-compatible response shape."""

    name = "flux"

    def __init__(self, config: ImageGenNodeConfigModel):
        super().__init__(config)
        self.api_key = config.api_key or os.environ.get("FLUX_API_KEY")
        self.base_url = (config.base_url or os.environ.get("FLUX_BASE_URL") or "https://api.bfl.ai/v1").rstrip("/")

    def validate_config(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, "FLUX_API_KEY not set"
        return True, ""

    def get_models(self) -> List[ImageModelInfoModel]:
        return [
            ImageModelInfoModel(
                id="flux-1.1-pro",
                name="FLUX 1.1 Pro",
                provider=self.name,
                supports_base64=True,
                supports_url=True,
                max_images=4,
            ),
            ImageModelInfoModel(
                id="flux-1-dev",
                name="FLUX 1 Dev",
                provider=self.name,
                supports_base64=True,
                supports_url=True,
                max_images=4,
            ),
        ]

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json", **self.config.headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _endpoint(self) -> str:
        endpoint = self.config.extra.get("endpoint", "/images/generations")
        if not isinstance(endpoint, str) or not endpoint.strip():
            endpoint = "/images/generations"
        if not endpoint.startswith("/"):
            endpoint = f"/{endpoint}"
        return endpoint

    def _build_payload(self, request: ImageGenRequestModel) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": request.model,
            "prompt": request.prompt,
            "size": request.size,
            "style": request.style,
            "quality": request.quality,
            "n": request.n,
            "response_format": request.response_format,
        }
        if request.negative_prompt:
            payload["negative_prompt"] = request.negative_prompt
        payload.update(request.extra)
        return payload

    async def generate(self, request: ImageGenRequestModel) -> ImageGenResponseModel:
        payload = self._build_payload(request)
        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            response = await client.post(
                f"{self.base_url}{self._endpoint()}",
                headers=self._headers(),
                json=payload,
            )
            if response.status_code >= 400:
                detail = response.text or f"HTTP {response.status_code}"
                raise RuntimeError(f"Flux image request failed: {detail}")
            response.raise_for_status()
            data = response.json()

        images = _extract_openai_style_images(data if isinstance(data, dict) else {})
        if not images and isinstance(data, dict):
            for item in data.get("images") or []:
                if isinstance(item, str) and item:
                    images.append(item)
        revised_prompt = None
        if isinstance(data, dict):
            raw_revised = data.get("revised_prompt")
            if isinstance(raw_revised, str):
                revised_prompt = raw_revised
        return ImageGenResponseModel(
            images=images,
            model=request.model,
            provider=self.name,
            revised_prompt=revised_prompt,
            raw=data if isinstance(data, dict) else {},
        )


class ImageProviderRegistry:
    """Runtime registry for image provider adapters."""

    _providers: Dict[str, Type[ImageProviderAdapter]] = {
        "openai": OpenAIImageProviderAdapter,
        "stability": StabilityImageProviderAdapter,
        "flux": FluxImageProviderAdapter,
    }

    @classmethod
    def get(cls, name: str, config: ImageGenNodeConfigModel) -> ImageProviderAdapter:
        provider_name = name.lower().strip()
        provider_cls = cls._providers.get(provider_name)
        if not provider_cls:
            raise ValueError(f"Unknown provider '{name}'. Available: {list(cls._providers.keys())}")
        return provider_cls(config)

    @classmethod
    def list_providers(cls) -> List[ImageProviderInfoModel]:
        providers: List[ImageProviderInfoModel] = []
        for name in SUPPORTED_IMAGE_PROVIDERS:
            provider_cls = cls._providers.get(name)
            if provider_cls is None:
                continue
            default_cfg = ImageGenNodeConfigModel(provider=name)
            provider = provider_cls(default_cfg)
            is_valid, reason = provider.validate_config()
            providers.append(
                ImageProviderInfoModel(
                    name=name,
                    configured=is_valid,
                    reason="" if is_valid else reason,
                    models=provider.get_models(),
                )
            )
        return providers


class ImageGenNodeApplet(BaseApplet):
    """Universal image generation node with provider adapter routing."""

    VERSION = "1.0.0"
    CAPABILITIES = [
        "image-generation",
        "multi-provider",
        "text-to-image",
    ]

    async def on_message(self, message: AppletMessage) -> AppletMessage:
        config = self._resolve_config(message)
        provider = ImageProviderRegistry.get(config.provider, config)
        is_valid, reason = provider.validate_config()
        if not is_valid:
            return AppletMessage(
                content={"error": f"Provider '{config.provider}' is not configured: {reason}"},
                context=message.context,
                metadata={"applet": IMAGE_NODE_TYPE, "status": "error"},
            )

        request = self._build_request(message, config, provider)
        response = await provider.generate(request)
        first_image = response.images[0] if response.images else ""

        output_context = {**message.context}
        output_context["last_image_response"] = response.model_dump()
        output_context["image_provider"] = response.provider
        output_context["image_model"] = response.model

        content: Dict[str, Any] = {
            "image": first_image,
            "images": response.images,
            "prompt": request.prompt,
            "provider": response.provider,
            "model": response.model,
        }
        if response.revised_prompt:
            content["revised_prompt"] = response.revised_prompt

        metadata = {
            "applet": IMAGE_NODE_TYPE,
            "provider": response.provider,
            "model": response.model,
            "image_count": len(response.images),
            "status": "success",
        }
        return AppletMessage(content=content, context=output_context, metadata=metadata)

    def _resolve_config(self, message: AppletMessage) -> ImageGenNodeConfigModel:
        node_data = message.metadata.get("node_data", {})
        if not isinstance(node_data, dict):
            node_data = {}

        context_config = message.context.get("image_config", {})
        if not isinstance(context_config, dict):
            context_config = {}

        metadata_config = message.metadata.get("image_config", {})
        if not isinstance(metadata_config, dict):
            metadata_config = {}

        merged = {**context_config, **metadata_config, **node_data}
        config_payload = {
            "label": merged.get("label", "Image Gen"),
            "provider": merged.get("provider", "openai"),
            "model": merged.get("model"),
            "size": merged.get("size", "1024x1024"),
            "style": merged.get("style", "photorealistic"),
            "quality": merged.get("quality", "standard"),
            "n": merged.get("n", 1),
            "response_format": merged.get("response_format", merged.get("responseFormat", "b64_json")),
            "api_key": merged.get("api_key", merged.get("apiKey")),
            "base_url": merged.get("base_url", merged.get("baseUrl")),
            "timeout_seconds": merged.get("timeout_seconds", merged.get("timeoutSeconds", 120.0)),
            "headers": merged.get("headers", {}),
            "extra": merged.get("extra", {}),
        }
        return ImageGenNodeConfigModel.model_validate(config_payload)

    def _build_request(
        self,
        message: AppletMessage,
        config: ImageGenNodeConfigModel,
        provider: ImageProviderAdapter,
    ) -> ImageGenRequestModel:
        prompt_text = _as_text(message.content).strip()
        if not prompt_text:
            prompt_text = "A beautiful landscape with mountains and a lake."

        negative_prompt = ""
        if isinstance(message.context, dict):
            raw_negative = message.context.get("negative_prompt", message.context.get("negativePrompt", ""))
            if isinstance(raw_negative, str):
                negative_prompt = raw_negative

        model_id = config.model or provider.default_model()
        return ImageGenRequestModel(
            prompt=prompt_text,
            negative_prompt=negative_prompt,
            model=model_id,
            size=config.size,
            style=config.style,
            quality=config.quality,
            n=config.n,
            response_format=config.response_format,
            extra=dict(config.extra),
        )


class HTTPRequestNodeApplet(BaseApplet):
    """Universal HTTP request node for calling external APIs."""

    VERSION = "1.0.0"
    CAPABILITIES = [
        "http-requests",
        "api-integration",
        "templated-headers",
        "templated-body",
    ]

    async def on_message(self, message: AppletMessage) -> AppletMessage:
        try:
            config = self._resolve_config(message)
        except Exception as exc:
            return AppletMessage(
                content={"error": f"Invalid HTTP request configuration: {exc}"},
                context=message.context,
                metadata={"applet": HTTP_REQUEST_NODE_TYPE, "status": "error"},
            )

        template_data = self._template_data(message)
        rendered_url_value = _render_template_payload(config.url, template_data)
        rendered_url = str(rendered_url_value).strip()
        if not rendered_url:
            return AppletMessage(
                content={"error": "HTTP request URL resolved to an empty value"},
                context=message.context,
                metadata={"applet": HTTP_REQUEST_NODE_TYPE, "status": "error"},
            )

        rendered_headers = self._normalize_headers(
            _render_template_payload(config.headers, template_data)
        )
        rendered_query_params = self._normalize_query_params(
            _render_template_payload(config.query_params, template_data)
        )

        body_template = config.body_template
        if body_template is None:
            body_template = self._default_body_template(message, config.method)
        rendered_body = (
            _render_template_payload(body_template, template_data)
            if body_template is not None
            else None
        )

        request_kwargs: Dict[str, Any] = {"headers": rendered_headers}
        if rendered_query_params:
            request_kwargs["params"] = rendered_query_params
        if config.body_type != "none" and rendered_body is not None:
            self._apply_body_payload(config.body_type, rendered_body, request_kwargs)
        request_kwargs.update(dict(config.extra))

        try:
            async with httpx.AsyncClient(
                timeout=config.timeout_seconds,
                follow_redirects=config.allow_redirects,
                verify=config.verify_ssl,
            ) as client:
                response = await client.request(config.method, rendered_url, **request_kwargs)
        except httpx.HTTPError as exc:
            logger.error("HTTP request node error: %s", exc)
            return AppletMessage(
                content={
                    "error": str(exc),
                    "url": rendered_url,
                    "method": config.method,
                },
                context=message.context,
                metadata={
                    "applet": HTTP_REQUEST_NODE_TYPE,
                    "status": "error",
                    "method": config.method,
                    "url": rendered_url,
                },
            )

        parsed_data = self._parse_response_data(response)
        response_headers: Dict[str, str] = (
            {key: value for key, value in response.headers.items()}
            if config.include_response_headers
            else {}
        )

        output_content: Dict[str, Any] = {
            "status_code": response.status_code,
            "ok": response.is_success,
            "url": str(response.url),
            "method": config.method,
            "data": parsed_data,
            "request": {
                "url": rendered_url,
                "method": config.method,
                "headers": rendered_headers,
                "query_params": rendered_query_params,
                "body": rendered_body,
            },
        }
        if response_headers:
            output_content["headers"] = response_headers

        output_context = {**message.context}
        output_context["last_http_response"] = output_content

        metadata = {
            "applet": HTTP_REQUEST_NODE_TYPE,
            "status": "success" if response.is_success else "error",
            "method": config.method,
            "status_code": response.status_code,
            "url": str(response.url),
        }
        return AppletMessage(content=output_content, context=output_context, metadata=metadata)

    def _resolve_config(self, message: AppletMessage) -> HTTPRequestNodeConfigModel:
        node_data = message.metadata.get("node_data", {})
        if not isinstance(node_data, dict):
            node_data = {}

        context_config = message.context.get("http_request_config", {})
        if not isinstance(context_config, dict):
            context_config = {}
        legacy_context_config = message.context.get("http_config", {})
        if isinstance(legacy_context_config, dict):
            context_config = {**legacy_context_config, **context_config}

        metadata_config = message.metadata.get("http_request_config", {})
        if not isinstance(metadata_config, dict):
            metadata_config = {}
        legacy_metadata_config = message.metadata.get("http_config", {})
        if isinstance(legacy_metadata_config, dict):
            metadata_config = {**legacy_metadata_config, **metadata_config}

        merged = {**context_config, **metadata_config, **node_data}
        config_payload = {
            "label": merged.get("label", "HTTP Request"),
            "url": merged.get("url"),
            "method": merged.get("method", "GET"),
            "headers": merged.get("headers", {}),
            "query_params": merged.get("query_params", merged.get("queryParams", merged.get("params", {}))),
            "body_template": merged.get("body_template", merged.get("bodyTemplate", merged.get("body"))),
            "body_type": merged.get("body_type", merged.get("bodyType", "auto")),
            "timeout_seconds": merged.get("timeout_seconds", merged.get("timeoutSeconds", 30.0)),
            "allow_redirects": merged.get("allow_redirects", merged.get("allowRedirects", True)),
            "verify_ssl": merged.get("verify_ssl", merged.get("verifySSL", True)),
            "include_response_headers": merged.get(
                "include_response_headers",
                merged.get("includeResponseHeaders", True),
            ),
            "extra": merged.get("extra", {}),
        }
        return HTTPRequestNodeConfigModel.model_validate(config_payload)

    def _template_data(self, message: AppletMessage) -> Dict[str, Any]:
        context = message.context if isinstance(message.context, dict) else {}
        results = context.get("results", {})
        if not isinstance(results, dict):
            results = {}
        return {
            "input": message.content,
            "content": message.content,
            "context": context,
            "results": results,
            "run_id": context.get("run_id", message.metadata.get("run_id")),
            "node_id": message.metadata.get("node_id"),
        }

    def _normalize_headers(self, raw_headers: Any) -> Dict[str, str]:
        if not isinstance(raw_headers, dict):
            return {}
        headers: Dict[str, str] = {}
        for key, value in raw_headers.items():
            if value is None:
                continue
            key_str = str(key).strip()
            if not key_str:
                continue
            if isinstance(value, (dict, list)):
                headers[key_str] = json.dumps(value, ensure_ascii=False)
            else:
                headers[key_str] = str(value)
        return headers

    def _normalize_query_params(self, raw_query_params: Any) -> Dict[str, Any]:
        if not isinstance(raw_query_params, dict):
            return {}
        query_params: Dict[str, Any] = {}
        for key, value in raw_query_params.items():
            key_str = str(key).strip()
            if not key_str or value is None:
                continue
            query_params[key_str] = value
        return query_params

    def _default_body_template(self, message: AppletMessage, method: str) -> Optional[Any]:
        if method not in {"POST", "PUT", "DELETE"}:
            return None
        if message.content is None:
            return None
        return message.content

    def _apply_body_payload(
        self,
        body_type: str,
        body: Any,
        request_kwargs: Dict[str, Any],
    ) -> None:
        if body_type == "json" or (body_type == "auto" and isinstance(body, (dict, list))):
            request_kwargs["json"] = body
            return

        if body_type == "form":
            if isinstance(body, dict):
                request_kwargs["data"] = {
                    str(key): (
                        json.dumps(value, ensure_ascii=False)
                        if isinstance(value, (dict, list))
                        else str(value)
                    )
                    for key, value in body.items()
                }
            else:
                request_kwargs["data"] = {"value": str(body)}
            return

        if body_type == "none":
            return

        if isinstance(body, (dict, list)):
            request_kwargs["content"] = json.dumps(body, ensure_ascii=False)
        else:
            request_kwargs["content"] = str(body)

    def _parse_response_data(self, response: httpx.Response) -> Any:
        content_type = response.headers.get("content-type", "").lower()
        if "application/json" in content_type:
            try:
                return response.json()
            except Exception:
                pass

        text_body = response.text
        if not text_body:
            return ""
        try:
            return json.loads(text_body)
        except Exception:
            return text_body


class TransformNodeApplet(BaseApplet):
    """Config-driven data transformation node."""

    VERSION = "1.0.0"
    CAPABILITIES = [
        "json-path-extract",
        "template-string",
        "regex-replace",
        "split-join",
        "config-driven-transform",
    ]

    async def on_message(self, message: AppletMessage) -> AppletMessage:
        try:
            config = self._resolve_config(message)
        except Exception as exc:
            return AppletMessage(
                content={"error": f"Invalid transform configuration: {exc}"},
                context=message.context,
                metadata={"applet": TRANSFORM_NODE_TYPE, "status": "error"},
            )

        template_data = self._template_data(message)
        source_value = _render_template_payload(config.source, template_data)

        try:
            output_value = self._apply_transform(
                config=config,
                source_value=source_value,
                template_data=template_data,
            )
        except Exception as exc:
            return AppletMessage(
                content={
                    "ok": False,
                    "operation": config.operation,
                    "input": source_value,
                    "error": str(exc),
                },
                context=message.context,
                metadata={
                    "applet": TRANSFORM_NODE_TYPE,
                    "status": "error",
                    "operation": config.operation,
                },
            )

        output_content = {
            "ok": True,
            "operation": config.operation,
            "input": source_value,
            "output": output_value,
        }
        output_context = {**message.context, "last_transform_response": output_content}
        output_metadata = {
            "applet": TRANSFORM_NODE_TYPE,
            "status": "success",
            "operation": config.operation,
        }
        return AppletMessage(content=output_content, context=output_context, metadata=output_metadata)

    def _resolve_config(self, message: AppletMessage) -> TransformNodeConfigModel:
        node_data = message.metadata.get("node_data", {})
        if not isinstance(node_data, dict):
            node_data = {}

        context_config = message.context.get("transform_config", {})
        if not isinstance(context_config, dict):
            context_config = {}
        legacy_context_config = message.context.get("transform", {})
        if isinstance(legacy_context_config, dict):
            context_config = {**legacy_context_config, **context_config}

        metadata_config = message.metadata.get("transform_config", {})
        if not isinstance(metadata_config, dict):
            metadata_config = {}
        legacy_metadata_config = message.metadata.get("transform", {})
        if isinstance(legacy_metadata_config, dict):
            metadata_config = {**legacy_metadata_config, **metadata_config}

        merged = {**context_config, **metadata_config, **node_data}
        config_payload = {
            "label": merged.get("label", "Transform"),
            "operation": merged.get("operation", "template"),
            "source": merged.get("source", merged.get("input", "{{content}}")),
            "json_path": merged.get("json_path", merged.get("jsonPath", merged.get("path", "$"))),
            "template": merged.get(
                "template",
                merged.get("template_string", merged.get("templateString", "{{source}}")),
            ),
            "regex_pattern": merged.get("regex_pattern", merged.get("regexPattern", merged.get("pattern", ""))),
            "regex_replacement": merged.get(
                "regex_replacement",
                merged.get("regexReplacement", merged.get("replacement", "")),
            ),
            "regex_flags": merged.get("regex_flags", merged.get("regexFlags", "")),
            "regex_count": merged.get("regex_count", merged.get("regexCount", 0)),
            "split_delimiter": merged.get(
                "split_delimiter",
                merged.get("splitDelimiter", merged.get("delimiter", ",")),
            ),
            "split_maxsplit": merged.get(
                "split_maxsplit",
                merged.get("splitMaxsplit", merged.get("maxsplit", -1)),
            ),
            "split_index": merged.get("split_index", merged.get("splitIndex")),
            "join_delimiter": merged.get(
                "join_delimiter",
                merged.get("joinDelimiter", merged.get("joiner", ",")),
            ),
            "return_list": merged.get("return_list", merged.get("returnList", False)),
            "strip_items": merged.get("strip_items", merged.get("stripItems", False)),
            "drop_empty": merged.get("drop_empty", merged.get("dropEmpty", False)),
            "extra": merged.get("extra", {}),
        }
        return TransformNodeConfigModel.model_validate(config_payload)

    def _template_data(self, message: AppletMessage) -> Dict[str, Any]:
        context = message.context if isinstance(message.context, dict) else {}
        results = context.get("results", {})
        if not isinstance(results, dict):
            results = {}
        return {
            "input": message.content,
            "content": message.content,
            "context": context,
            "results": results,
            "metadata": message.metadata,
            "run_id": context.get("run_id", message.metadata.get("run_id")),
            "node_id": message.metadata.get("node_id"),
        }

    def _apply_transform(
        self,
        config: TransformNodeConfigModel,
        source_value: Any,
        template_data: Dict[str, Any],
    ) -> Any:
        if config.operation == "json_path":
            return self._transform_json_path(source_value, config.json_path)
        if config.operation == "template":
            return self._transform_template(config.template, source_value, template_data)
        if config.operation == "regex_replace":
            return self._transform_regex_replace(
                source_value=source_value,
                pattern=config.regex_pattern,
                replacement=config.regex_replacement,
                flags_text=config.regex_flags,
                count=config.regex_count,
            )
        if config.operation == "split_join":
            return self._transform_split_join(source_value, config)
        raise ValueError(f"Unsupported transform operation: {config.operation}")

    def _transform_json_path(self, source_value: Any, json_path: str) -> Any:
        result, found = _resolve_json_path(source_value, json_path)
        if not found:
            raise ValueError(f"json_path not found: {json_path}")
        return result

    def _transform_template(
        self,
        template: str,
        source_value: Any,
        template_data: Dict[str, Any],
    ) -> Any:
        scope = {
            **template_data,
            "source": source_value,
            "value": source_value,
        }
        return _render_template_payload(template, scope)

    def _transform_regex_replace(
        self,
        source_value: Any,
        pattern: str,
        replacement: str,
        flags_text: str,
        count: int,
    ) -> str:
        if not pattern:
            raise ValueError("regex_pattern is required for regex_replace operation")

        flags = 0
        if "i" in flags_text:
            flags |= re.IGNORECASE
        if "m" in flags_text:
            flags |= re.MULTILINE
        if "s" in flags_text:
            flags |= re.DOTALL
        if "x" in flags_text:
            flags |= re.VERBOSE

        compiled = re.compile(pattern, flags=flags)
        source_text = _as_text(source_value)
        return compiled.sub(replacement, source_text, count=count)

    def _transform_split_join(
        self,
        source_value: Any,
        config: TransformNodeConfigModel,
    ) -> Any:
        if isinstance(source_value, list):
            parts = [_as_text(item) for item in source_value]
        else:
            source_text = _as_text(source_value)
            if config.split_delimiter == "":
                parts = list(source_text)
            else:
                maxsplit = config.split_maxsplit
                parts = source_text.split(config.split_delimiter, maxsplit)

        normalized: List[str] = []
        for item in parts:
            value = item.strip() if config.strip_items else item
            if config.drop_empty and value == "":
                continue
            normalized.append(value)

        if config.split_index is not None:
            if config.split_index >= len(normalized):
                raise ValueError("split_index is out of range")
            return normalized[config.split_index]

        if config.return_list:
            return normalized

        return config.join_delimiter.join(normalized)


class IfElseNodeApplet(BaseApplet):
    """Conditional routing node that evaluates data and selects a true/false branch."""

    VERSION = "1.0.0"
    CAPABILITIES = [
        "conditional-routing",
        "contains-condition",
        "equals-condition",
        "regex-condition",
        "json-path-condition",
    ]

    async def on_message(self, message: AppletMessage) -> AppletMessage:
        base_context = message.context if isinstance(message.context, dict) else {}

        try:
            config = self._resolve_config(message)
        except Exception as exc:
            error_content = {
                "ok": False,
                "operation": "unknown",
                "result": False,
                "branch": "false",
                "error": f"Invalid if/else configuration: {exc}",
            }
            return AppletMessage(
                content=error_content,
                context={**base_context, "last_if_else_response": error_content},
                metadata={
                    "applet": IF_ELSE_NODE_TYPE,
                    "status": "error",
                    "operation": "unknown",
                    "branch": "false",
                    "condition_result": False,
                },
            )

        template_data = self._template_data(message)
        source_value = _render_template_payload(config.source, template_data)
        expected_value = (
            _render_template_payload(config.value, template_data)
            if config.value is not None
            else None
        )

        try:
            result, details = self._evaluate_condition(config, source_value, expected_value)
            if config.negate:
                result = not result
                details["negated"] = True
        except Exception as exc:
            error_content = {
                "ok": False,
                "operation": config.operation,
                "source": source_value,
                "value": expected_value,
                "result": False,
                "branch": "false",
                "error": str(exc),
            }
            return AppletMessage(
                content=error_content,
                context={**base_context, "last_if_else_response": error_content},
                metadata={
                    "applet": IF_ELSE_NODE_TYPE,
                    "status": "error",
                    "operation": config.operation,
                    "branch": "false",
                    "condition_result": False,
                },
            )

        branch = "true" if result else "false"
        output_content = {
            "ok": True,
            "operation": config.operation,
            "source": source_value,
            "value": expected_value,
            "result": result,
            "branch": branch,
            "details": details,
        }
        output_context = {**base_context, "last_if_else_response": output_content}
        output_metadata = {
            "applet": IF_ELSE_NODE_TYPE,
            "status": "success",
            "operation": config.operation,
            "branch": branch,
            "condition_result": result,
        }
        return AppletMessage(content=output_content, context=output_context, metadata=output_metadata)

    def _resolve_config(self, message: AppletMessage) -> IfElseNodeConfigModel:
        context = message.context if isinstance(message.context, dict) else {}
        node_data = message.metadata.get("node_data", {})
        if not isinstance(node_data, dict):
            node_data = {}

        context_config: Dict[str, Any] = {}
        for key in ("if_else_config", "ifelse_config", "condition_config", "if_config"):
            candidate = context.get(key)
            if isinstance(candidate, dict):
                context_config = {**context_config, **candidate}

        metadata_config: Dict[str, Any] = {}
        for key in ("if_else_config", "ifelse_config", "condition_config", "if_config"):
            candidate = message.metadata.get(key)
            if isinstance(candidate, dict):
                metadata_config = {**metadata_config, **candidate}

        merged = {**context_config, **metadata_config, **node_data}
        config_payload = {
            "label": merged.get("label", "If / Else"),
            "operation": merged.get(
                "operation",
                merged.get("condition", merged.get("match_type", "equals")),
            ),
            "source": merged.get(
                "source",
                merged.get("left", merged.get("input", "{{content}}")),
            ),
            "value": merged.get(
                "value",
                merged.get("expected", merged.get("right")),
            ),
            "case_sensitive": merged.get(
                "case_sensitive",
                merged.get("caseSensitive", False),
            ),
            "negate": merged.get("negate", merged.get("not", False)),
            "regex_pattern": merged.get(
                "regex_pattern",
                merged.get("regexPattern", merged.get("pattern", "")),
            ),
            "regex_flags": merged.get("regex_flags", merged.get("regexFlags", "")),
            "json_path": merged.get("json_path", merged.get("jsonPath", merged.get("path", "$"))),
            "true_target": merged.get(
                "true_target",
                merged.get("trueTarget", merged.get("on_true")),
            ),
            "false_target": merged.get(
                "false_target",
                merged.get("falseTarget", merged.get("on_false")),
            ),
            "extra": merged.get("extra", {}),
        }
        return IfElseNodeConfigModel.model_validate(config_payload)

    def _template_data(self, message: AppletMessage) -> Dict[str, Any]:
        context = message.context if isinstance(message.context, dict) else {}
        results = context.get("results", {})
        if not isinstance(results, dict):
            results = {}
        return {
            "input": message.content,
            "content": message.content,
            "context": context,
            "results": results,
            "metadata": message.metadata,
            "run_id": context.get("run_id", message.metadata.get("run_id")),
            "node_id": message.metadata.get("node_id"),
        }

    def _evaluate_condition(
        self,
        config: IfElseNodeConfigModel,
        source_value: Any,
        expected_value: Any,
    ) -> tuple[bool, Dict[str, Any]]:
        if config.operation == "contains":
            matched = self._evaluate_contains(
                source_value=source_value,
                expected_value=expected_value,
                case_sensitive=config.case_sensitive,
            )
            return matched, {"mode": "contains"}

        if config.operation == "equals":
            matched = self._evaluate_equals(
                left=source_value,
                right=expected_value,
                case_sensitive=config.case_sensitive,
            )
            return matched, {"mode": "equals"}

        if config.operation == "regex":
            pattern = config.regex_pattern
            if not pattern and expected_value is not None:
                pattern = _as_text(expected_value)
            if not pattern:
                raise ValueError("regex_pattern is required for regex operation")

            flags = self._compile_regex_flags(config.regex_flags)
            source_text = _as_text(source_value)
            matched = re.search(pattern, source_text, flags=flags) is not None
            return matched, {"mode": "regex", "pattern": pattern}

        if config.operation == "json_path":
            matched_value, found = _resolve_json_path(source_value, config.json_path)
            if not found:
                return False, {"mode": "json_path", "found": False}
            if expected_value is None:
                return True, {"mode": "json_path", "found": True, "value": matched_value}
            matched = self._evaluate_equals(
                left=matched_value,
                right=expected_value,
                case_sensitive=config.case_sensitive,
            )
            return matched, {
                "mode": "json_path",
                "found": True,
                "value": matched_value,
            }

        raise ValueError(f"Unsupported if/else operation: {config.operation}")

    def _evaluate_contains(
        self,
        source_value: Any,
        expected_value: Any,
        case_sensitive: bool,
    ) -> bool:
        if expected_value is None:
            return False

        if isinstance(source_value, dict):
            if expected_value in source_value:
                return True
            return expected_value in source_value.values()

        if isinstance(source_value, (list, tuple, set)):
            return expected_value in source_value

        source_text = _as_text(source_value)
        expected_text = _as_text(expected_value)
        if not case_sensitive:
            source_text = source_text.lower()
            expected_text = expected_text.lower()
        return expected_text in source_text

    def _evaluate_equals(
        self,
        left: Any,
        right: Any,
        case_sensitive: bool,
    ) -> bool:
        if isinstance(left, str) and isinstance(right, str) and not case_sensitive:
            return left.lower() == right.lower()
        return left == right

    def _compile_regex_flags(self, flags_text: str) -> int:
        flags = 0
        if "i" in flags_text:
            flags |= re.IGNORECASE
        if "m" in flags_text:
            flags |= re.MULTILINE
        if "s" in flags_text:
            flags |= re.DOTALL
        if "x" in flags_text:
            flags |= re.VERBOSE
        return flags


class MergeNodeApplet(BaseApplet):
    """Fan-in node that merges multiple upstream branch outputs."""

    VERSION = "1.0.0"
    CAPABILITIES = [
        "fan-in",
        "merge-array",
        "merge-concatenate",
        "merge-first-wins",
    ]

    async def on_message(self, message: AppletMessage) -> AppletMessage:
        base_context = message.context if isinstance(message.context, dict) else {}

        try:
            config = self._resolve_config(message)
        except Exception as exc:
            error_content = {
                "ok": False,
                "strategy": "unknown",
                "output": None,
                "count": 0,
                "error": f"Invalid merge configuration: {exc}",
            }
            return AppletMessage(
                content=error_content,
                context={**base_context, "last_merge_response": error_content},
                metadata={"applet": MERGE_NODE_TYPE, "status": "error"},
            )

        inputs = self._normalize_inputs(message.content)
        merged_output = self._merge_inputs(
            strategy=config.strategy,
            inputs=inputs,
            delimiter=config.delimiter,
        )

        output_content = {
            "ok": True,
            "strategy": config.strategy,
            "count": len(inputs),
            "inputs": inputs,
            "output": merged_output,
        }
        output_context = {**base_context, "last_merge_response": output_content}
        output_metadata = {
            "applet": MERGE_NODE_TYPE,
            "status": "success",
            "strategy": config.strategy,
            "input_count": len(inputs),
        }
        return AppletMessage(content=output_content, context=output_context, metadata=output_metadata)

    def _resolve_config(self, message: AppletMessage) -> MergeNodeConfigModel:
        context = message.context if isinstance(message.context, dict) else {}
        node_data = message.metadata.get("node_data", {})
        if not isinstance(node_data, dict):
            node_data = {}

        context_config = context.get("merge_config", {})
        if not isinstance(context_config, dict):
            context_config = {}

        metadata_config = message.metadata.get("merge_config", {})
        if not isinstance(metadata_config, dict):
            metadata_config = {}

        merged = {**context_config, **metadata_config, **node_data}
        config_payload = {
            "label": merged.get("label", "Merge"),
            "strategy": merged.get("strategy", merged.get("merge_strategy", merged.get("mergeStrategy", "array"))),
            "delimiter": merged.get("delimiter", merged.get("join_delimiter", merged.get("joinDelimiter", "\n"))),
            "extra": merged.get("extra", {}),
        }
        return MergeNodeConfigModel.model_validate(config_payload)

    def _normalize_inputs(self, content: Any) -> List[Any]:
        if isinstance(content, dict):
            raw_inputs = content.get("inputs")
            if isinstance(raw_inputs, list):
                return list(raw_inputs)
            if raw_inputs is not None:
                return [raw_inputs]
            if "input" in content:
                return [content.get("input")]
            return [content]

        if isinstance(content, list):
            return list(content)
        if content is None:
            return []
        return [content]

    def _merge_inputs(
        self,
        strategy: str,
        inputs: List[Any],
        delimiter: str,
    ) -> Any:
        if strategy == "first_wins":
            return inputs[0] if inputs else None
        if strategy == "concatenate":
            return delimiter.join(_as_text(item) for item in inputs)
        return inputs


class CodeNodeApplet(BaseApplet):
    """Sandboxed code execution node for Python and JavaScript."""

    VERSION = "1.0.0"
    CAPABILITIES = [
        "code-execution",
        "python",
        "javascript",
        "sandboxed-subprocess",
        "resource-limits",
    ]

    async def on_message(self, message: AppletMessage) -> AppletMessage:
        try:
            config = self._resolve_config(message)
        except Exception as exc:
            return AppletMessage(
                content={"error": f"Invalid code node configuration: {exc}"},
                context=message.context,
                metadata={"applet": CODE_NODE_TYPE, "status": "error"},
            )

        code_text = config.code
        if not code_text.strip() and isinstance(message.content, dict):
            raw_code = message.content.get("code")
            if isinstance(raw_code, str):
                code_text = raw_code

        if not code_text.strip():
            return AppletMessage(
                content={"error": "No code provided"},
                context=message.context,
                metadata={"applet": CODE_NODE_TYPE, "status": "error"},
            )

        started_at = time.perf_counter()
        execution_result = await self._execute_sandboxed_code(message, config, code_text)
        duration_ms = round((time.perf_counter() - started_at) * 1000.0, 3)

        output_content = {
            **execution_result,
            "language": config.language,
            "duration_ms": duration_ms,
        }
        output_context = {**message.context, "last_code_response": output_content}
        output_metadata = {
            "applet": CODE_NODE_TYPE,
            "status": "success" if execution_result.get("ok") else "error",
            "language": config.language,
            "timed_out": execution_result.get("timed_out", False),
            "exit_code": execution_result.get("exit_code"),
            "duration_ms": duration_ms,
        }
        return AppletMessage(content=output_content, context=output_context, metadata=output_metadata)

    def _resolve_config(self, message: AppletMessage) -> CodeNodeConfigModel:
        node_data = message.metadata.get("node_data", {})
        if not isinstance(node_data, dict):
            node_data = {}

        context_config = message.context.get("code_config", {})
        if not isinstance(context_config, dict):
            context_config = {}

        metadata_config = message.metadata.get("code_config", {})
        if not isinstance(metadata_config, dict):
            metadata_config = {}

        merged = {**context_config, **metadata_config, **node_data}
        config_payload = {
            "label": merged.get("label", "Code"),
            "language": merged.get("language", merged.get("runtime", "python")),
            "code": merged.get("code", ""),
            "timeout_seconds": merged.get("timeout_seconds", merged.get("timeoutSeconds", 5.0)),
            "cpu_time_seconds": merged.get("cpu_time_seconds", merged.get("cpuTimeSeconds", 3)),
            "memory_limit_mb": merged.get("memory_limit_mb", merged.get("memoryLimitMb", 256)),
            "max_output_bytes": merged.get("max_output_bytes", merged.get("maxOutputBytes", 262144)),
            "working_dir": merged.get("working_dir", merged.get("workingDir", "/tmp")),
            "env": merged.get("env", {}),
            "extra": merged.get("extra", {}),
        }
        return CodeNodeConfigModel.model_validate(config_payload)

    async def _execute_sandboxed_code(
        self,
        message: AppletMessage,
        config: CodeNodeConfigModel,
        code_text: str,
    ) -> Dict[str, Any]:
        sandbox_dir = tempfile.mkdtemp(prefix="synapps-code-", dir="/tmp")
        requested_workdir = _safe_tmp_dir(config.working_dir)
        if requested_workdir == "/tmp":
            workdir = sandbox_dir
        else:
            workdir = requested_workdir
            Path(workdir).mkdir(parents=True, exist_ok=True)

        if config.language == "python":
            runner_path = Path(sandbox_dir) / "sandbox_runner.py"
            runner_path.write_text(PYTHON_CODE_WRAPPER, encoding="utf-8")
            executable = os.environ.get("CODE_NODE_PYTHON_BIN") or sys.executable or "python3"
            command = [executable, "-I", "-B", str(runner_path)]
        else:
            runner_path = Path(sandbox_dir) / "sandbox_runner.js"
            runner_path.write_text(JAVASCRIPT_CODE_WRAPPER, encoding="utf-8")
            executable = os.environ.get("CODE_NODE_NODE_BIN") or "node"
            command = [executable, str(runner_path)]

        payload = {
            "code": code_text,
            "data": message.content,
            "context": message.context,
            "metadata": message.metadata,
            "exec_timeout_ms": max(1, int(config.timeout_seconds * 1000)),
        }
        payload_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": "/tmp",
            "TMPDIR": "/tmp",
            "TMP": "/tmp",
            "TEMP": "/tmp",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUNBUFFERED": "1",
        }
        if isinstance(config.env, dict):
            for key, value in config.env.items():
                env[str(key)] = str(value)

        effective_memory_limit_mb = config.memory_limit_mb
        if config.language == "javascript":
            effective_memory_limit_mb = max(effective_memory_limit_mb, 768)

        preexec_fn = _sandbox_preexec_fn(
            cpu_time_seconds=config.cpu_time_seconds,
            memory_limit_mb=effective_memory_limit_mb,
            max_output_bytes=config.max_output_bytes,
        )

        timed_out = False
        stdout_text = ""
        stderr_text = ""
        stdout_truncated = False
        stderr_truncated = False
        return_code: Optional[int] = None

        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=workdir,
                env=env,
                preexec_fn=preexec_fn,
            )
        except FileNotFoundError:
            return {
                "ok": False,
                "timed_out": False,
                "exit_code": None,
                "stdout": "",
                "stderr": f"Runtime not found: {command[0]}",
                "result": None,
                "error": {"message": f"Runtime not found: {command[0]}"},
                "stdout_truncated": False,
                "stderr_truncated": False,
            }

        stdout_task = asyncio.create_task(_read_stream_limited(process.stdout, config.max_output_bytes))
        stderr_task = asyncio.create_task(_read_stream_limited(process.stderr, config.max_output_bytes))

        try:
            if process.stdin is not None:
                process.stdin.write(payload_bytes)
                await process.stdin.drain()
                process.stdin.close()

            await asyncio.wait_for(process.wait(), timeout=config.timeout_seconds)
            return_code = process.returncode
        except asyncio.TimeoutError:
            timed_out = True
            process.kill()
            await process.wait()
            return_code = process.returncode
        finally:
            stdout_bytes, stdout_truncated = await stdout_task
            stderr_bytes, stderr_truncated = await stderr_task
            stdout_text = stdout_bytes.decode("utf-8", errors="replace")
            stderr_text = stderr_bytes.decode("utf-8", errors="replace")

        if return_code in (-9, -24):
            timed_out = True

        cleaned_stdout, wrapper_payload = _extract_sandbox_result(stdout_text)
        wrapper_ok = bool(wrapper_payload and wrapper_payload.get("ok"))
        wrapper_error = wrapper_payload.get("error") if isinstance(wrapper_payload, dict) else None

        result_payload = {
            "ok": (not timed_out) and return_code == 0 and wrapper_ok,
            "timed_out": timed_out,
            "exit_code": return_code,
            "stdout": cleaned_stdout,
            "stderr": stderr_text,
            "result": wrapper_payload.get("result") if isinstance(wrapper_payload, dict) else None,
            "error": wrapper_error,
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
        }

        if timed_out and not result_payload.get("error"):
            result_payload["error"] = {"message": "Execution timed out"}
        elif return_code not in (0, None) and not result_payload.get("error"):
            result_payload["error"] = {"message": f"Process exited with code {return_code}"}
        elif wrapper_payload is None and not result_payload.get("error"):
            result_payload["error"] = {"message": "Sandbox wrapper did not return a structured result"}

        shutil.rmtree(sandbox_dir, ignore_errors=True)

        return result_payload


class ForEachNodeApplet(BaseApplet):
    """For-Each loop node that iterates over an array, executing downstream nodes per item."""

    VERSION = "1.0.0"
    CAPABILITIES = [
        "loop",
        "for-each",
        "array-iteration",
        "parallel-iteration",
        "max-iteration-limit",
    ]

    async def on_message(self, message: AppletMessage) -> AppletMessage:
        try:
            config = self._resolve_config(message)
        except Exception as exc:
            return AppletMessage(
                content={"error": f"Invalid for-each configuration: {exc}"},
                context=message.context,
                metadata={"applet": FOR_EACH_NODE_TYPE, "status": "error"},
            )

        template_data = self._template_data(message)
        resolved_source = _render_template_payload(config.array_source, template_data)

        items = self._coerce_to_list(resolved_source)
        if items is None:
            return AppletMessage(
                content={
                    "ok": False,
                    "error": "array_source did not resolve to an iterable array",
                    "resolved_value": resolved_source,
                },
                context=message.context,
                metadata={"applet": FOR_EACH_NODE_TYPE, "status": "error"},
            )

        total_items = len(items)
        effective_limit = min(total_items, config.max_iterations)
        truncated = total_items > config.max_iterations
        items = items[:effective_limit]

        node_id = message.metadata.get("node_id", "for_each")
        run_id = message.context.get("run_id", message.metadata.get("run_id"))

        if config.parallel:
            iteration_results = await self._run_parallel(
                items, message, config, node_id, run_id
            )
        else:
            iteration_results = await self._run_sequential(
                items, message, node_id, run_id
            )

        output_content = {
            "ok": True,
            "total_items": total_items,
            "iterated": len(items),
            "truncated": truncated,
            "max_iterations": config.max_iterations,
            "parallel": config.parallel,
            "results": iteration_results,
        }
        output_context = {
            **message.context,
            "last_for_each_response": output_content,
            "for_each_results": iteration_results,
        }
        output_metadata = {
            "applet": FOR_EACH_NODE_TYPE,
            "status": "success",
            "iterated": len(items),
            "truncated": truncated,
            "parallel": config.parallel,
        }
        return AppletMessage(
            content=output_content,
            context=output_context,
            metadata=output_metadata,
        )

    def _resolve_config(self, message: AppletMessage) -> ForEachNodeConfigModel:
        node_data = message.metadata.get("node_data", {})
        if not isinstance(node_data, dict):
            node_data = {}

        context_config = message.context.get("for_each_config", {})
        if not isinstance(context_config, dict):
            context_config = {}

        metadata_config = message.metadata.get("for_each_config", {})
        if not isinstance(metadata_config, dict):
            metadata_config = {}

        merged = {**context_config, **metadata_config, **node_data}
        config_payload = {
            "label": merged.get("label", "For-Each"),
            "array_source": merged.get(
                "array_source",
                merged.get("arraySource", merged.get("source", "{{input}}")),
            ),
            "max_iterations": merged.get(
                "max_iterations",
                merged.get("maxIterations", merged.get("limit", 1000)),
            ),
            "parallel": merged.get("parallel", False),
            "concurrency_limit": merged.get(
                "concurrency_limit",
                merged.get("concurrencyLimit", 10),
            ),
            "extra": merged.get("extra", {}),
        }
        return ForEachNodeConfigModel.model_validate(config_payload)

    def _template_data(self, message: AppletMessage) -> Dict[str, Any]:
        context = message.context if isinstance(message.context, dict) else {}
        results = context.get("results", {})
        if not isinstance(results, dict):
            results = {}
        return {
            "input": message.content,
            "content": message.content,
            "context": context,
            "results": results,
            "metadata": message.metadata,
            "run_id": context.get("run_id", message.metadata.get("run_id")),
            "node_id": message.metadata.get("node_id"),
        }

    @staticmethod
    def _coerce_to_list(value: Any) -> Optional[List[Any]]:
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            value = value.strip()
            if value.startswith("["):
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, list):
                        return parsed
                except (json.JSONDecodeError, ValueError):
                    pass
            return None
        if isinstance(value, dict):
            return None
        try:
            return list(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _build_iteration_message(
        item: Any,
        index: int,
        message: AppletMessage,
        node_id: str,
        run_id: Optional[str],
    ) -> AppletMessage:
        iteration_context = {
            **message.context,
            "for_each_item": item,
            "for_each_index": index,
            "for_each_node_id": node_id,
        }
        iteration_metadata = {
            **message.metadata,
            "for_each_item": item,
            "for_each_index": index,
            "parent_node_id": node_id,
        }
        return AppletMessage(
            content=item,
            context=iteration_context,
            metadata=iteration_metadata,
        )

    async def _execute_single_iteration(
        self,
        item: Any,
        index: int,
        message: AppletMessage,
        node_id: str,
        run_id: Optional[str],
    ) -> Dict[str, Any]:
        iteration_msg = self._build_iteration_message(item, index, message, node_id, run_id)

        downstream_nodes = self._get_downstream_nodes(message)

        if not downstream_nodes:
            return {"index": index, "item": item, "output": item}

        current_output: Any = item
        current_msg = iteration_msg

        for downstream in downstream_nodes:
            try:
                applet = await Orchestrator.load_applet(downstream["type"].lower())
                sub_metadata = {**current_msg.metadata}
                if "data" in downstream and isinstance(downstream["data"], dict):
                    sub_metadata["node_data"] = downstream["data"]
                sub_msg = AppletMessage(
                    content=current_msg.content,
                    context=current_msg.context,
                    metadata=sub_metadata,
                )
                response = await applet.on_message(sub_msg)
                current_output = response.content
                current_msg = response
            except Exception as exc:
                return {
                    "index": index,
                    "item": item,
                    "error": str(exc),
                    "failed_at_node": downstream.get("id", downstream.get("type")),
                }

        return {"index": index, "item": item, "output": current_output}

    def _get_downstream_nodes(self, message: AppletMessage) -> List[Dict[str, Any]]:
        node_data = message.metadata.get("node_data", {})
        if not isinstance(node_data, dict):
            return []
        sub_nodes = node_data.get("sub_nodes", node_data.get("subNodes", []))
        if isinstance(sub_nodes, list):
            return [n for n in sub_nodes if isinstance(n, dict) and "type" in n]
        return []

    async def _run_sequential(
        self,
        items: List[Any],
        message: AppletMessage,
        node_id: str,
        run_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for index, item in enumerate(items):
            result = await self._execute_single_iteration(
                item, index, message, node_id, run_id
            )
            results.append(result)
        return results

    async def _run_parallel(
        self,
        items: List[Any],
        message: AppletMessage,
        config: ForEachNodeConfigModel,
        node_id: str,
        run_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        semaphore = asyncio.Semaphore(config.concurrency_limit)

        async def _guarded(item: Any, index: int) -> Dict[str, Any]:
            async with semaphore:
                return await self._execute_single_iteration(
                    item, index, message, node_id, run_id
                )

        tasks = [_guarded(item, idx) for idx, item in enumerate(items)]
        return list(await asyncio.gather(*tasks))


applet_registry["llm"] = LLMNodeApplet
applet_registry[IMAGE_NODE_TYPE] = ImageGenNodeApplet
applet_registry[MEMORY_NODE_TYPE] = MemoryNodeApplet
applet_registry[HTTP_REQUEST_NODE_TYPE] = HTTPRequestNodeApplet
applet_registry[CODE_NODE_TYPE] = CodeNodeApplet
applet_registry[TRANSFORM_NODE_TYPE] = TransformNodeApplet
applet_registry[IF_ELSE_NODE_TYPE] = IfElseNodeApplet
applet_registry[MERGE_NODE_TYPE] = MergeNodeApplet
applet_registry[FOR_EACH_NODE_TYPE] = ForEachNodeApplet


# ============================================================
# Orchestrator Core
# ============================================================

class Orchestrator:
    """Core orchestration engine that executes applet flows."""

    @staticmethod
    async def load_applet(applet_type: str) -> BaseApplet:
        """Dynamically load an applet by type."""
        normalized_type = applet_type.strip().lower()
        if normalized_type in applet_registry:
            return applet_registry[normalized_type]()

        if normalized_type == LLM_NODE_TYPE:
            applet_registry[LLM_NODE_TYPE] = LLMNodeApplet
            return LLMNodeApplet()

        if normalized_type in {IMAGE_NODE_TYPE, "image_gen", "image-gen"}:
            applet_registry[IMAGE_NODE_TYPE] = ImageGenNodeApplet
            return ImageGenNodeApplet()

        if normalized_type in {MEMORY_NODE_TYPE, "memory_node", "memory-node"}:
            applet_registry[MEMORY_NODE_TYPE] = MemoryNodeApplet
            return MemoryNodeApplet()

        if normalized_type in {
            HTTP_REQUEST_NODE_TYPE,
            "http-request",
            "httprequest",
            "http_request_node",
            "http",
        }:
            applet_registry[HTTP_REQUEST_NODE_TYPE] = HTTPRequestNodeApplet
            return HTTPRequestNodeApplet()

        if normalized_type in {
            CODE_NODE_TYPE,
            "code-node",
            "code_node",
            "code_execution",
            "code-execution",
        }:
            applet_registry[CODE_NODE_TYPE] = CodeNodeApplet
            return CodeNodeApplet()

        if normalized_type in {
            TRANSFORM_NODE_TYPE,
            "transform-node",
            "transform_node",
            "transformer",
            "data_transform",
            "data-transform",
        }:
            applet_registry[TRANSFORM_NODE_TYPE] = TransformNodeApplet
            return TransformNodeApplet()

        if normalized_type in {
            IF_ELSE_NODE_TYPE,
            "ifelse",
            "if-else",
            "conditional",
            "condition",
            "condition_node",
            "condition-node",
        }:
            applet_registry[IF_ELSE_NODE_TYPE] = IfElseNodeApplet
            return IfElseNodeApplet()

        if normalized_type in {
            MERGE_NODE_TYPE,
            "fan_in",
            "fan-in",
            "merge_node",
            "merge-node",
            "fanin",
        }:
            applet_registry[MERGE_NODE_TYPE] = MergeNodeApplet
            return MergeNodeApplet()

        if normalized_type in {
            FOR_EACH_NODE_TYPE,
            "foreach",
            "for-each",
            "for_each_node",
            "for-each-node",
            "loop",
        }:
            applet_registry[FOR_EACH_NODE_TYPE] = ForEachNodeApplet
            return ForEachNodeApplet()

        try:
            module_path = f"apps.applets.{normalized_type}.applet"
            module = importlib.import_module(module_path)
            applet_class = getattr(module, f"{normalized_type.capitalize()}Applet")
            applet_registry[normalized_type] = applet_class
            return applet_class()
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to load applet '{normalized_type}': {e}")
            raise ValueError(f"Applet type '{normalized_type}' not found")

    @staticmethod
    def create_run_id() -> str:
        """Generate a unique run ID."""
        return str(uuid.uuid4())

    @staticmethod
    def _collect_outgoing_targets(outgoing_edges: List[Dict[str, Any]]) -> List[str]:
        """Collect unique target ids from outgoing edges while preserving order."""
        targets: List[str] = []
        for edge in outgoing_edges:
            target = edge.get("target")
            if isinstance(target, str) and target and target not in targets:
                targets.append(target)
        return targets

    @staticmethod
    def _branch_from_hint(hint: Any) -> Optional[str]:
        """Infer a branch from an arbitrary string hint."""
        if hint is None:
            return None
        text = str(hint).strip().lower()
        if not text:
            return None

        normalized = text.replace("-", "_")
        if normalized in _TRUE_BRANCH_HINTS:
            return "true"
        if normalized in _FALSE_BRANCH_HINTS:
            return "false"

        tokens = [token for token in re.split(r"[^a-z0-9]+", normalized) if token]
        has_true = any(token in _TRUE_BRANCH_HINTS for token in tokens)
        has_false = any(token in _FALSE_BRANCH_HINTS for token in tokens)
        if has_true and not has_false:
            return "true"
        if has_false and not has_true:
            return "false"
        return None

    @staticmethod
    def _extract_if_else_branch(response: Optional[AppletMessage]) -> str:
        """Resolve true/false routing branch from an if/else applet response."""
        if response is None:
            return "false"

        if isinstance(response.metadata, dict):
            branch = Orchestrator._branch_from_hint(response.metadata.get("branch"))
            if branch:
                return branch
            raw_result = response.metadata.get("condition_result")
            branch = Orchestrator._branch_from_hint(raw_result)
            if branch:
                return branch
            if isinstance(raw_result, bool):
                return "true" if raw_result else "false"
            if isinstance(raw_result, (int, float)):
                return "true" if raw_result != 0 else "false"

        if isinstance(response.content, dict):
            branch = Orchestrator._branch_from_hint(response.content.get("branch"))
            if branch:
                return branch
            raw_result = response.content.get("result")
            branch = Orchestrator._branch_from_hint(raw_result)
            if branch:
                return branch
            if isinstance(raw_result, bool):
                return "true" if raw_result else "false"
            if isinstance(raw_result, (int, float)):
                return "true" if raw_result != 0 else "false"

        return "false"

    @staticmethod
    def _branch_target_from_node_data(node: Dict[str, Any], branch: str) -> Optional[str]:
        """Read explicit branch target ids from if/else node data."""
        node_data = node.get("data", {})
        if not isinstance(node_data, dict):
            return None

        keys = (
            ("true_target", "trueTarget", "on_true", "onTrue", "then_target", "thenTarget")
            if branch == "true"
            else ("false_target", "falseTarget", "on_false", "onFalse", "else_target", "elseTarget")
        )
        for key in keys:
            value = node_data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    @staticmethod
    def _infer_edge_branch(edge: Dict[str, Any]) -> Optional[str]:
        """Infer true/false branch from edge metadata."""
        candidates: List[Any] = [
            edge.get("branch"),
            edge.get("label"),
            edge.get("sourceHandle"),
            edge.get("source_handle"),
            edge.get("targetHandle"),
            edge.get("target_handle"),
            edge.get("id"),
        ]

        edge_data = edge.get("data")
        if isinstance(edge_data, dict):
            candidates.extend(
                [
                    edge_data.get("branch"),
                    edge_data.get("label"),
                    edge_data.get("sourceHandle"),
                    edge_data.get("source_handle"),
                    edge_data.get("targetHandle"),
                    edge_data.get("target_handle"),
                ]
            )

        for candidate in candidates:
            branch = Orchestrator._branch_from_hint(candidate)
            if branch:
                return branch
        return None

    @staticmethod
    def _resolve_next_targets(
        node: Dict[str, Any],
        outgoing_edges: List[Dict[str, Any]],
        response: Optional[AppletMessage] = None,
    ) -> List[str]:
        """Resolve outgoing targets, applying conditional routing for if/else nodes."""
        default_targets = Orchestrator._collect_outgoing_targets(outgoing_edges)
        node_type = str(node.get("type", "")).strip().lower()
        if node_type != IF_ELSE_NODE_TYPE:
            return default_targets

        if not outgoing_edges:
            return []

        selected_branch = Orchestrator._extract_if_else_branch(response)
        explicit_target = Orchestrator._branch_target_from_node_data(node, selected_branch)
        if explicit_target and explicit_target in default_targets:
            return [explicit_target]

        branch_targets: List[str] = []
        for index, edge in enumerate(outgoing_edges):
            inferred_branch = Orchestrator._infer_edge_branch(edge)
            if inferred_branch is None:
                if len(outgoing_edges) >= 2:
                    inferred_branch = "true" if index == 0 else "false" if index == 1 else None
                else:
                    inferred_branch = "true" if index == 0 else None

            if inferred_branch != selected_branch:
                continue

            target = edge.get("target")
            if isinstance(target, str) and target and target not in branch_targets:
                branch_targets.append(target)

        if branch_targets:
            return [branch_targets[0]]

        if selected_branch == "true":
            return default_targets[:1]
        if len(default_targets) > 1:
            return [default_targets[1]]
        return []

    @staticmethod
    def _mark_animated_edges(
        flow_edges: List[Dict[str, Any]],
        source_node_id: str,
        selected_targets: List[str],
    ) -> None:
        """Mark selected edges as animated for runtime visualization."""
        if not selected_targets:
            return
        selected_target_set = set(selected_targets)
        for edge in flow_edges:
            if edge.get("source") != source_node_id:
                continue
            if edge.get("target") in selected_target_set:
                edge["animated"] = True

    @staticmethod
    def migrate_legacy_writer_nodes(flow: Dict[str, Any]) -> tuple[Dict[str, Any], bool]:
        """Convert legacy writer nodes into LLM nodes with OpenAI GPT-4o defaults."""
        if not isinstance(flow, dict):
            return flow, False

        nodes = flow.get("nodes")
        if not isinstance(nodes, list):
            return flow, False

        migrated = False
        migrated_nodes: List[Any] = []

        for node in nodes:
            if not isinstance(node, dict):
                migrated_nodes.append(node)
                continue

            node_type = str(node.get("type", "")).strip().lower()
            if node_type != LEGACY_WRITER_NODE_TYPE:
                migrated_nodes.append(node)
                continue

            node_data = node.get("data", {})
            migrated_data = dict(node_data) if isinstance(node_data, dict) else {}

            if "systemPrompt" in migrated_data and "system_prompt" not in migrated_data:
                migrated_data["system_prompt"] = migrated_data["systemPrompt"]
            if "maxTokens" in migrated_data and "max_tokens" not in migrated_data:
                migrated_data["max_tokens"] = migrated_data["maxTokens"]

            for key, value in LEGACY_WRITER_LLM_PRESET.items():
                migrated_data.setdefault(key, value)

            migrated_data.setdefault("legacy_applet", LEGACY_WRITER_NODE_TYPE)
            migrated_data.setdefault("migration_source", "T-052")

            migrated_node = dict(node)
            migrated_node["type"] = LLM_NODE_TYPE
            migrated_node["data"] = migrated_data
            migrated_nodes.append(migrated_node)
            migrated = True

        if not migrated:
            return flow, False

        migrated_flow = dict(flow)
        migrated_flow["nodes"] = migrated_nodes
        return migrated_flow, True

    @staticmethod
    def _resolve_legacy_artist_defaults(generator: Any) -> Dict[str, str]:
        """Map legacy artist generator values to image-provider defaults."""
        if isinstance(generator, str):
            value = generator.strip().lower()
            if value in {"openai", "dall-e-3", "dall-e3", "dalle3"}:
                return {"provider": "openai", "model": "dall-e-3"}
            if value in {"flux", "flux-1.1-pro", "flux-1-dev"}:
                return {"provider": "flux", "model": "flux-1.1-pro"}
        return {"provider": "stability", "model": "stable-diffusion-xl-1024-v1-0"}

    @staticmethod
    def migrate_legacy_artist_nodes(flow: Dict[str, Any]) -> tuple[Dict[str, Any], bool]:
        """Convert legacy artist nodes into image nodes with provider presets."""
        if not isinstance(flow, dict):
            return flow, False

        nodes = flow.get("nodes")
        if not isinstance(nodes, list):
            return flow, False

        migrated = False
        migrated_nodes: List[Any] = []

        for node in nodes:
            if not isinstance(node, dict):
                migrated_nodes.append(node)
                continue

            node_type = str(node.get("type", "")).strip().lower()
            if node_type != LEGACY_ARTIST_NODE_TYPE:
                migrated_nodes.append(node)
                continue

            node_data = node.get("data", {})
            migrated_data = dict(node_data) if isinstance(node_data, dict) else {}

            generator = migrated_data.get("provider")
            if not isinstance(generator, str):
                generator = migrated_data.get("generator", migrated_data.get("image_generator"))

            provider_defaults = Orchestrator._resolve_legacy_artist_defaults(generator)

            if "responseFormat" in migrated_data and "response_format" not in migrated_data:
                migrated_data["response_format"] = migrated_data["responseFormat"]

            for key, value in LEGACY_ARTIST_IMAGE_PRESET.items():
                migrated_data.setdefault(key, value)

            migrated_data.setdefault("provider", provider_defaults["provider"])
            migrated_data.setdefault("model", provider_defaults["model"])
            migrated_data.setdefault("legacy_applet", LEGACY_ARTIST_NODE_TYPE)
            migrated_data.setdefault("migration_source", "T-054")

            migrated_node = dict(node)
            migrated_node["type"] = IMAGE_NODE_TYPE
            migrated_node["data"] = migrated_data
            migrated_nodes.append(migrated_node)
            migrated = True

        if not migrated:
            return flow, False

        migrated_flow = dict(flow)
        migrated_flow["nodes"] = migrated_nodes
        return migrated_flow, True

    @staticmethod
    def _normalize_legacy_memory_backend(raw_backend: Any) -> str:
        """Map legacy memory backend names to supported memory-node backend ids."""
        if not isinstance(raw_backend, str):
            return "sqlite_fts"
        normalized = raw_backend.strip().lower()
        return LEGACY_MEMORY_BACKEND_ALIASES.get(normalized, "sqlite_fts")

    @staticmethod
    def migrate_legacy_memory_nodes(flow: Dict[str, Any]) -> tuple[Dict[str, Any], bool]:
        """Convert legacy memory applet nodes into memory nodes with persistent defaults."""
        if not isinstance(flow, dict):
            return flow, False

        nodes = flow.get("nodes")
        if not isinstance(nodes, list):
            return flow, False

        migrated = False
        migrated_nodes: List[Any] = []

        for node in nodes:
            if not isinstance(node, dict):
                migrated_nodes.append(node)
                continue

            raw_node_type = str(node.get("type", "")).strip()
            node_type = raw_node_type.lower()
            is_memory_node = node_type == MEMORY_NODE_TYPE
            is_memory_alias = node_type in {"memoryapplet", "memory_applet", "memory-applet"}
            if not (is_memory_node or is_memory_alias):
                migrated_nodes.append(node)
                continue

            node_data = node.get("data", {})
            migrated_data = dict(node_data) if isinstance(node_data, dict) else {}

            legacy_key_mappings = {
                "memoryKey": "key",
                "memory_key": "key",
                "memoryNamespace": "namespace",
                "memory_namespace": "namespace",
                "persistPath": "persist_path",
                "collectionName": "collection",
                "includeMetadata": "include_metadata",
                "topK": "top_k",
            }
            for legacy_key, modern_key in legacy_key_mappings.items():
                if legacy_key in migrated_data and modern_key not in migrated_data:
                    migrated_data[modern_key] = migrated_data[legacy_key]

            if "backend" in migrated_data:
                migrated_data["backend"] = Orchestrator._normalize_legacy_memory_backend(
                    migrated_data.get("backend")
                )
            else:
                migrated_data["backend"] = "sqlite_fts"

            operation = migrated_data.get("operation", "store")
            if isinstance(operation, str):
                normalized_operation = operation.strip().lower()
            else:
                normalized_operation = "store"
            if normalized_operation not in {"store", "retrieve", "delete", "clear"}:
                normalized_operation = "store"
            migrated_data["operation"] = normalized_operation

            namespace = migrated_data.get("namespace", DEFAULT_MEMORY_NAMESPACE)
            if not isinstance(namespace, str) or not namespace.strip():
                namespace = DEFAULT_MEMORY_NAMESPACE
            migrated_data["namespace"] = namespace.strip()

            if "tags" in migrated_data and isinstance(migrated_data["tags"], str):
                migrated_data["tags"] = [migrated_data["tags"]]

            if "top_k" in migrated_data:
                try:
                    migrated_data["top_k"] = max(1, min(50, int(migrated_data["top_k"])))
                except (TypeError, ValueError):
                    migrated_data["top_k"] = 5

            migrated_data.setdefault("label", "Memory")
            migrated_data.setdefault("include_metadata", False)
            migrated_data.setdefault("legacy_applet", LEGACY_MEMORY_NODE_TYPE)
            migrated_data.setdefault("migration_source", "T-056")

            migrated_node = dict(node)
            migrated_node["type"] = MEMORY_NODE_TYPE
            migrated_node["data"] = migrated_data
            migrated_nodes.append(migrated_node)

            if migrated_node != node:
                migrated = True

        if not migrated:
            return flow, False

        migrated_flow = dict(flow)
        migrated_flow["nodes"] = migrated_nodes
        return migrated_flow, True

    @staticmethod
    def migrate_legacy_nodes(flow: Dict[str, Any]) -> tuple[Dict[str, Any], bool]:
        """Apply all known legacy node migrations."""
        writer_migrated_flow, writer_migrated = Orchestrator.migrate_legacy_writer_nodes(flow)
        artist_migrated_flow, artist_migrated = Orchestrator.migrate_legacy_artist_nodes(writer_migrated_flow)
        fully_migrated_flow, memory_migrated = Orchestrator.migrate_legacy_memory_nodes(artist_migrated_flow)
        return fully_migrated_flow, (writer_migrated or artist_migrated or memory_migrated)

    @staticmethod
    async def auto_migrate_legacy_nodes(
        flow: Optional[Dict[str, Any]],
        persist: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Apply known legacy node migrations and optionally persist migrated flow."""
        if not flow:
            return flow

        migrated_flow, migrated = Orchestrator.migrate_legacy_nodes(flow)
        if migrated:
            logger.info(
                "Auto-migrated legacy nodes for flow '%s'",
                migrated_flow.get("id"),
            )
            if persist:
                await FlowRepository.save(migrated_flow)
        return migrated_flow

    @staticmethod
    async def auto_migrate_legacy_writer_nodes(
        flow: Optional[Dict[str, Any]],
        persist: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Backward-compatible alias for legacy migration helper."""
        return await Orchestrator.auto_migrate_legacy_nodes(flow, persist=persist)

    @staticmethod
    def _topological_layers(
        nodes_by_id: Dict[str, Dict[str, Any]],
        edges_by_source: Dict[str, List[Dict[str, Any]]],
        incoming_sources_by_target: Dict[str, List[str]],
    ) -> List[List[str]]:
        """Compute topological layers (Kahn's algorithm).

        Returns a list of layers, where each layer contains node IDs that
        have no unresolved dependencies and can execute in parallel.
        """
        in_degree: Dict[str, int] = {nid: 0 for nid in nodes_by_id}
        for target_id, sources in incoming_sources_by_target.items():
            if target_id in in_degree:
                in_degree[target_id] = len(sources)

        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        layers: List[List[str]] = []

        while queue:
            layers.append(list(queue))
            next_queue: List[str] = []
            for nid in queue:
                for edge in edges_by_source.get(nid, []):
                    target = edge.get("target")
                    if not isinstance(target, str) or target not in in_degree:
                        continue
                    in_degree[target] -= 1
                    if in_degree[target] == 0 and target not in next_queue:
                        next_queue.append(target)
            queue = next_queue

        # Any remaining nodes (cycles) get appended as a final layer
        remaining = [nid for nid, deg in in_degree.items() if deg > 0]
        if remaining:
            layers.append(remaining)

        return layers

    @staticmethod
    def _detect_parallel_groups(
        layer: List[str],
        nodes_by_id: Dict[str, Dict[str, Any]],
    ) -> List[List[str]]:
        """Split a topological layer into parallel-safe groups.

        Merge nodes and nodes with order dependencies are isolated;
        all other independent nodes are grouped together for parallel execution.
        """
        parallel_group: List[str] = []
        sequential: List[List[str]] = []

        for nid in layer:
            node = nodes_by_id.get(nid)
            if not isinstance(node, dict):
                continue
            node_type = str(node.get("type", "")).strip().lower()
            # Merge nodes must wait for inputs; run them individually
            if node_type == MERGE_NODE_TYPE:
                sequential.append([nid])
            else:
                parallel_group.append(nid)

        groups: List[List[str]] = []
        if parallel_group:
            groups.append(parallel_group)
        groups.extend(sequential)
        return groups

    @staticmethod
    async def execute_flow(flow: dict, input_data: Dict[str, Any]) -> str:
        """Execute a flow and return the run ID."""
        run_id = Orchestrator.create_run_id()
        start_time = time.time()
        flow_id = flow.get("id")
        initial_trace = _new_execution_trace(run_id, flow_id, input_data, start_time)

        status = {
            "run_id": run_id,
            "flow_id": flow_id,
            "status": "running",
            "current_applet": None,
            "progress": 0,
            "total_steps": len(flow.get("nodes", [])),
            "start_time": start_time,
            "results": {TRACE_RESULTS_KEY: initial_trace},
            "input_data": input_data,
        }

        status_dict = status.copy()

        flow_name = flow.get("name", "")
        if flow_name:
            metrics.record_template_run(flow_name)

        await emit_event("template_started", {
            "run_id": run_id, "flow_id": flow_id, "flow_name": flow_name,
        })

        workflow_run_repo = WorkflowRunRepository()
        logger.info(f"Starting workflow execution with run ID: {run_id}")
        await workflow_run_repo.save(status_dict)

        broadcast_status_dict = status_dict.copy()
        broadcast_status_dict["completed_applets"] = []
        await broadcast_status(broadcast_status_dict)

        asyncio.create_task(
            Orchestrator._execute_flow_async(
                run_id, flow, input_data, workflow_run_repo, broadcast_status
            )
        )

        return run_id

    @staticmethod
    async def _execute_flow_async(
        run_id: str,
        flow: dict,
        input_data: Dict[str, Any],
        workflow_run_repo: WorkflowRunRepository,
        broadcast_status_fn,
    ):
        """Execute a flow with parallel independent-node support.

        The engine performs a BFS traversal where each wave of ``current_nodes``
        is partitioned into *parallel groups* (independent nodes that share no
        mutual edges) and *sequential groups* (merge/fan-in nodes that need all
        inputs).  Parallel groups are dispatched concurrently via
        ``asyncio.gather`` with a configurable semaphore.
        """
        status = None
        memory_completed_applets: List[str] = []
        context: Dict[str, Any] = {}
        execution_trace = _new_execution_trace(run_id, flow.get("id"), input_data, time.time())

        def _ensure_trace_in_context_results() -> None:
            results = context.get("results")
            if not isinstance(results, dict):
                results = {}
                context["results"] = results
            results[TRACE_RESULTS_KEY] = execution_trace

        def _append_trace_error(payload: Any) -> None:
            errors = execution_trace.setdefault("errors", [])
            if isinstance(errors, list):
                errors.append(_trace_value(payload))

        async def _fail(error_msg: str, error_details: Optional[Dict[str, Any]] = None) -> None:
            nonlocal status
            end_time = time.time()
            if error_details:
                _append_trace_error(error_details)
            else:
                _append_trace_error({"message": error_msg})
            _finalize_execution_trace(execution_trace, "error", end_time)
            _ensure_trace_in_context_results()

            if status and isinstance(status, dict):
                status["status"] = "error"
                status["error"] = error_msg
                if error_details:
                    status["error_details"] = error_details
                status["end_time"] = end_time
                status["results"] = context.get("results", {})
                status["completed_applets"] = list(memory_completed_applets)
                await workflow_run_repo.save(status)
                await broadcast_status_fn(status)
            else:
                await broadcast_status_fn({
                    "run_id": run_id,
                    "status": "error",
                    "error": error_msg,
                    "completed_applets": list(memory_completed_applets),
                })

        try:
            status = await workflow_run_repo.get_by_run_id(run_id)

            if status and isinstance(status, dict):
                status_start_time = status.get("start_time")
                if isinstance(status_start_time, (int, float)):
                    execution_trace["started_at"] = float(status_start_time)

                status_input_data = status.get("input_data")
                if isinstance(status_input_data, dict):
                    execution_trace["input"] = _trace_value(status_input_data)

                status_results = status.get("results")
                if isinstance(status_results, dict):
                    existing_trace = status_results.get(TRACE_RESULTS_KEY)
                    if isinstance(existing_trace, dict):
                        normalized_trace = _trace_value(existing_trace)
                        if isinstance(normalized_trace, dict):
                            execution_trace = normalized_trace

            flow_nodes = flow.get("nodes", [])
            flow_edges = flow.get("edges", [])

            nodes_by_id: Dict[str, Dict[str, Any]] = {
                node["id"]: node
                for node in flow_nodes
                if isinstance(node, dict) and isinstance(node.get("id"), str)
            }

            edges_by_source: Dict[str, List[Dict[str, Any]]] = {}
            incoming_sources_by_target: Dict[str, List[str]] = {}
            for edge in flow_edges:
                if not isinstance(edge, dict):
                    continue
                source = edge.get("source")
                target = edge.get("target")
                if not isinstance(source, str) or not source:
                    continue
                if not isinstance(target, str) or not target:
                    continue
                edges_by_source.setdefault(source, []).append(edge)
                incoming_sources = incoming_sources_by_target.setdefault(target, [])
                if source not in incoming_sources:
                    incoming_sources.append(source)

            target_nodes = {
                edge.get("target")
                for edge in flow_edges
                if isinstance(edge, dict) and isinstance(edge.get("target"), str)
            }
            start_nodes = [
                node["id"]
                for node in flow_nodes
                if isinstance(node, dict) and isinstance(node.get("id"), str) and node["id"] not in target_nodes
            ]
            scheduled_nodes = set(start_nodes)

            if not start_nodes:
                await _fail("No start node found in flow")
                return

            context = {
                "input": input_data,
                "results": {TRACE_RESULTS_KEY: execution_trace},
                "run_id": run_id,
            }
            merge_inputs_by_node: Dict[str, List[Any]] = {}
            merge_input_sources_by_node: Dict[str, List[str]] = {}
            _ensure_trace_in_context_results()

            if status and isinstance(status, dict):
                status["input_data"] = input_data

            # Read configurable concurrency limit
            flow_concurrency = ENGINE_MAX_CONCURRENCY
            flow_meta = flow.get("data", flow.get("metadata", {}))
            if isinstance(flow_meta, dict):
                raw_conc = flow_meta.get(
                    "engine_max_concurrency",
                    flow_meta.get("engineMaxConcurrency"),
                )
                if raw_conc is not None:
                    try:
                        flow_concurrency = max(1, int(raw_conc))
                    except (ValueError, TypeError):
                        pass
            engine_semaphore = asyncio.Semaphore(flow_concurrency)

            visited: set = set()
            failed_node_id: Optional[str] = None

            # ----------------------------------------------------------------
            # _execute_single_node — extracted for reuse in parallel dispatch
            # ----------------------------------------------------------------
            async def _execute_single_node(node_id: str) -> Optional[List[str]]:
                """Execute one node and return its downstream target IDs.

                Returns ``None`` on fatal (non-fallback) error to signal abort.
                """
                nonlocal failed_node_id

                node = nodes_by_id.get(node_id)
                if not isinstance(node, dict):
                    return []
                node_type = str(node.get("type", "")).strip().lower()
                outgoing_edges = edges_by_source.get(node_id, [])
                node_started_at = time.time()
                node_trace: Dict[str, Any] = {
                    "node_id": node_id,
                    "node_type": node.get("type"),
                    "status": "running",
                    "input": None,
                    "output": None,
                    "attempts": 0,
                    "errors": [],
                    "started_at": node_started_at,
                    "ended_at": None,
                    "duration_ms": None,
                }

                # -- merge gate: defer if not all inputs arrived --
                if node_type == MERGE_NODE_TYPE:
                    required = incoming_sources_by_target.get(node_id, [])
                    received = merge_input_sources_by_node.get(node_id, [])
                    missing = [s for s in required if s not in received]
                    if missing and any(s in scheduled_nodes and s not in visited for s in missing):
                        return [node_id]  # re-enqueue

                visited.add(node_id)

                if status and isinstance(status, dict):
                    status["current_applet"] = node.get("type")
                    status["progress"] = status.get("progress", 0) + 1
                    status["results"] = context.get("results", {})

                if node_id not in memory_completed_applets:
                    memory_completed_applets.append(node_id)

                if status and isinstance(status, dict):
                    bcast = status.copy()
                    bcast["completed_applets"] = list(memory_completed_applets)
                    await workflow_run_repo.save(status)
                    await broadcast_status_fn(bcast)

                # -- start / end passthrough --
                if node_type in ("start", "end"):
                    if (
                        node_type == "start"
                        and isinstance(node.get("data"), dict)
                        and "parsedInputData" in node["data"]
                    ):
                        parsed_input = node["data"]["parsedInputData"]
                        if parsed_input and isinstance(parsed_input, dict):
                            context["input"] = parsed_input
                            if status and isinstance(status, dict):
                                status["input_data"] = parsed_input

                    next_targets = Orchestrator._collect_outgoing_targets(outgoing_edges)
                    passthrough_output = context.get("input", input_data)
                    node_ended_at = time.time()
                    node_trace["status"] = "success"
                    node_trace["input"] = _trace_value(context.get("input", input_data))
                    node_trace["output"] = _trace_value(passthrough_output)
                    node_trace["attempts"] = 1
                    node_trace["ended_at"] = node_ended_at
                    node_trace["duration_ms"] = max(
                        0.0,
                        (node_ended_at - node_started_at) * 1000.0,
                    )
                    context["results"][node_id] = {
                        "type": node["type"],
                        "input": _trace_value(context.get("input", input_data)),
                        "output": passthrough_output,
                        "status": "success",
                        "attempts": 1,
                        "errors": [],
                        "started_at": node_started_at,
                        "ended_at": node_ended_at,
                        "duration_ms": node_trace["duration_ms"],
                    }
                    trace_nodes = execution_trace.setdefault("nodes", [])
                    if isinstance(trace_nodes, list):
                        trace_nodes.append(node_trace)
                    _ensure_trace_in_context_results()

                    for tid in next_targets:
                        scheduled_nodes.add(tid)
                        tn = nodes_by_id.get(tid)
                        if isinstance(tn, dict) and str(tn.get("type", "")).strip().lower() == MERGE_NODE_TYPE:
                            merge_inputs_by_node.setdefault(tid, []).append(passthrough_output)
                            ms = merge_input_sources_by_node.setdefault(tid, [])
                            if node_id not in ms:
                                ms.append(node_id)
                    return next_targets

                # -- normal applet execution with retry / timeout / fallback --
                node_data = node.get("data", {})
                if not isinstance(node_data, dict):
                    node_data = {}

                timeout_seconds = float(node_data.get("timeout_seconds", 60.0))
                retry_config = node_data.get("retry_config", {})
                if not isinstance(retry_config, dict):
                    retry_config = {}
                max_retries = int(retry_config.get("max_retries", 0))
                retry_delay = float(retry_config.get("delay", 1.0))
                retry_backoff = float(retry_config.get("backoff", 2.0))
                fallback_node_id_cfg = node_data.get("fallback_node_id")

                attempts = 0
                last_error: Optional[NodeError] = None
                success = False
                response: Optional[AppletMessage] = None
                message_content_for_trace: Any = None

                while attempts <= max_retries:
                    try:
                        if attempts > 0:
                            wait_time = retry_delay * (retry_backoff ** (attempts - 1))
                            logger.info(
                                f"Retrying node {node_id} (attempt {attempts}/{max_retries}) after {wait_time}s"
                            )
                            await asyncio.sleep(wait_time)
                        attempt_number = attempts + 1

                        applet = await Orchestrator.load_applet(node_type)

                        message_content: Any = input_data
                        if node_type == MERGE_NODE_TYPE:
                            node_inputs = list(merge_inputs_by_node.get(node_id, []))
                            node_sources = list(merge_input_sources_by_node.get(node_id, []))
                            message_content = {
                                "inputs": node_inputs,
                                "sources": node_sources,
                                "count": len(node_inputs),
                                "input": input_data,
                            }
                        if message_content_for_trace is None:
                            message_content_for_trace = _trace_value(message_content)

                        message_metadata: Dict[str, Any] = {
                            "node_id": node_id,
                            "run_id": run_id,
                        }
                        message_metadata["node_data"] = node_data

                        if node_type == "writer" and "systemPrompt" in node_data:
                            message_metadata["system_prompt"] = node_data["systemPrompt"]
                        if node_type == "artist":
                            if "system_prompt" in node_data:
                                message_metadata["system_prompt"] = node_data["system_prompt"]
                            elif "systemPrompt" in node_data:
                                message_metadata["system_prompt"] = node_data["systemPrompt"]
                            if "generator" in node_data:
                                message_metadata["generator"] = node_data["generator"]

                        message = AppletMessage(
                            content=message_content,
                            context=context,
                            metadata=message_metadata,
                        )
                        node_trace["attempts"] = attempt_number

                        async with engine_semaphore:
                            response = await asyncio.wait_for(
                                applet.on_message(message),
                                timeout=timeout_seconds,
                            )

                        node_ended_at = time.time()
                        context["results"][node_id] = {
                            "type": node["type"],
                            "input": _trace_value(message_content),
                            "output": response.content,
                            "status": "success",
                            "attempts": attempt_number,
                            "errors": _trace_value(node_trace["errors"]),
                            "started_at": node_started_at,
                            "ended_at": node_ended_at,
                            "duration_ms": max(0.0, (node_ended_at - node_started_at) * 1000.0),
                        }
                        context.update(response.context)
                        if not isinstance(context.get("results"), dict):
                            context["results"] = {}
                        context["results"][node_id] = context["results"].get(node_id) or {
                            "type": node["type"],
                            "input": _trace_value(message_content),
                            "output": response.content,
                            "status": "success",
                            "attempts": attempt_number,
                            "errors": _trace_value(node_trace["errors"]),
                            "started_at": node_started_at,
                            "ended_at": node_ended_at,
                            "duration_ms": max(0.0, (node_ended_at - node_started_at) * 1000.0),
                        }
                        node_trace["status"] = "success"
                        node_trace["input"] = _trace_value(message_content)
                        node_trace["output"] = _trace_value(response.content)
                        node_trace["ended_at"] = node_ended_at
                        node_trace["duration_ms"] = max(0.0, (node_ended_at - node_started_at) * 1000.0)
                        trace_nodes = execution_trace.setdefault("nodes", [])
                        if isinstance(trace_nodes, list):
                            trace_nodes.append(node_trace)
                        _ensure_trace_in_context_results()
                        await emit_event("step_completed", {
                            "run_id": run_id, "node_id": node_id,
                            "node_type": node.get("type"),
                            "duration_ms": node_trace.get("duration_ms"),
                        })
                        success = True
                        break

                    except asyncio.TimeoutError:
                        last_error = NodeError(
                            NodeErrorCode.TIMEOUT,
                            f"Node execution timed out after {timeout_seconds}s",
                            node_id=node_id,
                        )
                        logger.warning(
                            f"Timeout in node {node_id} (attempt {attempt_number}/{max_retries + 1})"
                        )
                        errors_list = node_trace.setdefault("errors", [])
                        if isinstance(errors_list, list):
                            errors_list.append(
                                {
                                    "attempt": attempt_number,
                                    "time": time.time(),
                                    "error": last_error.to_dict(),
                                }
                            )
                    except Exception as e:
                        last_error = NodeError(
                            NodeErrorCode.EXECUTION_ERROR,
                            str(e),
                            node_id=node_id,
                        )
                        logger.error(
                            f"Error in node {node_id} (attempt {attempt_number}/{max_retries + 1}): {e}"
                        )
                        errors_list = node_trace.setdefault("errors", [])
                        if isinstance(errors_list, list):
                            errors_list.append(
                                {
                                    "attempt": attempt_number,
                                    "time": time.time(),
                                    "error": last_error.to_dict(),
                                }
                            )
                    attempts += 1

                if not success:
                    node_ended_at = time.time()
                    error_payload = last_error.to_dict() if last_error else {"message": "Unknown error"}
                    node_trace["input"] = _trace_value(message_content_for_trace)
                    node_trace["attempts"] = attempts
                    node_trace["ended_at"] = node_ended_at
                    node_trace["duration_ms"] = max(0.0, (node_ended_at - node_started_at) * 1000.0)

                    if fallback_node_id_cfg:
                        logger.info(f"Using fallback path for node {node_id} -> {fallback_node_id_cfg}")
                        node_trace["status"] = "fallback"
                        node_trace["output"] = {
                            "fallback_node_id": fallback_node_id_cfg,
                        }
                        context["results"][node_id] = {
                            "type": node["type"],
                            "input": _trace_value(message_content_for_trace),
                            "output": None,
                            "error": error_payload,
                            "status": "fallback",
                            "attempts": attempts,
                            "errors": _trace_value(node_trace["errors"]),
                            "started_at": node_started_at,
                            "ended_at": node_ended_at,
                            "duration_ms": node_trace["duration_ms"],
                        }
                        trace_nodes = execution_trace.setdefault("nodes", [])
                        if isinstance(trace_nodes, list):
                            trace_nodes.append(node_trace)
                        _append_trace_error({"node_id": node_id, "error": error_payload})
                        _ensure_trace_in_context_results()
                        if node_id not in memory_completed_applets:
                            memory_completed_applets.append(node_id)
                        if status and isinstance(status, dict):
                            status["results"] = context["results"]
                            status["completed_applets"] = list(memory_completed_applets)
                            await workflow_run_repo.save(status)
                            await broadcast_status_fn(status)
                        if fallback_node_id_cfg in nodes_by_id:
                            scheduled_nodes.add(fallback_node_id_cfg)
                            return [fallback_node_id_cfg]
                        return []

                    # Fatal failure — no fallback
                    failed_node_id = node_id
                    node_trace["status"] = "error"
                    node_trace["output"] = None
                    context["results"][node_id] = {
                        "type": node["type"],
                        "input": _trace_value(message_content_for_trace),
                        "output": None,
                        "error": error_payload,
                        "status": "error",
                        "attempts": attempts,
                        "errors": _trace_value(node_trace["errors"]),
                        "started_at": node_started_at,
                        "ended_at": node_ended_at,
                        "duration_ms": node_trace["duration_ms"],
                    }
                    trace_nodes = execution_trace.setdefault("nodes", [])
                    if isinstance(trace_nodes, list):
                        trace_nodes.append(node_trace)
                    _ensure_trace_in_context_results()
                    err_msg = (
                        f"Error in applet '{node['type']}': "
                        f"{last_error.message if last_error else 'Unknown error'}"
                    )
                    await emit_event("step_failed", {
                        "run_id": run_id, "node_id": node_id,
                        "node_type": node.get("type"),
                        "error": err_msg,
                    })
                    await _fail(err_msg, error_payload if isinstance(error_payload, dict) else None)
                    return None

                if node_id not in memory_completed_applets:
                    memory_completed_applets.append(node_id)

                if status and isinstance(status, dict):
                    status["results"] = context["results"]
                    status["completed_applets"] = list(memory_completed_applets)
                    await workflow_run_repo.save(status)
                    await broadcast_status_fn(status)

                selected_targets = Orchestrator._resolve_next_targets(
                    node=node,
                    outgoing_edges=outgoing_edges,
                    response=response,
                )
                for tid in selected_targets:
                    scheduled_nodes.add(tid)
                    tn = nodes_by_id.get(tid)
                    if isinstance(tn, dict) and str(tn.get("type", "")).strip().lower() == MERGE_NODE_TYPE:
                        merge_inputs_by_node.setdefault(tid, []).append(
                            response.content if response else None
                        )
                        ms = merge_input_sources_by_node.setdefault(tid, [])
                        if node_id not in ms:
                            ms.append(node_id)

                Orchestrator._mark_animated_edges(
                    flow_edges=flow_edges,
                    source_node_id=node_id,
                    selected_targets=selected_targets,
                )
                return selected_targets

            # ================================================================
            # Main execution loop — BFS with parallel independent nodes
            # ================================================================
            current_nodes = list(start_nodes)

            while current_nodes:
                next_nodes: List[str] = []

                # Partition into ready and deferred (merge nodes still waiting)
                ready: List[str] = []
                deferred: List[str] = []
                for nid in current_nodes:
                    if nid in visited:
                        continue
                    node = nodes_by_id.get(nid)
                    if not isinstance(node, dict):
                        continue
                    nt = str(node.get("type", "")).strip().lower()
                    if nt == MERGE_NODE_TYPE:
                        req = incoming_sources_by_target.get(nid, [])
                        recv = merge_input_sources_by_node.get(nid, [])
                        missing = [s for s in req if s not in recv]
                        if missing and any(s in scheduled_nodes and s not in visited for s in missing):
                            deferred.append(nid)
                            continue
                    ready.append(nid)

                # Group ready nodes into parallel-safe groups
                groups = Orchestrator._detect_parallel_groups(ready, nodes_by_id)

                for group in groups:
                    if failed_node_id:
                        break

                    if len(group) == 1:
                        # Single node — execute directly
                        result_targets = await _execute_single_node(group[0])
                        if result_targets is None:
                            break
                        for tid in result_targets:
                            if tid not in next_nodes:
                                next_nodes.append(tid)
                    else:
                        # Parallel group — execute concurrently
                        async def _run_node(nid: str) -> tuple:
                            targets = await _execute_single_node(nid)
                            return (nid, targets)

                        parallel_results = await asyncio.gather(
                            *[_run_node(nid) for nid in group],
                            return_exceptions=True,
                        )

                        for pr in parallel_results:
                            if failed_node_id:
                                break
                            if isinstance(pr, BaseException):
                                logger.error(f"Unexpected parallel node error: {pr}")
                                await _fail(f"Unexpected parallel execution error: {pr}")
                                failed_node_id = "parallel_group"
                                break
                            _nid, result_targets = pr
                            if result_targets is None:
                                break
                            for tid in result_targets:
                                if tid not in next_nodes:
                                    next_nodes.append(tid)

                if failed_node_id:
                    return

                # Re-enqueue deferred merge nodes
                for nid in deferred:
                    if nid not in next_nodes:
                        next_nodes.append(nid)

                current_nodes = next_nodes

            if status and isinstance(status, dict):
                end_time = time.time()
                _finalize_execution_trace(execution_trace, "success", end_time)
                _ensure_trace_in_context_results()
                status["status"] = "success"
                status["end_time"] = end_time
                status["results"] = context["results"]
                if "input_data" not in status or not status["input_data"]:
                    status["input_data"] = input_data
                await workflow_run_repo.save(status)
                broadcast_data = status.copy()
                broadcast_data["completed_applets"] = memory_completed_applets
                await broadcast_status_fn(broadcast_data)
                await emit_event("template_completed", {
                    "run_id": run_id, "flow_id": flow.get("id"),
                    "flow_name": flow.get("name", ""),
                    "duration_ms": execution_trace.get("duration_ms"),
                })

        except Exception as e:
            logger.error(f"Error executing workflow: {e}")
            end_time = time.time()
            _append_trace_error(
                {
                    "code": NodeErrorCode.UNKNOWN_ERROR,
                    "message": str(e),
                    "type": type(e).__name__,
                }
            )
            _finalize_execution_trace(execution_trace, "error", end_time)
            _ensure_trace_in_context_results()
            if status and isinstance(status, dict):
                status["status"] = "error"
                status["error"] = f"Workflow execution error: {str(e)}"
                status["end_time"] = end_time
                status["results"] = context.get("results", {})
                await workflow_run_repo.save(status)
                broadcast_data = status.copy()
                broadcast_data["completed_applets"] = memory_completed_applets
                await broadcast_status_fn(broadcast_data)
            else:
                await broadcast_status_fn({
                    "run_id": run_id,
                    "status": "error",
                    "error": f"Workflow execution error: {str(e)}",
                    "completed_applets": memory_completed_applets,
                })
            await emit_event("template_failed", {
                "run_id": run_id, "flow_id": flow.get("id"),
                "flow_name": flow.get("name", ""),
                "error": str(e),
            })




# ============================================================
# API Routes (v1)
# ============================================================

v1 = APIRouter(prefix="/api/v1", tags=["v1"])


@v1.post("/auth/register", response_model=AuthTokenResponseModel, status_code=201, tags=["Auth"])
async def register(body: AuthRegisterRequestStrict):
    """Register a new user account and receive JWT tokens."""
    now = _utc_now()
    async with get_db_session() as session:
        existing_result = await session.execute(
            select(AuthUser).where(AuthUser.email == body.email)
        )
        if existing_result.scalars().first():
            raise HTTPException(status_code=409, detail="Email already registered")

        user = AuthUser(
            id=str(uuid.uuid4()),
            email=body.email,
            password_hash=_hash_password(body.password),
            is_active=True,
            created_at=now,
            updated_at=now,
        )
        session.add(user)

    token_response, refresh_token, refresh_expires_at = _issue_api_tokens(user)
    await _store_refresh_token(user.id, refresh_token, refresh_expires_at)
    return token_response


@v1.post("/auth/login", response_model=AuthTokenResponseModel, tags=["Auth"])
async def login(body: AuthLoginRequestStrict):
    """Authenticate with email/password and receive JWT tokens."""
    async with get_db_session() as session:
        user_result = await session.execute(
            select(AuthUser).where(AuthUser.email == body.email)
        )
        user = user_result.scalars().first()
        if not user or not user.is_active:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        if not _verify_password(body.password, user.password_hash):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        user.updated_at = _utc_now()

    token_response, refresh_token, refresh_expires_at = _issue_api_tokens(user)
    await _store_refresh_token(user.id, refresh_token, refresh_expires_at)
    return token_response


@v1.post("/auth/refresh", response_model=AuthTokenResponseModel, tags=["Auth"])
async def refresh_token(body: AuthRefreshRequestStrict):
    """Rotate a refresh token and receive a new access/refresh pair."""
    raw_refresh = body.refresh_token.strip()
    payload = _decode_token(raw_refresh, expected_type="refresh")
    user_id = payload.get("sub")
    if not isinstance(user_id, str) or not user_id:
        raise HTTPException(status_code=401, detail="Invalid refresh token subject")

    refresh_hash = _hash_sha256(raw_refresh)
    now = _utc_now()

    async with get_db_session() as session:
        refresh_result = await session.execute(
            select(AuthRefreshToken).where(AuthRefreshToken.token_hash == refresh_hash)
        )
        stored_refresh = refresh_result.scalars().first()
        if not stored_refresh or stored_refresh.revoked:
            raise HTTPException(status_code=401, detail="Refresh token revoked")
        if stored_refresh.expires_at <= now:
            stored_refresh.revoked = True
            raise HTTPException(status_code=401, detail="Refresh token expired")
        if stored_refresh.user_id != user_id:
            raise HTTPException(status_code=401, detail="Refresh token user mismatch")

        user_result = await session.execute(
            select(AuthUser).where(AuthUser.id == user_id)
        )
        user = user_result.scalars().first()
        if not user or not user.is_active:
            raise HTTPException(status_code=401, detail="User is not active")

        stored_refresh.revoked = True
        stored_refresh.last_used_at = now
        user.updated_at = now

    token_response, new_refresh_token, refresh_expires_at = _issue_api_tokens(user)
    await _store_refresh_token(user.id, new_refresh_token, refresh_expires_at)
    return token_response


@v1.post("/auth/logout", tags=["Auth"])
async def logout(body: AuthRefreshRequestStrict):
    """Revoke a refresh token (log out)."""
    raw_refresh = body.refresh_token.strip()
    refresh_hash = _hash_sha256(raw_refresh)

    async with get_db_session() as session:
        refresh_result = await session.execute(
            select(AuthRefreshToken).where(AuthRefreshToken.token_hash == refresh_hash)
        )
        stored_refresh = refresh_result.scalars().first()
        if stored_refresh:
            stored_refresh.revoked = True
            stored_refresh.last_used_at = _utc_now()

    return {"message": "Logged out"}


@v1.get("/auth/me", response_model=UserProfileModel, tags=["Auth"])
async def auth_me(current_user: Dict[str, Any] = Depends(get_authenticated_user)):
    """Return the authenticated user's profile."""
    return UserProfileModel(
        id=current_user["id"],
        email=current_user["email"],
        is_active=current_user["is_active"],
        created_at=current_user["created_at"],
    )


@v1.post("/auth/api-keys", response_model=APIKeyCreateResponseModel, status_code=201, tags=["Auth"])
async def create_api_key(
    body: APIKeyCreateRequestStrict,
    current_user: Dict[str, Any] = Depends(get_authenticated_user),
):
    """Create an API key for X-API-Key header authentication."""
    plain_key = f"{API_KEY_VALUE_PREFIX}_{secrets.token_urlsafe(32)}"
    now = _utc_now()
    api_key_record = AuthUserAPIKey(
        id=str(uuid.uuid4()),
        user_id=current_user["id"],
        name=body.name,
        key_prefix=_api_key_lookup_prefix(plain_key),
        encrypted_key=_encrypt_api_key(plain_key),
        is_active=True,
        created_at=now,
        last_used_at=None,
    )

    async with get_db_session() as session:
        session.add(api_key_record)

    return APIKeyCreateResponseModel(
        id=api_key_record.id,
        name=api_key_record.name,
        key_prefix=api_key_record.key_prefix,
        is_active=api_key_record.is_active,
        created_at=api_key_record.created_at,
        last_used_at=api_key_record.last_used_at,
        api_key=plain_key,
    )


@v1.get("/auth/api-keys", tags=["Auth"])
async def list_api_keys(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    current_user: Dict[str, Any] = Depends(get_authenticated_user),
):
    """List active API keys for the authenticated user."""
    async with get_db_session() as session:
        result = await session.execute(
            select(AuthUserAPIKey).where(
                AuthUserAPIKey.user_id == current_user["id"],
                AuthUserAPIKey.is_active == True,  # noqa: E712 - SQLAlchemy boolean comparison
            )
        )
        records = result.scalars().all()
        items = [
            APIKeyResponseModel(
                id=record.id,
                name=record.name,
                key_prefix=record.key_prefix,
                is_active=record.is_active,
                created_at=record.created_at,
                last_used_at=record.last_used_at,
            ).model_dump()
            for record in records
        ]
        return paginate(items, page, page_size)


@v1.delete("/auth/api-keys/{api_key_id}", tags=["Auth"])
async def revoke_api_key(
    api_key_id: str,
    current_user: Dict[str, Any] = Depends(get_authenticated_user),
):
    """Revoke a user API key."""
    async with get_db_session() as session:
        result = await session.execute(
            select(AuthUserAPIKey).where(
                AuthUserAPIKey.id == api_key_id,
                AuthUserAPIKey.user_id == current_user["id"],
            )
        )
        record = result.scalars().first()
        if not record:
            raise HTTPException(status_code=404, detail="API key not found")
        record.is_active = False
        record.last_used_at = _utc_now()

    return {"message": "API key revoked"}


@v1.get("/applets", tags=["Applets"])
async def list_applets(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    current_user: Dict[str, Any] = Depends(get_authenticated_user),
):
    """List all registered applets with their metadata (paginated)."""
    result = []

    for applet_type, applet_class in applet_registry.items():
        result.append({
            "type": applet_type,
            **applet_class.get_metadata()
        })

    applets_dir = os.path.join(os.path.dirname(__file__), "..", "applets")
    if os.path.exists(applets_dir):
        for applet_dir in os.listdir(applets_dir):
            if applet_dir.startswith("__") or applet_dir.startswith("."):
                continue
            if applet_dir not in [a["type"] for a in result]:
                try:
                    applet = await Orchestrator.load_applet(applet_dir)
                    result.append({
                        "type": applet_dir,
                        **applet.get_metadata()
                    })
                except Exception as e:
                    logger.warning(f"Failed to load applet '{applet_dir}': {e}")

    return paginate(result, page, page_size)


@v1.get("/llm/providers", tags=["Providers"])
async def list_llm_providers(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    current_user: Dict[str, Any] = Depends(get_authenticated_user),
):
    """List supported LLM providers and model catalogs."""
    providers = [provider.model_dump() for provider in LLMProviderRegistry.list_providers()]
    return paginate(providers, page, page_size)


@v1.get("/image/providers", tags=["Providers"])
async def list_image_providers(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    current_user: Dict[str, Any] = Depends(get_authenticated_user),
):
    """List supported image generation providers and model catalogs."""
    providers = [provider.model_dump() for provider in ImageProviderRegistry.list_providers()]
    return paginate(providers, page, page_size)


# ---------------------------------------------------------------------------
# Provider auto-discovery + unified registry endpoints
# ---------------------------------------------------------------------------

from synapps.providers.llm import ProviderRegistry as SynappsProviderRegistry  # noqa: E402

# Run auto-discovery on startup
SynappsProviderRegistry.auto_discover()
_synapps_registry = SynappsProviderRegistry()
# Copy globally-discovered providers into the instance for the endpoints
for _pname in SynappsProviderRegistry.list_global():
    _synapps_registry.register(SynappsProviderRegistry.get_global(_pname))


@v1.get("/providers", tags=["Providers"])
async def list_discovered_providers(
    current_user: Dict[str, Any] = Depends(get_authenticated_user),
):
    """List all auto-discovered LLM providers with capabilities and status."""
    providers = _synapps_registry.all_providers_info()
    return {
        "providers": providers,
        "total": len(providers),
        "discovery": "filesystem",
    }


@v1.get("/providers/{name}/health", tags=["Providers"])
async def provider_health_check(
    name: str,
    current_user: Dict[str, Any] = Depends(get_authenticated_user),
):
    """Run a health check on a specific discovered provider."""
    try:
        health = _synapps_registry.provider_health(name)
    except Exception:
        raise HTTPException(status_code=404, detail=f"Provider '{name}' not found")
    return health


# ---------------------------------------------------------------------------
# Template validation
# ---------------------------------------------------------------------------

KNOWN_NODE_TYPES = frozenset({
    "start", "end", "llm", "image", "image_gen", "memory", "http_request",
    "code", "transform", "if_else", "merge", "for_each", "custom",
})


def validate_template(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a template/flow definition and return a structured report.

    Returns ``{"valid": True/False, "errors": [...], "warnings": [...], "summary": {...}}``.
    """
    errors: List[str] = []
    warnings: List[str] = []

    # --- Required top-level fields ---
    if not isinstance(data.get("name"), str) or not data["name"].strip():
        errors.append("Missing or empty required field: 'name'")
    if not isinstance(data.get("nodes"), list):
        errors.append("Missing or invalid required field: 'nodes' (must be a list)")
        # Can't continue structural checks without nodes
        return {
            "valid": False,
            "errors": errors,
            "warnings": warnings,
            "summary": {"node_count": 0, "edge_count": 0},
        }

    nodes = data["nodes"]
    edges = data.get("edges", [])
    if not isinstance(edges, list):
        errors.append("Invalid field: 'edges' (must be a list)")
        edges = []

    # --- Node validation ---
    node_ids: set[str] = set()
    has_start = False
    has_end = False
    for i, node in enumerate(nodes):
        if not isinstance(node, dict):
            errors.append(f"nodes[{i}]: must be an object")
            continue
        nid = node.get("id")
        if not nid or not isinstance(nid, str):
            errors.append(f"nodes[{i}]: missing or empty 'id'")
            continue
        if nid in node_ids:
            errors.append(f"nodes[{i}]: duplicate node id '{nid}'")
        node_ids.add(nid)

        ntype = node.get("type", "")
        if not ntype:
            errors.append(f"node '{nid}': missing 'type'")
        elif ntype not in KNOWN_NODE_TYPES:
            warnings.append(f"node '{nid}': unknown type '{ntype}'")

        if ntype == "start":
            has_start = True
        if ntype == "end":
            has_end = True

        pos = node.get("position")
        if not isinstance(pos, dict) or "x" not in pos or "y" not in pos:
            warnings.append(f"node '{nid}': missing or invalid 'position'")

    if not has_start:
        errors.append("Template must contain at least one 'start' node")
    if not has_end:
        errors.append("Template must contain at least one 'end' node")

    # --- Edge validation ---
    edge_ids: set[str] = set()
    adjacency: Dict[str, List[str]] = {nid: [] for nid in node_ids}
    for i, edge in enumerate(edges):
        if not isinstance(edge, dict):
            errors.append(f"edges[{i}]: must be an object")
            continue
        eid = edge.get("id", f"edge-{i}")
        if eid in edge_ids:
            errors.append(f"edges[{i}]: duplicate edge id '{eid}'")
        edge_ids.add(eid)

        src = edge.get("source", "")
        tgt = edge.get("target", "")
        if not src:
            errors.append(f"edge '{eid}': missing 'source'")
        elif src not in node_ids:
            errors.append(f"edge '{eid}': source '{src}' references unknown node")
        if not tgt:
            errors.append(f"edge '{eid}': missing 'target'")
        elif tgt not in node_ids:
            errors.append(f"edge '{eid}': target '{tgt}' references unknown node")

        if src and tgt and src == tgt:
            errors.append(f"edge '{eid}': self-loop (source == target: '{src}')")

        if src in node_ids and tgt in node_ids:
            adjacency.setdefault(src, []).append(tgt)

    # --- Circular dependency detection (DFS) ---
    WHITE, GRAY, BLACK = 0, 1, 2
    color: Dict[str, int] = {nid: WHITE for nid in node_ids}
    cycle_path: List[str] = []

    def _dfs(node: str) -> bool:
        color[node] = GRAY
        cycle_path.append(node)
        for neighbor in adjacency.get(node, []):
            if color[neighbor] == GRAY:
                # Found cycle — extract the cycle portion
                idx = cycle_path.index(neighbor)
                cycle = cycle_path[idx:] + [neighbor]
                errors.append(f"Circular dependency detected: {' -> '.join(cycle)}")
                return True
            if color[neighbor] == WHITE:
                if _dfs(neighbor):
                    return True
        cycle_path.pop()
        color[node] = BLACK
        return False

    for nid in node_ids:
        if color[nid] == WHITE:
            if _dfs(nid):
                break  # Report first cycle found

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "node_types": sorted({n.get("type", "") for n in nodes if isinstance(n, dict)}),
            "has_start": has_start,
            "has_end": has_end,
        },
    }


class ValidateTemplateRequest(BaseModel):
    """Request body for template validation."""
    model_config = ConfigDict(extra="allow")
    name: Optional[str] = None
    nodes: Optional[List[Dict[str, Any]]] = None
    edges: Optional[List[Dict[str, Any]]] = None


@v1.post("/templates/validate", tags=["Dashboard"])
async def validate_template_endpoint(
    payload: ValidateTemplateRequest,
    current_user: Dict[str, Any] = Depends(get_authenticated_user),
):
    """Dry-run validation of a template/flow definition without execution."""
    data = payload.model_dump()
    result = validate_template(data)
    return result


# ---------------------------------------------------------------------------
# Webhook CRUD endpoints
# ---------------------------------------------------------------------------

class RegisterWebhookRequest(StrictRequestModel):
    """Request body for webhook registration."""
    url: str = Field(..., min_length=1, description="Delivery URL")
    events: List[str] = Field(..., min_length=1, description="Event names to subscribe to")
    secret: Optional[str] = Field(None, description="HMAC-SHA256 signing secret")

    @field_validator("events")
    @classmethod
    def events_valid(cls, v):
        invalid = [e for e in v if e not in WEBHOOK_EVENTS]
        if invalid:
            raise ValueError(f"Invalid events: {invalid}. Valid: {sorted(WEBHOOK_EVENTS)}")
        return v


@v1.post("/webhooks", status_code=201, tags=["Dashboard"])
async def register_webhook(
    payload: RegisterWebhookRequest,
    current_user: Dict[str, Any] = Depends(get_authenticated_user),
):
    """Register a new webhook for event delivery."""
    hook = webhook_registry.register(
        url=payload.url,
        events=payload.events,
        secret=payload.secret,
    )
    return hook


@v1.get("/webhooks", tags=["Dashboard"])
async def list_webhooks(
    current_user: Dict[str, Any] = Depends(get_authenticated_user),
):
    """List all registered webhooks (secrets are not returned)."""
    hooks = webhook_registry.list_hooks()
    return {"webhooks": hooks, "total": len(hooks)}


@v1.delete("/webhooks/{hook_id}", tags=["Dashboard"])
async def delete_webhook(
    hook_id: str,
    current_user: Dict[str, Any] = Depends(get_authenticated_user),
):
    """Delete a webhook by ID."""
    if not webhook_registry.delete(hook_id):
        raise HTTPException(status_code=404, detail=f"Webhook '{hook_id}' not found")
    return {"message": "Webhook deleted", "id": hook_id}


# ---------------------------------------------------------------------------
# Async Task Queue endpoints
# ---------------------------------------------------------------------------

def _load_yaml_template(template_id: str) -> Optional[Dict[str, Any]]:
    """Load a YAML template by its ID (filename stem or 'id' field)."""
    import yaml

    templates_dir = project_root / "templates"
    if not templates_dir.is_dir():
        return None
    for path in templates_dir.glob("*.yaml"):
        try:
            with open(path) as f:
                data = yaml.safe_load(f)
            if not isinstance(data, dict):
                continue
            tid = data.get("id", path.stem)
            if tid == template_id:
                return data
        except Exception:
            continue
    return None


async def _run_task_background(task_id: str, template_data: Dict[str, Any], input_data: Dict[str, Any]) -> None:
    """Background coroutine that runs a template and updates task state."""
    task_queue.update(task_id, status="running", started_at=time.time(), progress_pct=10)
    try:
        flow_dict = {
            "id": str(uuid.uuid4()),
            "name": template_data.get("name", ""),
            "nodes": template_data.get("nodes", []),
            "edges": template_data.get("edges", []),
        }
        await FlowRepository.save(flow_dict)
        task_queue.update(task_id, progress_pct=30)

        run_id = await Orchestrator.execute_flow(flow_dict, input_data)
        task_queue.update(task_id, run_id=run_id, progress_pct=50)

        # Poll for completion (up to 60s)
        for _ in range(120):
            await asyncio.sleep(0.5)
            run = await WorkflowRunRepository.get_by_run_id(run_id)
            if run and run.get("status") in ("success", "error"):
                break

        run = await WorkflowRunRepository.get_by_run_id(run_id)
        run_status = run.get("status", "unknown") if run else "unknown"

        if run_status == "success":
            task_queue.update(
                task_id,
                status="completed",
                progress_pct=100,
                result={"run_id": run_id, "run_status": run_status},
                completed_at=time.time(),
            )
        else:
            task_queue.update(
                task_id,
                status="failed",
                progress_pct=100,
                error=run.get("error", "Execution failed") if run else "Run not found",
                result={"run_id": run_id, "run_status": run_status},
                completed_at=time.time(),
            )
    except Exception as e:
        task_queue.update(
            task_id,
            status="failed",
            progress_pct=100,
            error=str(e),
            completed_at=time.time(),
        )


class RunAsyncRequest(BaseModel):
    """Request body for async template execution."""
    model_config = ConfigDict(extra="forbid")
    input: Dict[str, Any] = Field(default_factory=dict, description="Input data for the workflow")


@v1.post("/templates/{template_id}/run-async", status_code=202, tags=["Runs"])
async def run_template_async(
    template_id: str,
    body: RunAsyncRequest,
    current_user: Dict[str, Any] = Depends(get_authenticated_user),
):
    """Run a YAML template asynchronously. Returns a task ID for polling."""
    template_data = _load_yaml_template(template_id)
    if not template_data:
        raise HTTPException(status_code=404, detail=f"Template '{template_id}' not found")

    task_id = task_queue.create(template_id, template_data.get("name", ""))
    asyncio.create_task(_run_task_background(task_id, template_data, body.input))
    return {"task_id": task_id, "status": "pending"}


@v1.get("/tasks/{task_id}", tags=["Runs"])
async def get_task(
    task_id: str,
    current_user: Dict[str, Any] = Depends(get_authenticated_user),
):
    """Get the status and result of an async task."""
    task = task_queue.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    return task


@v1.get("/tasks", tags=["Runs"])
async def list_tasks(
    status: Optional[str] = Query(None, description="Filter by status: pending, running, completed, failed"),
    current_user: Dict[str, Any] = Depends(get_authenticated_user),
):
    """List all async tasks, optionally filtered by status."""
    if status and status not in TASK_STATUSES:
        raise HTTPException(status_code=400, detail=f"Invalid status. Valid: {TASK_STATUSES}")
    tasks = task_queue.list_tasks(status=status)
    return {"tasks": tasks, "total": len(tasks)}


# ---------------------------------------------------------------------------
# Admin API Key Management endpoints (master-key-protected)
# ---------------------------------------------------------------------------


class AdminKeyCreateRequest(BaseModel):
    """Request body for creating an admin API key."""
    model_config = ConfigDict(extra="forbid")
    name: str = Field(..., min_length=1, max_length=128, description="Key name/label")
    scopes: Optional[List[str]] = Field(None, description="Scopes: read, write, admin")

    @field_validator("scopes")
    @classmethod
    def scopes_valid(cls, v):
        if v is not None:
            invalid = [s for s in v if s not in ADMIN_KEY_SCOPES]
            if invalid:
                raise ValueError(f"Invalid scopes: {invalid}. Valid: {sorted(ADMIN_KEY_SCOPES)}")
        return v


@v1.post("/admin/keys", status_code=201, tags=["Admin"])
async def create_admin_key(
    body: AdminKeyCreateRequest,
    _master: str = Depends(require_master_key),
):
    """Create an admin API key (requires master key)."""
    result = admin_key_registry.create(name=body.name, scopes=body.scopes)
    return result


@v1.get("/admin/keys", tags=["Admin"])
async def list_admin_keys(
    _master: str = Depends(require_master_key),
):
    """List all admin API keys (requires master key). Plain keys are never returned."""
    keys = admin_key_registry.list_keys()
    return {"keys": keys, "total": len(keys)}


@v1.delete("/admin/keys/{key_id}", tags=["Admin"])
async def delete_admin_key(
    key_id: str,
    _master: str = Depends(require_master_key),
):
    """Delete (revoke) an admin API key by ID (requires master key)."""
    if not admin_key_registry.delete(key_id):
        raise HTTPException(status_code=404, detail=f"Admin key '{key_id}' not found")
    return {"message": "Admin key deleted", "id": key_id}


@v1.post("/flows", status_code=201, tags=["Flows"])
async def create_flow(
    flow: CreateFlowRequest,
    current_user: Dict[str, Any] = Depends(get_authenticated_user),
):
    """Create or update a flow with strict validation."""
    flow_id = flow.id if flow.id else str(uuid.uuid4())

    flow_dict = flow.model_dump()
    flow_dict["id"] = flow_id
    flow_dict, migrated = Orchestrator.migrate_legacy_nodes(flow_dict)
    if migrated:
        logger.info("Applied legacy node migration while creating flow '%s'", flow_id)
    await FlowRepository.save(flow_dict)
    return {"message": "Flow created", "id": flow_id}


@v1.get("/flows", tags=["Flows"])
async def list_flows(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    current_user: Dict[str, Any] = Depends(get_authenticated_user),
):
    """List all flows with pagination."""
    flows = await FlowRepository.get_all()
    migrated_flows: List[Dict[str, Any]] = []
    for flow in flows:
        migrated_flow = await Orchestrator.auto_migrate_legacy_nodes(flow, persist=True)
        if isinstance(migrated_flow, dict):
            migrated_flows.append(migrated_flow)
    return paginate(migrated_flows, page, page_size)


@v1.get("/flows/{flow_id}", tags=["Flows"])
async def get_flow(
    flow_id: str,
    current_user: Dict[str, Any] = Depends(get_authenticated_user),
):
    """Get a flow by ID."""
    flow = await FlowRepository.get_by_id(flow_id)
    if not flow:
        raise HTTPException(status_code=404, detail="Flow not found")
    flow = await Orchestrator.auto_migrate_legacy_nodes(flow, persist=True)
    return flow


@v1.delete("/flows/{flow_id}", tags=["Flows"])
async def delete_flow(
    flow_id: str,
    current_user: Dict[str, Any] = Depends(get_authenticated_user),
):
    """Delete a flow."""
    flow = await FlowRepository.get_by_id(flow_id)
    if not flow:
        raise HTTPException(status_code=404, detail="Flow not found")
    await FlowRepository.delete(flow_id)
    return {"message": "Flow deleted"}


@v1.get("/flows/{flow_id}/export", tags=["Flows"])
async def export_flow(
    flow_id: str,
    current_user: Dict[str, Any] = Depends(get_authenticated_user),
):
    """Export a flow as a downloadable JSON file."""
    flow = await FlowRepository.get_by_id(flow_id)
    if not flow:
        raise HTTPException(status_code=404, detail="Flow not found")

    # Build a clean export payload (strip internal DB-only fields)
    export_data = {
        "synapps_version": "1.0.0",
        "name": flow["name"],
        "nodes": [
            {
                "id": n["id"],
                "type": n["type"],
                "position": n["position"],
                "data": n.get("data", {}),
            }
            for n in flow.get("nodes", [])
        ],
        "edges": [
            {
                "id": e["id"],
                "source": e["source"],
                "target": e["target"],
                "animated": e.get("animated", False),
            }
            for e in flow.get("edges", [])
        ],
    }

    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", flow["name"])[:60]
    return JSONResponse(
        content=export_data,
        headers={
            "Content-Disposition": f'attachment; filename="{safe_name}.synapps.json"',
        },
    )


class ImportFlowRequest(BaseModel):
    """Request body for importing a flow."""
    model_config = ConfigDict(strict=False)

    synapps_version: Optional[str] = None
    name: str = Field(..., min_length=1, max_length=200)
    nodes: List[Dict[str, Any]] = Field(default_factory=list)
    edges: List[Dict[str, Any]] = Field(default_factory=list)


@v1.post("/flows/import", status_code=201, tags=["Flows"])
async def import_flow(
    body: ImportFlowRequest,
    current_user: Dict[str, Any] = Depends(get_authenticated_user),
):
    """Import a flow from a JSON export. Assigns a new ID to avoid collisions."""
    new_id = str(uuid.uuid4())

    # Re-map node and edge IDs to avoid collisions with existing flows
    id_map: Dict[str, str] = {}
    new_nodes = []
    for node in body.nodes:
        old_id = node.get("id", str(uuid.uuid4()))
        new_node_id = str(uuid.uuid4())
        id_map[old_id] = new_node_id
        new_nodes.append({
            **node,
            "id": new_node_id,
        })

    new_edges = []
    for edge in body.edges:
        new_edges.append({
            "id": str(uuid.uuid4()),
            "source": id_map.get(edge.get("source", ""), edge.get("source", "")),
            "target": id_map.get(edge.get("target", ""), edge.get("target", "")),
            "animated": edge.get("animated", False),
        })

    flow_data = {
        "id": new_id,
        "name": body.name,
        "nodes": new_nodes,
        "edges": new_edges,
    }

    await FlowRepository.save(flow_data)
    return {"message": "Flow imported", "id": new_id}


async def _run_flow_impl(
    flow_id: str,
    body: RunFlowRequest,
    current_user: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Execute a flow and return a run identifier."""
    flow = await FlowRepository.get_by_id(flow_id)
    if not flow:
        raise HTTPException(status_code=404, detail="Flow not found")
    flow = await Orchestrator.auto_migrate_legacy_nodes(flow, persist=True)
    run_id = await Orchestrator.execute_flow(flow, body.input)
    return {"run_id": run_id}


@v1.post("/flows/{flow_id}/runs", status_code=202, tags=["Runs"])
async def create_flow_run(
    flow_id: str,
    body: RunFlowRequest,
    current_user: Dict[str, Any] = Depends(get_authenticated_user),
):
    """RESTful run creation endpoint for a flow."""
    return await _run_flow_impl(flow_id, body, current_user)


@v1.post("/flows/{flow_id}/run", deprecated=True, tags=["Runs"])
async def run_flow_legacy(
    flow_id: str,
    body: RunFlowRequest,
    current_user: Dict[str, Any] = Depends(get_authenticated_user),
):
    """Backward-compatible alias for flow execution."""
    return await _run_flow_impl(flow_id, body, current_user)


@v1.get("/runs", tags=["Runs"])
async def list_runs(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    current_user: Dict[str, Any] = Depends(get_authenticated_user),
):
    """List all workflow runs with pagination."""
    runs = await WorkflowRunRepository.get_all()
    return paginate(runs, page, page_size)


@v1.get("/runs/{run_id}", tags=["Runs"])
async def get_run(
    run_id: str,
    current_user: Dict[str, Any] = Depends(get_authenticated_user),
):
    """Get a workflow run by ID."""
    run = await WorkflowRunRepository.get_by_run_id(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@v1.get("/runs/{run_id}/trace", tags=["Runs"])
async def get_run_trace(
    run_id: str,
    current_user: Dict[str, Any] = Depends(get_authenticated_user),
):
    """Return normalized full execution trace for a workflow run."""
    run = await WorkflowRunRepository.get_by_run_id(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return _extract_trace_from_run(run)


@v1.get("/runs/{run_id}/diff", tags=["Runs"])
async def get_run_diff(
    run_id: str,
    other_run_id: str = Query(..., min_length=1, description="Run ID to compare against"),
    current_user: Dict[str, Any] = Depends(get_authenticated_user),
):
    """Return structural execution diff between two workflow runs."""
    if run_id == other_run_id:
        raise HTTPException(status_code=400, detail="Cannot diff a run against itself")

    base_run = await WorkflowRunRepository.get_by_run_id(run_id)
    if not base_run:
        raise HTTPException(status_code=404, detail="Base run not found")

    compare_run = await WorkflowRunRepository.get_by_run_id(other_run_id)
    if not compare_run:
        raise HTTPException(status_code=404, detail="Comparison run not found")

    diff_payload = _build_run_diff(base_run, compare_run)
    diff_payload["same_flow"] = base_run.get("flow_id") == compare_run.get("flow_id")
    return diff_payload


@v1.post("/runs/{run_id}/rerun", status_code=202, tags=["Runs"])
async def rerun_workflow(
    run_id: str,
    body: RerunFlowRequest,
    current_user: Dict[str, Any] = Depends(get_authenticated_user),
):
    """Re-run a previous workflow execution using original or overridden input."""
    source_run = await WorkflowRunRepository.get_by_run_id(run_id)
    if not source_run:
        raise HTTPException(status_code=404, detail="Run not found")

    flow_id = source_run.get("flow_id")
    if not isinstance(flow_id, str) or not flow_id:
        raise HTTPException(status_code=400, detail="Source run is missing a valid flow_id")

    flow = await FlowRepository.get_by_id(flow_id)
    if not flow:
        raise HTTPException(status_code=404, detail="Flow not found for source run")

    flow = await Orchestrator.auto_migrate_legacy_nodes(flow, persist=True)
    if not flow:
        raise HTTPException(status_code=404, detail="Flow not found for source run")

    source_input = source_run.get("input_data")
    if not isinstance(source_input, dict):
        source_trace = _extract_trace_from_run(source_run)
        trace_input = source_trace.get("input")
        source_input = trace_input if isinstance(trace_input, dict) else {}

    override_input = body.input if isinstance(body.input, dict) else {}
    if body.merge_with_original_input:
        rerun_input: Dict[str, Any] = dict(source_input)
        rerun_input.update(override_input)
    else:
        rerun_input = dict(override_input)

    rerun_input = _trace_value(rerun_input)
    if not isinstance(rerun_input, dict):
        rerun_input = {}

    new_run_id = await Orchestrator.execute_flow(flow, rerun_input)
    return {
        "run_id": new_run_id,
        "source_run_id": run_id,
        "flow_id": flow_id,
        "input": rerun_input,
    }


@v1.post("/ai/suggest", tags=["Applets"])
async def ai_suggest(
    body: AISuggestRequest,
    current_user: Dict[str, Any] = Depends(get_authenticated_user),
):
    """Generate code suggestions using AI."""
    raise HTTPException(
        status_code=501,
        detail="AI code suggestion is not implemented in the Alpha release. Please check back in a future version."
    )


# ---------------------------------------------------------------------------
# Workflow History + Audit Trail endpoints
# ---------------------------------------------------------------------------

HISTORY_VALID_STATUSES = frozenset({"idle", "running", "success", "error"})


async def _build_history_entry(run: Dict[str, Any]) -> Dict[str, Any]:
    """Build a history entry from a run dict, enriched with flow name."""
    flow_id = run.get("flow_id")
    flow_name = None
    node_count = 0
    if flow_id:
        flow = await FlowRepository.get_by_id(flow_id)
        if flow:
            flow_name = flow.get("name")
            node_count = len(flow.get("nodes", []))

    trace = _extract_trace_from_run(run)
    step_count = len(trace.get("nodes", []))
    steps_succeeded = sum(1 for n in trace.get("nodes", []) if n.get("status") == "success")
    steps_failed = sum(1 for n in trace.get("nodes", []) if n.get("status") == "error")

    input_data = run.get("input_data")
    input_summary = None
    if isinstance(input_data, dict):
        input_summary = {k: (str(v)[:100] if isinstance(v, str) and len(v) > 100 else v)
                         for k, v in list(input_data.items())[:10]}

    output_summary = None
    results = run.get("results")
    if isinstance(results, dict):
        output_keys = [k for k in results.keys() if k != TRACE_RESULTS_KEY]
        output_summary = {"keys": output_keys[:10], "total_keys": len(output_keys)}

    return {
        "run_id": run.get("run_id") or run.get("id"),
        "flow_id": flow_id,
        "flow_name": flow_name,
        "status": run.get("status"),
        "start_time": run.get("start_time"),
        "end_time": run.get("end_time"),
        "duration_ms": trace.get("duration_ms"),
        "node_count": node_count,
        "step_count": step_count,
        "steps_succeeded": steps_succeeded,
        "steps_failed": steps_failed,
        "error": run.get("error"),
        "input_summary": input_summary,
        "output_summary": output_summary,
    }


@v1.get("/history", tags=["History"])
async def list_execution_history(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    status: Optional[str] = Query(None, description="Filter by status: idle, running, success, error"),
    template: Optional[str] = Query(None, description="Filter by flow/template name (substring match)"),
    start_after: Optional[float] = Query(None, description="Filter runs started after this Unix timestamp"),
    start_before: Optional[float] = Query(None, description="Filter runs started before this Unix timestamp"),
    current_user: Dict[str, Any] = Depends(get_authenticated_user),
):
    """List past workflow executions with filtering by status, date range, and template name."""
    if status and status not in HISTORY_VALID_STATUSES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status. Valid: {sorted(HISTORY_VALID_STATUSES)}",
        )

    all_runs = await WorkflowRunRepository.get_all()

    # Sort by start_time descending (most recent first)
    all_runs.sort(key=lambda r: r.get("start_time", 0), reverse=True)

    # Apply filters
    if status:
        all_runs = [r for r in all_runs if r.get("status") == status]

    if start_after is not None:
        all_runs = [r for r in all_runs if (r.get("start_time") or 0) >= start_after]

    if start_before is not None:
        all_runs = [r for r in all_runs if (r.get("start_time") or 0) <= start_before]

    # Template/flow name filter requires flow lookup
    if template:
        template_lower = template.lower()
        filtered = []
        for r in all_runs:
            fid = r.get("flow_id")
            if fid:
                flow = await FlowRepository.get_by_id(fid)
                if flow and template_lower in (flow.get("name", "").lower()):
                    filtered.append(r)
        all_runs = filtered

    total = len(all_runs)

    # Paginate
    start = (page - 1) * page_size
    page_runs = all_runs[start: start + page_size]

    entries = [await _build_history_entry(r) for r in page_runs]

    return {
        "history": entries,
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@v1.get("/history/{run_id}", tags=["History"])
async def get_execution_detail(
    run_id: str,
    current_user: Dict[str, Any] = Depends(get_authenticated_user),
):
    """Get full execution detail for a past run, including step-by-step trace."""
    run = await WorkflowRunRepository.get_by_run_id(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    entry = await _build_history_entry(run)
    trace = _extract_trace_from_run(run)

    return {
        **entry,
        "input_data": run.get("input_data"),
        "trace": trace,
    }


@v1.get("/health", tags=["Health"])
async def health_v1():
    """Service health check — returns status, version, and uptime."""
    return _health_payload()


# ============================================================
# Portfolio Dashboard
# ============================================================

def _discover_yaml_templates() -> List[Dict[str, Any]]:
    """Scan templates/ for YAML workflow definitions."""
    import yaml

    templates_dir = project_root / "templates"
    results: List[Dict[str, Any]] = []
    if not templates_dir.is_dir():
        return results
    for path in sorted(templates_dir.glob("*.yaml")):
        try:
            with open(path) as f:
                data = yaml.safe_load(f)
            if not isinstance(data, dict):
                continue
            results.append({
                "id": data.get("id", path.stem),
                "name": data.get("name", path.stem),
                "description": data.get("description", ""),
                "tags": data.get("tags", []),
                "source": f"templates/{path.name}",
                "node_count": len(data.get("nodes", [])),
                "edge_count": len(data.get("edges", [])),
            })
        except Exception:
            continue
    return results


async def _get_last_run_for_flow_name(flow_name: str) -> Optional[Dict[str, Any]]:
    """Find the most recent run whose flow matches *flow_name*."""
    flows = await FlowRepository.get_all()
    matching_flow_ids = [f["id"] for f in flows if f.get("name") == flow_name]
    if not matching_flow_ids:
        return None
    runs = await WorkflowRunRepository.get_all()
    matching = [r for r in runs if r.get("flow_id") in matching_flow_ids]
    if not matching:
        return None
    matching.sort(key=lambda r: r.get("start_time", 0), reverse=True)
    latest = matching[0]
    return {
        "run_id": latest.get("run_id") or latest.get("id"),
        "flow_id": latest.get("flow_id"),
        "status": latest.get("status"),
        "started_at": latest.get("start_time"),
        "ended_at": latest.get("end_time"),
    }


@v1.get("/dashboard/portfolio", tags=["Dashboard"])
async def portfolio_dashboard(
    current_user: Dict[str, Any] = Depends(get_authenticated_user),
):
    """Portfolio dogfood dashboard — template statuses, provider registry, health."""

    # 1. Templates
    templates = _discover_yaml_templates()
    template_statuses = []
    for tmpl in templates:
        last_run = await _get_last_run_for_flow_name(tmpl["name"])
        template_statuses.append({
            **tmpl,
            "last_run": last_run,
        })

    # 2. Provider registry
    provider_status = []
    for info in LLMProviderRegistry.list_providers():
        d = info.model_dump()
        provider_status.append({
            "name": d["name"],
            "configured": d["configured"],
            "reason": d.get("reason", ""),
            "model_count": len(d.get("models", [])),
        })

    # 3. Health
    uptime_seconds = max(0, int(time.time() - APP_START_TIME))
    db_ok = True
    try:
        async with get_db_session() as session:
            await session.execute(select(1))
    except Exception:
        db_ok = False

    return {
        "templates": template_statuses,
        "template_count": len(template_statuses),
        "providers": provider_status,
        "provider_count": len(provider_status),
        "health": {
            "status": "healthy" if db_ok else "degraded",
            "database": "reachable" if db_ok else "unreachable",
            "uptime_seconds": uptime_seconds,
            "version": API_VERSION,
        },
    }


# ============================================================
# Health Detailed + Metrics
# ============================================================


@v1.get("/health/detailed", tags=["Health"])
async def health_detailed(
    current_user: Dict[str, Any] = Depends(get_authenticated_user),
):
    """Detailed health check with database, providers, and last template run."""
    uptime_seconds = max(0, int(time.time() - APP_START_TIME))

    # Database
    db_ok = True
    try:
        async with get_db_session() as session:
            await session.execute(select(1))
    except Exception:
        db_ok = False

    # Providers
    provider_checks = []
    for info in LLMProviderRegistry.list_providers():
        d = info.model_dump()
        provider_checks.append({
            "name": d["name"],
            "connected": d["configured"],
            "reason": d.get("reason", ""),
        })

    # Last template execution
    snap = metrics.snapshot()
    last_run_at = snap["last_template_run_at"]

    overall = "ok"
    if not db_ok:
        overall = "down"
    elif not any(p["connected"] for p in provider_checks):
        overall = "degraded"

    return {
        "status": overall,
        "uptime_seconds": uptime_seconds,
        "version": API_VERSION,
        "database": {"reachable": db_ok},
        "providers": provider_checks,
        "last_template_run_at": last_run_at,
    }


@v1.get("/metrics", tags=["Health"])
async def get_metrics(
    current_user: Dict[str, Any] = Depends(get_authenticated_user),
):
    """In-memory request metrics: counts, error rate, response time, provider usage."""
    return metrics.snapshot()


# Include versioned router
app.include_router(v1)


# ============================================================
# Health Check (unversioned)
# ============================================================


def _health_payload() -> Dict[str, Any]:
    uptime_seconds = max(0, int(time.time() - APP_START_TIME))
    return {
        "status": "healthy",
        "service": "SynApps Orchestrator API",
        "version": API_VERSION,
        "uptime": uptime_seconds,
    }


@app.get("/health")
async def health():
    """Unversioned health check endpoint."""
    return _health_payload()


@app.get("/")
async def health_root():
    """Root health check endpoint."""
    return _health_payload()


# ============================================================
# WebSocket (versioned, structured protocol with auth & recovery)
# ============================================================


async def _ws_authenticate(websocket: WebSocket) -> Optional[Dict[str, Any]]:
    """Authenticate a WebSocket connection via JWT, API key, or legacy WS token.

    The client must send an ``auth`` message within ``WS_AUTH_TIMEOUT_SECONDS``.
    Supported ``auth`` message shapes::

        {"type": "auth", "token": "<jwt_access_token>"}
        {"type": "auth", "api_key": "<api_key>"}
        {"type": "auth", "token": "<legacy_ws_auth_token>"}

    Returns the authenticated user dict or *None* on failure (connection closed).
    """
    if await _can_use_anonymous_bootstrap():
        try:
            raw = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
            msg = json.loads(raw)
            if msg.get("type") == "auth":
                user = await _ws_try_credentials(msg)
                if user:
                    return user
        except (asyncio.TimeoutError, json.JSONDecodeError, Exception):
            pass
        return {
            "id": "anonymous",
            "email": "anonymous@local",
            "is_active": True,
            "created_at": _utc_now(),
        }

    try:
        raw = await asyncio.wait_for(
            websocket.receive_text(), timeout=float(WS_AUTH_TIMEOUT_SECONDS)
        )
    except asyncio.TimeoutError:
        await websocket.send_json(_ws_message("error", {
            "code": "AUTH_TIMEOUT",
            "message": f"Send auth message within {WS_AUTH_TIMEOUT_SECONDS}s",
        }))
        await websocket.close(code=4002, reason="Authentication timeout")
        return None
    except Exception:
        await websocket.send_json(_ws_message("error", {
            "code": "AUTH_ERROR",
            "message": "Failed to read authentication message",
        }))
        await websocket.close(code=4003, reason="Auth read error")
        return None

    try:
        msg = json.loads(raw)
    except json.JSONDecodeError:
        await websocket.send_json(_ws_message("error", {
            "code": "AUTH_ERROR",
            "message": "Invalid authentication message format - expected JSON",
        }))
        await websocket.close(code=4003, reason="Invalid auth message")
        return None

    if msg.get("type") != "auth":
        await websocket.send_json(_ws_message("error", {
            "code": "AUTH_FAILED",
            "message": "First message must be of type 'auth'",
        }))
        await websocket.close(code=4001, reason="Authentication failed")
        return None

    user = await _ws_try_credentials(msg)
    if not user:
        await websocket.send_json(_ws_message("error", {
            "code": "AUTH_FAILED",
            "message": "Invalid credentials",
        }))
        await websocket.close(code=4001, reason="Authentication failed")
        return None

    return user


async def _ws_try_credentials(msg: dict) -> Optional[Dict[str, Any]]:
    """Try to authenticate from an auth message payload."""
    token = msg.get("token", "")
    if token:
        if WS_AUTH_TOKEN and token == WS_AUTH_TOKEN:
            return {
                "id": "ws_token_user",
                "email": "ws@local",
                "is_active": True,
                "created_at": _utc_now(),
            }
        try:
            return await _authenticate_user_by_jwt(token)
        except HTTPException:
            pass

    api_key = msg.get("api_key", "")
    if api_key:
        try:
            return await _authenticate_user_by_api_key(api_key)
        except HTTPException:
            pass

    return None


@app.websocket("/api/v1/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # --- Authentication phase ---
    user = await _ws_authenticate(websocket)
    if user is None:
        return

    user_id: str = user.get("id", "anonymous")

    # Accept session_id and optional last_seq from query params for reconnection.
    # If last_seq is explicitly provided (including "0"), replay logic will use it.
    requested_session_id = websocket.query_params.get("session_id")
    last_seq_raw = websocket.query_params.get("last_seq")
    last_seq: Optional[int] = None
    if last_seq_raw is not None:
        try:
            last_seq = int(last_seq_raw)
        except (ValueError, TypeError):
            last_seq = 0

    # --- Session creation / resumption ---
    session, reconnected = ws_manager.create_session(
        user_id=user_id,
        websocket=websocket,
        session_id=requested_session_id,
    )
    session_id = session.session_id
    if websocket not in connected_clients:
        connected_clients.append(websocket)

    logger.info(
        f"WebSocket client connected (session={session_id}, "
        f"user={user_id}, reconnected={reconnected}). "
        f"Total clients: {len(ws_manager.connected_websockets)}"
    )

    await websocket.send_json(_ws_message("auth.result", {
        "authenticated": True,
        "session_id": session_id,
        "user_id": user_id,
        "reconnected": reconnected,
        "server_seq": ws_manager.current_seq,
    }))

    # --- Replay missed messages on reconnect ---
    if reconnected and last_seq is not None:
        missed = ws_manager.get_missed_messages(last_seq)
        if missed:
            await websocket.send_json(_ws_message("replay.start", {
                "count": len(missed),
                "from_seq": last_seq + 1,
                "to_seq": missed[-1].get("_seq", 0),
            }))
            for m in missed:
                try:
                    await websocket.send_json(m)
                except Exception:
                    break
            await websocket.send_json(_ws_message("replay.end", {
                "count": len(missed),
            }))

    # --- Server heartbeat task ---
    async def _heartbeat():
        try:
            while True:
                await asyncio.sleep(WS_HEARTBEAT_INTERVAL)
                try:
                    await websocket.send_json(_ws_message("heartbeat", {
                        "server_seq": ws_manager.current_seq,
                    }))
                    session.last_active = time.time()
                except Exception:
                    break
        except asyncio.CancelledError:
            pass

    heartbeat_task = asyncio.create_task(_heartbeat())

    # --- Message loop ---
    try:
        while True:
            raw = await websocket.receive_text()
            session.last_active = time.time()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json(_ws_message("error", {
                    "code": "INVALID_MESSAGE",
                    "message": "Message must be valid JSON",
                }))
                continue

            msg_type = msg.get("type", "")
            msg_id = msg.get("id")

            if msg_type == "ping":
                await websocket.send_json(
                    _ws_message("pong", ref_id=msg_id)
                )

            elif msg_type == "subscribe":
                channel = msg.get("data", {}).get("channel", "")
                if channel:
                    session.subscriptions.add(channel)
                await websocket.send_json(_ws_message(
                    "subscribe.ack",
                    {"channel": channel},
                    ref_id=msg_id,
                ))

            elif msg_type == "unsubscribe":
                channel = msg.get("data", {}).get("channel", "")
                session.subscriptions.discard(channel)
                await websocket.send_json(_ws_message(
                    "unsubscribe.ack",
                    {"channel": channel},
                    ref_id=msg_id,
                ))

            elif msg_type == "get_state":
                await websocket.send_json(_ws_message(
                    "state",
                    {
                        "session_id": session_id,
                        "user_id": user_id,
                        "subscriptions": sorted(session.subscriptions),
                        "server_seq": ws_manager.current_seq,
                        "connected_at": session.connected_at,
                    },
                    ref_id=msg_id,
                ))

            else:
                await websocket.send_json(_ws_message("error", {
                    "code": "UNKNOWN_MESSAGE_TYPE",
                    "message": f"Unknown message type: {msg_type}",
                }, ref_id=msg_id))

    except WebSocketDisconnect:
        logger.info(f"Client disconnected (session={session_id})")
    except Exception as e:
        logger.error(f"WebSocket error (session={session_id}): {e}")
    finally:
        heartbeat_task.cancel()
        ws_manager.remove_session(websocket)
        if websocket in connected_clients:
            connected_clients.remove(websocket)
        logger.info(
            f"WebSocket session {session_id} cleaned up. "
            f"Remaining clients: {len(ws_manager.connected_websockets)}"
        )


# For direct execution
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
