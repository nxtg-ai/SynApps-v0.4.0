"""
SynApps Orchestrator - Core Module

This is the lightweight microkernel that routes messages between applets in a
defined sequence. The orchestrator's job is purely to pass messages and data
between applets.
"""
import asyncio
from abc import ABC, abstractmethod
import importlib
import json
import logging
import math
import os
import sqlite3
import sys
import time
import uuid
from enum import Enum
import threading
from typing import Any, AsyncIterator, Dict, List, Optional, Type
from pathlib import Path

# Load environment variables from .env.development file
from dotenv import load_dotenv

# Load .env.development file from the project root directory
project_root = Path(__file__).parent.parent.parent
dotenv_path = project_root / ".env.development"
load_dotenv(dotenv_path=dotenv_path)

from fastapi import (
    Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect,
    BackgroundTasks, APIRouter, Query, Request,
)
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import httpx
from pydantic import BaseModel, Field, field_validator

# Import database modules
from contextlib import asynccontextmanager
from apps.orchestrator.db import init_db, close_db_connections
from apps.orchestrator.repositories import FlowRepository, WorkflowRunRepository
from apps.orchestrator.models import (
    FlowModel,
    FlowNodeModel,
    FlowEdgeModel,
    WorkflowRunStatusModel,
    ImageGenNodeConfigModel,
    ImageGenRequestModel,
    ImageGenResponseModel,
    ImageModelInfoModel,
    ImageProviderInfoModel,
    LLMMessageModel,
    LLMModelInfoModel,
    MemoryNodeConfigModel,
    MemorySearchResultModel,
    LLMNodeConfigModel,
    LLMProviderInfoModel,
    LLMRequestModel,
    LLMResponseModel,
    LLMStreamChunkModel,
    LLMUsageModel,
    SUPPORTED_MEMORY_BACKENDS,
    SUPPORTED_IMAGE_PROVIDERS,
    SUPPORTED_LLM_PROVIDERS,
)

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
WS_AUTH_TOKEN = os.environ.get("WS_AUTH_TOKEN")
LEGACY_WRITER_NODE_TYPE = "writer"
LEGACY_ARTIST_NODE_TYPE = "artist"
LEGACY_MEMORY_NODE_TYPE = "memory"
LLM_NODE_TYPE = "llm"
IMAGE_NODE_TYPE = "image"
MEMORY_NODE_TYPE = "memory"
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

app = FastAPI(title="SynApps Orchestrator", version=API_VERSION, lifespan=lifespan)

# Configure CORS
backend_cors_origins = os.environ.get("BACKEND_CORS_ORIGINS", "")
backend_cors_origins = backend_cors_origins.split(",")
allowed_origins = [origin.strip() for origin in backend_cors_origins if origin.strip()]

if not allowed_origins:
    logger.warning("No CORS origins specified, allowing all origins in development mode")
    allowed_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

class FlowNodeRequest(BaseModel):
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


class FlowEdgeRequest(BaseModel):
    """Strictly validated flow edge for API requests."""
    id: str = Field(..., min_length=1, max_length=200)
    source: str = Field(..., min_length=1)
    target: str = Field(..., min_length=1)
    animated: bool = False


class CreateFlowRequest(BaseModel):
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


class RunFlowRequest(BaseModel):
    """Strictly validated request body for running a flow."""
    input: Dict[str, Any] = Field(default_factory=dict, description="Input data for the workflow")


class AISuggestRequest(BaseModel):
    """Strictly validated request body for AI suggestions."""
    prompt: str = Field(..., min_length=1, max_length=5000, description="The prompt for AI suggestion")
    context: Optional[str] = Field(None, max_length=10000, description="Optional context for the suggestion")


# ============================================================
# Internal Models
# ============================================================

class AppletStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"

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


# ============================================================
# WebSocket Protocol
# ============================================================

connected_clients: List[WebSocket] = []
ws_sessions: Dict[str, str] = {}  # session_id -> last known state for reconnection
applet_registry: Dict[str, Type['BaseApplet']] = {}


def _ws_message(msg_type: str, data: Optional[dict] = None) -> dict:
    """Create a structured WebSocket message with id, type, data, and timestamp."""
    return {
        "id": str(uuid.uuid4()),
        "type": msg_type,
        "data": data or {},
        "timestamp": time.time(),
    }


async def broadcast_status(status: Dict[str, Any]):
    """Broadcast workflow status to all connected clients using structured messages."""
    if not connected_clients:
        logger.warning("No connected clients to broadcast to")
        return

    broadcast_data = status.copy()
    if "completed_applets" not in broadcast_data:
        broadcast_data["completed_applets"] = []

    message = _ws_message("workflow.status", broadcast_data)

    disconnected = []
    for client in connected_clients:
        try:
            await client.send_json(message)
        except Exception as e:
            logger.error(f"Failed to send to client: {e}")
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
        if not connected_clients:
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
        disconnected = []
        for client in connected_clients:
            try:
                await client.send_json(message)
            except Exception:
                disconnected.append(client)
        for client in disconnected:
            if client in connected_clients:
                connected_clients.remove(client)

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


applet_registry["llm"] = LLMNodeApplet
applet_registry[IMAGE_NODE_TYPE] = ImageGenNodeApplet
applet_registry[MEMORY_NODE_TYPE] = MemoryNodeApplet


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
    async def execute_flow(flow: dict, input_data: Dict[str, Any]) -> str:
        """Execute a flow and return the run ID."""
        run_id = Orchestrator.create_run_id()

        status = {
            "run_id": run_id,
            "flow_id": flow["id"],
            "status": "running",
            "current_applet": None,
            "progress": 0,
            "total_steps": len(flow["nodes"]),
            "start_time": time.time(),
            "results": {},
            "input_data": input_data
        }

        memory_completed_applets = []
        status_dict = status.copy()

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
        status = None
        memory_completed_applets = []

        try:
            status = await workflow_run_repo.get_by_run_id(run_id)

            nodes_by_id = {node["id"]: node for node in flow["nodes"]}

            graph = {}
            for edge in flow["edges"]:
                if edge["source"] not in graph:
                    graph[edge["source"]] = []
                graph[edge["source"]].append(edge["target"])

            target_nodes = set(edge["target"] for edge in flow["edges"])
            start_nodes = [node["id"] for node in flow["nodes"] if node["id"] not in target_nodes]

            if not start_nodes:
                if status and isinstance(status, dict):
                    status["status"] = "error"
                    status["error"] = "No start node found in flow"
                    status["end_time"] = time.time()
                    await workflow_run_repo.save(status)
                    broadcast_data = status.copy()
                    broadcast_data["completed_applets"] = memory_completed_applets
                    await broadcast_status_fn(broadcast_data)
                else:
                    await broadcast_status_fn({
                        "run_id": run_id,
                        "status": "error",
                        "error": "No start node found in flow and status record is invalid",
                        "completed_applets": []
                    })
                return

            context = {
                "input": input_data,
                "results": {},
                "run_id": run_id
            }

            if status and isinstance(status, dict):
                status["input_data"] = input_data

            current_nodes = start_nodes
            visited = set()

            while current_nodes:
                next_nodes = []

                for node_id in current_nodes:
                    if node_id in visited:
                        continue

                    visited.add(node_id)
                    node = nodes_by_id[node_id]

                    if status and isinstance(status, dict):
                        status["current_applet"] = node["type"]
                        status["progress"] += 1

                    if node_id not in memory_completed_applets:
                        memory_completed_applets.append(node_id)

                    if status and isinstance(status, dict):
                        broadcast_data = status.copy()
                        broadcast_data["completed_applets"] = memory_completed_applets
                        await workflow_run_repo.save(status)
                        await broadcast_status_fn(broadcast_data)

                    if node["type"].lower() in ["start", "end"]:
                        if (
                            node["type"].lower() == "start"
                            and "data" in node
                            and "parsedInputData" in node["data"]
                        ):
                            parsed_input = node["data"]["parsedInputData"]
                            if parsed_input and isinstance(parsed_input, dict):
                                context["input"] = parsed_input
                                if status and isinstance(status, dict):
                                    status["input_data"] = parsed_input

                        if node_id in graph:
                            next_nodes.extend(graph[node_id])
                        continue

                    try:
                        applet = await Orchestrator.load_applet(node["type"].lower())

                        message_content = input_data
                        message_metadata = {"node_id": node_id, "run_id": run_id}

                        if "data" in node and isinstance(node["data"], dict):
                            message_metadata["node_data"] = node["data"]

                            if node["type"].lower() == "writer" and "systemPrompt" in node["data"]:
                                message_metadata["system_prompt"] = node["data"]["systemPrompt"]

                            if node["type"].lower() == "artist":
                                if "systemPrompt" in node["data"]:
                                    message_metadata["system_prompt"] = node["data"]["systemPrompt"]
                                if "generator" in node["data"]:
                                    message_metadata["generator"] = node["data"]["generator"]

                        message = AppletMessage(
                            content=message_content,
                            context=context,
                            metadata=message_metadata
                        )

                        response = await applet.on_message(message)

                        context["results"][node_id] = {
                            "type": node["type"],
                            "output": response.content
                        }
                        context.update(response.context)

                        if node_id not in memory_completed_applets:
                            memory_completed_applets.append(node_id)

                        if status and isinstance(status, dict):
                            broadcast_data = status.copy()
                            broadcast_data["completed_applets"] = memory_completed_applets
                            await broadcast_status_fn(broadcast_data)

                        if node_id in graph:
                            next_nodes.extend(graph[node_id])

                            for edge in flow["edges"]:
                                if edge["source"] == node_id:
                                    edge["animated"] = True

                    except Exception as e:
                        logger.error(f"Error executing applet '{node['type']}': {e}")
                        if status and isinstance(status, dict):
                            status["status"] = "error"
                            status["error"] = f"Error in applet '{node['type']}': {str(e)}"
                            status["end_time"] = time.time()
                            await workflow_run_repo.save(status)
                            broadcast_data = status.copy()
                            broadcast_data["completed_applets"] = memory_completed_applets
                            await broadcast_status_fn(broadcast_data)
                        return

                current_nodes = next_nodes

            if status and isinstance(status, dict):
                status["status"] = "success"
                status["end_time"] = time.time()
                status["results"] = context["results"]
                if "input_data" not in status or not status["input_data"]:
                    status["input_data"] = input_data
                await workflow_run_repo.save(status)
                broadcast_data = status.copy()
                broadcast_data["completed_applets"] = memory_completed_applets
                await broadcast_status_fn(broadcast_data)

        except Exception as e:
            logger.error(f"Error executing workflow: {e}")
            if status and isinstance(status, dict):
                status["status"] = "error"
                status["error"] = f"Workflow execution error: {str(e)}"
                status["end_time"] = time.time()
                await workflow_run_repo.save(status)
                broadcast_data = status.copy()
                broadcast_data["completed_applets"] = memory_completed_applets
                await broadcast_status_fn(broadcast_data)
            else:
                await broadcast_status_fn({
                    "run_id": run_id,
                    "status": "error",
                    "error": f"Workflow execution error: {str(e)}",
                    "completed_applets": memory_completed_applets
                })


# ============================================================
# API Routes (v1)
# ============================================================

v1 = APIRouter(prefix="/api/v1", tags=["v1"])


@v1.get("/applets")
async def list_applets(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
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


@v1.get("/llm/providers")
async def list_llm_providers():
    """List supported LLM providers and model catalogs."""
    providers = [provider.model_dump() for provider in LLMProviderRegistry.list_providers()]
    return {"providers": providers}


@v1.get("/image/providers")
async def list_image_providers():
    """List supported image generation providers and model catalogs."""
    providers = [provider.model_dump() for provider in ImageProviderRegistry.list_providers()]
    return {"providers": providers}


@v1.post("/flows", status_code=201)
async def create_flow(flow: CreateFlowRequest):
    """Create or update a flow with strict validation."""
    flow_id = flow.id if flow.id else str(uuid.uuid4())

    flow_dict = flow.model_dump()
    flow_dict["id"] = flow_id
    flow_dict, migrated = Orchestrator.migrate_legacy_nodes(flow_dict)
    if migrated:
        logger.info("Applied legacy node migration while creating flow '%s'", flow_id)
    await FlowRepository.save(flow_dict)
    return {"message": "Flow created", "id": flow_id}


@v1.get("/flows")
async def list_flows(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
):
    """List all flows with pagination."""
    flows = await FlowRepository.get_all()
    migrated_flows: List[Dict[str, Any]] = []
    for flow in flows:
        migrated_flow = await Orchestrator.auto_migrate_legacy_nodes(flow, persist=True)
        if isinstance(migrated_flow, dict):
            migrated_flows.append(migrated_flow)
    return paginate(migrated_flows, page, page_size)


@v1.get("/flows/{flow_id}")
async def get_flow(flow_id: str):
    """Get a flow by ID."""
    flow = await FlowRepository.get_by_id(flow_id)
    if not flow:
        raise HTTPException(status_code=404, detail="Flow not found")
    flow = await Orchestrator.auto_migrate_legacy_nodes(flow, persist=True)
    return flow


@v1.delete("/flows/{flow_id}")
async def delete_flow(flow_id: str):
    """Delete a flow."""
    flow = await FlowRepository.get_by_id(flow_id)
    if not flow:
        raise HTTPException(status_code=404, detail="Flow not found")
    await FlowRepository.delete(flow_id)
    return {"message": "Flow deleted"}


@v1.post("/flows/{flow_id}/run")
async def run_flow(flow_id: str, body: RunFlowRequest):
    """Run a flow with the given input data."""
    flow = await FlowRepository.get_by_id(flow_id)
    if not flow:
        raise HTTPException(status_code=404, detail="Flow not found")
    flow = await Orchestrator.auto_migrate_legacy_nodes(flow, persist=True)
    run_id = await Orchestrator.execute_flow(flow, body.input)
    return {"run_id": run_id}


@v1.get("/runs")
async def list_runs(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
):
    """List all workflow runs with pagination."""
    runs = await WorkflowRunRepository.get_all()
    return paginate(runs, page, page_size)


@v1.get("/runs/{run_id}")
async def get_run(run_id: str):
    """Get a workflow run by ID."""
    run = await WorkflowRunRepository.get_by_run_id(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@v1.post("/ai/suggest")
async def ai_suggest(body: AISuggestRequest):
    """Generate code suggestions using AI."""
    raise HTTPException(
        status_code=501,
        detail="AI code suggestion is not implemented in the Alpha release. Please check back in a future version."
    )


# Include versioned router
app.include_router(v1)


# ============================================================
# Health Check (unversioned)
# ============================================================

@app.get("/")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "SynApps Orchestrator API",
        "version": API_VERSION,
    }


# ============================================================
# WebSocket (versioned, structured protocol with auth)
# ============================================================

@app.websocket("/api/v1/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # Authentication phase
    authenticated = WS_AUTH_TOKEN is None  # Auto-auth if no token configured
    session_id: Optional[str] = None

    if not authenticated:
        try:
            raw = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
            msg = json.loads(raw)
            if msg.get("type") == "auth" and msg.get("token") == WS_AUTH_TOKEN:
                authenticated = True
                session_id = msg.get("session_id")
            else:
                await websocket.send_json(_ws_message("error", {
                    "code": "AUTH_FAILED",
                    "message": "Invalid or missing authentication token",
                }))
                await websocket.close(code=4001, reason="Authentication failed")
                return
        except asyncio.TimeoutError:
            await websocket.send_json(_ws_message("error", {
                "code": "AUTH_TIMEOUT",
                "message": "Authentication timeout - send auth message within 10 seconds",
            }))
            await websocket.close(code=4002, reason="Authentication timeout")
            return
        except (json.JSONDecodeError, Exception):
            await websocket.send_json(_ws_message("error", {
                "code": "AUTH_ERROR",
                "message": "Invalid authentication message format",
            }))
            await websocket.close(code=4003, reason="Invalid auth message")
            return

    # Assign or resume a session
    reconnected = False
    if session_id and session_id in ws_sessions:
        reconnected = True
    else:
        session_id = str(uuid.uuid4())
    ws_sessions[session_id] = "connected"

    # Register client
    connected_clients.append(websocket)
    logger.info(f"WebSocket client connected (session={session_id}, reconnected={reconnected}). Total clients: {len(connected_clients)}")

    await websocket.send_json(_ws_message("auth.result", {
        "authenticated": True,
        "session_id": session_id,
        "reconnected": reconnected,
    }))

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
                msg_type = msg.get("type", "")

                if msg_type == "ping":
                    await websocket.send_json(_ws_message("pong"))
                elif msg_type == "subscribe":
                    channel = msg.get("data", {}).get("channel", "")
                    await websocket.send_json(_ws_message("subscribe.ack", {"channel": channel}))
                else:
                    await websocket.send_json(_ws_message("error", {
                        "code": "UNKNOWN_MESSAGE_TYPE",
                        "message": f"Unknown message type: {msg_type}",
                    }))
            except json.JSONDecodeError:
                await websocket.send_json(_ws_message("error", {
                    "code": "INVALID_MESSAGE",
                    "message": "Message must be valid JSON",
                }))
    except WebSocketDisconnect:
        logger.info(f"Client disconnected (session={session_id})")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if websocket in connected_clients:
            connected_clients.remove(websocket)
        if session_id:
            ws_sessions[session_id] = "disconnected"
        logger.info(f"WebSocket client disconnected. Remaining clients: {len(connected_clients)}")


# For direct execution
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
