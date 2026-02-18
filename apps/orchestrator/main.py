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
import sys
import time
import uuid
from enum import Enum
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
    LLMMessageModel,
    LLMModelInfoModel,
    LLMNodeConfigModel,
    LLMProviderInfoModel,
    LLMRequestModel,
    LLMResponseModel,
    LLMStreamChunkModel,
    LLMUsageModel,
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


applet_registry["llm"] = LLMNodeApplet


# ============================================================
# Orchestrator Core
# ============================================================

class Orchestrator:
    """Core orchestration engine that executes applet flows."""

    @staticmethod
    async def load_applet(applet_type: str) -> BaseApplet:
        """Dynamically load an applet by type."""
        if applet_type in applet_registry:
            return applet_registry[applet_type]()

        if applet_type == "llm":
            applet_registry["llm"] = LLMNodeApplet
            return LLMNodeApplet()

        try:
            module_path = f"apps.applets.{applet_type}.applet"
            module = importlib.import_module(module_path)
            applet_class = getattr(module, f"{applet_type.capitalize()}Applet")
            applet_registry[applet_type] = applet_class
            return applet_class()
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to load applet '{applet_type}': {e}")
            raise ValueError(f"Applet type '{applet_type}' not found")

    @staticmethod
    def create_run_id() -> str:
        """Generate a unique run ID."""
        return str(uuid.uuid4())

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


@v1.post("/flows", status_code=201)
async def create_flow(flow: CreateFlowRequest):
    """Create or update a flow with strict validation."""
    flow_id = flow.id if flow.id else str(uuid.uuid4())

    flow_dict = flow.model_dump()
    flow_dict["id"] = flow_id
    await FlowRepository.save(flow_dict)
    return {"message": "Flow created", "id": flow_id}


@v1.get("/flows")
async def list_flows(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
):
    """List all flows with pagination."""
    flows = await FlowRepository.get_all()
    return paginate(flows, page, page_size)


@v1.get("/flows/{flow_id}")
async def get_flow(flow_id: str):
    """Get a flow by ID."""
    flow = await FlowRepository.get_by_id(flow_id)
    if not flow:
        raise HTTPException(status_code=404, detail="Flow not found")
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
