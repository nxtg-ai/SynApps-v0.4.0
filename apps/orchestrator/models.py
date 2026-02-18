"""
Database and API models for the SynApps orchestrator.

This module defines SQLAlchemy ORM models for database persistence
and Pydantic models for API validation.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import Boolean, Float, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""


class Flow(Base):
    """ORM model for workflow flows."""

    __tablename__ = "flows"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)

    nodes: Mapped[List["FlowNode"]] = relationship(
        "FlowNode",
        back_populates="flow",
        cascade="all, delete-orphan",
    )
    edges: Mapped[List["FlowEdge"]] = relationship(
        "FlowEdge",
        back_populates="flow",
        cascade="all, delete-orphan",
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert ORM model to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
        }


class FlowNode(Base):
    """ORM model for workflow nodes."""

    __tablename__ = "flow_nodes"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    flow_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("flows.id", ondelete="CASCADE"),
        nullable=False,
    )
    type: Mapped[str] = mapped_column(String, nullable=False)
    position_x: Mapped[float] = mapped_column(Float, nullable=False)
    position_y: Mapped[float] = mapped_column(Float, nullable=False)
    data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)

    flow: Mapped["Flow"] = relationship("Flow", back_populates="nodes")

    def to_dict(self) -> Dict[str, Any]:
        """Convert ORM model to dictionary."""
        return {
            "id": self.id,
            "type": self.type,
            "position": {
                "x": self.position_x,
                "y": self.position_y,
            },
            "data": self.data or {},
        }


class FlowEdge(Base):
    """ORM model for workflow edges."""

    __tablename__ = "flow_edges"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    flow_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("flows.id", ondelete="CASCADE"),
        nullable=False,
    )
    source: Mapped[str] = mapped_column(String, nullable=False)
    target: Mapped[str] = mapped_column(String, nullable=False)
    animated: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    flow: Mapped["Flow"] = relationship("Flow", back_populates="edges")

    def to_dict(self) -> Dict[str, Any]:
        """Convert ORM model to dictionary."""
        return {
            "id": self.id,
            "source": self.source,
            "target": self.target,
            "animated": self.animated,
        }


class WorkflowRun(Base):
    """ORM model for workflow runs."""

    __tablename__ = "workflow_runs"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    flow_id: Mapped[Optional[str]] = mapped_column(
        String,
        ForeignKey("flows.id", ondelete="SET NULL"),
        nullable=True,
    )
    status: Mapped[str] = mapped_column(String, nullable=False, default="idle")
    current_applet: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    progress: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_steps: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    start_time: Mapped[float] = mapped_column(Float, nullable=False)
    end_time: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    results: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    input_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    completed_applets: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert ORM model to dictionary."""
        return {
            "run_id": self.id,
            "flow_id": self.flow_id,
            "status": self.status,
            "current_applet": self.current_applet,
            "progress": self.progress,
            "total_steps": self.total_steps,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "results": self.results or {},
            "error": self.error,
            "input_data": self.input_data,
            "completed_applets": self.completed_applets or [],
        }


class FlowNodeModel(BaseModel):
    """API model for flow nodes."""

    id: str
    type: str
    position: Dict[str, float]
    data: Dict[str, Any] = Field(default_factory=dict)


class FlowEdgeModel(BaseModel):
    """API model for flow edges."""

    id: str
    source: str
    target: str
    animated: bool = False


class FlowModel(BaseModel):
    """API model for flows."""

    id: Optional[str] = None
    name: str
    nodes: List[FlowNodeModel] = Field(default_factory=list)
    edges: List[FlowEdgeModel] = Field(default_factory=list)


class WorkflowRunStatusModel(BaseModel):
    """API model for workflow run status."""

    run_id: str
    flow_id: str
    status: str = "idle"
    current_applet: Optional[str] = None
    progress: int = 0
    total_steps: int = 0
    start_time: float = Field(default_factory=lambda: time.time())
    end_time: Optional[float] = None
    results: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    input_data: Optional[Dict[str, Any]] = None
    completed_applets: List[str] = Field(default_factory=list)

    def model_dump(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Ensure `results` remains a dictionary in serialized output."""
        result = super().model_dump(*args, **kwargs)
        if result.get("results") is None:
            result["results"] = {}
        return result


SUPPORTED_LLM_PROVIDERS = ("openai", "anthropic", "google", "ollama", "custom")


class LLMMessageModel(BaseModel):
    """Provider-agnostic LLM message."""

    role: str = Field(..., min_length=1)
    content: str = Field(..., description="Message text content")

    @field_validator("role")
    @classmethod
    def validate_role(cls, value: str) -> str:
        role = value.strip().lower()
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError("role must be one of: system, user, assistant, tool")
        return role


class LLMNodeConfigModel(BaseModel):
    """Configuration schema for the universal LLM node."""

    model_config = ConfigDict(extra="allow")

    label: str = Field("LLM", max_length=100)
    provider: str = Field("openai")
    model: Optional[str] = None
    system_prompt: str = ""
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(1024, ge=1, le=32768)
    top_p: float = Field(1.0, ge=0.0, le=1.0)
    stop_sequences: List[str] = Field(default_factory=list)
    stream: bool = False
    structured_output: bool = False
    json_schema: Optional[Dict[str, Any]] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout_seconds: float = Field(120.0, gt=0.0, le=600.0)
    headers: Dict[str, str] = Field(default_factory=dict)
    extra: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, value: str) -> str:
        provider = value.strip().lower()
        if provider not in SUPPORTED_LLM_PROVIDERS:
            raise ValueError(
                f"provider must be one of: {', '.join(SUPPORTED_LLM_PROVIDERS)}"
            )
        return provider


class LLMRequestModel(BaseModel):
    """Provider-agnostic LLM completion request."""

    messages: List[LLMMessageModel] = Field(default_factory=list)
    model: str = Field(..., min_length=1)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(1024, ge=1, le=32768)
    top_p: float = Field(1.0, ge=0.0, le=1.0)
    stop_sequences: List[str] = Field(default_factory=list)
    stream: bool = False
    structured_output: bool = False
    json_schema: Optional[Dict[str, Any]] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


class LLMUsageModel(BaseModel):
    """Token usage summary."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMStreamChunkModel(BaseModel):
    """Single chunk from a streaming completion."""

    content: str = ""
    done: bool = False
    usage: Optional[LLMUsageModel] = None


class LLMResponseModel(BaseModel):
    """Provider-agnostic completion response."""

    content: str
    model: str
    provider: str
    usage: LLMUsageModel = Field(default_factory=LLMUsageModel)
    finish_reason: str = "stop"
    raw: Dict[str, Any] = Field(default_factory=dict)


class LLMModelInfoModel(BaseModel):
    """Metadata for a model exposed by a provider."""

    id: str
    name: str
    provider: str
    context_window: int = 0
    supports_streaming: bool = True
    supports_vision: bool = False
    max_output_tokens: Optional[int] = None


class LLMProviderInfoModel(BaseModel):
    """Provider availability and model catalog."""

    name: str
    configured: bool
    reason: str = ""
    models: List[LLMModelInfoModel] = Field(default_factory=list)


SUPPORTED_IMAGE_PROVIDERS = ("openai", "stability", "flux")


class ImageGenNodeConfigModel(BaseModel):
    """Configuration schema for the universal image generation node."""

    model_config = ConfigDict(extra="allow")

    label: str = Field("Image Gen", max_length=100)
    provider: str = Field("openai")
    model: Optional[str] = None
    size: str = Field("1024x1024")
    style: str = Field("photorealistic")
    quality: str = Field("standard")
    n: int = Field(1, ge=1, le=4)
    response_format: str = Field("b64_json")
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout_seconds: float = Field(120.0, gt=0.0, le=600.0)
    headers: Dict[str, str] = Field(default_factory=dict)
    extra: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, value: str) -> str:
        provider = value.strip().lower()
        if provider not in SUPPORTED_IMAGE_PROVIDERS:
            raise ValueError(
                f"provider must be one of: {', '.join(SUPPORTED_IMAGE_PROVIDERS)}"
            )
        return provider

    @field_validator("response_format")
    @classmethod
    def validate_response_format(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"b64_json", "url"}:
            raise ValueError("response_format must be one of: b64_json, url")
        return normalized


class ImageGenRequestModel(BaseModel):
    """Provider-agnostic image generation request."""

    prompt: str = Field(..., min_length=1, max_length=10000)
    negative_prompt: str = ""
    model: str = Field(..., min_length=1)
    size: str = "1024x1024"
    style: str = "photorealistic"
    quality: str = "standard"
    n: int = Field(1, ge=1, le=4)
    response_format: str = Field("b64_json")
    extra: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("response_format")
    @classmethod
    def validate_response_format(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"b64_json", "url"}:
            raise ValueError("response_format must be one of: b64_json, url")
        return normalized


class ImageGenResponseModel(BaseModel):
    """Provider-agnostic image generation response."""

    images: List[str] = Field(default_factory=list)
    model: str
    provider: str
    revised_prompt: Optional[str] = None
    raw: Dict[str, Any] = Field(default_factory=dict)


class ImageModelInfoModel(BaseModel):
    """Metadata for an image model exposed by a provider."""

    id: str
    name: str
    provider: str
    supports_base64: bool = True
    supports_url: bool = True
    max_images: int = 1


class ImageProviderInfoModel(BaseModel):
    """Provider availability and model catalog for image generation."""

    name: str
    configured: bool
    reason: str = ""
    models: List[ImageModelInfoModel] = Field(default_factory=list)


SUPPORTED_MEMORY_BACKENDS = ("sqlite_fts", "chroma")


class MemoryNodeConfigModel(BaseModel):
    """Configuration schema for the persistent memory node."""

    model_config = ConfigDict(extra="allow")

    label: str = Field("Memory", max_length=100)
    operation: str = Field("store")
    backend: str = Field("sqlite_fts")
    namespace: str = Field("default", min_length=1, max_length=200)
    key: Optional[str] = Field(None, max_length=200)
    query: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    top_k: int = Field(5, ge=1, le=50)
    persist_path: Optional[str] = None
    collection: str = Field("synapps_memory", min_length=1, max_length=200)
    include_metadata: bool = True
    extra: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("operation")
    @classmethod
    def validate_operation(cls, value: str) -> str:
        operation = value.strip().lower()
        if operation not in {"store", "retrieve", "delete", "clear"}:
            raise ValueError("operation must be one of: store, retrieve, delete, clear")
        return operation

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, value: str) -> str:
        backend = value.strip().lower()
        if backend not in SUPPORTED_MEMORY_BACKENDS:
            raise ValueError(
                f"backend must be one of: {', '.join(SUPPORTED_MEMORY_BACKENDS)}"
            )
        return backend

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("tags")
    @classmethod
    def normalize_tags(cls, value: List[str]) -> List[str]:
        unique: List[str] = []
        for item in value:
            cleaned = item.strip()
            if cleaned and cleaned not in unique:
                unique.append(cleaned)
        return unique


class MemorySearchResultModel(BaseModel):
    """Search result payload for memory retrieval responses."""

    key: str
    data: Any
    score: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


SUPPORTED_HTTP_METHODS = ("GET", "POST", "PUT", "DELETE")


class HTTPRequestNodeConfigModel(BaseModel):
    """Configuration schema for the HTTP request node."""

    model_config = ConfigDict(extra="allow")

    label: str = Field("HTTP Request", max_length=100)
    url: str = Field(..., min_length=1, max_length=5000)
    method: str = Field("GET")
    headers: Dict[str, str] = Field(default_factory=dict)
    query_params: Dict[str, Any] = Field(default_factory=dict)
    body_template: Optional[Any] = None
    body_type: str = Field("auto")
    timeout_seconds: float = Field(30.0, gt=0.0, le=600.0)
    allow_redirects: bool = True
    verify_ssl: bool = True
    include_response_headers: bool = True
    extra: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("url cannot be blank")
        return normalized

    @field_validator("method")
    @classmethod
    def validate_method(cls, value: str) -> str:
        method = value.strip().upper()
        if method not in SUPPORTED_HTTP_METHODS:
            raise ValueError(
                f"method must be one of: {', '.join(SUPPORTED_HTTP_METHODS)}"
            )
        return method

    @field_validator("body_type")
    @classmethod
    def validate_body_type(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"auto", "json", "text", "form", "none"}:
            raise ValueError("body_type must be one of: auto, json, text, form, none")
        return normalized


SUPPORTED_CODE_LANGUAGES = ("python", "javascript")


class CodeNodeConfigModel(BaseModel):
    """Configuration schema for the sandboxed code execution node."""

    model_config = ConfigDict(extra="allow")

    label: str = Field("Code", max_length=100)
    language: str = Field("python")
    code: str = Field("", max_length=200000)
    timeout_seconds: float = Field(5.0, gt=0.0, le=120.0)
    cpu_time_seconds: int = Field(3, ge=1, le=60)
    memory_limit_mb: int = Field(256, ge=64, le=2048)
    max_output_bytes: int = Field(262144, ge=1024, le=1048576)
    working_dir: str = Field("/tmp")
    env: Dict[str, str] = Field(default_factory=dict)
    extra: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("language")
    @classmethod
    def validate_language(cls, value: str) -> str:
        normalized = value.strip().lower()
        alias_map = {
            "py": "python",
            "python3": "python",
            "js": "javascript",
            "node": "javascript",
            "nodejs": "javascript",
        }
        normalized = alias_map.get(normalized, normalized)
        if normalized not in SUPPORTED_CODE_LANGUAGES:
            raise ValueError(
                f"language must be one of: {', '.join(SUPPORTED_CODE_LANGUAGES)}"
            )
        return normalized

    @field_validator("working_dir")
    @classmethod
    def validate_working_dir(cls, value: str) -> str:
        normalized = value.strip() or "/tmp"
        if not normalized.startswith("/tmp"):
            raise ValueError("working_dir must be under /tmp")
        return normalized


SUPPORTED_TRANSFORM_OPERATIONS = (
    "json_path",
    "template",
    "regex_replace",
    "split_join",
)


class TransformNodeConfigModel(BaseModel):
    """Configuration schema for the transform node."""

    model_config = ConfigDict(extra="allow")

    label: str = Field("Transform", max_length=100)
    operation: str = Field("template")
    source: Any = Field(default="{{content}}")
    json_path: str = Field("$")
    template: str = Field("{{source}}")
    regex_pattern: str = Field("", max_length=5000)
    regex_replacement: str = Field("")
    regex_flags: str = Field("")
    regex_count: int = Field(0, ge=0, le=10000)
    split_delimiter: str = Field(",")
    split_maxsplit: int = Field(-1, ge=-1, le=10000)
    split_index: Optional[int] = Field(None, ge=0, le=100000)
    join_delimiter: str = Field(",")
    return_list: bool = False
    strip_items: bool = False
    drop_empty: bool = False
    extra: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("operation")
    @classmethod
    def validate_operation(cls, value: str) -> str:
        normalized = value.strip().lower().replace("-", "_")
        alias_map = {
            "jsonpath": "json_path",
            "json_extract": "json_path",
            "extract": "json_path",
            "template_string": "template",
            "string_template": "template",
            "format": "template",
            "regex": "regex_replace",
            "replace": "regex_replace",
            "split": "split_join",
            "join": "split_join",
            "splitjoin": "split_join",
        }
        normalized = alias_map.get(normalized, normalized)
        if normalized not in SUPPORTED_TRANSFORM_OPERATIONS:
            raise ValueError(
                f"operation must be one of: {', '.join(SUPPORTED_TRANSFORM_OPERATIONS)}"
            )
        return normalized

    @field_validator("json_path")
    @classmethod
    def validate_json_path(cls, value: str) -> str:
        normalized = value.strip() or "$"
        if not normalized.startswith("$"):
            normalized = f"${normalized if normalized.startswith('.') else f'.{normalized}'}"
        return normalized

    @field_validator("regex_flags")
    @classmethod
    def validate_regex_flags(cls, value: str) -> str:
        normalized = value.strip().lower()
        allowed = {"i", "m", "s", "x"}
        unique = []
        for flag in normalized:
            if flag not in allowed:
                raise ValueError("regex_flags may contain only: i, m, s, x")
            if flag not in unique:
                unique.append(flag)
        return "".join(unique)
