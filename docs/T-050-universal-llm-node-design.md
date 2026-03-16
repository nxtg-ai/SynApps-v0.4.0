# T-050: Universal LLM Node and Provider Adapter Design

**Author:** Claude (Forge Architect)
**Date:** 2026-02-17
**Status:** Design Complete

---

## 1. Problem Statement

The current `WriterApplet` is hardcoded to OpenAI's `gpt-4o` model. Users cannot:
- Choose a different provider (Anthropic, Google, Ollama, custom endpoints)
- Select a specific model (e.g., Claude 3.5 Sonnet vs GPT-4o)
- Configure generation parameters per node (temperature, max tokens, etc.)
- Use local models via Ollama for privacy-sensitive workflows
- Use their own API keys at the node level (currently env-var-only)

The `ArtistApplet` already demonstrates a multi-provider pattern (Stability + OpenAI fallback) but implements it ad-hoc with interleaved provider logic rather than a clean abstraction.

This design replaces the single-provider `WriterApplet` with a **universal LLM Node** backed by a **provider adapter pattern**, making the system model-agnostic.

---

## 2. Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                     LLM Node (Applet)                    │
│                                                          │
│  ┌────────────────┐   ┌──────────────────────────────┐   │
│  │  on_message()  │──>│  ProviderRegistry.get(name)  │   │
│  │  - extract cfg │   │  - returns LLMProvider       │   │
│  │  - build req   │   └──────────────────────────────┘   │
│  │  - call adapter│                                      │
│  │  - format resp │   ┌──────────────────────────────┐   │
│  └────────────────┘   │   LLMProvider (ABC)          │   │
│                       │   .complete(request) -> resp  │   │
│                       │   .stream(request) -> chunks  │   │
│                       │   .get_models() -> list       │   │
│                       │   .validate_config() -> bool  │   │
│                       └──────────┬───────────────────┘   │
│                                  │                       │
│         ┌────────────┬───────────┼───────────┬────────┐  │
│         │            │           │           │        │  │
│    ┌────▼────┐ ┌─────▼───┐ ┌────▼────┐ ┌────▼──┐ ┌───▼─┐│
│    │ OpenAI  │ │Anthropic│ │ Google  │ │Ollama │ │Cust.││
│    │ Adapter │ │ Adapter │ │ Adapter │ │Adapter│ │Adpt.││
│    └─────────┘ └─────────┘ └─────────┘ └───────┘ └─────┘│
└──────────────────────────────────────────────────────────┘
```

**Key principle:** The LLM Node itself contains zero provider-specific logic. All provider differences are encapsulated in adapters behind the `LLMProvider` interface.

---

## 3. Provider Interface (ABC)

### 3.1 Core Types

```python
# File: apps/applets/llm/providers/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Optional


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    """A single message in a conversation."""
    role: Role
    content: str


@dataclass
class LLMRequest:
    """Provider-agnostic request to an LLM."""
    messages: list[Message]
    model: str
    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 1.0
    stop_sequences: list[str] = field(default_factory=list)
    # Provider-specific passthrough (e.g., OpenAI's response_format, Anthropic's thinking)
    extra: dict = field(default_factory=dict)


@dataclass
class LLMResponse:
    """Provider-agnostic response from an LLM."""
    content: str
    model: str
    provider: str
    usage: "TokenUsage"
    finish_reason: str = "stop"
    # Raw provider response for debugging/advanced use
    raw: dict = field(default_factory=dict)


@dataclass
class TokenUsage:
    """Token usage statistics."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class StreamChunk:
    """A single chunk from a streaming response."""
    content: str
    done: bool = False
    usage: Optional[TokenUsage] = None


@dataclass
class ModelInfo:
    """Metadata about a model available from a provider."""
    id: str
    name: str
    provider: str
    context_window: int
    supports_streaming: bool = True
    supports_vision: bool = False
    max_output_tokens: Optional[int] = None
```

### 3.2 Abstract Provider

```python
# File: apps/applets/llm/providers/base.py (continued)

class LLMProvider(ABC):
    """Abstract base class for LLM provider adapters."""

    # Provider identity
    name: str  # e.g., "openai", "anthropic"

    @abstractmethod
    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Send a completion request and return the full response."""
        ...

    @abstractmethod
    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        """Send a completion request and stream the response."""
        ...

    @abstractmethod
    def get_models(self) -> list[ModelInfo]:
        """Return the list of models this provider supports."""
        ...

    @abstractmethod
    def validate_config(self) -> tuple[bool, str]:
        """Check that the provider is properly configured.

        Returns (True, "") on success, (False, "reason") on failure.
        """
        ...

    def default_model(self) -> str:
        """Return the default model ID for this provider."""
        models = self.get_models()
        return models[0].id if models else ""
```

### 3.3 Design Decisions for the Interface

| Decision | Rationale |
|----------|-----------|
| **`messages` list (not single prompt)** | Supports multi-turn context. The orchestrator can inject conversation history from upstream nodes. Future-proofs for chat-based workflows. |
| **`extra` dict on LLMRequest** | Each provider has unique features (OpenAI's JSON mode, Anthropic's extended thinking, Google's safety settings). The `extra` dict passes these through without polluting the common interface. |
| **`raw` dict on LLMResponse** | Preserves the full provider response for debugging, billing analysis, or provider-specific post-processing without coupling the interface to any provider's schema. |
| **Separate `complete` vs `stream`** | Not all workflows need streaming. `complete` is simpler and returns a single object. `stream` is opt-in for real-time UI updates. |
| **`validate_config` method** | Enables the `/applets` endpoint to report which providers are available vs misconfigured, so the UI can disable unavailable options. |
| **`get_models` method** | Lets the frontend dynamically populate model dropdowns per provider, rather than hardcoding model lists. |

---

## 4. Provider Adapters

### 4.1 OpenAI Adapter

```python
# File: apps/applets/llm/providers/openai_provider.py

class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or "https://api.openai.com/v1"

    async def complete(self, request: LLMRequest) -> LLMResponse:
        payload = {
            "model": request.model,
            "messages": [{"role": m.role.value, "content": m.content} for m in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p,
        }
        if request.stop_sequences:
            payload["stop"] = request.stop_sequences
        payload.update(request.extra)  # response_format, tools, etc.

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        choice = data["choices"][0]
        usage = data.get("usage", {})
        return LLMResponse(
            content=choice["message"]["content"],
            model=data["model"],
            provider=self.name,
            usage=TokenUsage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            ),
            finish_reason=choice.get("finish_reason", "stop"),
            raw=data,
        )

    # stream() implementation: SSE parsing of /chat/completions?stream=true
    # get_models(): returns gpt-4o, gpt-4o-mini, gpt-4.1, o3-mini, etc.
    # validate_config(): checks self.api_key is set

    def get_models(self) -> list[ModelInfo]:
        return [
            ModelInfo(id="gpt-4o", name="GPT-4o", provider=self.name, context_window=128000, supports_vision=True, max_output_tokens=16384),
            ModelInfo(id="gpt-4o-mini", name="GPT-4o Mini", provider=self.name, context_window=128000, supports_vision=True, max_output_tokens=16384),
            ModelInfo(id="gpt-4.1", name="GPT-4.1", provider=self.name, context_window=1047576, supports_vision=True, max_output_tokens=32768),
            ModelInfo(id="o3-mini", name="o3-mini", provider=self.name, context_window=200000, supports_vision=False, max_output_tokens=100000),
        ]

    def validate_config(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, "OPENAI_API_KEY not set"
        return True, ""
```

### 4.2 Anthropic Adapter

```python
# File: apps/applets/llm/providers/anthropic_provider.py

class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

    async def complete(self, request: LLMRequest) -> LLMResponse:
        # Anthropic API separates system from messages
        system_text = ""
        messages = []
        for m in request.messages:
            if m.role == Role.SYSTEM:
                system_text += m.content + "\n"
            else:
                messages.append({"role": m.role.value, "content": m.content})

        payload = {
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
        }
        if system_text.strip():
            payload["system"] = system_text.strip()
        if request.stop_sequences:
            payload["stop_sequences"] = request.stop_sequences
        payload.update(request.extra)  # thinking, tool_use, etc.

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        # Anthropic returns content as list of blocks
        content_text = ""
        for block in data.get("content", []):
            if block["type"] == "text":
                content_text += block["text"]

        usage = data.get("usage", {})
        return LLMResponse(
            content=content_text,
            model=data["model"],
            provider=self.name,
            usage=TokenUsage(
                prompt_tokens=usage.get("input_tokens", 0),
                completion_tokens=usage.get("output_tokens", 0),
                total_tokens=usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            ),
            finish_reason=data.get("stop_reason", "end_turn"),
            raw=data,
        )

    def get_models(self) -> list[ModelInfo]:
        return [
            ModelInfo(id="claude-sonnet-4-6", name="Claude Sonnet 4.6", provider=self.name, context_window=200000, supports_vision=True, max_output_tokens=16000),
            ModelInfo(id="claude-haiku-4-5-20251001", name="Claude Haiku 4.5", provider=self.name, context_window=200000, supports_vision=True, max_output_tokens=8192),
            ModelInfo(id="claude-opus-4-6", name="Claude Opus 4.6", provider=self.name, context_window=200000, supports_vision=True, max_output_tokens=32000),
        ]

    def validate_config(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, "ANTHROPIC_API_KEY not set"
        return True, ""
```

### 4.3 Google (Gemini) Adapter

```python
# File: apps/applets/llm/providers/google_provider.py

class GoogleProvider(LLMProvider):
    name = "google"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")

    async def complete(self, request: LLMRequest) -> LLMResponse:
        # Google Gemini API uses a different message format
        contents = []
        system_instruction = None
        for m in request.messages:
            if m.role == Role.SYSTEM:
                system_instruction = {"parts": [{"text": m.content}]}
            else:
                role = "user" if m.role == Role.USER else "model"
                contents.append({"role": role, "parts": [{"text": m.content}]})

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": request.temperature,
                "maxOutputTokens": request.max_tokens,
                "topP": request.top_p,
            },
        }
        if system_instruction:
            payload["systemInstruction"] = system_instruction
        if request.stop_sequences:
            payload["generationConfig"]["stopSequences"] = request.stop_sequences

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{request.model}:generateContent?key={self.api_key}"

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        candidate = data["candidates"][0]
        content_text = ""
        for part in candidate["content"]["parts"]:
            content_text += part.get("text", "")

        usage = data.get("usageMetadata", {})
        return LLMResponse(
            content=content_text,
            model=request.model,
            provider=self.name,
            usage=TokenUsage(
                prompt_tokens=usage.get("promptTokenCount", 0),
                completion_tokens=usage.get("candidatesTokenCount", 0),
                total_tokens=usage.get("totalTokenCount", 0),
            ),
            finish_reason=candidate.get("finishReason", "STOP"),
            raw=data,
        )

    def get_models(self) -> list[ModelInfo]:
        return [
            ModelInfo(id="gemini-2.5-flash", name="Gemini 2.5 Flash", provider=self.name, context_window=1048576, supports_vision=True, max_output_tokens=65536),
            ModelInfo(id="gemini-2.5-pro", name="Gemini 2.5 Pro", provider=self.name, context_window=1048576, supports_vision=True, max_output_tokens=65536),
        ]

    def validate_config(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, "GOOGLE_API_KEY not set"
        return True, ""
```

### 4.4 Ollama Adapter (Local Models)

```python
# File: apps/applets/llm/providers/ollama_provider.py

class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

    async def complete(self, request: LLMRequest) -> LLMResponse:
        # Ollama uses OpenAI-compatible /api/chat endpoint
        payload = {
            "model": request.model,
            "messages": [{"role": m.role.value, "content": m.content} for m in request.messages],
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
                "top_p": request.top_p,
            },
        }
        if request.stop_sequences:
            payload["options"]["stop"] = request.stop_sequences

        async with httpx.AsyncClient(timeout=300.0) as client:  # Longer timeout for local models
            resp = await client.post(f"{self.base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()

        return LLMResponse(
            content=data["message"]["content"],
            model=data.get("model", request.model),
            provider=self.name,
            usage=TokenUsage(
                prompt_tokens=data.get("prompt_eval_count", 0),
                completion_tokens=data.get("eval_count", 0),
                total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
            ),
            finish_reason="stop" if data.get("done") else "length",
            raw=data,
        )

    def get_models(self) -> list[ModelInfo]:
        # Dynamic: query Ollama for installed models
        # Fallback to common defaults if Ollama is not running
        return [
            ModelInfo(id="llama3.1", name="Llama 3.1 8B", provider=self.name, context_window=131072, max_output_tokens=4096),
            ModelInfo(id="mistral", name="Mistral 7B", provider=self.name, context_window=32768, max_output_tokens=4096),
            ModelInfo(id="codellama", name="Code Llama", provider=self.name, context_window=16384, max_output_tokens=4096),
        ]

    def validate_config(self) -> tuple[bool, str]:
        # No API key needed, but check if Ollama is reachable
        # This is a sync check; real implementation would cache the result
        return True, ""  # Defer actual connectivity check to first use
```

### 4.5 Custom/OpenAI-Compatible Adapter

Handles any endpoint that speaks the OpenAI chat completions API (vLLM, Together AI, Fireworks, Groq, LM Studio, etc.).

```python
# File: apps/applets/llm/providers/custom_provider.py

class CustomProvider(LLMProvider):
    """Adapter for any OpenAI-compatible API endpoint."""
    name = "custom"

    def __init__(self, base_url: str, api_key: str = "", model_id: str = "default"):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_id = model_id

    async def complete(self, request: LLMRequest) -> LLMResponse:
        # Reuse OpenAI-format payload
        payload = {
            "model": request.model or self.model_id,
            "messages": [{"role": m.role.value, "content": m.content} for m in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        choice = data["choices"][0]
        usage = data.get("usage", {})
        return LLMResponse(
            content=choice["message"]["content"],
            model=data.get("model", request.model),
            provider=self.name,
            usage=TokenUsage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            ),
            finish_reason=choice.get("finish_reason", "stop"),
            raw=data,
        )

    def get_models(self) -> list[ModelInfo]:
        return [ModelInfo(id=self.model_id, name=self.model_id, provider=self.name, context_window=0)]

    def validate_config(self) -> tuple[bool, str]:
        if not self.base_url:
            return False, "base_url is required for custom provider"
        return True, ""
```

---

## 5. Provider Registry

```python
# File: apps/applets/llm/providers/__init__.py

from .base import LLMProvider
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider
from .google_provider import GoogleProvider
from .ollama_provider import OllamaProvider
from .custom_provider import CustomProvider


class ProviderRegistry:
    """Registry of available LLM providers."""

    _providers: dict[str, type[LLMProvider]] = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "google": GoogleProvider,
        "ollama": OllamaProvider,
        "custom": CustomProvider,
    }

    @classmethod
    def get(cls, name: str, **kwargs) -> LLMProvider:
        """Instantiate a provider by name.

        Args:
            name: Provider identifier (e.g., "openai", "anthropic")
            **kwargs: Provider-specific init arguments (api_key, base_url, etc.)

        Raises:
            ValueError: If provider name is not registered.
        """
        provider_cls = cls._providers.get(name)
        if not provider_cls:
            raise ValueError(
                f"Unknown provider '{name}'. Available: {list(cls._providers.keys())}"
            )
        return provider_cls(**kwargs)

    @classmethod
    def register(cls, name: str, provider_cls: type[LLMProvider]) -> None:
        """Register a new provider at runtime."""
        cls._providers[name] = provider_cls

    @classmethod
    def list_providers(cls) -> list[dict]:
        """Return metadata for all registered providers with config status."""
        result = []
        for name, provider_cls in cls._providers.items():
            try:
                instance = provider_cls()
                valid, reason = instance.validate_config()
                models = instance.get_models()
            except Exception as e:
                valid, reason, models = False, str(e), []

            result.append({
                "name": name,
                "configured": valid,
                "reason": reason if not valid else "",
                "models": [{"id": m.id, "name": m.name, "context_window": m.context_window} for m in models],
            })
        return result
```

---

## 6. LLM Node (Applet Implementation)

```python
# File: apps/applets/llm/applet.py

import logging
from typing import Dict, Any
from apps.orchestrator.main import BaseApplet, AppletMessage
from apps.applets.llm.providers import ProviderRegistry
from apps.applets.llm.providers.base import LLMRequest, Message, Role

logger = logging.getLogger("llm-applet")


class LlmApplet(BaseApplet):
    """
    Universal LLM Node - Generates text using any supported LLM provider.

    Supports OpenAI, Anthropic, Google, Ollama, and custom OpenAI-compatible endpoints.
    Provider and model are configurable per node instance.
    """

    VERSION = "1.0.0"
    CAPABILITIES = ["text-generation", "summarization", "content-creation", "multi-provider"]

    async def on_message(self, message: AppletMessage) -> AppletMessage:
        """Process a message using the configured LLM provider."""
        content = message.content
        context = message.context
        node_data = message.metadata.get("node_data", {})

        # --- 1. Extract the user prompt ---
        input_text = self._extract_prompt(content)

        # --- 2. Resolve provider configuration ---
        config = self._resolve_config(context, node_data)

        # --- 3. Build the LLM request ---
        messages = []
        if config["system_prompt"]:
            messages.append(Message(role=Role.SYSTEM, content=config["system_prompt"]))
        messages.append(Message(role=Role.USER, content=input_text))

        request = LLMRequest(
            messages=messages,
            model=config["model"],
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
            top_p=config["top_p"],
            stop_sequences=config["stop_sequences"],
            extra=config["extra"],
        )

        # --- 4. Call the provider ---
        provider_kwargs = {}
        if config.get("api_key"):
            provider_kwargs["api_key"] = config["api_key"]
        if config.get("base_url"):
            provider_kwargs["base_url"] = config["base_url"]

        try:
            provider = ProviderRegistry.get(config["provider"], **provider_kwargs)
            valid, reason = provider.validate_config()
            if not valid:
                return AppletMessage(
                    content=f"Provider '{config['provider']}' is not configured: {reason}",
                    context={**context},
                    metadata={"applet": "llm", "status": "error"},
                )

            response = await provider.complete(request)
        except Exception as e:
            logger.error(f"LLM provider error ({config['provider']}): {e}")
            return AppletMessage(
                content=f"Error calling {config['provider']}: {str(e)}",
                context={**context},
                metadata={"applet": "llm", "status": "error"},
            )

        # --- 5. Return the response ---
        return AppletMessage(
            content=response.content,
            context={**context},
            metadata={
                "applet": "llm",
                "provider": response.provider,
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                "finish_reason": response.finish_reason,
            },
        )

    def _extract_prompt(self, content: Any) -> str:
        """Extract the user prompt from various input formats."""
        if isinstance(content, str):
            return content
        if isinstance(content, dict):
            return content.get("prompt") or content.get("text") or content.get("input", "")
        return str(content) if content else ""

    def _resolve_config(self, context: dict, node_data: dict) -> dict:
        """Resolve LLM configuration from node data, context, and defaults.

        Priority (highest to lowest):
        1. Node data (per-node config set in the UI)
        2. Context (upstream node or workflow-level config)
        3. Defaults
        """
        def pick(key: str, default):
            return node_data.get(key, context.get(key, default))

        return {
            "provider": pick("provider", "openai"),
            "model": pick("model", "gpt-4o"),
            "system_prompt": pick("system_prompt", pick("systemPrompt", "")),
            "temperature": float(pick("temperature", 0.7)),
            "max_tokens": int(pick("max_tokens", pick("maxTokens", 1024))),
            "top_p": float(pick("top_p", pick("topP", 1.0))),
            "stop_sequences": pick("stop_sequences", []),
            "api_key": node_data.get("api_key") or context.get("api_key"),
            "base_url": node_data.get("base_url") or context.get("base_url"),
            "extra": node_data.get("extra", {}),
        }
```

---

## 7. Node Configuration Schema

This is the complete set of options available when configuring an LLM node in the workflow editor.

### 7.1 Full Configuration Model

```python
# File: apps/applets/llm/config.py

from pydantic import BaseModel, Field
from typing import Optional


class LLMNodeConfig(BaseModel):
    """Configuration schema for the LLM Node.

    This is stored in FlowNode.data and sent as node_data to the applet.
    """
    # Display
    label: str = Field("LLM", max_length=100, description="Display name for this node")

    # Provider selection
    provider: str = Field("openai", description="LLM provider: openai, anthropic, google, ollama, custom")
    model: str = Field("gpt-4o", description="Model ID (e.g., gpt-4o, claude-sonnet-4-6, gemini-2.5-flash)")

    # Prompt
    system_prompt: str = Field("", description="System instructions for the LLM")

    # Generation parameters
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(1024, ge=1, le=200000, description="Maximum tokens to generate")
    top_p: float = Field(1.0, ge=0.0, le=1.0, description="Nucleus sampling threshold")
    stop_sequences: list[str] = Field(default_factory=list, description="Stop sequences")

    # Advanced / per-node credentials (optional)
    api_key: Optional[str] = Field(None, description="Override API key for this node")
    base_url: Optional[str] = Field(None, description="Custom endpoint URL (for custom/ollama)")

    # Provider-specific extras
    extra: dict = Field(default_factory=dict, description="Provider-specific parameters")
```

### 7.2 Configuration Options Table

| Option | Type | Default | Description | UI Widget |
|--------|------|---------|-------------|-----------|
| `label` | string | "LLM" | Node display name | Text input |
| `provider` | enum | "openai" | Provider selection | Dropdown: OpenAI, Anthropic, Google, Ollama, Custom |
| `model` | string | "gpt-4o" | Model ID | Dropdown (dynamic, filtered by provider) |
| `system_prompt` | string | "" | System instructions | Textarea |
| `temperature` | float | 0.7 | Randomness (0-2) | Slider |
| `max_tokens` | int | 1024 | Max output length | Number input |
| `top_p` | float | 1.0 | Nucleus sampling | Slider (advanced) |
| `stop_sequences` | string[] | [] | Stop sequences | Tag input (advanced) |
| `api_key` | string? | null | Per-node API key override | Password input (advanced) |
| `base_url` | string? | null | Custom endpoint URL | Text input (shown for custom/ollama) |
| `extra` | dict | {} | Provider-specific params | JSON editor (advanced) |

### 7.3 Frontend Config Panel Behavior

The node config modal should be **provider-aware**:

1. User selects `provider` → model dropdown refreshes to show that provider's models
2. If `provider = "custom"` → show `base_url` input (required) and `api_key` input
3. If `provider = "ollama"` → show `base_url` with default `http://localhost:11434`
4. Temperature and max_tokens sliders always visible
5. `top_p`, `stop_sequences`, `api_key`, and `extra` hidden behind "Advanced" toggle

---

## 8. API Endpoints

### 8.1 New endpoint: List LLM providers and models

```
GET /api/v1/llm/providers
```

Response:
```json
{
  "providers": [
    {
      "name": "openai",
      "configured": true,
      "reason": "",
      "models": [
        {"id": "gpt-4o", "name": "GPT-4o", "context_window": 128000},
        {"id": "gpt-4o-mini", "name": "GPT-4o Mini", "context_window": 128000}
      ]
    },
    {
      "name": "anthropic",
      "configured": false,
      "reason": "ANTHROPIC_API_KEY not set",
      "models": [
        {"id": "claude-sonnet-4-6", "name": "Claude Sonnet 4.6", "context_window": 200000}
      ]
    }
  ]
}
```

This endpoint is called by the frontend to populate the provider/model dropdowns and show configuration status.

### 8.2 Orchestrator integration

Add to `main.py`:
```python
@v1.get("/llm/providers")
async def list_llm_providers():
    """List available LLM providers and their models."""
    from apps.applets.llm.providers import ProviderRegistry
    return {"providers": ProviderRegistry.list_providers()}
```

---

## 9. Orchestrator Changes

### 9.1 Node data passthrough

The orchestrator currently passes `systemPrompt` and `generator` from `node["data"]` into `message_metadata` with type-specific `if` branches (lines 516-524 of `main.py`). This needs to change to a generic pattern:

**Before** (type-specific):
```python
if node["type"].lower() == "writer" and "systemPrompt" in node["data"]:
    message_metadata["system_prompt"] = node["data"]["systemPrompt"]
```

**After** (generic):
```python
# Always pass node.data as node_data in metadata
if "data" in node:
    message_metadata["node_data"] = node["data"]
```

This is the only change needed in the orchestrator execution engine. The LLM Node reads its configuration from `message.metadata["node_data"]`, not from hardcoded metadata keys.

### 9.2 Backward compatibility

The generic `node_data` passthrough is **backward compatible** because:
- Existing WriterApplet/ArtistApplet read from `context["system_prompt"]` and `context["image_generator"]`, which are unaffected
- The new LlmApplet reads from `metadata["node_data"]`
- Both patterns coexist during migration

---

## 10. Migration Plan: WriterApplet to LLM Node

### Phase 1: Add LLM Node alongside WriterApplet (non-breaking)

1. Create `apps/applets/llm/` directory with provider adapters
2. Register `llm` as a new applet type in the orchestrator
3. Add `/api/v1/llm/providers` endpoint
4. Add `llm` node type to the frontend node palette
5. Create the LLM node config modal with provider/model selection
6. **WriterApplet continues working unchanged**

### Phase 2: Frontend parity

7. Add LLM node to AppletNode component (icon, color, description)
8. Build provider-aware NodeConfigModal for the LLM node type
9. Add provider status indicators in the UI (configured vs not)
10. Test all 5 providers end-to-end

### Phase 3: Deprecate WriterApplet

11. Add deprecation notice to WriterApplet: "Use LLM node instead"
12. Add a migration utility that converts existing `writer` nodes in saved flows to `llm` nodes with `provider: "openai", model: "gpt-4o"` config
13. Mark `writer` as deprecated in the applet registry metadata

### Phase 4: Remove WriterApplet

14. Remove `apps/applets/writer/` directory
15. Update all workflow templates that use `writer` to use `llm`
16. Remove writer-specific branches in AppletNode.tsx and NodeConfigModal.tsx

### Migration utility for existing flows

```python
async def migrate_writer_to_llm(flow: dict) -> dict:
    """Convert writer nodes to llm nodes in a saved flow."""
    for node in flow["nodes"]:
        if node["type"] == "writer":
            node["type"] = "llm"
            node["data"] = {
                **node.get("data", {}),
                "provider": "openai",
                "model": "gpt-4o",
                "temperature": 0.7,
                "max_tokens": 1000,
            }
            # Rename systemPrompt -> system_prompt for consistency
            if "systemPrompt" in node["data"]:
                node["data"]["system_prompt"] = node["data"].pop("systemPrompt")
    return flow
```

---

## 11. File Structure

```
apps/applets/llm/
├── __init__.py
├── applet.py                        # LlmApplet (BaseApplet implementation)
├── config.py                        # LLMNodeConfig (Pydantic validation)
└── providers/
    ├── __init__.py                   # ProviderRegistry + re-exports
    ├── base.py                       # LLMProvider ABC + data types
    ├── openai_provider.py            # OpenAI adapter
    ├── anthropic_provider.py         # Anthropic adapter
    ├── google_provider.py            # Google Gemini adapter
    ├── ollama_provider.py            # Ollama adapter
    └── custom_provider.py            # OpenAI-compatible custom endpoints
```

---

## 12. Frontend Changes

### 12.1 Type additions

```typescript
// Add to src/types/index.ts

export interface LLMProviderInfo {
  name: string;
  configured: boolean;
  reason: string;
  models: LLMModelInfo[];
}

export interface LLMModelInfo {
  id: string;
  name: string;
  context_window: number;
}

export interface LLMNodeData {
  label: string;
  provider: string;
  model: string;
  system_prompt: string;
  temperature: number;
  max_tokens: number;
  top_p: number;
  stop_sequences: string[];
  api_key?: string;
  base_url?: string;
  extra: Record<string, any>;
}
```

### 12.2 AppletNode updates

Add to the icon/color/description switch statements in `AppletNode.tsx`:

```typescript
case 'llm':
  icon = '🤖';
  color = 'rgba(139, 92, 246, 0.05)';    // Purple
  accentColor = '#8b5cf6';
  description = 'Generates text with any LLM';
```

### 12.3 NodeConfigModal updates

Add an `llm` case to `renderFormFields()` that:
1. Shows a provider dropdown (fetches from `/api/v1/llm/providers`)
2. Shows a model dropdown (filtered by selected provider)
3. Shows temperature slider, max_tokens input, system_prompt textarea
4. Shows "Advanced" expandable section with top_p, stop_sequences, api_key, base_url

### 12.4 API service addition

```typescript
// Add to ApiService.ts
public async getLLMProviders(): Promise<LLMProviderInfo[]> {
  const response = await this.api.get('/llm/providers');
  return response.data.providers;
}
```

---

## 13. Testing Strategy

### 13.1 Unit tests per provider

Each adapter gets its own test file with mocked HTTP responses:

```python
# apps/orchestrator/tests/test_llm_providers.py

@pytest.mark.asyncio
async def test_openai_complete():
    provider = OpenAIProvider(api_key="test")
    # Mock httpx response
    # Assert LLMResponse fields are correctly mapped

@pytest.mark.asyncio
async def test_anthropic_system_prompt_extraction():
    provider = AnthropicProvider(api_key="test")
    # Verify system messages are separated from user messages

@pytest.mark.asyncio
async def test_ollama_timeout():
    provider = OllamaProvider(base_url="http://localhost:99999")
    # Assert proper error handling for unreachable Ollama

@pytest.mark.asyncio
async def test_custom_provider_no_auth():
    provider = CustomProvider(base_url="http://localhost:8080")
    # Verify no Authorization header when api_key is empty
```

### 13.2 LLM Node integration tests

```python
# apps/orchestrator/tests/test_llm_applet.py

@pytest.mark.asyncio
async def test_llm_applet_default_config():
    # No node_data → defaults to openai/gpt-4o

@pytest.mark.asyncio
async def test_llm_applet_anthropic_provider():
    # node_data = {provider: "anthropic", model: "claude-sonnet-4-6"}

@pytest.mark.asyncio
async def test_llm_applet_invalid_provider():
    # node_data = {provider: "nonexistent"} → error message

@pytest.mark.asyncio
async def test_llm_applet_unconfigured_provider():
    # Provider exists but API key missing → descriptive error

@pytest.mark.asyncio
async def test_config_priority():
    # node_data overrides context overrides defaults
```

### 13.3 Migration tests

```python
@pytest.mark.asyncio
async def test_migrate_writer_to_llm():
    flow = {"nodes": [{"id": "1", "type": "writer", "data": {"systemPrompt": "Be helpful"}}]}
    migrated = await migrate_writer_to_llm(flow)
    assert migrated["nodes"][0]["type"] == "llm"
    assert migrated["nodes"][0]["data"]["provider"] == "openai"
    assert migrated["nodes"][0]["data"]["system_prompt"] == "Be helpful"
    assert "systemPrompt" not in migrated["nodes"][0]["data"]
```

---

## 14. Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Provider API rate limits** | Medium | Token usage is tracked in `LLMResponse.usage`; future: add rate-limit middleware per provider |
| **API key leakage via per-node keys** | High | Per-node `api_key` is never persisted to the database in plain text. The frontend masks the field. Keys should be stored in a secrets manager in production. For MVP, accept the risk with a warning in docs. |
| **Ollama cold-start latency** | Low | Ollama adapter uses 300s timeout. UI should show "Loading model..." state for local providers. |
| **Provider API changes break adapters** | Medium | Each adapter is isolated. A breaking change in one provider's API affects only that adapter file. Pin provider SDK versions if we add them later. |
| **WriterApplet removal breaks saved flows** | High | Phase 3 migration utility converts flows automatically. Keep WriterApplet working during transition. Add an `aliases` map in the orchestrator that routes `writer` → `llm` for backward compatibility. |
| **Context window exceeded** | Medium | `ModelInfo.context_window` enables future validation before sending requests. For now, rely on provider-side error messages. |

---

## 15. Future Extensions (Out of Scope)

These are explicitly **not** part of this design but are enabled by it:

- **Streaming to frontend via WebSocket** - `LLMProvider.stream()` is defined but not wired to the UI yet
- **Vision/multimodal** - `ModelInfo.supports_vision` flag is present; `Message.content` could become `str | list[ContentBlock]`
- **Tool use / function calling** - Pass through via `LLMRequest.extra`
- **Token budget management** - `TokenUsage` tracking enables cost controls
- **Provider fallback chains** - Like ArtistApplet's Stability→OpenAI pattern, but generalized (try provider A, fall back to B)
- **Model comparison mode** - Send same prompt to multiple providers, show results side-by-side
