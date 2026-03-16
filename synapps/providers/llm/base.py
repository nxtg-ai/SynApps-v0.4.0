"""Abstract base class for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class ProviderError(Exception):
    """Raised when a provider encounters an error during completion."""


class ProviderNotFoundError(ProviderError):
    """Raised when a requested provider is not registered."""


@dataclass
class ModelInfo:
    """Metadata for a model offered by a provider."""

    id: str
    name: str
    context_window: int = 0
    max_output_tokens: Optional[int] = None
    supports_streaming: bool = True
    supports_vision: bool = False


@dataclass
class LLMResponse:
    """Provider-agnostic completion response."""

    content: str
    model: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    finish_reason: str = "stop"
    raw: Dict[str, Any] = field(default_factory=dict)


class BaseLLMProvider(ABC):
    """Common interface every LLM provider must implement.

    Subclasses MUST set ``name`` as a class attribute and implement
    :meth:`complete`, :meth:`get_models`, and :meth:`validate`.
    """

    name: str = ""

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        model: str,
        *,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> LLMResponse:
        """Run a non-streaming completion and return the result."""

    @abstractmethod
    def get_models(self) -> List[ModelInfo]:
        """Return the list of models this provider exposes."""

    @abstractmethod
    def validate(self) -> tuple[bool, str]:
        """Check whether the provider is properly configured.

        Returns ``(True, "")`` on success, or ``(False, reason)`` on failure.
        """
