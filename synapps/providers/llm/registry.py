"""Provider registry with auto-discovery and fallback support."""

from __future__ import annotations

import importlib
import inspect
import logging
import pathlib
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Type

from synapps.providers.llm.base import BaseLLMProvider, ProviderNotFoundError

logger = logging.getLogger(__name__)


class ProviderRegistry:
    """Thread-safe registry that maps provider names to classes.

    Supports both class-level (shared) and instance-level registries.
    Use the class methods for the global shared registry, or instantiate
    for an isolated registry (useful in tests).
    """

    _global_providers: Dict[str, Type[BaseLLMProvider]] = {}

    def __init__(self) -> None:
        self._providers: Dict[str, Type[BaseLLMProvider]] = {}

    # --- Instance-level API (isolated registries) ---

    def register(self, provider_cls: Type[BaseLLMProvider]) -> None:
        """Register a provider class. Uses ``provider_cls.name`` as the key."""
        name = provider_cls.name.lower().strip()
        if not name:
            raise ValueError("Provider class must have a non-empty 'name' attribute")
        self._providers[name] = provider_cls

    def get(self, name: str) -> Type[BaseLLMProvider]:
        """Look up a provider class by name. Raises ProviderNotFoundError if missing."""
        key = name.lower().strip()
        cls = self._providers.get(key)
        if cls is None:
            available = list(self._providers.keys())
            raise ProviderNotFoundError(
                f"Unknown provider '{name}'. Available: {available}"
            )
        return cls

    def unregister(self, name: str) -> bool:
        """Remove a provider. Returns True if it existed, False otherwise."""
        return self._providers.pop(name.lower().strip(), None) is not None

    def list_providers(self) -> List[str]:
        """Return sorted list of registered provider names."""
        return sorted(self._providers.keys())

    def has(self, name: str) -> bool:
        """Check whether a provider is registered."""
        return name.lower().strip() in self._providers

    def clear(self) -> None:
        """Remove all registered providers."""
        self._providers.clear()

    def get_with_fallback(
        self, name: str, fallback: Optional[str] = None
    ) -> Type[BaseLLMProvider]:
        """Try ``name`` first; fall back to ``fallback`` if unavailable."""
        key = name.lower().strip()
        cls = self._providers.get(key)
        if cls is not None:
            return cls
        if fallback:
            fb_cls = self._providers.get(fallback.lower().strip())
            if fb_cls is not None:
                return fb_cls
        raise ProviderNotFoundError(
            f"Provider '{name}' not found and fallback '{fallback}' also unavailable"
        )

    def provider_info(self, name: str) -> Dict[str, Any]:
        """Return detailed info for a single registered provider."""
        provider_cls = self.get(name)
        instance = provider_cls()
        connected, reason = instance.validate()
        models = instance.get_models()
        return {
            "name": provider_cls.name,
            "connected": connected,
            "reason": reason,
            "model_count": len(models),
            "models": [asdict(m) for m in models],
        }

    def all_providers_info(self) -> List[Dict[str, Any]]:
        """Return info for every registered provider."""
        return [self.provider_info(n) for n in self.list_providers()]

    def provider_health(self, name: str) -> Dict[str, Any]:
        """Run a health check on a single provider."""
        provider_cls = self.get(name)
        instance = provider_cls()
        connected, reason = instance.validate()
        models = instance.get_models()
        return {
            "name": provider_cls.name,
            "status": "ok" if connected else "unavailable",
            "connected": connected,
            "reason": reason,
            "model_count": len(models),
        }

    # --- Class-level (global) API ---

    @classmethod
    def register_global(cls, provider_cls: Type[BaseLLMProvider]) -> None:
        """Register a provider in the global shared registry."""
        name = provider_cls.name.lower().strip()
        if not name:
            raise ValueError("Provider class must have a non-empty 'name' attribute")
        cls._global_providers[name] = provider_cls

    @classmethod
    def get_global(cls, name: str) -> Type[BaseLLMProvider]:
        """Look up a provider from the global registry."""
        key = name.lower().strip()
        provider = cls._global_providers.get(key)
        if provider is None:
            available = list(cls._global_providers.keys())
            raise ProviderNotFoundError(
                f"Unknown provider '{name}'. Available: {available}"
            )
        return provider

    @classmethod
    def list_global(cls) -> List[str]:
        """Return sorted list of globally registered provider names."""
        return sorted(cls._global_providers.keys())

    @classmethod
    def clear_global(cls) -> None:
        """Remove all globally registered providers."""
        cls._global_providers.clear()

    @classmethod
    def auto_discover(cls) -> None:
        """Scan providers/ directory and register every BaseLLMProvider subclass."""
        providers_dir = pathlib.Path(__file__).resolve().parent
        _scan_directory(providers_dir, "synapps.providers.llm", cls)

    @classmethod
    def auto_discover_directory(cls, directory: pathlib.Path, base_module: str) -> int:
        """Scan an arbitrary directory for BaseLLMProvider subclasses.

        Returns the number of providers newly registered.
        """
        before = len(cls._global_providers)
        _scan_directory(directory, base_module, cls)
        return len(cls._global_providers) - before


def _scan_directory(
    directory: pathlib.Path, base_module: str, registry: type[ProviderRegistry]
) -> None:
    """Walk *directory* for .py files, import them, and register providers."""
    if not directory.is_dir():
        return
    for py_file in sorted(directory.glob("*.py")):
        if py_file.name.startswith("_"):
            continue
        module_name = f"{base_module}.{py_file.stem}"
        try:
            mod = importlib.import_module(module_name)
        except Exception:
            logger.warning("Failed to import provider module %s", module_name)
            continue
        for _attr_name, obj in inspect.getmembers(mod, inspect.isclass):
            if (
                issubclass(obj, BaseLLMProvider)
                and obj is not BaseLLMProvider
                and getattr(obj, "name", "")
            ):
                registry.register_global(obj)
                logger.debug("Auto-discovered provider: %s", obj.name)
