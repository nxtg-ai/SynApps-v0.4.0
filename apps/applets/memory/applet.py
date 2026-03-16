"""
MemoryApplet - legacy compatibility layer for memory nodes.

Memory persistence is now routed through the universal Memory Node.
"""
import logging
from typing import Any, Dict

from apps.orchestrator.main import AppletMessage, BaseApplet, MemoryNodeApplet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("memory-applet")


class MemoryApplet(BaseApplet):
    """
    Legacy memory applet wrapper backed by the persistent memory node.

    Capabilities:
    - Context storage
    - Information retrieval
    - Memory management
    """

    VERSION = "1.0.0"
    CAPABILITIES = [
        "context-storage",
        "information-retrieval",
        "memory-management",
        "memory-node-compatible",
    ]

    def __init__(self):
        self._memory_applet = MemoryNodeApplet()

    async def on_message(self, message: AppletMessage) -> AppletMessage:
        """Process an incoming message by routing to the persistent memory node."""
        logger.info("Memory Applet received message")

        context = message.context if isinstance(message.context, dict) else {}
        metadata = message.metadata if isinstance(message.metadata, dict) else {}
        operation = self._resolve_operation(message.content)

        if operation not in {"store", "retrieve", "delete", "clear"}:
            return AppletMessage(
                content={"error": f"Invalid operation: {operation}"},
                context=context,
                metadata={"applet": "memory", "status": "error"},
            )

        memory_config = self._resolve_memory_config(message)
        routed_content = self._resolve_routed_content(message.content, operation)

        routed_message = AppletMessage(
            content=routed_content,
            context={**context, "memory_config": memory_config},
            metadata={**metadata, "memory_config": memory_config},
        )
        response = await self._memory_applet.on_message(routed_message)

        response_metadata = dict(response.metadata) if isinstance(response.metadata, dict) else {}
        response_metadata["applet"] = "memory"
        response_metadata.setdefault("operation", operation)
        response_metadata.setdefault("migrated_to", "memory")

        return AppletMessage(
            content=response.content,
            context=response.context,
            metadata=response_metadata,
        )

    def _resolve_operation(self, content: Any) -> str:
        if isinstance(content, dict) and "operation" in content:
            raw_operation = content.get("operation")
            if isinstance(raw_operation, str):
                return raw_operation.strip().lower()
        return "store"

    def _resolve_routed_content(self, content: Any, operation: str) -> Dict[str, Any]:
        if isinstance(content, dict):
            routed = dict(content)
            routed["operation"] = operation
            return routed
        if operation == "store":
            return {"operation": operation, "data": {"value": content}}
        return {"operation": operation}

    def _resolve_memory_config(self, message: AppletMessage) -> Dict[str, Any]:
        context = message.context if isinstance(message.context, dict) else {}
        metadata = message.metadata if isinstance(message.metadata, dict) else {}

        context_cfg = context.get("memory_config", {})
        if not isinstance(context_cfg, dict):
            context_cfg = {}

        metadata_cfg = metadata.get("memory_config", {})
        if not isinstance(metadata_cfg, dict):
            metadata_cfg = {}

        content_cfg: Dict[str, Any] = {}
        if isinstance(message.content, dict):
            for key in ("backend", "namespace", "persist_path", "collection", "top_k"):
                if key in message.content:
                    content_cfg[key] = message.content[key]

        merged = {**context_cfg, **metadata_cfg, **content_cfg}
        if "backend" not in merged:
            merged["backend"] = "sqlite_fts"
        if "namespace" not in merged:
            merged["namespace"] = "default"
        # Preserve legacy MemoryApplet output shape by default.
        merged.setdefault("include_metadata", False)
        merged.setdefault("label", "Memory")
        return merged


if __name__ == "__main__":
    import asyncio

    async def test_memory():
        applet = MemoryApplet()

        store_response = await applet.on_message(
            AppletMessage(
                content={
                    "operation": "store",
                    "data": {"name": "John", "age": 30},
                    "key": "user_profile",
                    "tags": ["user", "profile"],
                },
                context={},
                metadata={},
            )
        )
        print(f"Store response: {store_response.content}")

        retrieve_response = await applet.on_message(
            AppletMessage(
                content={"operation": "retrieve", "key": "user_profile"},
                context={},
                metadata={},
            )
        )
        print(f"Retrieve response: {retrieve_response.content}")

    asyncio.run(test_memory())
