"""
MemoryApplet - Vector store for maintaining context between steps

This applet stores and retrieves information to maintain context between workflow steps.
"""
import os
import json
import logging
import uuid
from typing import Dict, Any, List, Optional

# Import base applet from orchestrator
from apps.orchestrator.main import BaseApplet, AppletMessage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("memory-applet")

class MemoryApplet(BaseApplet):
    """
    Memory Applet that stores and retrieves information.
    
    Capabilities:
    - Context storage
    - Information retrieval
    - Memory management
    """
    
    VERSION = "0.1.0"
    CAPABILITIES = ["context-storage", "information-retrieval", "memory-management"]
    
    def __init__(self):
        """Initialize the Memory Applet."""
        # For MVP, we'll use an in-memory store
        # In a production system, this would be a vector database like Pinecone or FAISS
        self.memory_store = {}
    
    async def on_message(self, message: AppletMessage) -> AppletMessage:
        """Process an incoming message to store or retrieve memory."""
        logger.info("Memory Applet received message")
        
        # Extract content from message
        content = message.content
        context = message.context
        
        # Determine the operation (store or retrieve)
        operation = "store"  # Default operation
        if isinstance(content, dict) and "operation" in content:
            operation = content["operation"]
        
        if operation.lower() == "store":
            return await self._handle_store(message)
        elif operation.lower() == "retrieve":
            return await self._handle_retrieve(message)
        elif operation.lower() == "delete":
            return await self._handle_delete(message)
        elif operation.lower() == "clear":
            return await self._handle_clear(message)
        else:
            # Invalid operation
            return AppletMessage(
                content={"error": f"Invalid operation: {operation}"},
                context=context,
                metadata={"applet": "memory", "status": "error"}
            )
    
    async def _handle_store(self, message: AppletMessage) -> AppletMessage:
        """Handle a memory storage operation."""
        content = message.content
        context = message.context
        
        # Extract data to store
        data = {}
        if isinstance(content, dict) and "data" in content:
            data = content["data"]
        elif isinstance(content, dict):
            # If no specific "data" field, use the whole content
            data = content
        else:
            # If content is not a dict, wrap it
            data = {"value": content}
        
        # Generate a key if not provided
        key = str(uuid.uuid4())
        if isinstance(content, dict) and "key" in content:
            key = content["key"]
        
        # Store in memory
        self.memory_store[key] = {
            "data": data,
            "metadata": {
                "timestamp": context.get("timestamp", None),
                "run_id": context.get("run_id", None),
                "tags": content.get("tags", []) if isinstance(content, dict) else []
            }
        }
        
        logger.info(f"Stored memory with key: {key}")
        
        # Return success response
        return AppletMessage(
            content={"key": key, "status": "stored"},
            context={**context, "memory_key": key},  # Add key to context
            metadata={"applet": "memory", "operation": "store"}
        )
    
    async def _handle_retrieve(self, message: AppletMessage) -> AppletMessage:
        """Handle a memory retrieval operation."""
        content = message.content
        context = message.context
        
        # Extract key to retrieve
        key = None
        if isinstance(content, dict) and "key" in content:
            key = content["key"]
        elif "memory_key" in context:
            key = context["memory_key"]
        
        # If key is provided, retrieve specific memory
        if key and key in self.memory_store:
            memory_data = self.memory_store[key]["data"]
            
            logger.info(f"Retrieved memory with key: {key}")
            
            return AppletMessage(
                content=memory_data,
                context={**context, "memory_retrieved": True},
                metadata={"applet": "memory", "operation": "retrieve", "key": key}
            )
        
        # If no key or not found, try to match by tags
        tags = []
        if isinstance(content, dict) and "tags" in content:
            tags = content["tags"]
        
        if tags:
            # Find memories with matching tags
            matched_memories = {}
            for mem_key, mem_value in self.memory_store.items():
                mem_tags = mem_value["metadata"].get("tags", [])
                if any(tag in mem_tags for tag in tags):
                    matched_memories[mem_key] = mem_value["data"]
            
            if matched_memories:
                logger.info(f"Retrieved {len(matched_memories)} memories by tags")
                
                return AppletMessage(
                    content={"memories": matched_memories},
                    context={**context, "memory_retrieved": True},
                    metadata={"applet": "memory", "operation": "retrieve", "by_tags": tags}
                )
        
        # Nothing found
        logger.warning("No matching memories found")
        
        return AppletMessage(
            content={"status": "not_found"},
            context={**context, "memory_retrieved": False},
            metadata={"applet": "memory", "operation": "retrieve", "status": "not_found"}
        )

    async def _handle_delete(self, message: AppletMessage) -> AppletMessage:
        """Handle a memory deletion operation."""
        content = message.content
        context = message.context
        
        key = None
        if isinstance(content, dict) and "key" in content:
            key = content["key"]
            
        if key and key in self.memory_store:
            del self.memory_store[key]
            return AppletMessage(
                content={"status": "deleted", "key": key},
                context=context,
                metadata={"applet": "memory", "operation": "delete", "status": "success"}
            )
            
        return AppletMessage(
            content={"status": "not_found", "key": key},
            context=context,
            metadata={"applet": "memory", "operation": "delete", "status": "not_found"}
        )
        
    async def _handle_clear(self, message: AppletMessage) -> AppletMessage:
        """Handle clearing all memories."""
        self.memory_store.clear()
        return AppletMessage(
            content={"status": "cleared"},
            context=message.context,
            metadata={"applet": "memory", "operation": "clear", "status": "success"}
        )

# For testing
if __name__ == "__main__":
    import asyncio
    
    async def test_memory():
        applet = MemoryApplet()
        
        # Test storing
        store_message = AppletMessage(
            content={
                "operation": "store",
                "data": {"name": "John", "age": 30},
                "key": "user_profile",
                "tags": ["user", "profile"]
            },
            context={},
            metadata={}
        )
        store_response = await applet.on_message(store_message)
        print(f"Store response: {store_response.content}")
        
        # Test retrieving
        retrieve_message = AppletMessage(
            content={
                "operation": "retrieve",
                "key": "user_profile"
            },
            context={},
            metadata={}
        )
        retrieve_response = await applet.on_message(retrieve_message)
        print(f"Retrieve response: {retrieve_response.content}")
        
        # Test retrieving by tags
        tag_retrieve_message = AppletMessage(
            content={
                "operation": "retrieve",
                "tags": ["profile"]
            },
            context={},
            metadata={}
        )
        tag_response = await applet.on_message(tag_retrieve_message)
        print(f"Tag retrieve response: {tag_response.content}")
    
    asyncio.run(test_memory())
