# SynApps Applet Interface Specification

This document provides a detailed technical specification for the SynApps Applet Interface.

## Overview

SynApps applets are modular AI agents that implement a standardized interface to enable seamless communication through the SynApps Orchestrator. Each applet is a self-contained unit that performs a specific function, such as text generation, image creation, or memory management.

## Base Applet Interface

All applets must implement the `BaseApplet` interface:

```python
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
```

## Message Format

Communication between applets uses the `AppletMessage` class:

```python
class AppletMessage(BaseModel):
    content: Any
    context: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}
```

Where:
- `content`: The primary payload of the message (can be any data type)
- `context`: Shared context that is passed between applets in a workflow
- `metadata`: Additional information about the message or its processing

## Lifecycle

1. **Initialization**: The Orchestrator instantiates an applet when needed.
2. **Message Reception**: The applet receives a message through the `on_message` method.
3. **Processing**: The applet processes the message content.
4. **Response**: The applet returns a new message with the processed content.
5. **Context Preservation**: The applet should preserve and update the context as needed.

## Applet Requirements

### 1. Statelessness

Applets should be designed to be stateless, with any required state stored in the message context. This allows the Orchestrator to easily manage applet instances and scale horizontally.

### 2. Error Handling

Applets should handle errors gracefully and return appropriate error messages in the response rather than raising exceptions. The Orchestrator will catch any raised exceptions, but properly handled errors provide a better user experience.

Example:

```python
async def on_message(self, message: AppletMessage) -> AppletMessage:
    try:
        # Process message
        result = process_content(message.content)
        return AppletMessage(
            content=result,
            context=message.context,
            metadata={"status": "success"}
        )
    except Exception as e:
        return AppletMessage(
            content={"error": str(e)},
            context=message.context,
            metadata={"status": "error"}
        )
```

### 3. Asynchronous Processing

Applets should use asynchronous programming for I/O operations to ensure optimal performance. All external API calls, database operations, etc., should be done using `async/await` patterns.

### 4. Context Management

Applets should respect the context passed in messages, which may contain information from previous steps in the workflow. Applets may add to or modify the context but should be careful not to remove existing context unless explicitly instructed to do so.

### 5. Metadata Documentation

Applets should document the metadata they produce and expect in their docstrings. This helps users and other developers understand how to properly interact with the applet.

## Standard Applet Types

### WriterApplet

The WriterApplet generates text content using language models.

**Input Content:**
- String: Text prompt/topic
- Dictionary: Contains "prompt" or "text" keys with prompt content

**Output Content:**
- String: Generated text content

**Context Keys:**
- `system_prompt`: Optional system instruction for the language model
- `temperature`: Optional sampling temperature (default: 0.7)
- `max_tokens`: Optional max tokens to generate (default: 1000)

**Metadata Keys:**
- `model`: The language model used (e.g., "gpt-4o")
- `applet`: Always "writer"

### ArtistApplet

The ArtistApplet generates images from text descriptions.

**Input Content:**
- String: Image description/prompt
- Dictionary: Contains "prompt" or "text" keys with image description

**Output Content:**
- Dictionary:
  - `image`: Base64 encoded image data
  - `prompt`: The original prompt
  - `generator`: The image generator used (e.g., "stability", "openai")

**Context Keys:**
- `image_generator`: Optional preference for generator (default: "stability")
- `style`: Optional style hint (default: "photorealistic")

**Metadata Keys:**
- `generator`: The image generator used
- `applet`: Always "artist"

### MemoryApplet

The MemoryApplet stores and retrieves information.

**Input Content:**
- Dictionary:
  - `operation`: "store" or "retrieve"
  - `key`: Optional key for storage/retrieval
  - `data`: Data to store (for "store" operation)
  - `tags`: Optional tags for content categorization

**Output Content:**
- For "store": `{"key": string, "status": "stored"}`
- For "retrieve": Either the retrieved data or `{"status": "not_found"}`

**Context Keys:**
- `memory_key`: Key for the most recently stored/retrieved memory
- `memory_retrieved`: Boolean indicating retrieval success

**Metadata Keys:**
- `operation`: The operation performed ("store" or "retrieve")
- `key`: The key used (if provided)
- `by_tags`: Tags used for retrieval (if applicable)
- `applet`: Always "memory"

## Creating a New Applet

To create a new applet:

1. Create a new directory in `apps/applets/` with your applet name
2. Create an `applet.py` file implementing the BaseApplet interface
3. Create a `setup.py` file for dependencies
4. Implement the `get_metadata` and `on_message` methods
5. Add any additional helper methods or classes needed

Example directory structure:

```
apps/applets/my-applet/
├── applet.py
└── setup.py
```

## Testing Applets

Each applet should include a test section in the main script for quick verification:

```python
if __name__ == "__main__":
    import asyncio
    
    async def test_applet():
        applet = MyApplet()
        message = AppletMessage(
            content="test input",
            context={},
            metadata={}
        )
        response = await applet.on_message(message)
        print(response.content)
    
    asyncio.run(test_applet())
```

## Best Practices

1. **Keep dependencies minimal:** Only include packages that are absolutely necessary.
2. **Document applet behavior:** Include detailed docstrings explaining what the applet does, expected inputs, and outputs.
3. **Graceful fallbacks:** If an external service is unavailable, provide graceful fallback behavior.
4. **Input validation:** Validate all inputs and provide helpful error messages.
5. **Logging:** Include appropriate logging to help with debugging.
6. **Version compatibility:** Document compatibility with orchestrator versions.
7. **Configuration:** Make parameters configurable through environment variables when appropriate.
