# Applets Documentation

## Introduction

Applets are modular, AI-powered components designed to perform specific tasks within a SynApps workflow. They are the building blocks of any AI automation process within the platform, allowing for flexible and extensible system design. Each applet is self-contained, focusing on a single responsibility, and communicates with other applets via messages routed by the Orchestrator.

## Key Concepts

*   **Modularity:** Applets are independent units, making workflows easy to design, understand, and maintain.
*   **AI-Powered:** Many applets leverage AI models (including Large Language Models) to perform their designated tasks.
*   **Message-Driven:** Applets communicate by sending and receiving `AppletMessage` objects, enabling asynchronous and decoupled operations.
*   **Dynamic Loading:** Applets are dynamically loaded by the Orchestrator as needed, based on the workflow definition.

## For Users: How to Use Applets

SynApps workflows are constructed by chaining together various applets. When designing a workflow in the SynApps interface, you will select and configure applets based on your needs.

*   **Selecting Applets:** Browse the available applet library, each designed for a specific purpose (e.g., `WriterApplet` for text generation, `MemoryApplet` for data storage).
*   **Configuring Applets:** Each applet may have specific configuration parameters (e.g., prompts for an LLM applet, database connection details for a data applet). These parameters are set within the workflow editor.
*   **Connecting Applets:** Define the flow of information between applets by connecting their inputs and outputs. The Orchestrator ensures messages are passed correctly.

## For Developers: Building Custom Applets

Developers can extend SynApps' capabilities by creating custom applets. All applets must inherit from the `BaseApplet` class and implement specific methods.

### `BaseApplet` Interface

All custom applets must inherit from `apps.orchestrator.main.BaseApplet`.

```python
class BaseApplet:
    """Base class that all applets must implement."""

    @classmethod
    def get_metadata(cls) -> Dict[str, Any]:
        """Return applet metadata."""
        raise NotImplementedError("Applets must implement get_metadata")

    async def on_message(self, message: AppletMessage) -> AppletMessage:
        """Process an incoming message and return a response message."""
        raise NotImplementedError("Applets must implement on_message")
```

### Implementing `get_metadata`

The `get_metadata` class method should return a dictionary containing information about the applet, such as its name, description, version, and capabilities. This metadata is used by the SynApps platform to display and manage the applet.

Example:

```python
@classmethod
def get_metadata(cls) -> Dict[str, Any]:
    return {
        "name": "Writer Applet",
        "description": "Generates text using an LLM.",
        "version": "0.1.0",
        "capabilities": ["text_generation", "llm_interaction"]
    }
```

### Implementing `on_message`

The `on_message` asynchronous method is the core logic of the applet. It receives an `AppletMessage` as input, performs its designated task, and returns an `AppletMessage` as output.

The `AppletMessage` object typically contains:

*   `payload`: The main data being passed between applets.
*   `context`: Workflow-specific context information.
*   `current_applet`: Identifier of the applet currently processing the message.

Example (simplified):

```python
async def on_message(self, message: AppletMessage) -> AppletMessage:
    input_text = message.payload.get("text_input")
    if not input_text:
        raise ValueError("No text_input found in message payload.")

    # Perform LLM call (example using an abstract LLM interface)
    generated_text = await self.llm_service.generate_text(input_text, message.context)

    return AppletMessage(
        payload={"generated_text": generated_text},
        context=message.context,
        current_applet="writer"
    )
```

### Applet Structure

Applets are typically organized within the `apps/applets` directory, with each applet residing in its own subdirectory.

Example:

```
apps/
└── applets/
    └── writer/
        ├── __init__.py
        └── applet.py
    └── memory/
        ├── __init__.py
        └── applet.py
```

The `applet.py` file should contain the main applet class inheriting from `BaseApplet`.

## Applet Limitations (Free Tier)

For users on the free tier, there might be limitations on the number of applets that can be used within a single workflow. Please refer to your plan details for more information.