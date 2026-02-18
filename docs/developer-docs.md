# SynApps Developer Documentation

This document provides detailed information for developers working on the SynApps project.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Backend (Orchestrator)](#backend-orchestrator)
3. [Applets](#applets)
4. [Frontend](#frontend)
5. [Development Setup](#development-setup)
6. [Workflow](#workflow)
7. [Testing](#testing)
8. [Deployment](#deployment)
9. [Best Practices](#best-practices)

## Architecture Overview

SynApps follows a microkernel architecture pattern:

- **Microkernel (Orchestrator):** The lightweight core that handles message routing between applets
- **Applets:** Pluggable modules that implement specific AI capabilities
- **Frontend:** Visual interface for designing and monitoring workflows

The system is designed to be modular and extensible, allowing new applet types to be added without changing the core orchestration logic.

### Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        Web Frontend                         │
│                                                             │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐   │
│  │  Workflow     │  │  Template     │  │  Monitoring   │   │
│  │  Designer     │  │  Gallery      │  │  Dashboard    │   │
│  └───────────────┘  └───────────────┘  └───────────────┘   │
│                                                             │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            │ REST API / WebSockets
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                         Orchestrator                        │
│                                                             │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐   │
│  │  Message      │  │  Flow         │  │  Status       │   │
│  │  Router       │  │  Executor     │  │  Tracker      │   │
│  └───────────────┘  └───────────────┘  └───────────────┘   │
│                                                             │
└─┬───────────────────┬───────────────────┬──────────────────┘
  │                   │                   │
  │                   │                   │
┌─▼─────────────┐ ┌───▼───────────┐ ┌────▼──────────┐
│ WriterApplet  │ │ ArtistApplet  │ │ MemoryApplet  │
└───────────────┘ └───────────────┘ └───────────────┘
```

## Backend (Orchestrator)

The Orchestrator is the core component that manages the execution of workflows by routing messages between applets.

### Key Components

- **Flow Executor:** Handles the execution of workflows, traversing the graph of applets
- **Message Router:** Routes messages between applets based on the flow definition
- **Status Tracker:** Tracks and reports the status of running workflows
- **REST API:** Provides endpoints for managing flows and triggering executions
- **WebSocket Server:** Provides real-time updates on workflow status

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check and version info |
| `/applets` | GET | List available applets |
| `/flows` | GET | List all flows |
| `/flows` | POST | Create or update a flow |
| `/flows/{flow_id}` | GET | Get a specific flow |
| `/flows/{flow_id}` | DELETE | Delete a flow |
| `/flows/{flow_id}/run` | POST | Run a flow with input data |
| `/runs` | GET | List all workflow runs |
| `/runs/{run_id}` | GET | Get details of a specific run |
| `/ai/suggest` | POST | Generate AI code suggestions |

### WebSocket Protocol

The Orchestrator uses WebSockets for real-time communication with the frontend:

- Connect to `/ws` endpoint
- Messages follow the format: `{ "type": "event_type", "data": { ... } }`
- Event types:
  - `workflow.status`: Updates on workflow execution status
  - `agent.status`: Status updates from the build agent

## Applets

Applets are self-contained modules that implement specific AI capabilities. They follow a standard interface to integrate with the Orchestrator.

### Applet Interface

Each applet must implement the `BaseApplet` interface:

```python
class BaseApplet:
    @classmethod
    def get_metadata(cls) -> Dict[str, Any]:
        """Return applet metadata."""
        return {
            "name": cls.__name__,
            "description": cls.__doc__,
            "version": getattr(cls, "VERSION", "0.1.0"),
            "capabilities": getattr(cls, "CAPABILITIES", []),
        }
    
    async def on_message(self, message: AppletMessage) -> AppletMessage:
        """Process an incoming message and return a response."""
        raise NotImplementedError("Applets must implement on_message")
```

### Message Format

Messages between applets follow the `AppletMessage` format:

```python
class AppletMessage(BaseModel):
    content: Any  # The main content of the message
    context: Dict[str, Any] = {}  # Shared context between applets
    metadata: Dict[str, Any] = {}  # Metadata about the message
```

### Creating a New Applet

To create a new applet:

1. Create a new directory under `apps/applets/`
2. Implement a class that extends `BaseApplet`
3. Implement the `on_message` method to process incoming messages
4. Add appropriate metadata (name, description, version, capabilities)

Example:

```python
class MyNewApplet(BaseApplet):
    """
    My new applet that does something amazing.
    """
    
    VERSION = "0.1.0"
    CAPABILITIES = ["capability1", "capability2"]
    
    async def on_message(self, message: AppletMessage) -> AppletMessage:
        # Process the message
        result = do_something_with(message.content)
        
        # Return the result
        return AppletMessage(
            content=result,
            context=message.context,  # Preserve context
            metadata={"applet": "my-new-applet"}
        )
```

## Frontend

The frontend is a React application that provides a visual interface for creating, running, and monitoring workflows.

### Key Components

- **WorkflowCanvas:** Visual editor for designing workflows using React Flow
- **TemplateLoader:** Gallery of pre-built workflow templates
- **CodeEditor:** Monaco-based editor for customizing applet code
- **NotificationCenter:** System for notifying users of workflow status
- **ApiService:** Service for communicating with the backend REST API
- **WebSocketService:** Service for real-time updates via WebSockets

### Pages

- **Dashboard:** Landing page with recent workflows and templates
- **Editor:** Visual workflow editor
- **History:** List of past workflow runs with details
- **AppletLibrary:** Gallery of available applets
- **Settings:** User and application settings

### State Management

The frontend uses React's state management with hooks:

- Component-level state for UI components
- Context API for shared state where needed
- Local Storage for persistent settings

### Styling

The application uses CSS modules for styling:

- Each component has its own CSS file
- Global styles are defined in `index.css`
- CSS variables are used for theming

## Development Setup

### Prerequisites

- [Node.js](https://nodejs.org/) (v16 or later)
- [Python](https://python.org/) (v3.11 or later)
- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/) (optional)

### Stack Upgrade

The project has recently undergone a major stack upgrade. For more details on the new versions, migration steps, and breaking changes, please refer to the [Stack Upgrade Guide](stack-upgrade.md).

### Backend Setup

1. Create a virtual environment:
   ```bash
   cd apps/orchestrator
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -e .
   ```

3. Run the development server:
   ```bash
   uvicorn main:app --reload
   ```

### Frontend Setup

1. Install dependencies:
   ```bash
   cd apps/web-frontend
   npm install
   ```

2. Run the development server:
   ```bash
   npm start
   ```

### Docker Setup

To run the entire stack using Docker:

1. Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   STABILITY_API_KEY=your_stability_api_key
   ```

2. Run Docker Compose:
   ```bash
   docker-compose -f infra/docker/docker-compose.yml up
   ```

## Workflow

### Git Workflow

We follow a feature branch workflow:

1. Create a feature branch from `main`: `git checkout -b feature/my-feature`
2. Make your changes
3. Commit your changes: `git commit -m "Add my feature"`
4. Push to your branch: `git push origin feature/my-feature`
5. Create a pull request to `main`
6. After review and approval, the branch will be merged

### Code Style

- **Python:** Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- **TypeScript/JavaScript:** Use [Prettier](https://prettier.io/) for formatting
- **Commit Messages:** Follow [Conventional Commits](https://www.conventionalcommits.org/)

## Testing

### Backend Tests

Backend tests use pytest:

```bash
cd apps/orchestrator
pytest
```

### Frontend Tests

Frontend tests use Jest and React Testing Library:

```bash
cd apps/web-frontend
npm test
```

### End-to-End Tests

End-to-end tests use Cypress (planned for future implementation).

## Deployment

### Docker Deployment

Build and push the Docker images:

```bash
docker build -f infra/docker/Dockerfile.orchestrator -t synapps-orchestrator:latest .
docker build -f infra/docker/Dockerfile.frontend -t synapps-frontend:latest .
```

### Kubernetes Deployment

Apply the Kubernetes configuration:

```bash
kubectl apply -f infra/k8s/
```

### Vercel Deployment (Frontend)

Deploy the frontend to Vercel:

```bash
cd apps/web-frontend
vercel
```

### Fly.io Deployment (Backend)

Deploy the backend to Fly.io:

```bash
fly deploy
```

## Best Practices

### Security

- Do not commit API keys or sensitive information
- Use environment variables for configuration
- Validate all user input

### Performance

- Keep applet implementations lightweight
- Use asynchronous code for I/O operations
- Implement caching where appropriate

### Documentation

- Document all public functions and methods
- Keep README and documentation up to date
- Use comments for complex logic

### Error Handling

- Implement proper error handling in all applets
- Return meaningful error messages
- Log errors with appropriate context
