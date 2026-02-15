# SynApps Architecture

This document outlines the architecture of the SynApps platform, a visual workflow system for modular AI agents.

## System Overview

SynApps follows a microkernel architecture with the following key components:

1. **Orchestrator**: A lightweight message routing core that sequences interactions between applets.
2. **Applets**: Self-contained AI micro-agents with specialized skills (Writer, Artist, Memory, etc.).
3. **Frontend**: A React-based web application for visually designing and monitoring workflows.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│                        Frontend (React)                         │
│                                                                 │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐     │
│  │ Dashboard │  │  Editor   │  │  History  │  │  Library  │     │
│  └───────────┘  └───────────┘  └───────────┘  └───────────┘     │
│                                                                 │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            │ HTTP/WebSocket
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                                                                 │
│                      Orchestrator (FastAPI)                     │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                                                           │  │
│  │                    Message Router                         │  │
│  │                                                           │  │
│  └───────────┬───────────────┬─────────────────┬─────────────┘  │
│              │               │                 │                │
│  ┌───────────▼───┐   ┌───────▼─────┐   ┌───────▼────┐           │
│  │ Flow Manager  │   │ Run Tracker │   │ API Gateway│           │
│  └───────────────┘   └─────────────┘   └────────────┘           │
│                                                                 │
└───────┬─────────────────┬──────────────────┬────────────────────┘
        │                 │                  │
        │                 │                  │
┌───────▼─────┐   ┌───────▼─────┐    ┌───────▼─────┐
│             │   │             │    │             │
│   Writer    │   │   Artist    │    │   Memory    │
│   Applet    │   │   Applet    │    │   Applet    │
│             │   │             │    │             │
└─────────────┘   └──────┬──────┘    └─────────────┘
                         │
                         │ External API
                         │
                  ┌──────▼──────┐
                  │             │
                  │  Stable     │
                  │  Diffusion  │
                  │             │
                  └─────────────┘
```

## Component Details

### 1. Frontend (React/TypeScript)

The frontend is a modern React application that provides a visual interface for creating and monitoring AI workflows.

Key components:
- **WorkflowCanvas**: Visual editor for creating and connecting applets.
- **TemplateLoader**: One-click template selection for common workflows.
- **CodeEditor**: Monaco-based editor for customizing applet code.
- **NotificationCenter**: Real-time status updates and alerts.

Technologies:
- React 18 with TypeScript
- React Flow for workflow visualization
- anime.js for animations
- Monaco Editor for code editing
- WebSocket for real-time updates

### 2. Orchestrator (Python/FastAPI)

The orchestrator is the core of the SynApps platform, responsible for routing messages between applets and managing workflow execution.

Key components:
- **Message Router**: Routes messages between applets in a defined sequence.
- **Flow Manager**: Stores and retrieves workflow definitions.
- **Run Tracker**: Tracks the status and results of workflow runs.
- **API Gateway**: Exposes the orchestrator functionality via REST API.

Technologies:
- Python 3.9+
- FastAPI for API endpoints
- WebSockets for real-time communication
- Async/await for non-blocking execution

### 3. Applets

Applets are self-contained AI micro-agents that implement a standard interface for receiving messages, processing them, and producing outputs.

Core applets:
- **WriterApplet**: Generates text using gpt-4o or similar LLMs.
- **ArtistApplet**: Creates images using Stable Diffusion or DALL·E 3.
- **MemoryApplet**: Stores and retrieves information to maintain context.

Technologies:
- Python classes implementing the BaseApplet interface
- External API integrations (OpenAI, Stability.ai)
- Async processing for non-blocking execution

## Data Flow

1. **Flow Creation**:
   - User creates a workflow in the visual editor
   - Frontend sends the flow definition to the orchestrator via API
   - Orchestrator stores the flow definition

2. **Flow Execution**:
   - User triggers a workflow run with input data
   - Orchestrator creates a run record and starts execution
   - For each node in the flow:
     - Orchestrator loads the appropriate applet
     - Sends input and context to the applet
     - Applet processes the input and returns output
     - Orchestrator updates the run status and sends a WebSocket update
     - Orchestrator passes the output to the next applet in the sequence
   - When all nodes have been processed, the run is marked as complete

3. **Real-time Updates**:
   - Frontend connects to the orchestrator via WebSocket
   - Orchestrator sends status updates as the workflow progresses
   - Frontend updates the visual representation of the workflow
   - On completion, a notification is shown to the user

For a detailed step-by-step view of how a request moves from the client to the database, see [request-flow.md](request-flow.md).

## Scalability Considerations

### Current Implementation (MVP)

The MVP implementation uses a single-instance orchestrator that runs workflows in-memory. This is sufficient for low to moderate loads but has limitations for scale.

### Future Scalability

For production scale, the following modifications would be needed:

1. **Stateless Orchestrator**:
   - Move workflow state to a distributed database (e.g., Redis, MongoDB)
   - Allow multiple orchestrator instances to run in parallel

2. **Message Queue**:
   - Implement a message queue (e.g., RabbitMQ, Kafka) for applet communication
   - Decouple the orchestrator from applet execution

3. **Containerized Applets**:
   - Run each applet as a separate microservice
   - Scale applets independently based on demand

4. **Horizontal Scaling**:
   - Deploy multiple instances of the orchestrator behind a load balancer
   - Implement session affinity for WebSocket connections

## Security Considerations

### Authentication and Authorization

- User authentication via OAuth or similar
- Role-based access control for flows and applets
- API key management for external services

### Data Protection

- Encryption of sensitive data at rest and in transit
- Isolation of user data in multi-tenant deployments
- Rate limiting and throttling to prevent abuse

### Applet Isolation

- Sandbox execution for custom applet code
- Resource limits to prevent excessive consumption
- Validation of inputs and outputs to prevent injection attacks

## Monitoring and Observability

- Structured logging for troubleshooting
- Metrics collection for performance monitoring
- Distributed tracing for request flows
- Alerting for critical errors or performance degradation

## Deployment Architecture

### Development Environment

- Local Docker Compose setup
- Local development servers with hot reloading
- Mock APIs for external services

### Production Environment

- Kubernetes for container orchestration
- CI/CD pipelines for automated deployment
- Vercel for frontend hosting
- Fly.io or Kubernetes for backend deployment
- CDN for static assets
- Managed databases for persistence

## Future Extensions

1. **Advanced Workflow Features**:
   - Conditional branching based on applet outputs
   - Parallel execution of independent applets
   - Error handling and recovery strategies

2. **Applet Marketplace**:
   - User-created applet sharing
   - Versioning and dependency management
   - Rating and review system

3. **Enhanced Collaboration**:
   - Real-time collaborative editing
   - Sharing and permissions
   - Team workspaces

4. **Advanced AI Features**:
   - Custom model fine-tuning
   - Advanced prompt engineering
   - Model chaining and ensemble methods
