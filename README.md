
![SynApps Logo](logo192.png)

# SynApps [![Star on GitHub](https://img.shields.io/github/stars/nxtg-ai/SynApps-v0.4.0?style=social)](https://github.com/nxtg-ai/SynApps-v0.4.0/stargazers)

A web-based visual AI workflow builder where users drag-and-drop AI agent nodes, connect them on a canvas, and execute workflows in real-time.

## Introduction

SynApps is a **web-based visual platform for modular AI agents called Snaplets**. Its mission is to let indie creators build autonomous AI snaplets like LEGO blocks -- each snaplet is a small agent with a specialized skill. A lightweight **SynApps Orchestrator** routes messages between these snaplets, sequencing their interactions to solve tasks collaboratively. SynApps connects AI "synapses" (snaplets) in real time, forming an intelligent network that can tackle complex workflows.

## Features

- **Visual Workflow Builder:** Drag-and-drop AI nodes onto a canvas and connect them to build workflows.
- **Autonomous & Collaborative Snaplets:** Each snaplet runs autonomously but can pass data to others via the orchestrator.
- **Real-Time Visual Feedback:** See the AI snaplets at work with an animated graph of nodes and connections.
- **Background Execution & Notifications:** Snaplets run in the background once triggered, with notifications for status changes.
- **Extensibility:** 9 built-in node types (LLM, ImageGen, Code, HTTP, Transform, IfElse, Merge, ForEach, Memory) with support for custom logic via the Code node.

## Quick Start

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/)
- For local development: Node.js 20+ and Python 3.11+

### Running with Docker

1. Clone the repository:
   ```bash
   git clone https://github.com/nxtg-ai/synapps.git
   cd synapps
   ```

2. Create a `.env` file in the root of the project:
   ```
   OPENAI_API_KEY=your_openai_api_key
   STABILITY_API_KEY=your_stability_api_key
   ```

3. Build and run the containers (PostgreSQL + orchestrator + frontend):
   ```bash
   docker-compose -f infra/docker/docker-compose.yml up --build
   ```

4. Open your browser and navigate to [http://localhost:3000](http://localhost:3000)

### Local Development

#### Backend (Orchestrator)

```bash
# From the repo root
cd apps/orchestrator && pip install -e . && cd ../..

# Set up your environment variables
cp .env.example .env
# Then edit .env with your actual API keys

# Run database migrations
alembic upgrade head

# Start the dev server
PYTHONPATH=. uvicorn apps.orchestrator.main:app --reload --port 8000
```

#### Frontend

```bash
cd apps/web-frontend
npm install
npm run dev    # Starts Vite dev server on :3000
```

## Architecture

SynApps follows a microkernel architecture:

- **Orchestrator:** A FastAPI backend that routes messages between applets and manages workflow execution. All applet logic, auth, and WebSocket handlers live in `apps/orchestrator/main.py`.
- **Applets (Nodes):** Self-contained AI micro-agents implementing a standard `BaseApplet` interface.
- **Frontend:** React 18 + TypeScript app with a visual workflow editor built on @xyflow/react, styled with Tailwind CSS, state managed by Zustand.
- **Database:** SQLite (dev via aiosqlite) / PostgreSQL (prod via asyncpg) with async SQLAlchemy 2.0 ORM.

## Node Types

| Node | Description |
|------|-------------|
| **LLM** | Text generation via LLM (e.g. GPT-4o) |
| **ImageGen** | Image generation from text prompts |
| **Code** | Execute custom Python/JavaScript logic |
| **HTTP** | Make HTTP requests to external APIs |
| **Transform** | Transform and reshape data between nodes |
| **IfElse** | Conditional branching based on expressions |
| **Merge** | Combine outputs from multiple branches |
| **ForEach** | Iterate over collections |
| **Memory** | Store/retrieve context using SQLite FTS or ChromaDB vector store |

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.11+, FastAPI 0.115+, Pydantic v2, SQLAlchemy 2.0 (async) |
| Frontend | React 18, TypeScript (strict), Vite 6, Tailwind CSS 4, Zustand, @xyflow/react v12 |
| Database | SQLite (dev), PostgreSQL (prod) |
| Testing | pytest + pytest-asyncio (backend), Vitest + React Testing Library (frontend), Playwright (E2E) |
| Linting | Ruff (Python), ESLint 9 flat config + Prettier (TypeScript) |
| CI/CD | GitHub Actions, Codecov, Docker |
| Deploy | Fly.io (backend), Vercel (frontend) |

## Portfolio Templates

SynApps ships with workflow templates that validate the platform against real NXTG.AI portfolio use cases.

| Template | Consumer | Pipeline | Nodes |
|----------|----------|----------|-------|
| **2Brain Inbox Triage** | 2Brain (P-13) | Capture → Classify → Structure → Store | Start → LLM → Code → Memory → End |
| **Content Engine Pipeline** | nxtg-content-engine (P-14) | Research → Summarize → Format → Store | Start → HTTP → LLM → Code → Memory → End |

Templates are available in the frontend gallery (`apps/web-frontend/src/templates/`) and as standalone YAML definitions (`templates/`).

## Deployment

- **Frontend:** Vercel
- **Backend:** Fly.io

CI/CD pipelines are set up using GitHub Actions.

## Testing

### Backend

```bash
# Run all tests (from repo root)
PYTHONPATH=. pytest apps/orchestrator/tests/ -v

# With coverage
PYTHONPATH=. pytest apps/orchestrator/tests/ --cov=apps/orchestrator --cov-report=term-missing
```

### Frontend

```bash
cd apps/web-frontend
npm test                        # Vitest (single run)
npm run typecheck               # TypeScript type checking
```

### E2E

```bash
cd apps/web-frontend
npx playwright test             # Run all E2E tests
npx playwright test --headed    # Run with browser visible
```

## Linting & Formatting

```bash
# Backend (from repo root)
ruff check apps/orchestrator --config apps/orchestrator/pyproject.toml
ruff format apps/orchestrator --config apps/orchestrator/pyproject.toml

# Frontend (from apps/web-frontend/)
npm run lint
npm run format:check

# All at once via pre-commit
pre-commit run --all-files
```

## Development Scripts

A convenience script starts both servers (requires `concurrently` and `kill-port` installed globally):

```bash
.scripts/start-dev.sh
```

> **Note:** The script may use older invocation patterns. If you encounter import errors, use the manual backend/frontend commands documented above.

## Development Workflow

1. Create a feature branch from `master`
2. Make your changes
3. Write tests for your changes
4. Run `pre-commit run --all-files` to check linting
5. Submit a pull request to `master`

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [@xyflow/react](https://reactflow.dev/) for the workflow visualization
- [anime.js](https://animejs.com/) for animations
- [FastAPI](https://fastapi.tiangolo.com/) for the backend
- [Monaco Editor](https://microsoft.github.io/monaco-editor/) for the code editor
- [Tailwind CSS](https://tailwindcss.com/) for styling
- [Zustand](https://zustand-demo.pmnd.rs/) for state management
