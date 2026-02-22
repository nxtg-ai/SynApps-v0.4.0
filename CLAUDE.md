# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SynApps is a web-based visual AI workflow builder. Users drag-and-drop AI agent nodes ("snaplets"), connect them on a canvas, and execute workflows in real-time. The architecture is a microkernel: a lightweight orchestrator routes messages between self-contained applets.

**Current state:** v0.5.x alpha transitioning to v1.0. See `SPEC.md` for the full v1.0 upgrade specification.

## Repository Structure

```
apps/
  orchestrator/       # Python backend (FastAPI) — the core API and execution engine
    main.py           # ~6000 lines: all routes, applets, engine, auth, WebSocket
    models.py         # SQLAlchemy ORM models + Pydantic request/response models
    db.py             # Async engine/session factory (SQLite dev, PostgreSQL prod)
    repositories.py   # Repository pattern for Flow and WorkflowRun persistence
    middleware/        # rate_limiter.py, billing_guard.py
    tests/            # pytest test suite
    migrations/       # Alembic migrations (orchestrator-local)
  applets/            # Legacy applet implementations (writer, artist, memory)
  web-frontend/       # React 18 + TypeScript frontend
    src/
      components/     # UI components (WorkflowCanvas, CodeEditor, Layout, ui/)
      pages/          # Route pages (Dashboard, Editor, History, Settings, AppletLibrary)
      stores/         # Zustand stores (workflowStore, executionStore, settingsStore)
      services/       # ApiService (axios), WebSocketService
      types/          # TypeScript type definitions
      templates/      # Workflow templates
    e2e/              # Playwright E2E tests
migrations/           # Root-level Alembic migrations
infra/docker/         # Dockerfiles for orchestrator and frontend
```

**Key architectural note:** `main.py` is a monolith containing routes, all applet implementations (LLM, ImageGen, Code, HTTP, Transform, IfElse, Merge, ForEach), the execution engine, auth system, and WebSocket handlers. Applet classes in `apps/applets/` are the original v0.4 implementations; the active ones are defined inline in `main.py`.

## Development Commands

### Backend (from repo root)

```bash
# Install (editable mode — run once)
cd apps/orchestrator && pip install -e . && cd ../..

# Run backend dev server
PYTHONPATH=. uvicorn apps.orchestrator.main:app --reload --port 8000

# Run all backend tests
PYTHONPATH=. pytest apps/orchestrator/tests/ -v

# Run a single test file
PYTHONPATH=. pytest apps/orchestrator/tests/test_main.py -v

# Run a single test by name
PYTHONPATH=. pytest apps/orchestrator/tests/test_main.py -v -k "test_health_check"

# Backend tests with coverage
PYTHONPATH=. pytest apps/orchestrator/tests/ --cov=apps/orchestrator --cov-report=term-missing

# Lint backend (ruff)
ruff check apps/orchestrator --config apps/orchestrator/pyproject.toml
ruff format apps/orchestrator --config apps/orchestrator/pyproject.toml
```

### Frontend (from `apps/web-frontend/`)

```bash
npm install          # Install dependencies (once)
npm run dev          # Start Vite dev server on :3000
npm run build        # Production build
npm test             # Run vitest (single run)
npm run lint         # ESLint
npm run format:check # Prettier check
npm run typecheck    # TypeScript type checking
```

### E2E Tests (from `apps/web-frontend/`)

```bash
npx playwright test              # Run all E2E tests
npx playwright test --headed     # Run with browser visible
```

### Database Migrations (from repo root)

```bash
alembic upgrade head             # Apply all migrations
alembic revision --autogenerate -m "description"  # Generate new migration
```

### Docker (full stack)

```bash
docker-compose up --build        # Start PostgreSQL + orchestrator + frontend
```

### Pre-commit hooks

```bash
pre-commit run --all-files       # Run all hooks manually
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.11+, FastAPI 0.115+, Pydantic v2, SQLAlchemy 2.0 (async) |
| Frontend | React 18, TypeScript (strict), Vite 6, Tailwind CSS 4, Zustand, @xyflow/react v12 |
| Database | SQLite (dev via aiosqlite), PostgreSQL (prod via asyncpg) |
| Testing | pytest + pytest-asyncio (backend), Vitest + React Testing Library (frontend), Playwright (E2E) |
| Linting | Ruff (Python), ESLint 9 flat config + Prettier (TypeScript) |
| CI/CD | GitHub Actions, Codecov, Docker |
| Deploy | Fly.io (backend), Vercel (frontend) |

## Key Conventions

- **Backend tests** use in-memory SQLite (`sqlite+aiosqlite:///:memory:`) — set automatically by `conftest.py`. The `PYTHONPATH` must include the repo root so `apps.orchestrator` imports resolve.
- **Frontend path alias:** `@/` maps to `src/` (configured in `vite.config.ts` and `tsconfig.json`).
- **Environment variables:** Backend reads from `.env.development` at project root. Frontend uses `VITE_` prefixed vars (legacy `REACT_APP_` also supported via Vite config).
- **API versioning:** All backend endpoints are under `/api/v1/`. Health check at `/api/v1/health`.
- **Ruff config** lives in `apps/orchestrator/pyproject.toml` (line length 100, Python 3.11 target).
- **ESLint** uses flat config at `apps/web-frontend/eslint.config.mjs`. `@typescript-eslint/no-explicit-any` is off.
- **Pre-commit hooks** run ruff (lint + format) on backend, ESLint + Prettier + typecheck on frontend.

## ASIF Governance

This project is **P-10** in the ASIF portfolio (Developer Tools vertical). On every session:
1. Read `.asif/NEXUS.md` — check the `## CoS Directives` section
2. Execute any **PENDING** directives before other work (unless Asif explicitly overrides)
3. Write your response inline under each directive's `**Response**` section
4. Update initiative statuses in NEXUS if your work changes them
5. If you have questions for the CoS, add them under `## Team Questions` in NEXUS

**Escalation via Team Questions**: When you hit a blocker, need an architecture review, or have a portfolio-level question, add it under `## Team Questions` in your `.asif/NEXUS.md`. Your CoS checks these 3x daily during scheduled enrichment cycles and will respond inline or issue follow-up directives.
