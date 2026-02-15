# SynApps v1.0 — SOTA Upgrade Specification

## Vision

SynApps is a free, open-source visual AI workflow builder — the answer to "why is LangFlow so complicated?" and "why am I paying for N8N?" Users drag-and-drop AI agent nodes, connect them visually, hit run, and watch the workflow execute in real-time. No code required. No subscription wall.

v1.0 takes the working v0.5 alpha and makes it production-grade: modern stack, beautiful UI, hardened execution engine, and FM-agnostic so users can swap Claude, GPT, Gemini, or local models per node.

## Current State (v0.5.2)

- **Backend**: FastAPI 0.68, Pydantic v1, SQLAlchemy 1.4 (async), Python 3.9+
- **Frontend**: React 18, CRA (react-scripts), CSS modules, ReactFlow 11, TypeScript
- **Applets**: 3 (Writer/Artist/Memory), hardcoded to OpenAI + Stability AI
- **Execution**: Sequential only, no parallel branches, no conditionals
- **Auth**: None
- **Tests**: ~10% coverage
- **Deploy**: Docker, Fly.io (backend), Vercel (frontend)

## Target State (v1.0)

### 1. Stack Modernization

#### Backend
- **Python**: 3.11+ (for performance + modern syntax)
- **FastAPI**: Latest (0.115+)
- **Pydantic**: v2 (remove all v1 compat hacks like `model_to_dict`)
- **SQLAlchemy**: 2.0 (new-style ORM, mapped_column)
- **Alembic**: Latest with auto-migration generation
- **Database**: SQLite for dev, PostgreSQL for prod (already supported, just needs testing)
- Remove deprecated `@app.on_event` patterns (already migrated to lifespan)

#### Frontend
- **Build tool**: CRA (react-scripts) -> Vite 6
- **Styling**: CSS modules -> Tailwind CSS 4 + shadcn/ui components
- **State management**: Add Zustand for global state (workflow state, execution state, settings)
- **ReactFlow**: Update to latest (v12 has better performance and TypeScript support)
- **TypeScript**: Strict mode enabled
- Keep: React 18, Monaco Editor, anime.js (animations are good)

#### Dev Tooling
- Add ESLint 9 (flat config) + Prettier
- Add Ruff for Python linting/formatting
- Add pre-commit hooks (lint + format + type check)

### 2. UI/UX Overhaul

#### Design System
- Tailwind + shadcn/ui for consistent, accessible components
- Dark mode as default (toggle available)
- Design tokens for colors, spacing, typography
- Responsive layout (works on tablet, graceful on mobile)

#### Workflow Editor (Main Screen)
- **Left sidebar**: Collapsible node palette with search/filter, grouped by category (AI, Logic, Data, I/O)
- **Center canvas**: ReactFlow with minimap, better zoom controls, snap-to-grid
- **Right sidebar**: Node inspector panel — shows config for selected node, live output preview
- **Bottom panel**: Execution log with timestamps, expandable per-node details
- **Top bar**: Workflow name (editable inline), run button (prominent), save status indicator

#### Node Design
- Rounded cards with category-colored left border (AI = blue, Logic = purple, Data = green, I/O = orange)
- Status indicator dot (idle/running/success/error) with pulse animation
- Mini-preview of last output inside node (truncated text for Writer, thumbnail for Artist)
- Port indicators (input/output) with type hints on hover
- Handle validation — prevent invalid connections (e.g., image output to text-only input)

#### Execution Visualization
- Animated edge flow (particles moving along edges during execution)
- Node glow effect when actively processing
- Progress indicator per node (spinner → checkmark/X)
- Execution timeline bar showing total progress

#### Dashboard
- Workflow cards with last-run status, run count, last modified
- Quick-run button per card
- Search and filter workflows
- Usage stats (total runs, success rate, avg execution time)

#### New Pages
- **Settings**: API key management (encrypted at rest), theme toggle, default model selection
- **Applet Library**: Browse installed applets with capability descriptions, install new ones

### 3. Execution Engine Upgrades

#### Parallel Execution
- Fan-out: One node output feeds multiple downstream nodes simultaneously
- Fan-in: Multiple node outputs merge into a single downstream node (configurable merge strategy: concatenate, array, first-wins)
- Topological sort with parallel group detection (nodes with no dependencies on each other run concurrently)

#### Conditional Routing
- **If/Else Node**: Evaluate a condition on the input (contains, equals, regex match, JSON path check) and route to one of two output branches
- **Switch Node**: Multi-branch routing based on a value (like a switch statement)
- Visual indication of which branch was taken after execution

#### Loop Support
- **For-Each Node**: Iterate over an array input, executing a sub-workflow per item
- Max iteration limit (configurable, default 100) to prevent infinite loops
- Parallel iteration option (process N items concurrently)

#### Error Handling
- Per-node retry configuration (max retries, delay between retries)
- Per-node timeout (default 60s, configurable)
- Fallback path: If a node fails after retries, route to a fallback branch instead of failing the entire workflow
- Structured error codes (not just string messages)
- Execution state checkpointing — resume from last successful node on retry

#### Execution History
- Full execution trace per run (input/output for every node, timing, errors)
- Diff view between runs
- Re-run with modified inputs from history page

### 4. FM-Agnostic LLM Node

Replace the hardcoded Writer applet with a universal **LLM Node** that supports:

#### Providers
- **OpenAI**: GPT-4o, GPT-4o-mini, o1, o3
- **Anthropic**: Claude 4.5 Sonnet, Claude Opus 4
- **Google**: Gemini 2.5 Pro, Gemini 2.5 Flash
- **Local**: Ollama (llama, mistral, etc.) via OpenAI-compatible API
- **Custom**: Any OpenAI-compatible endpoint (user provides base URL + API key)

#### Configuration (per node)
- Provider dropdown (OpenAI / Anthropic / Google / Ollama / Custom)
- Model selector (populated dynamically based on provider)
- System prompt (text area)
- Temperature, max tokens, top-p sliders
- Structured output toggle (JSON mode with schema)
- Streaming toggle (show output as it generates in the node preview)

#### Implementation
- Provider adapter pattern: each provider implements a common interface
- API keys stored in settings (encrypted), not per-node
- Streaming via SSE to the frontend (not just final result)

### 5. New Applet Types

#### Core Applets (ship with v1.0)

| Applet | Description |
|--------|-------------|
| **LLM Node** | Universal FM node (replaces Writer). See section 4. |
| **Image Gen Node** | Universal image generation (replaces Artist). Supports DALL-E 3, Stability AI, Flux. |
| **Memory Node** | Upgraded: persistent vector store via ChromaDB or SQLite FTS (replaces in-memory dict) |
| **HTTP Request Node** | Make HTTP requests (GET/POST/PUT/DELETE). Config: URL, method, headers, body template. Useful for calling external APIs. |
| **Code Node** | Execute Python or JavaScript code. Input available as `data` variable. Sandboxed execution (subprocess with timeout). |
| **Transform Node** | Data transformation: JSON path extract, template string, regex replace, split/join. No code needed — config-driven. |
| **If/Else Node** | Conditional branching (see Execution Engine section) |
| **Merge Node** | Fan-in merge point for parallel branches |

#### Legacy Migration
- **WriterApplet** → deprecated, auto-migrates to LLM Node with OpenAI/GPT-4o preset
- **ArtistApplet** → deprecated, auto-migrates to Image Gen Node with Stability AI preset
- **MemoryApplet** → deprecated, auto-migrates to Memory Node with SQLite FTS backend

### 6. Authentication & Security

#### Auth
- JWT-based authentication with refresh tokens
- Email/password registration + login
- OAuth2 providers: Google, GitHub (stretch goal)
- API key authentication for programmatic access

#### Security
- API keys encrypted at rest (Fernet symmetric encryption)
- CORS properly configured per environment
- Rate limiting on all endpoints (configurable per-user)
- Code Node sandboxing (subprocess with resource limits, no filesystem access beyond /tmp)
- Input sanitization on all user-provided content
- HTTPS enforced in production

### 7. Testing & Quality

#### Backend
- pytest with async support (pytest-asyncio)
- Test coverage target: 80%+
- Unit tests for: every applet, orchestrator execution engine, all API endpoints, auth flow
- Integration tests for: full workflow execution (create → run → verify output)
- Fixtures for database setup/teardown

#### Frontend
- Vitest (replacing Jest, since we're moving to Vite)
- React Testing Library for component tests
- Test coverage target: 70%+
- Unit tests for: all components, services, state management
- Integration tests for: workflow creation flow, execution flow, settings flow
- E2E tests with Playwright: happy path workflow creation and execution

#### CI/CD
- GitHub Actions: lint → type-check → test → build on every PR
- Coverage reporting (Codecov or similar)
- Separate deploy pipelines for staging and production

### 8. API Improvements

#### RESTful API Hardening
- OpenAPI spec auto-generated (FastAPI gives this for free)
- Consistent error response format: `{ "error": { "code": "string", "message": "string", "details": {} } }`
- Pagination on list endpoints (offset/limit with total count)
- Request validation with Pydantic v2 (strict mode)
- API versioning: `/api/v1/...`

#### WebSocket Protocol
- Structured message format: `{ "type": "node_status" | "workflow_complete" | "error", "data": {} }`
- Reconnection with state recovery (client reconnects and gets current execution state)
- Authentication on WebSocket connections (JWT in query param or first message)

### 9. Performance

- **Backend**: async everywhere, connection pooling for PostgreSQL, lazy applet loading
- **Frontend**: Code splitting per page (React.lazy + Suspense), tree-shaking with Vite, ReactFlow virtualization for large workflows
- **Execution**: Parallel node execution with configurable concurrency limit
- **Database**: Indexes on frequently queried columns (flow_id, status, created_at), query optimization for history page

### 10. Deployment

- Docker Compose for local dev (orchestrator + frontend + PostgreSQL)
- Fly.io for backend (existing, update Dockerfile for Python 3.11)
- Vercel for frontend (existing, update for Vite build output)
- Environment-based configuration (dev/staging/prod)
- Health check endpoint: `GET /health` returning `{ "status": "ok", "version": "1.0.0", "uptime": "..." }`

## Competitive Positioning

Based on competitive analysis (see `docs/competitive-analysis-2026.md`):

### Landscape (Feb 2026)
| Competitor | Stars | Weakness SynApps exploits |
|-----------|-------|--------------------------|
| **n8n** (174k) | Automation tool that added AI. AI features feel bolted-on. Critical RCE vulnerabilities. | SynApps is AI-native from day one. Security-first design. |
| **LangFlow** (140k) | Performance ceiling (10-15s delays). Critical RCE vuln. Upgrades break state. | SynApps async execution is fast. Sandboxed execution. |
| **Dify** (127k) | Best execution engine, but weak frontend story. No embeddable UI. | SynApps: execution visualization directly in the canvas. |
| **Flowise** (47k) | Beginner-friendly but shallow. LangChain dependency leaks through. | SynApps: easy on-ramp AND depth. No framework lock-in. |
| **Rivet** (4.2k) | Best UX (real-time tracing) but desktop-only, tiny community. | SynApps: Rivet's vision as a web-native platform. |

### Differentiators to Implement in v1.0
1. **Real-time execution visualization in-canvas** — streaming tokens, data flow, execution state directly in nodes. No platform does this well on the web.
2. **Selective re-computation** — only re-execute nodes whose inputs changed (ComfyUI pattern). No LLM workflow builder does this yet.
3. **Security-first execution** — sandboxed Code Node (subprocess with resource limits). n8n and LangFlow both had CVSS 9+ RCE vulnerabilities.
4. **JSON workflow export** — shareable, versionable, loadable (ComfyUI pattern). Essential for open-source community.

### Features Inspired by Competitors
- **From Dify**: SSE structured events per node (workflow_started, node_finished, workflow_finished)
- **From Flowise**: HITL checkpoint nodes for human approval gates
- **From ComfyUI**: Cached intermediate results, selective re-execution
- **From Rivet**: Graph-in-graph nesting (reusable sub-workflows) — stretch goal for v1.0
- **From Google ADK**: Agent orchestration primitives (Sequential, Loop, Parallel as node types)

### Positioning Statement
> SynApps: The visual AI workflow builder that's free, fast, and FM-agnostic. Real-time execution tracing, parallel branches, multi-provider models. No subscription. No lock-in.

## Out of Scope (v1.x / v2.0)

- MCP tool integration (applets invoking MCP servers)
- Google A2A protocol support
- Google ADK integration
- Applet marketplace / community sharing
- Real-time collaborative editing (multiplayer)
- Custom model fine-tuning
- Mobile app
- Self-hosted installer / one-click deploy
- Workflow versioning / git integration
- Usage-based billing system
- Multi-tenant / team workspaces

## Success Criteria

1. `docker-compose up` starts the full stack in < 30 seconds
2. Create a 5-node workflow, run it, see animated execution — under 10 seconds total UX
3. Swap LLM provider mid-workflow (e.g., GPT-4o for node 1, Claude for node 2) — works seamlessly
4. Backend tests: 80%+ coverage, 0 failures
5. Frontend tests: 70%+ coverage, 0 failures
6. Lighthouse score: > 90 performance, > 90 accessibility
7. No hardcoded API keys, models, or provider-specific logic outside adapter layer
8. Works in Chrome, Firefox, Safari (latest 2 versions)

## Non-Goals

- This is NOT an enterprise platform. No RBAC, no audit logs, no compliance features.
- This is NOT a LangChain replacement. No Python SDK, no programmatic API for building workflows in code.
- This is NOT a managed service. Users self-host or deploy to their own Fly.io/Vercel.
- Keep it simple. If a feature adds complexity without clear user value, cut it.
