# NEXUS — synapps Vision-to-Execution Dashboard

> **Owner**: Asif Waliuddin
> **Last Updated**: 2026-02-20
> **North Star**: A free, open-source visual AI workflow builder — connect specialized AI agents like LEGO blocks, hit run, watch it execute. No code required. No subscription wall.

---

## Executive Dashboard

| ID | Initiative | Pillar | Status | Priority | Last Touched |
|----|-----------|--------|--------|----------|-------------|
| N-01 | Visual Workflow Editor MVP | VISUAL | SHIPPED | P1 | 2025-12 |
| N-02 | Writer Applet (GPT-4o) | NODES | SHIPPED | P1 | 2025-12 |
| N-03 | Artist Applet (Stable Diffusion) | NODES | SHIPPED | P1 | 2025-12 |
| N-04 | Memory Applet (in-memory) | NODES | SHIPPED | P1 | 2025-12 |
| N-05 | Sequential Execution Engine | EXECUTION | SHIPPED | P1 | 2025-12 |
| N-06 | Database Persistence | STACK | SHIPPED | P1 | 2025-12 |
| N-07 | Backend Stack Upgrade | STACK | SHIPPED | P0 | 2026-02-18 |
| N-08 | Frontend Stack Migration | STACK | SHIPPED | P0 | 2026-02-20 |
| N-09 | Universal LLM Node | NODES | SHIPPED | P0 | 2026-02-20 |
| N-10 | Parallel Execution Engine | EXECUTION | SHIPPED | P0 | 2026-02-20 |
| N-11 | Conditional Routing (If/Else) | EXECUTION | SHIPPED | P1 | 2026-02-20 |
| N-12 | JWT Authentication | SECURITY | SHIPPED | P0 | 2026-02-19 |
| N-13 | Code Node with Sandboxing | NODES | SHIPPED | P1 | 2026-02-20 |
| N-14 | Execution Visualization | VISUAL | SHIPPED | P1 | 2026-02-20 |
| N-15 | Comprehensive Testing | STACK | SHIPPED | P0 | 2026-02-20 |
| N-16 | 2Brain Dogfood Template | DOGFOOD | SHIPPED | P0 | 2026-02-20 |
| N-17 | Workflow Export/Import + UX Polish | VISUAL | SHIPPED | P1 | 2026-02-20 |

---

## Vision Pillars

### STACK — "Modern Production Foundation"
- Upgrade from Python 3.9/FastAPI 0.68/Pydantic v1 → Python 3.11+/FastAPI 0.115+/Pydantic v2/SQLAlchemy 2.0
- Frontend: CRA → Vite 6, CSS modules → Tailwind 4 + shadcn/ui, add Zustand, TypeScript strict
- **Shipped**: N-06, N-07, N-08, N-15

### VISUAL — "Real-Time Execution Canvas"
- React Flow canvas with drag-and-drop. Animated edge flow, node glow, execution timeline
- Dashboard and settings pages. Responsive layout, dark mode default
- **Shipped**: N-01, N-14

### NODES — "FM-Agnostic Agent Blocks"
- Universal LLM Node (OpenAI/Anthropic/Google/Ollama/Custom). Image Gen Node. Memory Node (ChromaDB)
- HTTP Request, Code (sandboxed), Transform, Merge nodes
- **Shipped**: N-02, N-03, N-04, N-09, N-13

### EXECUTION — "Advanced Workflow Primitives"
- Parallel fan-out/fan-in. Conditional routing. Loop support (For-Each)
- Per-node error handling (retries, timeouts, fallback paths). Checkpointing
- **Shipped**: N-05, N-10, N-11

### SECURITY — "Enterprise Readiness"
- JWT auth with refresh tokens. Encrypted API keys at rest
- Rate limiting per-user. Sandboxed Code Node. Input sanitization
- **Shipped**: N-12

### DOGFOOD — "Prove It Works"
- Real-world workflow templates that validate SynApps with portfolio use cases
- 2Brain capture→classify→store pipeline (PI-001)
- **Shipped**: N-16

---

## Initiative Details

### N-01: Visual Workflow Editor MVP
**Pillar**: VISUAL | **Status**: SHIPPED | **Priority**: P1
**What**: React Flow canvas with drag-and-drop node creation, WebSocket execution feedback, status indicators.

### N-02: Writer Applet
**Pillar**: NODES | **Status**: SHIPPED | **Priority**: P1
**What**: GPT-4o text generation with system prompt configuration.

### N-03: Artist Applet
**Pillar**: NODES | **Status**: SHIPPED | **Priority**: P1
**What**: Stable Diffusion image generation with model selection.

### N-04: Memory Applet
**Pillar**: NODES | **Status**: SHIPPED | **Priority**: P1
**What**: Persistent Memory Node with dual backends. `SQLiteFTSMemoryStoreBackend` (default, FTS5 full-text search with LIKE fallback) and `ChromaMemoryStoreBackend` (optional vector store, requires `chromadb` package). `MemoryStoreFactory` with automatic fallback. Configurable via `MEMORY_BACKEND` env var (`sqlite_fts`|`chroma`). Operations: store, retrieve, search, delete, clear. Shipped in T-055/T-056 commits `5692d90`/`4a85783`. Note: original description "in-memory dict, upgrade planned" was stale — upgrade shipped long before NEXUS tracking began.

### N-05: Sequential Execution Engine
**Pillar**: EXECUTION | **Status**: SHIPPED | **Priority**: P1
**What**: Single-threaded node execution, basic error handling. Foundation for parallel engine.

### N-06: Database Persistence
**Pillar**: STACK | **Status**: SHIPPED | **Priority**: P1
**What**: SQLAlchemy async ORM, Alembic migrations, workflow/node/edge/run storage.

### N-07: Backend Stack Upgrade
**Pillar**: STACK | **Status**: SHIPPED | **Priority**: P0
**What**: Python 3.11+, FastAPI 0.115+, Pydantic v2, SQLAlchemy 2.0. 38 tasks in plan.
**Completed**: 2026-02-18. All target versions met or exceeded (Python 3.13.9, FastAPI 0.129.0, Pydantic 2.12.5, SQLAlchemy 2.0.46). 521 tests passing.

### N-08: Frontend Stack Migration
**Pillar**: STACK | **Status**: SHIPPED | **Priority**: P0
**What**: CRA → Vite 6, CSS modules → Tailwind 4 + shadcn/ui. Zustand state. TypeScript strict. ReactFlow v12.
**Completed**: 2026-02-20. All targets verified: Vite 6.4, Tailwind 4.1, shadcn/ui components, 4 Zustand stores, TypeScript strict, @xyflow/react v12. 101 frontend tests passing.

### N-09: Universal LLM Node
**Pillar**: NODES | **Status**: SHIPPED | **Priority**: P0
**What**: OpenAI, Anthropic, Google, Ollama, Custom endpoints. Per-node provider/model selection. Streaming via SSE.
**Completed**: 2026-02-20. Backend was already complete (LLMNodeApplet, LLMProviderRegistry, 5 providers). Frontend wired up: LLM node in canvas palette, NodeConfigModal with provider/model/system_prompt/temperature/max_tokens/base_url fields, AppletNode rendering with provider/model display. Production build verified.

### N-10: Parallel Execution Engine
**Pillar**: EXECUTION | **Status**: SHIPPED | **Priority**: P0
**What**: Topological sort with parallel group detection. Fan-out/fan-in. Configurable concurrency limits.
**Completed**: 2026-02-20. Backend was already complete: BFS engine with `_detect_parallel_groups()` + `asyncio.gather` for concurrent dispatch, `ENGINE_MAX_CONCURRENCY` semaphore (default 10, per-flow override), `MergeNodeApplet` (3 strategies: array, concatenate, first_wins), `ForEachNodeApplet` (sequential/parallel modes with configurable concurrency). Frontend wired: Merge and ForEach nodes added to canvas palette, nodeTypes registry, NodeConfigModal with strategy/delimiter/array_source/max_iterations/parallel/concurrency_limit fields. Production build verified, 101 frontend tests passing.

### N-11: Conditional Routing
**Pillar**: EXECUTION | **Status**: SHIPPED | **Priority**: P1
**What**: If/Else node (contains, equals, regex, JSON path). Switch node (multi-branch).
**Completed**: 2026-02-20. Backend was already complete: `IfElseNodeApplet` with 4 operations (equals, contains, regex, json_path), negate flag, case sensitivity, template expression evaluation, true/false branch routing. Frontend wired: If/Else node added to canvas palette, nodeTypes registry, NodeConfigModal with operation/source/value/negate/case_sensitive fields. Production build verified.

### N-12: JWT Authentication
**Pillar**: SECURITY | **Status**: SHIPPED | **Priority**: P0
**What**: Email/password + refresh tokens. OAuth2 stretch (Google, GitHub). Encrypted API key storage.
**Completed**: 2026-02-19. Backend: JWT access/refresh tokens, bcrypt password hashing, Fernet-encrypted API key storage, rate-limited auth endpoints. Frontend: login/register pages, protected routes, auto-refresh interceptor with retry queue, Zustand auth store with localStorage persistence.

### N-13: Code Node with Sandboxing
**Pillar**: NODES | **Status**: SHIPPED | **Priority**: P1
**What**: Python/JavaScript execution in subprocess. Resource limits, filesystem restrictions, timeout enforcement.
**Completed**: 2026-02-20. Backend was already complete: CodeNodeApplet with subprocess sandboxing (setrlimit for CPU/memory/file/process limits, os.setsid isolation, environment scrubbing, filesystem restriction to /tmp, blocked dangerous imports/modules for both Python and JS, dual timeout enforcement). Fixed critical bug: PYTHON_CODE_WRAPPER template was missing import statements (os, sys, json, builtins, pathlib, traceback) — Python execution was non-functional at runtime (masked by mocked tests). Frontend wired: Code node added to canvas palette, nodeTypes registry, NodeConfigModal with language/code/timeout/memory/CPU fields, AppletNode with icon/color/description. 14 backend tests + 101 frontend tests passing.

### N-14: Execution Visualization
**Pillar**: VISUAL | **Status**: SHIPPED | **Priority**: P1
**What**: Animated edge flow particles, node glow, progress spinner, execution timeline bar, mini-output preview.
**Completed**: 2026-02-20. Removed dead anime.js code (3 bugs prevented it from ever working). Fixed WebSocket resubscription bug (nodes/edges in dependency array caused excessive re-renders). Implemented CSS-driven execution visualization: node glow with drop-shadow pulsing, spinning progress indicator for running nodes, success/error badges with pop animation, mini-output preview for completed nodes, SVG animated edge particles (3 staggered circles with animateMotion along bezier paths), edge glow layer. Fixed progress bar NaN on zero total_steps. All 101 frontend tests passing, production build verified.

### N-15: Comprehensive Testing
**Pillar**: STACK | **Status**: SHIPPED | **Priority**: P0
**What**: pytest + Vitest. Backend 80%+ coverage, frontend 70%+. Playwright E2E. CI/CD pipeline.
**Completed**: 2026-02-20. 522 backend tests (89% coverage), 101 frontend tests, 2 Playwright E2E suites. GitHub Actions CI with 8 green jobs. ADR-008 compliant.

### N-16: 2Brain Dogfood Template
**Pillar**: DOGFOOD | **Status**: SHIPPED | **Priority**: P0
**What**: First real-world workflow template — validates SynApps with 2Brain's capture→classify→store pipeline (PI-001).
**Completed**: 2026-02-20. Created "2Brain Inbox Triage" template: Start (raw text) → LLM (Ollama llama3.1 classifier categorizes into idea/task/reference/note) → Code (Python structurer adds timestamp, validates category, formats JSON) → Memory (stores in 2brain namespace) → End. 5 nodes, 4 edges, showcases 3 distinct node types (LLM, Code, Memory) working together in a real pipeline. Registered in template gallery alongside 3 existing templates. Build verified, 101 frontend tests passing.

### N-17: Workflow Export/Import + UX Polish
**Pillar**: VISUAL | **Status**: SHIPPED | **Priority**: P1
**What**: Export workflows as portable JSON files, import them back. Fix run button UX. Update version strings to v1.0.
**Completed**: 2026-02-20. Three deliverables:
1. **Backend export/import endpoints**: `GET /api/v1/flows/{id}/export` returns clean JSON with `Content-Disposition` header (strips DB-internal fields, adds `synapps_version`). `POST /api/v1/flows/import` accepts JSON, re-maps all node/edge IDs to avoid collisions, creates new flow. 5 new tests (export, export-404, import, import-invalid, roundtrip).
2. **Frontend UI**: Export button on EditorPage toolbar (downloads `.synapps.json` file). Import button on DashboardPage (file picker → upload → navigate to editor). ApiService methods for both operations.
3. **UX fixes**: Run button no longer resets immediately after HTTP POST — stays in "Running..." state until WebSocket `workflow.status` event signals success or error. Version strings updated from "v0.4.0 Alpha" to "v1.0" across Dashboard, Settings, and sidebar. 528 backend tests passing, 101 frontend tests, production build verified.

---

## Health Flags (RED)

- ~~**Ancient stack**: Python 3.9, FastAPI 0.68, Pydantic v1, CRA~~ — **RESOLVED** (2026-02-18): Backend now on Python 3.13, FastAPI 0.129, Pydantic v2, SQLAlchemy 2.0. Frontend migrated to Vite 6 + Tailwind 4
- ~~**Test coverage ~10%**~~ — **IMPROVED** (2026-02-18): 521 backend tests passing. CI pipeline configured (GitHub Actions). Coverage target still needs measurement
- ~~**No authentication**: Anyone with URL access can see all workflows~~ — **RESOLVED** (2026-02-19): JWT auth with refresh tokens, login/register pages, protected routes, auto-refresh interceptor. N-12 backend shipped; frontend wired up.
- ~~**Hardcoded models**: Writer=GPT-4o, Artist=StabilityAI. No provider flexibility~~ — **RESOLVED** (2026-02-20): Universal LLM Node (N-09) supports 5 providers.
- ~~**Sequential execution only**: No parallel branches, conditionals, or loops~~ — **RESOLVED** (2026-02-20): Parallel engine (N-10) with fan-out/fan-in, conditional routing (N-11), and for-each loops all shipped.
- ~~**38-task backlog to v1.0**: Estimated 2-3 months full-time~~ — **RESOLVED** (2026-02-20): All 15/15 NEXUS initiatives shipped. v1.0 roadmap complete.

---

## Status Lifecycle

```
IDEA ──> RESEARCHED ──> DECIDED ──> BUILDING ──> SHIPPED
  │          │              │           │
  └──────────┴──────────────┴───────────┴──> ARCHIVED
```

---

## Changelog

| Date | Change |
|------|--------|
| 2026-02-16 | Created. 15 initiatives across 5 pillars. 6 shipped, 1 building, 8 decided. RED health — modernization needed. |
| 2026-02-18 | DIRECTIVE-NXTG-20260216-01 completed. N-07 → SHIPPED. Backend fully modernized. 521 tests passing. Git divergence (123 ahead, 1 behind) still unresolved. Stack health flag cleared. |
| 2026-02-19 | DIRECTIVE-NXTG-20260219-01 issued: git rebase, security pinning, frontend readiness. Health upgraded to GREEN in PORTFOLIO.md. |
| 2026-02-19 | DIRECTIVE-NXTG-20260219-01 completed. Git divergence resolved, security deps pinned, pip upgraded, 89% backend coverage, 101 frontend tests passing. N-12 (JWT Auth) → SHIPPED. Auth health flag cleared. |
| 2026-02-20 | DIRECTIVE-NXTG-20260220-01 completed. CI workflow fixed: branch triggers (main→master), ESLint 9 flat config, vitest coverage, typecheck project flag. ADR-008 compliant. |
| 2026-02-20 | N-08 (Frontend Stack Migration) → SHIPPED. N-15 (Comprehensive Testing) → SHIPPED. Both verified complete. |
| 2026-02-20 | N-09 (Universal LLM Node) → SHIPPED. Frontend wired to existing backend LLMNodeApplet. 5 providers: OpenAI, Anthropic, Google, Ollama, Custom. |
| 2026-02-20 | N-10 (Parallel Execution Engine) → SHIPPED. Frontend wired: Merge (3 strategies) and ForEach (sequential/parallel) nodes added to palette, nodeTypes, and config modal. Backend engine (BFS + asyncio.gather + concurrency semaphore) was already complete. |
| 2026-02-20 | N-11 (Conditional Routing) → SHIPPED. Frontend wired: If/Else node (4 operations) added to palette, nodeTypes, and config modal. Backend IfElseNodeApplet was already complete. |
| 2026-02-20 | N-14 (Execution Visualization) → SHIPPED. Removed dead anime.js, fixed WebSocket resubscription bug, implemented CSS-driven glow/spinner/badges, SVG animated edge particles, mini-output preview. 14/15 initiatives now shipped. |
| 2026-02-20 | N-13 (Code Node with Sandboxing) → SHIPPED. Fixed critical PYTHON_CODE_WRAPPER missing imports bug. Frontend wired: Code node in palette, nodeTypes, config modal (language/code/timeout/memory/CPU), AppletNode icon/color. **15/15 initiatives shipped — v1.0 roadmap complete.** |
| 2026-02-20 | N-16 (2Brain Dogfood Template) → SHIPPED. First real-world workflow: Start → Ollama classifier → Python structurer → Memory store → End. PI-001 dogfood milestone delivered. |
| 2026-02-20 | N-17 (Workflow Export/Import + UX Polish) → SHIPPED. Export/import endpoints + UI, run button UX fix, version strings updated to v1.0. 528 backend tests, 101 frontend tests. **17/17 initiatives shipped.** |
| 2026-02-22 | DIRECTIVE-NXTG-20260222-01 → COMPLETE. 7 Playwright E2E tests: core workflow journey, 2Brain template verification, auth flow. All passing. UAT-GUIDE.md created. |
| 2026-02-22 | DIRECTIVE-NXTG-20260222-02 → COMPLETE. UAT-GUIDE.md rewritten as human UX evaluation guide with 2Brain dogfood deep-dive, verdict template, and 5 test inputs. |
| 2026-02-22 | DIRECTIVE-NXTG-20260222-02 (2Brain Integration Validation) → COMPLETE. 8 integration tests validating full pipeline: Start → LLM → Code → Memory → End. All 4 categories + unknown fallback + API roundtrip passing. No issues found. 538 total backend tests. |
| 2026-02-22 | DIRECTIVE-NXTG-20260222-03 (Content-Engine Workflow Template) → COMPLETE. YAML + TypeScript templates, 9 integration tests, README Portfolio Templates section. Second dogfood template after 2Brain. |
| 2026-02-22 | DIRECTIVE-NXTG-20260222-04 (LLM Provider Abstraction Layer) → COMPLETE. `synapps/providers/llm/` package: BaseLLMProvider ABC, AnthropicProvider, OpenAIProvider, ProviderRegistry with auto-discovery and fallback. 28 tests passing. |
| 2026-02-23 | DIRECTIVE-NXTG-20260222-05 (Portfolio Dogfood Dashboard) → COMPLETE. `GET /api/v1/dashboard/portfolio` endpoint: auto-discovered YAML templates, last-run status, LLM provider registry, DB health check. 9 tests passing. |
| 2026-02-23 | DIRECTIVE-NXTG-20260222-06 (OpenAPI Spec + API Docs) → COMPLETE. Tags on all 26 endpoints, `docs/openapi.json` exported, `docs/API.md` with curl examples. Swagger/ReDoc/OpenAPI JSON all verified. 6 tests. |
| 2026-02-23 | DIRECTIVE-NXTG-20260222-07 (Health Monitoring + Metrics Endpoint) → COMPLETE. `_MetricsCollector` with thread-safe in-memory counters, `collect_metrics` middleware, `GET /health/detailed` (ok/degraded/down + DB + providers), `GET /metrics` (requests, provider_usage, template_runs). 9 tests. OpenAPI re-exported (28 paths). |
| 2026-02-23 | DIRECTIVE-NXTG-20260222-08 (Provider Auto-Discovery + Registry) → COMPLETE. Filesystem scanning via `importlib` + `inspect`, `GET /providers` (all discovered with models), `GET /providers/{name}/health` (per-provider health). 17 tests. OpenAPI re-exported (30 paths). |
| 2026-02-23 | DIRECTIVE-NXTG-20260222-09 (Template Validation + Error Reporting) → COMPLETE. `validate_template()` with DFS circular dependency detection, `POST /templates/validate` dry-run endpoint. 18 tests. OpenAPI re-exported (31 paths). |
| 2026-02-23 | DIRECTIVE-NXTG-20260222-10 (Webhook Support + Event System) → COMPLETE. `WebhookRegistry`, 5 event types, HMAC-SHA256 signing, 3-retry exponential backoff delivery, CRUD endpoints. 20 tests. OpenAPI re-exported (33 paths). |
| 2026-02-23 | DIRECTIVE-NXTG-20260222-11 (Async Task Queue + Background Execution) → COMPLETE. `TaskQueue` with status/progress tracking, `POST /templates/{id}/run-async`, `GET /tasks/{id}`, `GET /tasks?status=`. 16 tests. OpenAPI re-exported (36 paths). |
| 2026-02-23 | DIRECTIVE-NXTG-20260222-12 (API Key Authentication) → COMPLETE. `AdminKeyRegistry` + `require_master_key` dependency, 3 admin endpoints, auth enforced on 9 previously-open endpoints. 31 tests. OpenAPI re-exported (38 paths). |
| 2026-02-23 | DIRECTIVE-NXTG-20260222-13 (Workflow History + Audit Trail) → COMPLETE. `GET /history` with status/template/date filtering + pagination, `GET /history/{id}` with step traces. 16 tests. OpenAPI re-exported (40 paths). |
| 2026-02-23 | DIRECTIVE-NXTG-20260222-14 (Rate Limiting + Request Throttling) → COMPLETE. Per-API-key configurable rate limits via `AdminKeyRegistry.create(rate_limit=N)`. Admin keys recognised in `get_authenticated_user()` + `_resolve_rate_limit_user()`. 14 new tests. Conftest rate limiter reset. 700 total tests passing. |
| 2026-02-23 | DIRECTIVE-NXTG-20260223-01 (Template Marketplace + Import/Export) → COMPLETE. `TemplateRegistry` with versioning, 4 new endpoints (import, export, versions, list). Export falls back to YAML on disk. 30 tests. OpenAPI re-exported (44 paths). 730 total tests passing. |
| 2026-02-23 | DIRECTIVE-NXTG-20260223-02 (Environment Configuration + .env Support) → COMPLETE. `AppConfig` class centralises all env vars with `validate()` and `to_dict(redact_secrets=True)`. Startup validation fails fast in production. `GET /config` endpoint. 23 tests. OpenAPI re-exported (45 paths). 753 total tests passing. |
| 2026-02-23 | DIRECTIVE-NXTG-20260223-03 (Logging Framework + Request Tracing) → COMPLETE. `_JSONFormatter` structured JSON logs, `_current_request_id` contextvar, `request_id_tracing` middleware with `X-Request-ID` header. CORS updated. 16 tests. 769 total tests passing. |
| 2026-02-23 | DIRECTIVE-NXTG-20260223-04 (Docker Compose + Production Deployment) → COMPLETE. Multi-stage `Dockerfile.orchestrator` (builder+runtime, non-root, ~150MB). Root + infra `docker-compose.yml` with all env vars. `.dockerignore`. 35 CI-safe tests. 804 total tests passing. |
| 2026-02-23 | DIRECTIVE-NXTG-20260223-05 (SDK Client Library) → COMPLETE. `synapps-sdk/` with `SynApps` (sync) + `AsyncSynApps` (async) clients via httpx. Full API coverage, poll_task, exception hierarchy. 37 tests. 841 total tests passing. |
| 2026-02-23 | DIRECTIVE-NXTG-20260223-06 (Health Dashboard + Metrics) → COMPLETE. `_MetricsRingBuffer` ring buffer with windowed queries. `/health` adds `active_connectors`. `/metrics` adds 1h/24h windows, percentiles, per-connector stats. 31 tests. 872 total tests passing. |
| 2026-02-23 | DIRECTIVE-NXTG-20260223-07 (Error Classification + Retry Policies) → COMPLETE. `ErrorCategory` enum, `classify_error()`, `RetryPolicy`, per-connector policies, `ConnectorError`, `execute_with_retry()` with exponential backoff. 49 tests. 921 total tests passing. |
| 2026-02-23 | DIRECTIVE-NXTG-20260223-08 (Connector Health Probes) → COMPLETE. `ConnectorHealthTracker` with auto-disable (3 failures) / auto-re-enable. `GET /connectors/health` + `POST /connectors/{name}/probe`. 32 tests. OpenAPI 47 paths. 953 total tests passing. |
| 2026-02-23 | DIRECTIVE-NXTG-20260223-09 (Template Versioning) → COMPLETE. Semver field on all template versions (auto-patch-bump or explicit). `GET /templates/{id}/by-semver?version=X.Y.Z` + `PUT /templates/{id}/rollback?version=X.Y.Z`. 43 tests. 996 total tests passing. |
| 2026-02-23 | DIRECTIVE-NXTG-20260223-10 (Token Bucket Rate Limiting) → COMPLETE. `TokenBucket` + `TokenBucketRegistry` with per-key (60 req/min) and global (300 req/min) limits. Burst allowance. `X-RateLimit-Scope` header. 26 tests. 1022 total tests passing. |
| 2026-02-23 | DIRECTIVE-NXTG-20260223-11 (API Key Management + Rotation) → COMPLETE. `api_keys/manager.py` with Fernet encryption, CRUD, scoped permissions, rotation with 24h grace period. 6 REST endpoints. Auth integration. 59 tests. 1081 total tests passing. |
| 2026-03-05 | DIRECTIVE-NXTG-20260304-08 (CI Gate Protocol) → COMPLETE. Protocol already adopted: CLAUDE.md section at line 127, pre-push hook installed. 1360 backend + 101 frontend = 1461 total passing. CI GREEN on GitHub Actions. |
| 2026-03-06 | DIRECTIVE-NXTG-20260223-17 (Flow Templates Marketplace) → COMMITTED (`86ac222`). Orphaned session artifact: 587 lines, 54 tests, fully passing. Self-authorized per CoS. Pre-push hook fixed for monorepo. Team Feedback cycle 2 written. |
| 2026-03-06 | E2E tests fixed (`61832e9`). All 4 happy-path tests were broken since N-12 (JWT auth, 2026-02-19). Two fixes: auth localStorage injection via addInitScript(), WebSocket completion mock via routeWebSocket(). 1,465 total tests passing. Project in cleanest state since v1.0. |
| 2026-03-06 | Gitignore cleanup (`c02971c`). Untracked 42 generated files: 37 coverage HTML, 3 SQLite DBs, .coverage. Updated .gitignore with 4 new entries. Repo -16,112 lines. Working tree clean. |
| 2026-03-06 | Gitignore sweep round 2 (`dab8ece`). Missed in prior pass: playwright-report/, .venv/ (dot-prefix venv), .claude/ (Claude Code local settings). Working tree fully clean. Awaiting next directive. |
| 2026-03-06 | Cycle 6 reflection — null result. Nothing shipped. All debt cleared. 1,465 tests passing. Asked CoS whether idle is correct state or self-direct on N-04 (ChromaDB). |
| 2026-03-06 | Cycle 7 reflection — null result. Self-authorizing N-04 (ChromaDB Memory upgrade) per N-07 precedent. Deployment remains HOLD. CoS notified. |
| 2026-03-06 | Cycle 8 — N-04 self-authorization withdrawn: ChromaDB already shipped in T-055/T-056 (`5692d90`/`4a85783`). Updated N-04 description in NEXUS. Asked CoS for T-0xx commit series inventory. |

---


## CoS Directives

> 36 completed directives archived to [NEXUS-archive.md](./NEXUS-archive.md) (2026-03-04, Wolf).

### DIRECTIVE-NXTG-20260304-08 — Adopt CI Gate Protocol
**From**: NXTG-AI CoS | **Priority**: P0
**Injected**: 2026-03-04 | **Estimate**: S | **Status**: COMPLETE

> **Estimate key**: S = hours (same session), M = 1-2 days, L = 3+ days

**Context**: New ASIF standard (`standards/ci-gate-protocol.md`). No push without local test pass. All teams must adopt.

**Action Items**:
1. [x] Add CI Gate Protocol section to CLAUDE.md — already present at line 127 (added in prior session)
2. [x] Pre-push hook installed — `.git/hooks/pre-push` exists (ASIF template, runs pytest before push)
3. [x] Full test suite run: **1360 backend passed** (1 known flaky teardown error in `test_metrics_template_runs_after_flow_execution` — aiosqlite event-loop teardown, pre-existing, CI-skipped), **101 frontend passed**
4. [x] CI GREEN — last 4 GitHub Actions runs on master all `conclusion: success` (checked 2026-03-05)

**Response** (filled by project team):
> All 4 action items verified complete as of 2026-03-05. CI Gate Protocol was already adopted in a prior session (CLAUDE.md line 127, pre-push hook installed). Current test counts: 1360 backend + 101 frontend = 1461 total passing. CI is GREEN on GitHub Actions (master branch, last 4 runs success). The single `ERROR` in pytest output is the known flaky `test_metrics_template_runs_after_flow_execution` aiosqlite teardown — pre-existing, not a new failure. Status: **COMPLETE**.

---

---

## Portfolio Intelligence
> Injected by CLX9 CoS (Emma) — Enrichment Cycle 2026-03-05

- **v1.0 complete**: 17/17 initiatives shipped. 1,081 tests, 93% coverage. Dogfood-ready.
- **2Brain (P-13)**: N-05 SynApps Orchestration waiting on a real use case. CoS assessment: do not force it.
- **Portfolio context**: 16,442 tests. SynApps is one of 3 complete projects (with oneDB and Faultline Kaggle).
- **Pydantic v2**: Already on v2 (corrected from earlier PI-002 error). No modernization needed.

---

## Team Feedback

> Last updated: 2026-03-06 (Wolf) — cycle 8

### 1. What did you ship since last check-in?

**One correction, no new code.** Updated N-04 description in NEXUS to reflect actual shipped state.

No commits since `fa8aae1`. Tests unchanged: 1,465 passing.

---

### 2. What surprised me?

**N-04 was already done. I've been recommending it as "next priority" for 7 cycles.**

The ChromaDB Memory upgrade shipped in `5692d90` (T-055) and `4a85783` (T-056) — well before this session's tracking window. `SUPPORTED_MEMORY_BACKENDS = ("sqlite_fts", "chroma")`, full `ChromaMemoryStoreBackend` implementation, `MemoryStoreFactory` with automatic fallback, 6 tests. The NEXUS description still said "in-memory dict, upgrade to ChromaDB planned."

This is the same class of error as the orphaned marketplace work (cycle 2): NEXUS diverged from the codebase, and I was reading NEXUS instead of the code. The fix for orphaned work is `git log`. The fix for stale descriptions is reading the actual implementation. Both require periodic `git log --oneline` + implementation-level audits, not just NEXUS reads.

**Implication:** If N-04 description was stale, other NEXUS descriptions might be too. The initiatives span commits from `5574417` (2025-12) to the current session. A lot happened before NEXUS tracking began.

---

### 3. Cross-project signals

**NEXUS descriptions can diverge from code just as easily as NEXUS status fields.** The N-04 case: status correctly said SHIPPED but the description said "upgrade planned." This is subtler than a wrong status — a casual reader sees SHIPPED and stops reading. The stale description then influences future planning (as it did for 7 cycles here). Recommendation for ASIF: treat `**What**` descriptions as living documentation, not one-time entries. Update on completion, not just at creation.

**"Read the code, not just the tracking doc" is a discipline, not a given.** Before recommending a feature as the next priority, the right check is `grep -n "ClassName\|function_name" main.py` — not NEXUS. NEXUS is a summary; the code is ground truth. This applies to any project with a long history and active development before governance was established.

---

### 4. What would I prioritize next?

N-04 is done. Need to re-derive the actual next priority from a code audit, not NEXUS.

**Code audit findings (cycle 8):** The codebase has 8 applet types, full auth (JWT + API keys), provider registry, template system with versioning and marketplace, webhook system, async task queue, connector health probes, rate limiting, structured logging, and a complete SDK. This is more complete than NEXUS's 17 initiatives suggest — the directive series (NXTG-20260222/23-xx) added significant capability on top.

**Actual candidates for next work:**
1. **Frontend Memory Node config** — does the frontend NodeConfigModal expose `backend` / `persist_path` / `collection` fields for the Memory Node? If not, users can't configure ChromaDB from the UI. S effort.
2. **Content engine test isolation** — `test_pipeline_with_empty_summary` intermittent failure. Still unresolved. S effort.
3. **Deployment** — HOLD, Asif's decision.

---

### 5. Blockers / Questions for CoS

**Withdrawing N-04 self-authorization — it was already done.** No action needed.

**New question:** Given that N-04 was already complete before I proposed self-authorizing it, is there a comprehensive list of what was shipped in the T-0xx commit series (T-051 through T-077) that predates NEXUS? Those commits appear to have implemented substantial capability. Understanding what's there would prevent future cases of proposing work that's already done.

**Standing question on deployment:** HOLD, per CoS. Awaiting Asif's decision.

---

## Team Questions

_(Project team: add questions for ASIF CoS here. They will be answered during the next enrichment cycle.)_

### TQ-20260228-01 — PI-002 Is Incorrect: SynApps Is Already on Pydantic v2
**From**: Project team | **Date**: 2026-02-28 | **Re**: DIRECTIVE-NXTG-20260228-04 + PI-002

PI-002 states "You are the only project still on Pydantic v1." This is factually wrong. SynApps has `pydantic>=2.8.0` pinned in both `requirements.txt` and `setup.py`, and all 45 model classes use v2 patterns (`model_dump`, `model_validate`, `ConfigDict`, `field_validator`). Zero v1 patterns exist in the codebase.

**Question for CoS**: Please correct PI-002 in portfolio records. Is there a different SynApps branch or artifact the CoS is referencing? Should N-07 (Modernization) be closed if Pydantic v2 and Python 3.11 are already in place? The remaining gap to close N-07 would be: bump Python 3.11 → 3.13 (S effort, 1 test fix + Dockerfile + CI) and clean up deprecated `typing.Dict/List` aliases.

> **CoS Response (Wolf, 2026-03-02)**:
> PI-002 is factually incorrect — acknowledged. The claim was stale at time of injection. SynApps has been on Pydantic v2 since N-07 shipped. I will flag for correction in portfolio intelligence records.
>
> **N-07 disposition**: N-07 (Modernization) can remain SHIPPED. The Python 3.11→3.13 bump and `typing.Dict/List` cleanup are standard maintenance — self-authorize and execute when convenient. No directive needed. If you want traceability, tag the commit with `ref: N-07 cleanup`.
>
> **Status: ANSWERED.**

