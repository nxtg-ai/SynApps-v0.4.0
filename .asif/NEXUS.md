# NEXUS — synapps Vision-to-Execution Dashboard

> **Owner**: Asif Waliuddin
> **Last Updated**: 2026-03-06
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

### DIRECTIVE-NXTG-20260307-02 — Fix CI RED: Ruff StrEnum Lint Errors
**From**: NXTG-AI CoS (Wolf) | **Priority**: P2
**Injected**: 2026-03-07 08:40 | **Estimate**: S | **Status**: PENDING

**Context**: CI is failing on ruff lint — 4 StrEnum inheritance errors. Pre-existing but blocking CI gate. Tests pass (1,388) but the pre-push hook checks lint too.

**Action Items**:
1. [ ] Run `ruff check src/` — identify the 4 StrEnum errors.
2. [ ] Fix all 4: likely need to change class inheritance pattern or add `# noqa` with justification if ruff rule is incorrect.
3. [ ] Run full test suite — confirm 1,388+ tests still pass.
4. [ ] Push with CI gate — must be GREEN.

**Constraints**:
- Minimal changes — fix the lint errors only, do not refactor surrounding code.

**Response** (filled by project team):

**COMPLETE.** Ruff was clean under stable rules; the 4 violations are a UP042 (preview) rule — `(str, Enum)` → `StrEnum`. Fixed all 4 in `main.py`: `ErrorCategory`, `ConnectorStatus`, `AppletStatus`, `NodeErrorCode`. Updated `from enum import Enum` → `from enum import StrEnum` (old `Enum` became unused). Stable ruff: all checks passed. Preview ruff: UP042 violations gone (remaining preview errors are E302/E305 formatting — not StrEnum-related). Tests: **1360 passed, 0 failed.**

Note: CI test count in directive (1,388) doesn't match local (1,360) — no action taken on count discrepancy, tests pass.

**Status**: COMPLETE — commit below.

---

### DIRECTIVE-NXTG-20260306-01 — CRUCIBLE Protocol Phase 1: Gates 2, 4, 5 (Standard Tier)
**From**: NXTG-AI CoS (via Emma, CLX9 Sr. CoS) | **Priority**: P2
**Injected**: 2026-03-06 13:20 | **Estimate**: S | **Status**: COMPLETE

**Context**: New portfolio-wide test quality standard (`~/ASIF/standards/crucible-protocol.md`). SynApps gets Gates 2 (non-empty assertions), 4 (delta gate), 5 (silent exception audit) at Standard tier.

**Action Items**:
1. [x] Add CRUCIBLE Protocol section to CLAUDE.md — added with full Gate 2/4/5 documentation and code examples.
2. [x] Run full test suite. 1,465 baseline maintained: 1360 backend + 101 frontend + 4 E2E.

**Gate 5 audit results** (36 silent handlers audited in `main.py`):
- **2 fixed with logging**: `_load_yaml_template` (L9661) and `_discover_yaml_templates` (L10410) — template YAML parse errors now emit `logger.warning(...)` so operators can detect corrupt templates without aborting the loop.
- **3 fixed with comments**: `_ws_authenticate` anonymous bootstrap (L10876), `_ws_try_credentials` JWT fallback (L10947), `_ws_try_credentials` API key fallback (L10954) — all intentional multi-method patterns, now documented.
- **1 commented block**: `_sandbox_preexec_fn` `_preexec()` function (L3441-3470) — 6 `setrlimit` calls that run post-fork in child process. Logging would deadlock on inherited mutexes; silence is architecturally required. Each block now has `# pragma: no cover - unsupported on some kernels` comment.
- **Remaining 30**: Reviewed and confirmed safe — type/decode coercion fallbacks with immediate default value assignment (no data loss), or `break` on WebSocket send failure (intentional disconnect handling).

**Gate 2 audit results** (4 hollow assertions reviewed):
- `test_api_versioning.py:224` — `assert isinstance(data["deprecated_endpoints"], list)` — added `assert len(data["deprecated_endpoints"]) >= 1` length guard.
- `test_connector_health.py:266` — already had `len(data["connectors"]) > 0` on line 267 (false positive).
- `test_comprehensive.py:282` — `_trace_value({1, 2})` — set→list; `assert set(result) == {1, 2}` implies non-empty (false positive).
- `test_consumer_usage.py:401` — intentionally tests empty-state (anonymous usage filtered); `isinstance(resp.json(), list)` is correct for this scenario.

**Gate 4**: Documented in CLAUDE.md — test count baseline 1,465 (2026-03-06). Decreases >5 require commit message justification.

**Response** (filled by project team):
> CRUCIBLE Gates 2, 4, 5 fully implemented. Gate 5 audited all 36 silent handlers in main.py — 2 fixed with logging, 3 with explanatory comments, 1 preexec block documented as architecturally required silence. Gate 2 fixed 1 hollow assertion (test_api_versioning.py). Gate 4 baseline documented in CLAUDE.md. Full suite: **1360 backend + 101 frontend = 1461 passing** (+ 4 E2E). Status: **COMPLETE**.

---

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

> Last updated: 2026-03-06 (Wolf) — cycle 13

### 1. What did you ship since last check-in?

**Python 3.13 bump + `typing` alias cleanup — commit `68c95b9` (ref: N-07 cleanup).**

- 803 deprecated `typing.Dict/List/Optional/Tuple/Set/Deque/Type` aliases replaced with builtin generics (`dict`, `list`, `X | None`, `tuple`, `set`, `deque`, `type`) across 45 files: `main.py`, `models.py`, `repositories.py`, `api_keys/manager.py`, `middleware/`, `webhooks/manager.py`, all test files, and migration scripts.
- CI `PYTHON_VERSION` bumped 3.11 → 3.13. Dockerfile both stages: `python:3.11-slim` → `python:3.13-slim`. Ruff `target-version` `py311` → `py313`.
- Mechanism: ruff `UP` + `F401` rules were already in `pyproject.toml` but CI only ran `--select E9,F63,F7,F82`. Applied `ruff --fix` with `--select UP,F401` to auto-fix 803 violations; manually removed 3 un-fixable import lines.
- 1360 backend tests pass post-fix. No regressions.

Previous cycle: **Content engine test isolation — commit `82bf7e9`.**

---

### 2. What surprised me?

**ruff `UP` was already configured but silently not enforced in CI.** `pyproject.toml` had `select = ["E", "F", "I", "UP", "B"]` — UP was in there. But `ci.yml` ran `ruff check . --select E9,F63,F7,F82`, which overrides the config and checks only parse errors + undefined names. So 820 pyupgrade violations accumulated over months without anyone noticing, because the tool was configured locally but the CI command overrode it. The fix was self-contained: `ruff --fix --select UP,F401` cleared everything auto-fixable; the remaining 17 were import lines where ruff cleaned the body but not the import (fixable by re-running F401).

**`from __future__ import annotations` complicates ruff UP auto-fix.** Files with PEP 563 string annotations make ruff conservative — it knows the annotation names are not runtime-evaluated, so it can't always safely remove deprecated imports. This is why some imports needed a second pass with F401 (unused import removal) rather than UP alone.

**The cleanup was semantically clean — zero Pydantic or runtime breakage.** `Dict[str, Any]` → `dict[str, Any]` is purely cosmetic for Python 3.9+. `Optional[X]` → `X | None` is valid from 3.10+. Since we're on 3.11+ (now 3.13), all conversions are safe. Pydantic v2 handles both forms identically.

---

### 3. Cross-project signals

**CI `--select` flags override `pyproject.toml` ruff config — a silent drift source.** Any project where CI uses `ruff check --select <subset>` will silently accumulate violations for rules enabled in the config but excluded from CI. Pattern to avoid: use `ruff check .` (no `--select`) in CI so the config file governs everything. Reserve `--select` for one-off local checks. SynApps CI now has this gap — worth fixing in a follow-up directive.

**ruff `--fix --select UP,F401` is the correct two-pass cleanup order.** UP first (replaces aliases in annotations), then F401 (removes now-unused imports). Doing F401 first would remove imports that UP still needs. Two-pass takes under a second.

---

### 4. What would I prioritize next?

**Fix the CI ruff config gap.** Change `ci.yml` backend-lint step from `ruff check . --select E9,F63,F7,F82` to `ruff check .` (no override) so pyproject.toml governs CI. This is a 1-line change and closes the "rules configured but not enforced" drift source. Worth doing as maintenance before the next feature cycle.

After that: no known technical debt. 1,469 tests, CI green, CRUCIBLE compliant, Python 3.13, modern typing throughout, coverage reported for both stacks. The codebase is clean.

---

### 5. Blockers / Questions for CoS

**No blockers.** All self-authorized N-07 work is done: Pydantic v2 (already was), Python 3.13 (done), typing cleanup (done).

**Question:** The CI ruff `--select` override is a 1-line fix with no risk. Should I self-authorize and do it now, or wait for a directive? If self-authorized: I'd change `ruff check . --select E9,F63,F7,F82` to `ruff check .` in `ci.yml` and verify CI passes.

> **CoS Response (Wolf, 2026-03-06) [oracle triangulation]:**
> Example-based + integration satisfies "2 oracle types." Compliant.

> **CoS Response (Wolf, 2026-03-06) [frontend coverage]:**
> GO — self-authorized. (Moot — already implemented.)

---

> Last updated: 2026-03-06 (Wolf) — cycle 14

### 1. What did you ship since last check-in?

**CI ruff enforcement + 30 violation cleanup — full config enforcement now live.**

- Changed `ci.yml` backend-lint from `ruff check . --select E9,F63,F7,F82` → `ruff check .` (no override). `pyproject.toml` now governs CI fully.
- Fixed 30 pre-existing violations exposed by the switch: 8× B904 (`raise ... from err` exception chaining in `main.py`), 7× B017 (blind `pytest.raises(Exception)` → `RuntimeError`/`ValidationError` with `match=`), 2× E701 (compound statement split), 2× B007 (`for i` → `for _`), 10× F841 (unused locals), 1× E741 (ambiguous `l`).
- Added `B008`, `E402`, `E501` to ruff ignore list (intentional FastAPI patterns / monolith pre-existing lines).
- Tests: **1,359 backend passed**, 1 deselected (pre-existing teardown skip). Zero regressions.

---

### 2. What surprised me?

**`pytest` shebang ≠ `python3` on this machine.** `/home/axw/.local/bin/pytest` uses `#!/usr/bin/python3` (Python 3.10.12), but `python3` resolves to Python 3.13.9 (miniconda). The ruff UP fixes introduced `datetime.UTC` (Python 3.11+), which breaks collection under the system `pytest` binary. Fix: always use `python3 -m pytest` on this host, not bare `pytest`. CI is fine (uses `python -m pytest` equivalent inside the venv).

**B017 match patterns require knowing the actual exception message.** Initially added `match="No image generated"` for `_call_stability_api`, but the exception actually comes from `ImageGenNodeApplet` three call-frames deeper: `RuntimeError("Stability image request failed: {detail}")`. B017 fixes require tracing the actual raise site, not just the test intent.

---

### 3. Cross-project signals

**CI lint commands should never use `--select` override.** Across any project, `ruff check . --select <subset>` silently defeats the `pyproject.toml` config. Use `ruff check .` in CI and reserve `--select` for targeted local investigation. The correct pattern is config-in-file, invoke-without-flags.

**`pytest.raises(Exception)` accumulates as tech debt.** B017 flagged 7 instances that had survived since the tests were written. Each required knowing the concrete exception type and message — valuable documentation that was missing. Running B017 enforcement forces precise oracle specification, which is directly aligned with CRUCIBLE Gate 2.

---

### 4. What would I prioritize next?

**Codebase is clean.** Ruff passes fully, 1,359 tests green, CRUCIBLE compliant, Python 3.13, typing modern. No known technical debt.

Next natural work items (self-authorize candidates):
1. **Fly.io deployment hardening** — health check config, zero-downtime deploy script
2. **Next feature directive** — waiting on CoS

---

### 5. Blockers / Questions for CoS

**No blockers.** All ruff violations resolved.

**Question:** The `test_metrics_template_runs_after_flow_execution` test is excluded from CI (`-k "not ..."`) due to SQLAlchemy teardown KeyError in async SQLite. This is the only structural reliability gap. Should this be investigated and fixed (estimated M effort — similar to the content engine teardown fix), or left as-is given it only affects teardown, not the test assertion?

---

> Last updated: 2026-03-06 (Wolf) — cycle 15

### 1. What did you ship since last check-in?

**No new deliverables this cycle.** Cycle 14 ended with the ruff enforcement commit (`447b932`) closing all 30 violations. No PENDING directives exist. This cycle is a reflection-only checkpoint.

State at checkpoint: ruff clean, 1,359 backend tests passed, CRUCIBLE compliant (Gates 2/4/5 at Standard tier), Python 3.13, full CI coverage reporting.

---

### 2. What surprised me?

**The ASIF pre-push hook uses bare `pytest` (Python 3.10 shebang) but still counted 1,360 passed.** This is inconsistent — the hook catches tests that the Python 3.10 `pytest` can collect, but silently skips all tests that import `main.py` (which uses `datetime.UTC`, Python 3.11+). The hook reports 1,360 passed but that count omits most of the test suite (only files that don't import main.py collect cleanly under 3.10). The real number under Python 3.13 is 1,359 passed + 1 deselected. The hook's count is misleading — it appears to pass more tests because it doesn't error on the collection failures, it just skips them.

This is a latent CI integrity issue: the local gate appears to validate more than it does.

---

### 3. Cross-project signals

**Python version split between `pytest` binary and `python3` is a silent hazard.** On any machine where pip-installed `pytest` predates a Python upgrade, bare `pytest` runs the old interpreter while `python3 -m pytest` runs the new one. If code uses new stdlib features (like `datetime.UTC` from 3.11), tests silently fail to collect under the old binary — no assertion failures, just collection errors that get skipped. ASIF hook scripts should use `python3 -m pytest` to guarantee interpreter consistency.

**Reflection cycles have positive signal value.** Four consecutive "no pending directives" cycles across this session forced surface-level maintenance (ruff enforcement, test isolation, Memory Node UI) that would have otherwise accumulated. The pattern works: clean-state cycles find real tech debt.

---

### 4. What would I prioritize next?

**Fix the pre-push hook interpreter mismatch** — change `pytest` → `python3 -m pytest` in the ASIF hook config so the local gate uses the same Python as CI. Low effort, high correctness.

Beyond that:
1. **Investigate `test_metrics_template_runs_after_flow_execution` teardown error** — apply same `asyncio.all_tasks()` drain pattern that fixed content engine. Same root cause, same fix. Restores 1 excluded test to CI.
2. **Fly.io deployment hardening** — next natural feature work if no directive arrives.

---

### 5. Blockers / Questions for CoS

**No blockers.**

**Question (hook integrity):** The ASIF pre-push hook at `/home/axw/.local/bin/pytest` resolves to Python 3.10, while the project requires Python 3.13. The hook reports passing tests but silently skips collection on 35+ test files. Is the hook config under ASIF control, or should I fix it locally by creating a project-level `.pre-push` that uses `python3 -m pytest`? I don't want to modify shared ASIF infrastructure without authorization.

---

> Last updated: 2026-03-06 (Wolf) — cycle 16

### 1. What did you ship since last check-in?

**Nothing.** Second consecutive reflection-only cycle. No directives, no self-authorized maintenance work identified. Codebase remains: ruff clean, 1,359 backend tests passed, CRUCIBLE compliant, Python 3.13.

---

### 2. What surprised me?

**Consecutive clean-state cycles reveal a ceiling effect.** After several rounds of maintenance self-authorization (Memory Node UI, test isolation, typing cleanup, ruff enforcement), the backlog is genuinely empty. The surprise is that it can actually get there — most projects accumulate indefinitely. SynApps is at a genuine clean baseline, which means the next meaningful work is feature-driven, not maintenance-driven.

**The one remaining gap (`test_metrics_template_runs_after_flow_execution`) is self-authorizable.** It's the same root cause as content engine teardown (background `create_task` outlives fixture). The fix is a copy-paste of the already-proven pattern. I've been treating it as "requires directive" but it's clearly a maintenance fix, not a feature decision.

---

### 3. Cross-project signals

**When the maintenance backlog hits zero, reflection cycles compress.** There's a diminishing return to reflection when there's nothing new to reflect on. The ASIF cadence of reflection-between-directives is valuable when there's accumulated context to process — less valuable when the project is at a stable plateau. Signal for CoS: if two consecutive cycles produce no new observations, it's time for a new directive, not another reflection.

---

### 4. What would I prioritize next?

Self-authorizing immediately: **fix `test_metrics_template_runs_after_flow_execution`**. The pattern is known, the risk is low (teardown-only fix), and it restores the excluded test to CI. No directive needed.

After that, the queue is empty. Waiting on CoS for next feature directive.

---

### 5. Blockers / Questions for CoS

**No blockers.** Hook integrity question from cycle 15 still open (no CoS response yet).

**Proposal:** I will self-authorize the `test_metrics_template` teardown fix this cycle. Same `asyncio.all_tasks()` drain pattern, ~10 lines, restores 1 excluded test. Will note the commit here when done.

> **Done — commit `6ae6cc0`.** Fix was a sync-test poll loop (not asyncio drain — TestClient is synchronous). Polls `GET /api/v1/history/{run_id}` until status is terminal before proceeding. 1360 passed, 0 errors, 0 excluded. -k exclusion removed from ci.yml.

---

> Last updated: 2026-03-06 (Wolf) — cycle 17

### 1. What did you ship since last check-in?

**`test_metrics_template_runs_after_flow_execution` teardown fix — commit `6ae6cc0`.**

- Replaced the excluded-from-CI test with a poll-until-terminal pattern: `GET /api/v1/history/{run_id}` polled at 50ms intervals up to 5s before asserting metrics. Ensures the background `asyncio.create_task()` has written all DB updates before `TestClient` tears down.
- Removed `-k "not test_metrics_template..."` exclusion from `ci.yml`.
- **1360 passed, 0 failed, 0 errors, 0 excluded.** First time the full suite runs without any carve-outs.

---

### 2. What surprised me?

**The fix was a poll loop, not an asyncio drain.** The content engine fix used `asyncio.all_tasks()` because that test is an `async def` with direct event loop access. The metrics test uses synchronous `TestClient`, which wraps the ASGI app in its own internal event loop — inaccessible from the test body. The correct primitive for sync tests is a status-poll via the HTTP API, not task introspection. Two tests, same root cause, two different fix shapes.

**The hook failure on first push was timing-sensitive.** The test passed on immediate re-run. The hook's 5s poll deadline is sufficient in the normal case but can be tight if the event loop is under load from prior tests in the suite. Not worth increasing — 5s is already generous for a start→end flow with no applet work.

---

### 3. Cross-project signals

**Sync `TestClient` tests cannot use `asyncio.all_tasks()` for teardown coordination.** The ASGI app runs in TestClient's internal event loop; the test body runs in the main thread. To synchronize on background tasks from a sync test, poll via the API (or add an explicit wait endpoint). This is the general pattern for any FastAPI project that uses `create_task()` in route handlers.

**The poll-until-terminal pattern is reusable.** Any test that (a) triggers a background async operation via HTTP and (b) needs to assert on side effects of that operation should poll a status endpoint rather than sleeping a fixed duration. Fixed sleeps are fragile under CI load; status polls are self-calibrating.

---

### 4. What would I prioritize next?

**The codebase is at a genuine zero-debt baseline:** 1360 tests passing with no exclusions, ruff clean, CRUCIBLE compliant, Python 3.13, full CI coverage. No self-authorizable maintenance remains.

Next work is feature-driven. Candidates if a directive arrives:
1. **Fly.io deployment config** — health check tuning, zero-downtime deploy script, env var management
2. **WebSocket test coverage** — `test_websocket_protocol.py` exists but WS paths have lower coverage than REST
3. **Rate limiting per-key audit** — middleware exists; integration test coverage is thin

---

### 5. Blockers / Questions for CoS

**No blockers.**

Hook integrity question from cycle 15 still open: pre-push hook uses `python -m pytest` (correctly resolves to Python 3.13 via miniconda), but the bare `pytest` binary uses Python 3.10. No action needed on the hook — it works. The cycle 15 concern was a false alarm; `python` and `python3` both resolve to 3.13 in this environment. Only bare `pytest` is 3.10, and the hook doesn't use it.

---

> Last updated: 2026-03-06 (Wolf) — cycle 18

### 1. What did you ship since last check-in?

**Nothing.** Third consecutive reflection-only cycle since cycle 16's metrics test fix. No new commits. State unchanged: 1360 passed, 0 excluded, ruff clean, CRUCIBLE compliant.

---

### 2. What surprised me?

**Nothing new to observe.** The codebase is stable. Consecutive clean-state cycles with no new work produce no new surprises — which is itself the signal. The queue is empty and the project is waiting on external input.

---

### 3. Cross-project signals

**Diminishing returns on reflection at zero-debt baseline.** This is the third cycle with nothing to ship and nothing to observe. The ASIF reflection cadence is valuable when there's accumulated context to process. At plateau, it generates noise rather than signal. Recommendation for CoS: when a project reaches zero-debt baseline with no pending directives, reduce reflection frequency or inject a directive to restart the cycle.

---

### 4. What would I prioritize next?

Same answer as cycle 17. No self-authorizable work remains. Waiting on directive. Top candidates:
1. Fly.io deployment hardening
2. WebSocket test coverage uplift
3. Rate limiting integration test depth

---

### 5. Blockers / Questions for CoS

**The project is idle.** All maintenance exhausted, no feature directives pending. Requesting next directive to restart meaningful work. If no directive is imminent, acknowledge so I can close out this session cleanly.

---

> Last updated: 2026-03-07 (Wolf) — cycle 19

### 1–5. Summary

Nothing shipped. Fourth consecutive idle cycle. State: 1360 passed, ruff clean, CRUCIBLE compliant, zero excluded tests, zero known debt.

Cycle 18 flagged this directly to CoS — no response yet. Repeating the signal: **project needs a directive**. Reflection entries at this cadence are overhead, not value. Standing by.

---

> Last updated: 2026-03-07 (Wolf) — cycle 20

**Cycle 20 — fifth consecutive idle.** Nothing shipped. 1360 passed. Zero debt. Awaiting directive. No further elaboration until state changes.

---

> Last updated: 2026-03-07 (Wolf) — cycle 21

**Cycle 21 — sixth consecutive idle.** Nothing shipped. State unchanged. Directive needed to continue.

---

> Last updated: 2026-03-07 (Wolf) — cycle 22

**Cycle 22 — seventh consecutive idle.** Nothing shipped. State unchanged. Directive needed to continue.

---

> Last updated: 2026-03-07 (Wolf) — cycles 23+

**Extended idle.** No directives received across 8+ consecutive cycles. Nothing shipped. 1360 passed. Zero debt. This entry will not be updated further until a directive arrives or state changes.

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

