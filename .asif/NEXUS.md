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
**What**: Vector-based context storage (in-memory dict). Upgrade to ChromaDB planned.

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

---


## CoS Directives

> 36 completed directives archived to [NEXUS-archive.md](./NEXUS-archive.md) (2026-03-04, Wolf).

### DIRECTIVE-NXTG-20260304-08 — Adopt CI Gate Protocol
**From**: NXTG-AI CoS | **Priority**: P0
**Injected**: 2026-03-04 | **Estimate**: S | **Status**: PENDING

> **Estimate key**: S = hours (same session), M = 1-2 days, L = 3+ days

**Context**: New ASIF standard (`standards/ci-gate-protocol.md`). No push without local test pass. All teams must adopt.

**Action Items**:
1. [ ] Add CI Gate Protocol section to CLAUDE.md:
   ```
   ## CI Gate Protocol (ASIF Standard)
   Before EVERY `git push`, you MUST:
   1. Run the full test suite: `pytest --tb=short -q` (backend) and `npx vitest run` (frontend)
   2. Verify ZERO failures (xfail/skip OK, failures NOT OK)
   3. If tests fail → fix before pushing. No exceptions.
   4. Include test count in commit message: "Tests: X passed, Y skipped"
   Violating this protocol means broken CI, which means Asif gets spammed.
   ```
2. [ ] Install pre-push hook: `cp ~/ASIF/scripts/templates/pre-push-hook.sh .git/hooks/pre-push && chmod +x .git/hooks/pre-push`
3. [ ] Run full test suite. Report current passing count.
4. [ ] Verify CI is GREEN on GitHub Actions. If RED, fix the root cause before any other work.

**Response** (filled by project team):
>

---

---

## Portfolio Intelligence

_Cross-project insights injected by ASIF CoS. Read these for awareness — they inform your priorities._

### PI-001: 2Brain Is Your First Dogfood Consumer (2026-02-17)
**From**: NXTG-AI CoS + Asif (founder decision, 2026-02-17)

2Brain (P-13) has been stale for 38 days, blocked waiting for SynApps. Asif has confirmed: **SynApps' first dogfood use case is 2Brain's capture→classify→store pipeline.** This means:

1. Your visual workflow builder will be validated with a real portfolio use case
2. 2Brain's pipeline (capture text → Ollama classification → storage) is the test scenario
3. CLX9 CoS directive to build a standalone Python replacement (DIRECTIVE-CLX9-20260216-03) has been **DEFERRED** — SynApps IS the path forward

Ship the dogfood milestone. 2Brain is waiting.

### PI-002: Portfolio Standard Is Pydantic v2 + FastAPI Modern (2026-02-17)
~~You are the only project still on Pydantic v1.~~ **CORRECTION (Wolf, 2026-03-02):** SynApps has been on Pydantic v2 since N-07 shipped (2026-02-18). All 45 model classes use v2 patterns. PI-002 was stale at time of injection. The portfolio standard is Pydantic v2 + FastAPI 0.115+ + Python 3.11+ — **SynApps is fully compliant.**

### PI-003: Podcast-Pipeline Has a Shipped DAG Execution Engine (2026-02-18)
Podcast-Pipeline (P-04) shipped a stage graph orchestrator (N-01) with topological sort, dependency resolution, and parallel level computation — 14 tests passing. Your N-10 (Parallel Execution Engine) needs the same thing: topological sort with parallel group detection, fan-out/fan-in. Both are Python. Reference their implementation before building from scratch.

---


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

