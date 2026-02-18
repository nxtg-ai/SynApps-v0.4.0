# NEXUS — synapps Vision-to-Execution Dashboard

> **Owner**: Asif Waliuddin
> **Last Updated**: 2026-02-16
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
| N-07 | Backend Stack Upgrade | STACK | BUILDING | P0 | 2026-02 |
| N-08 | Frontend Stack Migration | STACK | DECIDED | P0 | — |
| N-09 | Universal LLM Node | NODES | DECIDED | P0 | — |
| N-10 | Parallel Execution Engine | EXECUTION | DECIDED | P0 | — |
| N-11 | Conditional Routing (If/Else) | EXECUTION | DECIDED | P1 | — |
| N-12 | JWT Authentication | SECURITY | DECIDED | P0 | — |
| N-13 | Code Node with Sandboxing | NODES | DECIDED | P1 | — |
| N-14 | Execution Visualization | VISUAL | DECIDED | P1 | — |
| N-15 | Comprehensive Testing | STACK | DECIDED | P0 | — |

---

## Vision Pillars

### STACK — "Modern Production Foundation"
- Upgrade from Python 3.9/FastAPI 0.68/Pydantic v1 → Python 3.11+/FastAPI 0.115+/Pydantic v2/SQLAlchemy 2.0
- Frontend: CRA → Vite 6, CSS modules → Tailwind 4 + shadcn/ui, add Zustand, TypeScript strict
- **Shipped**: N-06
- **Building**: N-07
- **Decided**: N-08, N-15

### VISUAL — "Real-Time Execution Canvas"
- React Flow canvas with drag-and-drop. Animated edge flow, node glow, execution timeline
- Dashboard and settings pages. Responsive layout, dark mode default
- **Shipped**: N-01
- **Decided**: N-14

### NODES — "FM-Agnostic Agent Blocks"
- Universal LLM Node (OpenAI/Anthropic/Google/Ollama/Custom). Image Gen Node. Memory Node (ChromaDB)
- HTTP Request, Code (sandboxed), Transform, Merge nodes
- **Shipped**: N-02, N-03, N-04
- **Decided**: N-09, N-13

### EXECUTION — "Advanced Workflow Primitives"
- Parallel fan-out/fan-in. Conditional routing. Loop support (For-Each)
- Per-node error handling (retries, timeouts, fallback paths). Checkpointing
- **Shipped**: N-05
- **Decided**: N-10, N-11

### SECURITY — "Enterprise Readiness"
- JWT auth with refresh tokens. Encrypted API keys at rest
- Rate limiting per-user. Sandboxed Code Node. Input sanitization
- **Decided**: N-12

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
**Pillar**: STACK | **Status**: BUILDING | **Priority**: P0
**What**: Python 3.11+, FastAPI 0.115+, Pydantic v2, SQLAlchemy 2.0. 38 tasks in plan.
**Next step**: Complete migration sequence per .forge/plan.md.

### N-08: Frontend Stack Migration
**Pillar**: STACK | **Status**: DECIDED | **Priority**: P0
**What**: CRA → Vite 6, CSS modules → Tailwind 4 + shadcn/ui. Zustand state. TypeScript strict. ReactFlow v12.

### N-09: Universal LLM Node
**Pillar**: NODES | **Status**: DECIDED | **Priority**: P0
**What**: OpenAI, Anthropic, Google, Ollama, Custom endpoints. Per-node provider/model selection. Streaming via SSE.

### N-10: Parallel Execution Engine
**Pillar**: EXECUTION | **Status**: DECIDED | **Priority**: P0
**What**: Topological sort with parallel group detection. Fan-out/fan-in. Configurable concurrency limits.

### N-11: Conditional Routing
**Pillar**: EXECUTION | **Status**: DECIDED | **Priority**: P1
**What**: If/Else node (contains, equals, regex, JSON path). Switch node (multi-branch).

### N-12: JWT Authentication
**Pillar**: SECURITY | **Status**: DECIDED | **Priority**: P0
**What**: Email/password + refresh tokens. OAuth2 stretch (Google, GitHub). Encrypted API key storage.

### N-13: Code Node with Sandboxing
**Pillar**: NODES | **Status**: DECIDED | **Priority**: P1
**What**: Python/JavaScript execution in subprocess. Resource limits, filesystem restrictions, timeout enforcement.

### N-14: Execution Visualization
**Pillar**: VISUAL | **Status**: DECIDED | **Priority**: P1
**What**: Animated edge flow particles, node glow, progress spinner, execution timeline bar, mini-output preview.

### N-15: Comprehensive Testing
**Pillar**: STACK | **Status**: DECIDED | **Priority**: P0
**What**: pytest + Vitest. Backend 80%+ coverage, frontend 70%+. Playwright E2E. CI/CD pipeline.

---

## Health Flags (RED)

- **Ancient stack**: Python 3.9, FastAPI 0.68, Pydantic v1, CRA — security vulnerabilities, no modern async
- **Test coverage ~10%**: No CI integration. Regression risk
- **No authentication**: Anyone with URL access can see all workflows
- **Hardcoded models**: Writer=GPT-4o, Artist=StabilityAI. No provider flexibility
- **Sequential execution only**: No parallel branches, conditionals, or loops
- **38-task backlog to v1.0**: Estimated 2-3 months full-time

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

---

## CoS Directives

### DIRECTIVE-NXTG-20260216-01 — Modernization Progress Check
**From**: NXTG-AI CoS | **Date**: 2026-02-16 | **Status**: PENDING
**Priority**: P0

**Action Items**:
1. [ ] Report current Python version, FastAPI version, Pydantic version, SQLAlchemy version. Are any of the N-07 upgrades completed? Update N-07 status.
2. [ ] Run existing tests (`pytest` or whatever test runner is configured) and report pass/fail counts. Update N-15 with baseline.
3. [ ] Check if the merge conflict with `origin/main` has been resolved (repo had diverged history with 5 local vs 1 remote commits as of 2026-02-16). Report git status.
4. [ ] List all security vulnerabilities from outdated dependencies. Run `pip audit` or equivalent if available.

**Constraints**:
- Health is RED. The #1 priority is N-07 (backend stack upgrade). Do NOT add new features until the stack is modernized.
- Upgrade order: Python 3.9→3.11+ first, then Pydantic v1→v2, then FastAPI 0.68→0.115+, then SQLAlchemy 1.4→2.0.
- Do NOT break existing shipped features (N-01 through N-06) during upgrade.

**Portfolio context**:
- synapps (P-10) is in the Developer Tools vertical alongside NXTG-Forge (P-03).
- The portfolio standard is Pydantic v2 + FastAPI modern + Python 3.11+. synapps is the only project still on v1.
- Frontend migration (N-08, CRA→Vite) should happen AFTER backend stabilizes.

**Response** (filled by project team):
> _(pending)_

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
You are the only project still on Pydantic v1. The portfolio standard is Pydantic v2 + FastAPI 0.115+ + Python 3.11+. Your modernization (N-07) isn't just tech debt — it's alignment with the entire ecosystem. oneDB, threedb, and Podcast-Pipeline all use Pydantic v2.

---

## Team Questions

_(Project team: add questions for ASIF CoS here. They will be answered during the next enrichment cycle.)_
