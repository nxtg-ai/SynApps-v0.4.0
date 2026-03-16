# SynApps v0.5 to v1.0 — Detailed Migration Plan

## Quick Summary

| Migration | Files Changed | Risk | Effort |
|-----------|--------------|------|--------|
| CRA to Vite | 11 files | LOW | 2-3h |
| Pydantic v1 to v2 | 3 files (5 changes) | LOW | 30min |
| SQLAlchemy 1.4 to 2.0 | 5 files | MEDIUM | 2h |
| CSS to Tailwind | 16 CSS files (~2,200 lines) | MEDIUM | 6-8h |
| main.py decomposition | 1 file -> 7 modules | MEDIUM | 3-4h |
| Test coverage (15% -> 80%) | ~20 new test files | HIGH effort | 8-12h |

## Critical Issues Found

### 1. API Keys Committed to Repo
**File:** `.env.development` lines 8-9 contain real API keys:
```
OPENAI_API_KEY=sk-proj-v96plU...
STABILITY_API_KEY=sk-D7OWl4...
```
**Action:** Rotate keys immediately. Add `.env.development` to `.gitignore`.

### 2. Dead Dependencies
- `socket.io-client` (^4.5.3) — never imported anywhere. WebSocket service uses native `WebSocket` API.
- `react-flow-renderer` (^10.3.17) — legacy package. Code uses `reactflow` (v11) exclusively.
**Action:** Remove both from `package.json`.

### 3. Latent Bug in main.py
Line 438: `broadcast_status` is used as both a function parameter name and a variable name, shadowing the imported function. If an error occurs during applet execution, line 438 tries to call a dict as a function → `TypeError: 'dict' object is not callable`.

---

## 1. CRA to Vite Migration

### Files to Change

| File | Action |
|------|--------|
| `apps/web-frontend/package.json` | Rewrite deps/scripts, remove eslintConfig/browserslist |
| `apps/web-frontend/package-lock.json` | Delete and regenerate |
| `apps/web-frontend/tsconfig.json` | Update target, module, moduleResolution, add vite/client |
| `apps/web-frontend/public/index.html` | Move to project root, strip `%PUBLIC_URL%`, add script tag |
| `apps/web-frontend/vite.config.ts` | NEW: Create with React plugin + proxy + vitest config |
| `apps/web-frontend/src/vite-env.d.ts` | NEW: TypeScript types for import.meta.env |
| `apps/web-frontend/src/services/ApiService.ts:18` | `process.env.REACT_APP_API_URL` -> `import.meta.env.VITE_API_URL` |
| `apps/web-frontend/src/services/WebSocketService.ts:30` | `process.env.REACT_APP_WEBSOCKET_URL` -> `import.meta.env.VITE_WEBSOCKET_URL` |
| `apps/web-frontend/src/.../AppletNode.test.tsx` | `jest.mock` -> `vi.mock`, `jest.fn` -> `vi.fn` |
| `apps/web-frontend/src/.../HistoryPage.test.tsx` | Same Jest -> Vitest migration |
| `.env.example` | Rename `REACT_APP_*` to `VITE_*` |

### New Dependencies
- Add: `vite`, `@vitejs/plugin-react`, `vitest`, `@vitest/coverage-v8`, `jsdom`
- Remove: `react-scripts`, `webpack-dev-server`, `@testing-library/jest-dom`, `@types/jest`, `nth-check`
- Remove (dead): `socket.io-client`, `react-flow-renderer`

---

## 2. Pydantic v1 to v2 Migration

Only 5 changes needed — very clean codebase:

| File | Line | Change |
|------|------|--------|
| `models.py` | 160-166 | `.dict()` override -> `.model_dump()` override |
| `main.py` | 200-205 | DELETE `model_to_dict()` compat shim |
| `main.py` | 257 | `model_to_dict(status)` -> `status` (already a dict) |
| `main.py` | 510 | `model_to_dict(flow)` -> `flow.model_dump()` |
| `requirements.txt` | 3 | `pydantic>=1.8.2` -> `pydantic>=2.0.0` |

No `@validator`, no `class Config`, no `orm_mode` — all clean BaseModel usage.

---

## 3. SQLAlchemy 1.4 to 2.0 Migration

| File | Change |
|------|--------|
| `models.py:10,15` | `declarative_base()` -> `class Base(DeclarativeBase)` |
| `models.py:22-103` | `Column()` -> `mapped_column()` + `Mapped[]` annotations (4 models) |
| `repositories.py:13` | `from sqlalchemy.future import select` -> `from sqlalchemy import select` |
| `db.py:43-49` | `sessionmaker(class_=AsyncSession)` -> `async_sessionmaker` |
| `migrations/env.py:86-92` | `AsyncEngine(engine_from_config(...))` -> `create_async_engine(...)` |
| `requirements.txt:7` | `sqlalchemy>=1.4.0` -> `sqlalchemy>=2.0.0` |

Good news: Repository layer already uses `select()` not `session.query()` — already 2.0 style.

---

## 4. CSS to Tailwind Migration

### CSS File Inventory (16 files, ~2,200 lines)

| File | Lines | Complexity |
|------|-------|------------|
| `index.css` | 82 | LOW — global resets, CSS variables |
| `App.css` | 5 | TRIVIAL |
| `CodeEditor.css` | 203 | MEDIUM |
| `MainLayout.css` | 108 | MEDIUM |
| `Notifications.css` | 189 | MEDIUM |
| `TemplateLoader.css` | 123 | LOW |
| `NodeConfigModal.css` | 220 | HIGH — glassmorphism, animations |
| `NodeContextMenu.css` | 83 | LOW |
| `WorkflowCanvas.css` | 264 | HIGH — ReactFlow overrides |
| `Nodes.css` | 270 | HIGH — pulse/shake/glow animations |
| `AppletLibraryPage.css` | 272 | MEDIUM |
| `DashboardPage.css` | 345 | MEDIUM |
| `EditorPage.css` | 355 | MEDIUM |
| `HistoryPage.css` | 377 | MEDIUM |
| `NotFoundPage.css` | 71 | LOW |
| `SettingsPage.css` | 177 | LOW |

### Hard-to-Migrate Patterns
1. **8 keyframe animations** — define in `tailwind.config.ts` `theme.extend.keyframes`
2. **ReactFlow CSS overrides** (`.react-flow__node`, etc.) — MUST remain as plain CSS
3. **`::before` pseudo-elements** with custom content — keep as component CSS
4. **Staggered animation delays** (`:nth-child`) — use inline styles

### Strategy
Phase 1: Install Tailwind + theme config (keep existing CSS)
Phase 2: Convert simple pages first (NotFound, Settings)
Phase 3: Convert complex components last (WorkflowCanvas, Nodes)
Phase 4: Extract ReactFlow overrides + animations into dedicated CSS files
Phase 5: Delete original CSS files one by one

---

## 5. main.py Decomposition (572 -> ~40 lines)

### Target Structure
```
apps/orchestrator/
  main.py              -> Slim entry point (~40 lines)
  core/
    config.py          -> CORS, env loading, logging (lines 1-27, 42-46, 67-87)
    schemas.py         -> Pydantic models (lines 89-130)
  routes/
    flows.py           -> Flow CRUD (lines 502-534)
    execution.py       -> Run endpoints (lines 536-556)
    applets.py         -> Applet listing (lines 474-500)
    ai.py              -> AI suggest stub (lines 558-566)
  services/
    orchestrator.py    -> Orchestrator class + BaseApplet (lines 182-468)
    websocket.py       -> WebSocket manager (lines 131-180)
```

---

## 6. Test Coverage Assessment

### Current: ~15% backend, ~10% frontend

**Backend (6 tests):** Only basic CRUD endpoints. Zero coverage on:
- Orchestrator execution engine (262 lines!)
- All 3 applets
- WebSocket broadcasting
- Repositories
- Error paths

**Frontend (14 tests):** Only AppletNode + HistoryPage. Zero coverage on:
- EditorPage (the main screen)
- DashboardPage
- WorkflowCanvas
- ApiService (9 methods)
- WebSocketService
- All utilities

### Priority Test Plan

**Backend Phase 1 (Critical):**
1. `test_orchestrator.py` — execute_flow with mocked applets
2. `test_repositories.py` — all CRUD operations
3. `test_applets.py` — each applet with mocked APIs
4. `test_websocket.py` — broadcast with mock clients

**Frontend Phase 1 (Critical):**
1. `ApiService.test.ts` — all 9 API methods
2. `WebSocketService.test.ts` — connect, reconnect, dispatch
3. `flowUtils.test.ts` — all 4 utilities
4. `EditorPage.test.tsx` — render, save, run
