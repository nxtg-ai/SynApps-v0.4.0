# T-040: Backend Stack Upgrade Plan

**Author:** Claude (automated design task)
**Date:** 2026-02-17
**Status:** Complete

---

## 1. Executive Summary

This document plans the migration of the SynApps orchestrator backend from its current dependency floor to modern versions. The upgrade targets:

| Dependency | Current Floor | Target |
|---|---|---|
| Python | 3.9 | 3.11+ |
| FastAPI | 0.68.0 | 0.115+ |
| Pydantic | 1.8.2 (mixed v1/v2 usage) | 2.x (pure v2) |
| SQLAlchemy | 1.4.0 | 2.0+ |
| Alembic | 1.7.0 | 1.13+ (latest) |
| uvicorn | 0.15.0 | 0.30+ |

The codebase is small (~1,200 lines of application code across 10 files) making this a low-risk, high-value upgrade. Most breaking changes are mechanical find-and-replace operations.

---

## 2. Current State Analysis

### 2.1 Dependency Versions (requirements.txt)

```
fastapi>=0.68.0
uvicorn>=0.15.0
pydantic>=1.8.2
websockets>=10.0
python-dotenv>=0.19.0
httpx>=0.23.0
sqlalchemy>=1.4.0
alembic>=1.7.0
psycopg2-binary>=2.9.0
python-multipart>=0.0.5
aiosqlite>=0.19.0
```

### 2.2 Files in Scope

| File | Lines | Role | Impact |
|---|---|---|---|
| `apps/orchestrator/main.py` | 831 | FastAPI app, routes, Pydantic models, WebSocket | HIGH |
| `apps/orchestrator/models.py` | 167 | SQLAlchemy ORM + Pydantic API models | HIGH |
| `apps/orchestrator/db.py` | 81 | Database engine, session factory | HIGH |
| `apps/orchestrator/repositories.py` | 142 | Repository pattern over SQLAlchemy | MEDIUM |
| `apps/orchestrator/migrations/env.py` | 105 | Alembic migration runner | HIGH |
| `apps/orchestrator/middleware/billing_guard.py` | 160 | Starlette middleware | LOW |
| `apps/orchestrator/migrate_db.py` | 55 | Ad-hoc SQLite migration script | LOW (delete candidate) |
| `apps/applets/writer/applet.py` | 124 | Writer applet (uses Pydantic via BaseApplet) | LOW |
| `apps/applets/artist/applet.py` | 202 | Artist applet (uses Pydantic via BaseApplet) | LOW |
| `apps/applets/memory/applet.py` | 240 | Memory applet (uses Pydantic via BaseApplet) | LOW |
| `apps/orchestrator/setup.py` | 28 | Package metadata | MEDIUM |
| `infra/docker/Dockerfile.orchestrator` | 25 | Docker build | MEDIUM |
| `apps/orchestrator/alembic.ini` | 85 | Alembic config | LOW |
| `apps/orchestrator/tests/*.py` | ~350 | Test suite | MEDIUM |

---

## 3. Migration Plan by Dependency

### 3.1 Python 3.9 → 3.11+

**Risk: LOW** — No breaking changes, purely additive.

#### 3.1.1 Changes Required

**`infra/docker/Dockerfile.orchestrator:2`**
```dockerfile
# FROM python:3.9-slim as orchestrator
FROM python:3.11-slim as orchestrator
```

**`apps/orchestrator/setup.py:22-26`** — Update classifiers:
```python
classifiers=[
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
],
```

#### 3.1.2 Optional Modernizations (not blocking)

These are nice-to-have cleanups for a follow-up PR, not required for the upgrade:

- Replace `typing.Dict`, `typing.List`, `typing.Optional` with built-in `dict`, `list`, `X | None` (PEP 604/585). Affects `main.py`, `models.py`, `repositories.py`, `db.py`, all applets.
- Use `tomllib` (stdlib in 3.11) if TOML config is ever needed.
- Use `ExceptionGroup` / `except*` for parallel applet error aggregation (future T-023 execution engine work).

#### 3.1.3 Compatibility Notes

- The venv currently uses Python 3.13 (`venv/lib/python3.13/`), so the runtime already exceeds 3.11. The pinned floor in `requirements.txt` and `setup.py` is what needs updating.
- `tuple[str, str]` in `artist/applet.py:81` already uses PEP 585 syntax — works fine on 3.9+.

---

### 3.2 Pydantic 1.x → 2.x (pure v2)

**Risk: MEDIUM** — The codebase already uses some v2 patterns but has leftover v1 compatibility code.

#### 3.2.1 Already v2-Compatible (no changes needed)

The following patterns in `main.py` are already v2-native:
- `field_validator` with `@classmethod` decorator (lines 184, 207, 213)
- `Field(...)` with `min_length`, `max_length` constraints
- `model_dump()` call at line 643
- `BaseModel` imports from `pydantic`

#### 3.2.2 Breaking Changes to Fix

**A. Remove `model_to_dict()` v1/v2 compatibility shim**

`main.py:345-349`:
```python
def model_to_dict(model):
    """Convert a Pydantic model to a dictionary, handling both v1 and v2 Pydantic."""
    if isinstance(model, dict):
        return model
    return model.model_dump() if hasattr(model, 'model_dump') else model.dict()
```

**Action:** Replace the function body with just `model.model_dump()`. Better yet, since `model_to_dict` is only called at `main.py:398` where `status` is already a plain `dict`, this function can be simplified or calls can be inlined. Audit all call sites:

- `main.py:398`: `status_dict = model_to_dict(status)` — `status` is a plain `dict` here, so `model_to_dict` is a no-op. Remove the call entirely and use `status` directly.

**B. Remove `.dict()` override in WorkflowRunStatusModel**

`models.py:160-166`:
```python
def dict(self, *args, **kwargs) -> Dict[str, Any]:
    """Override dict method to provide a consistent output."""
    result = super().dict(*args, **kwargs)
    if result.get("results") is None:
        result["results"] = {}
    return result
```

**Action:** In Pydantic v2, `.dict()` is deprecated in favor of `.model_dump()`. The `results` field already has `default_factory=dict`, so `None` should never occur. Remove this override entirely. If the safeguard is needed, use a `@field_validator` or `@model_validator` instead:

```python
@field_validator("results", mode="before")
@classmethod
def ensure_results_not_none(cls, v):
    return v if v is not None else {}
```

**C. Verify `BaseModel` usage in all Pydantic models**

All Pydantic models in `main.py` and `models.py` already use v2-compatible patterns. No `validator` (v1) or `Config` inner class usage detected. No `.schema()` calls. This is clean.

#### 3.2.3 Requirements Update

```
pydantic>=2.0.0
```

---

### 3.3 SQLAlchemy 1.4 → 2.0

**Risk: HIGH** — This is the largest migration. SA 2.0 deprecates the `declarative_base()` function and the `Column()` pattern in favor of `DeclarativeBase` class and `Mapped`/`mapped_column()`.

#### 3.3.1 Breaking Changes

**A. Replace `declarative_base()` with `DeclarativeBase`**

`models.py:10,15`:
```python
# BEFORE (deprecated)
from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()

# AFTER (SA 2.0)
from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    pass
```

**Impact:** Every file that imports `Base` (`db.py:13`, `migrations/env.py:16`) will get the new class automatically since the import path (`from apps.orchestrator.models import Base`) stays the same.

**B. Replace `Column()` with `mapped_column()` and `Mapped[]`**

`models.py` — all four ORM models (`Flow`, `FlowNode`, `FlowEdge`, `WorkflowRun`):

```python
# BEFORE (SA 1.4 pattern)
from sqlalchemy import Column, String, Integer, Float, Boolean, ForeignKey, JSON
from sqlalchemy.orm import relationship

class Flow(Base):
    __tablename__ = "flows"
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    nodes = relationship("FlowNode", back_populates="flow", cascade="all, delete-orphan")

# AFTER (SA 2.0 pattern)
from typing import Optional, List
from sqlalchemy import String, Integer, Float, Boolean, ForeignKey, JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Flow(Base):
    __tablename__ = "flows"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String)
    nodes: Mapped[List["FlowNode"]] = relationship(back_populates="flow", cascade="all, delete-orphan")
```

**Full model rewrites needed for:**
- `Flow` (models.py:18-36) — 2 columns, 2 relationships
- `FlowNode` (models.py:39-63) — 5 columns, 1 relationship
- `FlowEdge` (models.py:66-86) — 4 columns, 1 relationship
- `WorkflowRun` (models.py:89-119) — 10 columns, 0 relationships

**C. Replace `sqlalchemy.future.select` with `sqlalchemy.select`**

`repositories.py:13`:
```python
# BEFORE
from sqlalchemy.future import select

# AFTER
from sqlalchemy import select
```

The `sqlalchemy.future` module was a forward-compatibility bridge in SA 1.4. In SA 2.0, `select` is in the main namespace. All existing `select()` usage patterns (in `repositories.py`) are already SA 2.0 compatible in syntax.

**D. Replace `sessionmaker` with `async_sessionmaker`**

`db.py:43-49`:
```python
# BEFORE
from sqlalchemy.orm import sessionmaker
async_session = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# AFTER
from sqlalchemy.ext.asyncio import async_sessionmaker
async_session = async_sessionmaker(
    engine,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)
```

Note: `async_sessionmaker` already returns `AsyncSession` instances, so `class_=AsyncSession` is no longer needed.

**E. Update `AsyncEngine` construction in Alembic env.py**

`migrations/env.py:86-92`:
```python
# BEFORE
connectable = AsyncEngine(
    engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
)

# AFTER
from sqlalchemy.ext.asyncio import create_async_engine
connectable = create_async_engine(
    config.get_main_option("sqlalchemy.url"),
    poolclass=pool.NullPool,
)
```

The `AsyncEngine(sync_engine)` wrapping pattern is fragile and deprecated. Use `create_async_engine` directly.

#### 3.3.2 Database Compatibility

- **Schema unchanged.** The `Mapped`/`mapped_column` migration is purely Python-side. The generated SQL DDL is identical. No new Alembic migration is needed for this upgrade.
- **Existing migrations remain valid.** Migration files use `sa.Column()` which still works in SA 2.0 for DDL operations.

#### 3.3.3 Requirements Update

```
sqlalchemy>=2.0.0
```

---

### 3.4 FastAPI 0.68 → 0.115+

**Risk: LOW** — The codebase already uses modern FastAPI patterns.

#### 3.4.1 Already Compatible (no changes needed)

- `lifespan` async context manager (main.py:62-71) — modern pattern, introduced in FastAPI 0.93.
- `APIRouter` with prefix and tags (main.py:603).
- `Query()` with constraints (main.py:608-609).
- `RequestValidationError` exception handler (main.py:135).
- `WebSocket` and `WebSocketDisconnect` handling (main.py:738-824).

#### 3.4.2 Changes to Consider

**A. Starlette `BaseHTTPMiddleware` deprecation warning**

`middleware/billing_guard.py:12`:
```python
from starlette.middleware.base import BaseHTTPMiddleware
```

Starlette has deprecated `BaseHTTPMiddleware` in favor of pure ASGI middleware. However, FastAPI still supports it and it won't break. This is a non-blocking cleanup for a future PR.

**B. Pydantic v2 integration**

FastAPI 0.100+ natively supports Pydantic v2. With the Pydantic v2 migration above, FastAPI will automatically use v2 serialization. No additional changes needed.

#### 3.4.3 Requirements Update

```
fastapi>=0.115.0
uvicorn>=0.30.0
```

---

### 3.5 Alembic 1.7 → 1.13+ (latest)

**Risk: LOW** — Alembic has been very stable.

#### 3.5.1 Changes Required

**A. Update `env.py` for SA 2.0 compatibility** (covered in 3.3.1E above)

**B. Update `alembic.ini` sqlalchemy.url** — Currently hardcoded:
```ini
sqlalchemy.url = sqlite+aiosqlite:///synapps.db
```
This is overridden at runtime in `env.py:24`, so no change needed. The override pattern is correct.

**C. Consider removing `migrate_db.py`**

`apps/orchestrator/migrate_db.py` is an ad-hoc SQLite migration script that adds a `completed_applets` column using raw `sqlite3`. This column isn't in the ORM models and isn't used by the application code. This file should be deleted — all migrations should go through Alembic.

#### 3.5.2 Requirements Update

```
alembic>=1.13.0
```

---

## 4. Deprecated Patterns to Remove

| Pattern | Location | Replacement |
|---|---|---|
| `declarative_base()` | `models.py:10,15` | `class Base(DeclarativeBase): pass` |
| `Column()` ORM declarations | `models.py:22-103` | `Mapped[T]` + `mapped_column()` |
| `from sqlalchemy.future import select` | `repositories.py:13` | `from sqlalchemy import select` |
| `sessionmaker(..., class_=AsyncSession)` | `db.py:9,43-49` | `async_sessionmaker(...)` |
| `AsyncEngine(engine_from_config(...))` | `migrations/env.py:86-92` | `create_async_engine(url)` |
| `model_to_dict()` v1/v2 shim | `main.py:345-349` | Remove function, inline dict usage |
| `.dict()` override | `models.py:160-166` | Remove (use `@field_validator` if needed) |
| `from sqlalchemy.ext.declarative import declarative_base` | `models.py:10` | `from sqlalchemy.orm import DeclarativeBase` |
| Ad-hoc SQLite migration script | `migrate_db.py` (entire file) | Delete — use Alembic exclusively |
| Python 3.8/3.9 classifiers | `setup.py:24-25` | Update to 3.11/3.12 |
| `python:3.9-slim` Docker base | `Dockerfile.orchestrator:2` | `python:3.11-slim` |

---

## 5. Migration Execution Order

The upgrade must be done in a specific order due to inter-dependency:

### Phase 1: Python Version (no code changes required)
1. Update `Dockerfile.orchestrator` base image to `python:3.11-slim`
2. Update `setup.py` classifiers
3. Verify all tests pass on Python 3.11

### Phase 2: SQLAlchemy 2.0 (largest change, do first)
1. Pin `sqlalchemy>=2.0.0` in requirements.txt
2. Rewrite `models.py`:
   - Replace `declarative_base()` with `DeclarativeBase` class
   - Rewrite all 4 ORM models to use `Mapped[]` + `mapped_column()`
   - Keep `to_dict()` methods unchanged
3. Update `db.py`:
   - Replace `sessionmaker` with `async_sessionmaker`
   - Remove `class_=AsyncSession` parameter
4. Update `repositories.py`:
   - Change `from sqlalchemy.future import select` to `from sqlalchemy import select`
5. Update `migrations/env.py`:
   - Replace `AsyncEngine(engine_from_config(...))` with `create_async_engine()`
   - Update `from models import Base` import path if needed
6. Run existing test suite — schema is unchanged, all tests should pass
7. Run `alembic check` to verify no schema drift

### Phase 3: Pydantic v2 (cleanup)
1. Pin `pydantic>=2.0.0` in requirements.txt
2. Remove `model_to_dict()` function from `main.py`
3. Remove `.dict()` override from `WorkflowRunStatusModel` in `models.py`
4. (Optional) Add `@field_validator` for `results` if None safeguard is needed
5. Run test suite

### Phase 4: FastAPI + Uvicorn
1. Pin `fastapi>=0.115.0` and `uvicorn>=0.30.0` in requirements.txt
2. No code changes required — existing patterns are compatible
3. Run full test suite including WebSocket tests

### Phase 5: Alembic + Cleanup
1. Pin `alembic>=1.13.0` in requirements.txt
2. Delete `migrate_db.py`
3. Run `alembic upgrade head` and `alembic check` to verify
4. Update all version pins in `setup.py` `install_requires`

### Phase 6: Final Validation
1. Run full test suite: `pytest apps/orchestrator/tests/ -v`
2. Start server locally and test all REST endpoints
3. Test WebSocket connection and workflow execution
4. Build Docker image and verify container starts
5. Run Alembic migration from scratch on a fresh database

---

## 6. Updated requirements.txt (Target State)

```
fastapi>=0.115.0
uvicorn>=0.30.0
pydantic>=2.0.0
websockets>=10.0
python-dotenv>=0.19.0
httpx>=0.23.0
sqlalchemy>=2.0.0
alembic>=1.13.0
psycopg2-binary>=2.9.0
python-multipart>=0.0.5
aiosqlite>=0.19.0
```

---

## 7. Updated setup.py install_requires (Target State)

```python
install_requires=[
    "fastapi>=0.115.0",
    "uvicorn>=0.30.0",
    "pydantic>=2.0.0",
    "websockets>=10.0",
    "python-dotenv>=0.19.0",
    "httpx>=0.23.0",
    "sqlalchemy>=2.0.0",
    "alembic>=1.13.0",
],
```

---

## 8. Rewritten models.py (Target State Reference)

This is the full target state for `models.py` after the SA 2.0 + Pydantic v2 migration:

```python
"""
Database and API models for the SynApps orchestrator.
"""
from typing import Any, Optional
import time
from sqlalchemy import String, Integer, Float, Boolean, ForeignKey, JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from pydantic import BaseModel, Field, field_validator


class Base(DeclarativeBase):
    pass


class Flow(Base):
    __tablename__ = "flows"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String)

    nodes: Mapped[list["FlowNode"]] = relationship(
        back_populates="flow", cascade="all, delete-orphan"
    )
    edges: Mapped[list["FlowEdge"]] = relationship(
        back_populates="flow", cascade="all, delete-orphan"
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
        }


class FlowNode(Base):
    __tablename__ = "flow_nodes"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    flow_id: Mapped[str] = mapped_column(
        String, ForeignKey("flows.id", ondelete="CASCADE")
    )
    type: Mapped[str] = mapped_column(String)
    position_x: Mapped[float] = mapped_column(Float)
    position_y: Mapped[float] = mapped_column(Float)
    data: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    flow: Mapped["Flow"] = relationship(back_populates="nodes")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "position": {"x": self.position_x, "y": self.position_y},
            "data": self.data or {},
        }


class FlowEdge(Base):
    __tablename__ = "flow_edges"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    flow_id: Mapped[str] = mapped_column(
        String, ForeignKey("flows.id", ondelete="CASCADE")
    )
    source: Mapped[str] = mapped_column(String)
    target: Mapped[str] = mapped_column(String)
    animated: Mapped[bool] = mapped_column(Boolean, default=False)

    flow: Mapped["Flow"] = relationship(back_populates="edges")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "target": self.target,
            "animated": self.animated,
        }


class WorkflowRun(Base):
    __tablename__ = "workflow_runs"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    flow_id: Mapped[Optional[str]] = mapped_column(
        String, ForeignKey("flows.id", ondelete="SET NULL"), nullable=True
    )
    status: Mapped[str] = mapped_column(String, default="idle")
    current_applet: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    progress: Mapped[int] = mapped_column(Integer, default=0)
    total_steps: Mapped[int] = mapped_column(Integer, default=0)
    start_time: Mapped[float] = mapped_column(Float)
    end_time: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    results: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    input_data: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.id,
            "flow_id": self.flow_id,
            "status": self.status,
            "current_applet": self.current_applet,
            "progress": self.progress,
            "total_steps": self.total_steps,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "results": self.results or {},
            "error": self.error,
            "input_data": self.input_data,
        }


# Pydantic API Models (unchanged, already v2-compatible)
class FlowNodeModel(BaseModel):
    id: str
    type: str
    position: dict[str, float]
    data: dict[str, Any] = Field(default_factory=dict)


class FlowEdgeModel(BaseModel):
    id: str
    source: str
    target: str
    animated: bool = False


class FlowModel(BaseModel):
    id: str | None = None
    name: str
    nodes: list[FlowNodeModel] = Field(default_factory=list)
    edges: list[FlowEdgeModel] = Field(default_factory=list)


class WorkflowRunStatusModel(BaseModel):
    run_id: str
    flow_id: str
    status: str = "idle"
    current_applet: str | None = None
    progress: int = 0
    total_steps: int = 0
    start_time: float = Field(default_factory=lambda: time.time())
    end_time: float | None = None
    results: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
```

---

## 9. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| SA 2.0 `Mapped[]` type causes Alembic autogenerate to detect false changes | Medium | Low | Run `alembic check` after migration; suppress with `compare_type=False` temporarily |
| Test database isolation breaks with `async_sessionmaker` | Low | Medium | Tests already use per-function fixtures with tempfile DBs |
| Pydantic v2 serialization differences in API responses | Low | Medium | Run integration tests against known API response shapes |
| Docker build fails with new Python version | Low | Low | Pin exact Python minor (3.11) not just major |
| `BaseHTTPMiddleware` deprecation in billing_guard | Low | Low | Non-blocking; move to pure ASGI middleware in future PR |

---

## 10. Files NOT Requiring Changes

These files are already compatible and need no modifications:
- `apps/applets/writer/applet.py` — uses `BaseApplet` and `AppletMessage` only
- `apps/applets/artist/applet.py` — same
- `apps/applets/memory/applet.py` — same
- `apps/orchestrator/middleware/__init__.py` — empty init
- `apps/orchestrator/tests/__init__.py` — empty init
- `apps/orchestrator/alembic.ini` — config file, no code
- Existing Alembic migration version files — they use `sa.Column()` DDL which still works in SA 2.0

---

## 11. Test Plan

After each phase:

1. **Unit tests:** `pytest apps/orchestrator/tests/ -v`
2. **Import check:** `python -c "from apps.orchestrator.main import app; print('OK')"`
3. **Alembic check:** `cd apps/orchestrator && alembic check`
4. **Fresh DB test:** Delete `synapps.db`, run `alembic upgrade head`, verify tables created
5. **Docker build:** `docker build -f infra/docker/Dockerfile.orchestrator .`
6. **Smoke test:** Start server, hit `GET /`, verify `{"status": "healthy"}`
