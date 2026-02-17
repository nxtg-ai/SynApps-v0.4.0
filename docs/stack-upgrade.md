# Stack Upgrade and Migration Guide

This document outlines the stack upgrade for the SynApps orchestrator backend and provides migration steps and breaking changes for contributors.

## 1. Upgrade Overview

The SynApps backend is being upgraded to modern versions of its core dependencies to improve performance, type safety, and maintainability.

### 1.1 Target Versions

| Dependency | Current Floor | Target |
|---|---|---|
| Python | 3.9 | 3.11+ |
| FastAPI | 0.68.0 | 0.115+ |
| Pydantic | 1.8.2 | 2.x (pure v2) |
| SQLAlchemy | 1.4.0 | 2.0+ |
| Alembic | 1.7.0 | 1.13+ |
| uvicorn | 0.15.0 | 0.30+ |

## 2. Migration Phases

The upgrade is executed in phases to ensure stability and allow for incremental testing.

### Phase 1: Python Version Update
- Update `Dockerfile.orchestrator` base image to `python:3.11-slim`.
- Update `setup.py` classifiers to include Python 3.11 and 3.12.
- Ensure all tests pass on Python 3.11+.

### Phase 2: SQLAlchemy 2.0 Migration
- Update `requirements.txt` to `sqlalchemy>=2.0.0`.
- **Breaking Change:** Replace `declarative_base()` with `DeclarativeBase` class in `models.py`.
- **Breaking Change:** Migrate all ORM models to use `Mapped[T]` and `mapped_column()` syntax.
- Update `db.py` to use `async_sessionmaker`.
- Update `repositories.py` to use `select` from `sqlalchemy` instead of `sqlalchemy.future`.
- Update `migrations/env.py` to use `create_async_engine`.

### Phase 3: Pydantic v2 Migration
- Update `requirements.txt` to `pydantic>=2.0.0`.
- Remove `model_to_dict()` compatibility shims.
- Replace `.dict()` calls with `.model_dump()`.
- Update validators to use `@field_validator` and `@model_validator`.

### Phase 4: FastAPI & Uvicorn Update
- Update `requirements.txt` to `fastapi>=0.115.0` and `uvicorn>=0.30.0`.
- Verify Pydantic v2 integration with FastAPI.

### Phase 5: Alembic & Cleanup
- Update `requirements.txt` to `alembic>=1.13.0`.
- Remove legacy migration scripts (e.g., `migrate_db.py`).
- Verify database migrations and schema consistency.

## 3. Breaking Changes for Contributors

### 3.1 SQLAlchemy 2.0 Syntax

Contributors must use the new 2.0 style for all ORM models and queries.

**Old Style (1.4):**
```python
from sqlalchemy import Column, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True)
```

**New Style (2.0):**
```python
from sqlalchemy import String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String, primary_key=True)
```

### 3.2 Pydantic v2 Changes

- Use `.model_dump()` instead of `.dict()`.
- Use `.model_dump_json()` instead of `.json()`.
- Use `@field_validator` instead of `@validator`.
- Model configuration is now done via `model_config` attribute instead of an inner `Config` class.

### 3.3 Database Sessions

Always use `async_sessionmaker` for creating asynchronous sessions.

```python
from sqlalchemy.ext.asyncio import async_sessionmaker
# session_factory is an instance of async_sessionmaker
async with session_factory() as session:
    # Use session
    ...
```

## 4. Migration Steps for Local Development

To update your local development environment:

1. **Update Python:** Ensure you are using Python 3.11 or later.
2. **Refresh Virtual Environment:**
   ```bash
   rm -rf .venv
   python3.11 -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```
3. **Run Migrations:**
   ```bash
   cd apps/orchestrator
   alembic upgrade head
   ```
4. **Verify Installation:**
   ```bash
   pytest
   ```

## 5. Verification Checklist

Before submitting changes related to the stack upgrade, ensure:
- [ ] `pytest` passes with 0 failures.
- [ ] `alembic check` reports no schema drift.
- [ ] `ruff check .` passes (or your preferred linter).
- [ ] The application starts successfully and `GET /` returns a healthy status.
- [ ] WebSocket connections are functional.

## 6. FAQ

**Q: Do I need to rewrite existing migrations?**
A: No. Alembic migrations using `sa.Column` remain compatible with SQLAlchemy 2.0. Only the application-level ORM models need to be updated.

**Q: Can I still use Python 3.9?**
A: No. The project now officially requires Python 3.11+ to take advantage of modern features and better performance.

**Q: What happened to `migrate_db.py`?**
A: It has been removed. All database schema changes must now be managed exclusively through Alembic migrations.
