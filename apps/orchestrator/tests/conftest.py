import pytest
import os
import importlib
import apps.orchestrator.db as db_module


@pytest.fixture(autouse=True)
def setup_default_db_env(monkeypatch):
    """
    Ensures that the default DATABASE_URL for tests is always in-memory SQLite.
    This runs before any other fixture, setting the environment variable.
    """
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
    # Reload db module so the engine picks up the in-memory URL
    importlib.reload(db_module)
