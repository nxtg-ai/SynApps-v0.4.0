import pytest
import os
import importlib
import apps.orchestrator.db as db_module
from apps.orchestrator.middleware.rate_limiter import _SlidingWindowCounter


@pytest.fixture(autouse=True)
def setup_default_db_env(monkeypatch):
    """
    Ensures that the default DATABASE_URL for tests is always in-memory SQLite.
    This runs before any other fixture, setting the environment variable.
    """
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
    # Reload db module so the engine picks up the in-memory URL
    importlib.reload(db_module)


@pytest.fixture(autouse=True)
def _reset_rate_limit_counter(monkeypatch):
    """Reset the rate limiter sliding window counter between tests.

    Without this, tests sharing the same anonymous IP key accumulate
    requests across the suite and hit 429 after 30 total requests.
    """
    monkeypatch.setattr(
        "apps.orchestrator.middleware.rate_limiter._counter",
        _SlidingWindowCounter(),
    )
