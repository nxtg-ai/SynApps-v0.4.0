"""
Database migration script to add the completed_applets column to workflow_runs.

This script is idempotent and uses SQLAlchemy 2.0 async engine patterns.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import create_async_engine


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("db_migration")

DATABASE_PATH = os.environ.get("DATABASE_PATH", "synapps.db")
DATABASE_URL = os.environ.get("DATABASE_URL")


def _resolve_database_url() -> str:
    """Resolve database URL, honoring DATABASE_URL then DATABASE_PATH."""
    runtime_database_path = os.environ.get("DATABASE_PATH", DATABASE_PATH)
    runtime_database_url = os.environ.get("DATABASE_URL", DATABASE_URL)

    # Preserve backward compatibility for scripts/tests that override DATABASE_PATH.
    if runtime_database_path and runtime_database_path != "synapps.db":
        return f"sqlite+aiosqlite:///{runtime_database_path}"

    if runtime_database_url:
        return runtime_database_url

    return f"sqlite+aiosqlite:///{runtime_database_path}"


def _has_column(sync_connection: Any, table_name: str, column_name: str) -> bool:
    """Check whether a column exists using SQLAlchemy inspector."""
    inspector = inspect(sync_connection)
    try:
        columns = inspector.get_columns(table_name)
    except Exception:
        return False
    return any(column.get("name") == column_name for column in columns)


async def add_completed_applets_column() -> bool:
    """Add the completed_applets column to workflow_runs if missing."""
    engine = create_async_engine(_resolve_database_url(), pool_pre_ping=True)
    try:
        async with engine.begin() as connection:
            has_column = await connection.run_sync(
                lambda sync_connection: _has_column(
                    sync_connection,
                    "workflow_runs",
                    "completed_applets",
                )
            )

            if has_column:
                logger.info("Column completed_applets already exists")
                return True

            logger.info("Adding completed_applets column to workflow_runs table...")
            await connection.execute(
                text("ALTER TABLE workflow_runs ADD COLUMN completed_applets JSON")
            )
            logger.info("Column added successfully")
            return True
    except Exception as exc:
        logger.error("Migration failed: %s", exc)
        return False
    finally:
        await engine.dispose()


async def main() -> None:
    """Run the migration."""
    logger.info("Starting database migration...")
    success = await add_completed_applets_column()
    if success:
        logger.info("Migration completed successfully")
    else:
        logger.error("Migration failed")


if __name__ == "__main__":
    asyncio.run(main())
