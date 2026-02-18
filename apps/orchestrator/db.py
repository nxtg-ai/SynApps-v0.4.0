"""
Database connection module.

This module provides utilities for database connections and session management.
"""
import os
import logging
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.exc import SQLAlchemyError

from apps.orchestrator.models import Base

# Configure logging
logger = logging.getLogger("db")

# Get database URL from environment or use SQLite as default
DATABASE_URL = os.environ.get(
    "DATABASE_URL", 
    "sqlite+aiosqlite:///synapps.db"
)

# Create async engine
if DATABASE_URL.startswith("sqlite"):
    # SQLite-specific options
    engine = create_async_engine(
        DATABASE_URL, 
        connect_args={"check_same_thread": False},
        pool_pre_ping=True
    )
else:
    # PostgreSQL or other database options
    engine = create_async_engine(
        DATABASE_URL,
        pool_size=10,
        max_overflow=20,
        pool_recycle=3600,
        pool_pre_ping=True
    )

# Create async session factory
async_session = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)

@asynccontextmanager
async def get_db_session():
    """Async context manager for database sessions."""
    session = async_session()
    try:
        yield session
        await session.commit()
    except Exception as e:
        await session.rollback()
        logger.error(f"Database session error: {e}")
        raise
    finally:
        await session.close()


async def init_db():
    """Initialize database schema."""
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database schema initialized successfully")
    except SQLAlchemyError as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


async def close_db_connections():
    """Close all database connections in the pool."""
    await engine.dispose()
    logger.info("All database connections closed")
