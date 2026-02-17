import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from sqlalchemy.exc import SQLAlchemyError
from apps.orchestrator.db import get_db_session, init_db, Base, async_session, DATABASE_URL, close_db_connections, engine
from sqlalchemy.ext.asyncio import create_async_engine as original_create_async_engine
import os
import importlib
import apps.orchestrator.db as db_module_to_patch



@pytest.mark.asyncio
async def test_get_db_session_rollback():
    """Test that get_db_session rolls back on exception."""
    mock_session = AsyncMock()
    mock_session.commit.side_effect = Exception("Test exception")
    
    with patch('apps.orchestrator.db.async_session', return_value=mock_session):
        with pytest.raises(Exception, match="Test exception"):
            async with get_db_session() as session:
                assert session == mock_session
        
        mock_session.rollback.assert_called_once()
        mock_session.close.assert_called_once()

@pytest.mark.asyncio
async def test_init_db_sqlalchemy_error():
    """Test init_db handles SQLAlchemyError during schema creation."""
    mock_connection = AsyncMock()
    mock_connection.run_sync.side_effect = SQLAlchemyError("DB init failed")

    mock_engine_begin = AsyncMock() # This will be the awaitable returned by engine.begin()
    mock_engine_begin.__aenter__.return_value = mock_connection
    mock_engine_begin.__aexit__.return_value = False

    mock_engine = MagicMock()
    mock_engine.begin.return_value = mock_engine_begin # engine.begin() returns the awaitable

    with patch('apps.orchestrator.db.engine', new=mock_engine):
        with pytest.raises(SQLAlchemyError, match="DB init failed"):
            await init_db()
        mock_engine.begin.assert_called_once()
        mock_connection.run_sync.assert_called_once_with(Base.metadata.create_all)


@pytest.mark.asyncio
async def test_init_db_success():
    """Test init_db successfully initializes the schema."""
    mock_connection = AsyncMock()
    mock_connection.run_sync.return_value = None 

    mock_engine_begin = AsyncMock()
    mock_engine_begin.__aenter__.return_value = mock_connection
    mock_engine_begin.__aexit__.return_value = False

    mock_engine = MagicMock()
    mock_engine.begin.return_value = mock_engine_begin

    with patch('apps.orchestrator.db.engine', new=mock_engine):
        await init_db()
        mock_engine.begin.assert_called_once()
        mock_connection.run_sync.assert_called_once_with(Base.metadata.create_all)

@pytest.mark.asyncio
async def test_postgres_engine_creation(monkeypatch):
    """Test that the PostgreSQL engine branch is taken when DATABASE_URL starts with postgres."""
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://user:password@host:5432/dbname")
    
    # Reload the db module to pick up the new environment variable
    importlib.reload(db_module_to_patch)
    
    assert db_module_to_patch.DATABASE_URL.startswith("postgresql")
    assert db_module_to_patch.engine.sync_engine.url.port == 5432
    # Check for parameters passed to create_async_engine, not on the engine object itself
    # This is a bit tricky since create_async_engine is called at module level.
    # The best way to test the arguments would be to patch create_async_engine during the reload.
    
    # Simpler assertion for now: check the dialect name
    assert db_module_to_patch.engine.dialect.name == "postgresql"

    # Clean up environment variable (important for other tests)
    monkeypatch.undo() 
    # Reload again to reset to original DATABASE_URL and engine for subsequent tests
    importlib.reload(db_module_to_patch)

@pytest.mark.asyncio
async def test_close_db_connections_success():
    """Test that close_db_connections disposes the engine."""
    mock_engine = AsyncMock()
    with patch('apps.orchestrator.db.engine', new=mock_engine):
        await close_db_connections()
        mock_engine.dispose.assert_called_once()
