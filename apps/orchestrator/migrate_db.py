"""
Database migration script to add the completed_applets column to workflow_runs table.

This script should be run once to update the database schema.
"""
import asyncio
import logging
import sqlite3
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("db_migration")

# Get database path from environment or use default
DATABASE_PATH = os.environ.get("DATABASE_PATH", "synapps.db")

async def add_completed_applets_column():
    """Add the completed_applets column to the workflow_runs table."""
    try:
        # Connect to SQLite database
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        # Check if the column already exists
        cursor.execute("PRAGMA table_info(workflow_runs)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if "completed_applets" not in columns:
            logger.info("Adding completed_applets column to workflow_runs table...")
            # Add the column (SQLite doesn't support adding JSON columns directly, so we use TEXT)
            cursor.execute("ALTER TABLE workflow_runs ADD COLUMN completed_applets TEXT")
            conn.commit()
            logger.info("Column added successfully")
        else:
            logger.info("Column completed_applets already exists")
        
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return False

async def main():
    """Run the migration."""
    logger.info("Starting database migration...")
    success = await add_completed_applets_column()
    if success:
        logger.info("Migration completed successfully")
    else:
        logger.error("Migration failed")

if __name__ == "__main__":
    asyncio.run(main())
