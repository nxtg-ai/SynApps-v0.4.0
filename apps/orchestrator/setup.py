"""
Setup script for SynApps Orchestrator
"""
from setuptools import setup, find_packages

setup(
    name="synapps-orchestrator",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.115.0",
        "uvicorn>=0.30.0",
        "pydantic>=2.8.0",
        "sqlalchemy>=2.0.30",
        "alembic>=1.16.0",
        "websockets>=12.0",
        "python-dotenv>=1.0.1",
        "httpx>=0.27.0",
        "python-multipart>=0.0.9",
        "aiosqlite>=0.20.0",
        "asyncpg>=0.30.0",
        "psycopg2-binary>=2.9.9",
    ],
    description="SynApps Orchestrator - Lightweight message routing for AI applets",
    author="SynApps Team",
    author_email="synapps.info@nxtg.ai",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
