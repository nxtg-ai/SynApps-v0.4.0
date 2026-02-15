"""
Setup script for SynApps Orchestrator
"""
from setuptools import setup, find_packages

setup(
    name="synapps-orchestrator",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "pydantic>=1.8.2",
        "websockets>=10.0",
        "python-dotenv>=0.19.0",
        "httpx>=0.23.0"
    ],
    description="SynApps Orchestrator - Lightweight message routing for AI applets",
    author="SynApps Team",
    author_email="synapps.info@nxtg.ai",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
