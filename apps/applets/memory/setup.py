"""
Setup script for Memory Applet
"""
from setuptools import setup, find_packages

setup(
    name="synapps-memory-applet",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[],
    description="SynApps Memory Applet - Context storage and retrieval",
    author="SynApps Team",
    author_email="synapps.info@nxtg.ai",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
