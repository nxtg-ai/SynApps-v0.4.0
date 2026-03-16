"""
Setup script for Artist Applet
"""
from setuptools import setup, find_packages

setup(
    name="synapps-artist-applet",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "httpx>=0.23.0",
    ],
    description="SynApps Artist Applet - Image generation using Stable Diffusion or dall-e-3",
    author="SynApps Team",
    author_email="synapps.info@nxtg.ai",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
