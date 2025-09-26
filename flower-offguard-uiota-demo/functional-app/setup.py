#!/usr/bin/env python3
"""
Setup script for the functional federated learning application
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements_path = this_directory / "containers" / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = requirements_path.read_text().strip().split('\n')
    requirements = [req.strip() for req in requirements if req.strip() and not req.startswith('#')]

setup(
    name="functional-fl-app",
    version="1.0.0",
    description="Functional Federated Learning Application with Real ML Training",
    long_description=long_description,
    long_description_content_type="text/markdown",

    author="FL Development Team",
    author_email="fl-team@example.com",
    url="https://github.com/example/functional-fl-app",

    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,

    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Distributed Computing",
    ],

    keywords="federated-learning machine-learning pytorch flower distributed-systems",

    entry_points={
        "console_scripts": [
            "fl-server=run_server:main",
            "fl-client=run_client:main",
            "fl-demo=run_demo:main",
        ],
    },

    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-asyncio>=0.21",
            "black>=22.0",
            "flake8>=5.0",
            "mypy>=1.0",
        ],
        "privacy": [
            "opacus>=1.4.0",
        ],
        "viz": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
        ],
    },

    package_data={
        "app": ["templates/*.html", "static/css/*.css", "static/js/*.js"],
        "containers": ["*.yml", "*.conf", "Dockerfile.*"],
    },

    include_package_data=True,
    zip_safe=False,
)