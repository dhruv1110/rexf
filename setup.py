"""Setup configuration for rexf - Reproducible Experiments Framework."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="rexf",
    version="0.1.0",
    author="Rexf Contributors",
    author_email="contact@rexf.dev",
    description="A lightweight Python library for reproducible experiments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rexf/rexf",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "matplotlib>=3.3.0",
        "pandas>=1.3.0",
        "pyyaml>=6.0",
        "gitpython>=3.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.910",
        ],
    },
)
