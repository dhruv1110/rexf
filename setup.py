"""Setup configuration for rexf - Reproducible Experiments Framework.

This setup.py file is maintained for compatibility with older build systems.
The primary build configuration is in pyproject.toml following PEP 518.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read version from package
def get_version():
    """Extract version from package __init__.py."""
    init_file = this_directory / "rexf" / "__init__.py"
    for line in init_file.read_text().splitlines():
        if line.startswith("__version__"):
            return line.split('"')[1]
    return "0.1.0"

setup(
    name="rexf",
    version=get_version(),
    author="dhruv1110",
    author_email="dhruv1110@users.noreply.github.com",
    description="A lightweight Python library for reproducible experiments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dhruv1110/rexf",
    project_urls={
        "Documentation": "https://github.com/dhruv1110/rexf#readme",
        "Repository": "https://github.com/dhruv1110/rexf.git",
        "Bug Reports": "https://github.com/dhruv1110/rexf/issues",
        "Changelog": "https://github.com/dhruv1110/rexf/blob/main/CHANGELOG.md",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    include_package_data=True,
    package_data={
        "rexf": ["py.typed"],
    },
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
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="experiments reproducibility research tracking visualization",
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
            "pytest-cov>=2.10",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.910",
            "isort>=5.0",
            "pre-commit>=2.0",
        ],
        "test": [
            "pytest>=6.0",
            "pytest-cov>=2.10",
        ],
    },
    zip_safe=False,
)
