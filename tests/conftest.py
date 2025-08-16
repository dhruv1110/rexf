"""Pytest configuration and fixtures for rexf tests."""

import shutil
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_experiment_data():
    """Sample experiment data for testing."""
    return {
        "parameters": {"x": 5.0, "y": 3.0},
        "results": {"output": 8.0},
        "metrics": {"accuracy": 0.95, "loss": 0.05},
    }
