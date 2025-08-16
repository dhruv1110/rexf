"""Pytest configuration and shared fixtures."""

import tempfile
import uuid
from pathlib import Path
from typing import Generator

import pytest

from rexf import experiment


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def temp_db_path(temp_dir: Path) -> str:
    """Create a temporary database path."""
    return str(temp_dir / "test_experiments.db")


@pytest.fixture
def unique_db_path(temp_dir: Path) -> str:
    """Create a unique database path for isolated tests."""
    return str(temp_dir / f"test_{uuid.uuid4().hex[:8]}.db")


# Sample experiment functions for testing
@experiment
def _sample_ml_experiment(learning_rate=0.01, batch_size=32, epochs=10):
    """Sample ML experiment for testing."""
    import time

    time.sleep(0.01)  # Simulate computation

    # Simulate performance based on parameters
    accuracy = 0.8 + (0.001 - learning_rate) * 100
    accuracy = max(0.5, min(0.99, accuracy))

    return {
        "accuracy": round(accuracy, 4),
        "loss": round(1 - accuracy, 4),
        "training_time": 0.01,
    }


@experiment
def _sample_math_experiment(x=1.0, y=2.0):
    """Sample mathematical experiment for testing."""
    result = x * y + 0.5
    return {
        "result": result,
        "sum": x + y,
        "product": x * y,
    }


@experiment("custom_named_experiment")
def sample_named_experiment(param1=1, param2=2):
    """Sample experiment with custom name."""
    return {"output": param1 + param2}


@experiment
def _sample_failing_experiment(should_fail=False):
    """Sample experiment that can fail for testing error handling."""
    if should_fail:
        raise ValueError("Intentional test failure")
    return {"success": True}


@experiment
def _sample_optimization_experiment(algorithm="gd", step_size=0.1, iterations=10):
    """Sample optimization experiment."""
    # Simulate optimization convergence
    final_value = 1.0 + step_size * 0.1
    return {
        "final_value": final_value,
        "iterations": iterations,
        "converged": final_value < 1.1,
    }


# Fixture versions of the sample experiments (matching test expectations)
@pytest.fixture
def sample_ml_experiment():
    """Fixture for sample ML experiment."""
    return _sample_ml_experiment


@pytest.fixture
def sample_math_experiment():
    """Fixture for sample math experiment."""
    return _sample_math_experiment


@pytest.fixture
def sample_failing_experiment():
    """Fixture for sample failing experiment."""
    return _sample_failing_experiment


@pytest.fixture
def sample_optimization_experiment():
    """Fixture for sample optimization experiment."""
    return _sample_optimization_experiment


# Pytest collection configuration
def pytest_collection_modifyitems(config, items):
    """Add custom markers to tests based on their names."""
    for item in items:
        # Mark slow tests
        if "slow" in item.nodeid or "exploration" in item.nodeid:
            item.add_marker(pytest.mark.slow)

        # Mark integration tests
        if "integration" in item.nodeid or "dashboard" in item.nodeid:
            item.add_marker(pytest.mark.integration)

        # Mark unit tests (default)
        if not any(
            mark.name in ["slow", "integration"] for mark in item.iter_markers()
        ):
            item.add_marker(pytest.mark.unit)


# Custom markers
pytestmark = [
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
    pytest.mark.filterwarnings("ignore::UserWarning"),
]
