"""Tests for the new simple API."""

import tempfile
from pathlib import Path

from rexf import experiment, run


@experiment
def simple_test_function(x, y=2.0):
    """Simple test function."""
    result = x * y + 0.1
    return {"score": result, "product": x * y}


@experiment("custom_name")
def named_test_function(param1, param2=1.0):
    """Named test function."""
    return {"accuracy": param1 + param2 * 0.1}


def test_simple_experiment_decorator():
    """Test the @experiment decorator works correctly."""
    from rexf.core.simple_api import get_experiment_metadata, is_experiment

    # Test that functions are properly decorated
    assert is_experiment(simple_test_function)
    assert is_experiment(named_test_function)

    # Test metadata extraction
    metadata = get_experiment_metadata(simple_test_function)
    assert metadata["name"] == "simple_test_function"
    assert "auto_params" in metadata
    assert "x" in metadata["auto_params"]
    assert "y" in metadata["auto_params"]

    # Test custom name
    named_metadata = get_experiment_metadata(named_test_function)
    assert named_metadata["name"] == "custom_name"


def test_simple_run_interface():
    """Test the run module interface."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize a runner with explicit path instead of changing directories
        from rexf.run import ExperimentRunner

        test_runner = ExperimentRunner(storage_path=str(Path(temp_dir) / "test.db"))

        # Use the test runner directly instead of global run module
        import rexf.run as run_module

        old_runner = run_module._default_runner
        run_module._default_runner = test_runner

        try:
            # Test single experiment run
            run_id = run.single(simple_test_function, x=3.0, y=4.0)
            assert isinstance(run_id, str)
            assert len(run_id) > 10  # UUID should be reasonably long

            # Test another experiment
            run_id2 = run.single(named_test_function, param1=0.8, param2=2.0)
            assert run_id2 != run_id  # Should be different

            # Test best experiments
            best_exps = run.best(top=2)
            assert len(best_exps) >= 1  # At least one experiment with metrics

            # Test insights
            insights_data = run.insights()
            assert "total_experiments" in insights_data
            assert insights_data["total_experiments"] == 2
            assert insights_data["completed"] == 2
            assert insights_data["failed"] == 0

            # Test recent experiments
            recent_exps = run.recent(hours=1)
            assert len(recent_exps) == 2

            # Test failed experiments (should be empty)
            failed_exps = run.failed()
            assert len(failed_exps) == 0

        finally:
            # Restore original runner
            run_module._default_runner = old_runner


def test_auto_categorization():
    """Test automatic result categorization."""
    from rexf.core.simple_api import auto_track_returns

    # Test dictionary with mixed types
    result = {
        "accuracy": 0.95,
        "loss": 0.1,
        "model_name": "test",
        "data_file": "test.csv",
    }
    categorized = auto_track_returns(result, "test_exp")

    # Numbers should go to metrics
    assert "accuracy" in categorized["metrics"]
    assert "loss" in categorized["metrics"]

    # Strings should go to results
    assert "model_name" in categorized["results"]

    # Test single number
    single_result = 0.95
    categorized_single = auto_track_returns(single_result, "test_exp")
    assert categorized_single["metrics"]["result"] == 0.95

    # Test None result
    none_result = auto_track_returns(None, "test_exp")
    assert none_result["metrics"] == {}
    assert none_result["results"] == {}


def test_parameter_extraction():
    """Test parameter extraction from function calls."""
    from rexf.core.simple_api import extract_parameter_values

    # Test with all parameters provided
    params = extract_parameter_values(simple_test_function, 5.0, y=3.0)
    assert params["x"] == 5.0
    assert params["y"] == 3.0

    # Test with default parameters
    params_default = extract_parameter_values(simple_test_function, 5.0)
    assert params_default["x"] == 5.0
    assert params_default["y"] == 2.0  # Default value

    # Test with kwargs
    params_kwargs = extract_parameter_values(simple_test_function, x=7.0, y=8.0)
    assert params_kwargs["x"] == 7.0
    assert params_kwargs["y"] == 8.0


if __name__ == "__main__":
    test_simple_experiment_decorator()
    test_simple_run_interface()
    test_auto_categorization()
    test_parameter_extraction()
    print("✅ All simple API tests passed!")
