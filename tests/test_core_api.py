"""Test core API functionality - the simple @experiment decorator and basic operations."""

import pytest

from rexf import experiment, run
from rexf.core.simple_api import (
    auto_track_returns,
    extract_parameter_values,
    get_experiment_metadata,
    is_experiment,
)


class TestExperimentDecorator:
    """Test the @experiment decorator functionality."""

    def test_basic_decorator(self):
        """Test basic @experiment decorator usage."""
        
        @experiment
        def test_func(x=1, y=2):
            return {"result": x + y}
        
        # Check decorator was applied
        assert is_experiment(test_func)
        assert hasattr(test_func, "_experiment_metadata")
        assert hasattr(test_func, "_is_rexf_experiment")

    def test_named_decorator(self):
        """Test @experiment decorator with custom name."""
        
        @experiment("custom_name")
        def test_func(x=1):
            return {"result": x}
        
        metadata = get_experiment_metadata(test_func)
        assert metadata["name"] == "custom_name"

    def test_optimize_for_parameter(self):
        """Test @experiment decorator with optimize_for parameter."""
        
        @experiment(optimize_for="accuracy")
        def test_func(x=1):
            return {"accuracy": x * 0.9}
        
        metadata = get_experiment_metadata(test_func)
        assert metadata["optimize_for"] == "accuracy"

    def test_parameter_detection(self):
        """Test automatic parameter detection."""
        
        @experiment
        def test_func(a, b=10, c: int = 20, d: float = 0.5):
            return {"result": a + b + c + d}
        
        metadata = get_experiment_metadata(test_func)
        params = metadata["auto_params"]
        
        assert "a" in params
        assert "b" in params
        assert "c" in params
        assert "d" in params
        
        # Check defaults
        assert params["b"]["default"] == 10
        assert params["c"]["default"] == 20
        assert params["d"]["default"] == 0.5

    def test_extract_parameter_values(self):
        """Test parameter value extraction."""
        
        @experiment
        def test_func(x=1, y=2, z=3):
            return {"result": x + y + z}
        
        # Test with partial parameters
        values = extract_parameter_values(test_func, x=10, y=20)
        assert values == {"x": 10, "y": 20, "z": 3}  # z should use default
        
        # Test with all parameters
        values = extract_parameter_values(test_func, x=5, y=6, z=7)
        assert values == {"x": 5, "y": 6, "z": 7}


class TestAutoTrackReturns:
    """Test automatic return value tracking."""

    def test_single_number_return(self):
        """Test single numeric return value."""
        result = auto_track_returns(0.95, "test_exp")
        
        assert result["metrics"]["result"] == 0.95
        assert result["results"] == {}
        assert result["artifacts"] == {}

    def test_single_string_return(self):
        """Test single string return value."""
        result = auto_track_returns("success", "test_exp")
        
        assert result["metrics"] == {}
        assert result["results"]["result"] == "success"
        assert result["artifacts"] == {}

    def test_dict_return_auto_categorization(self):
        """Test automatic categorization of dictionary returns."""
        return_value = {
            "accuracy": 0.95,
            "loss": 0.05,
            "model_path": "/tmp/model.pkl",
            "status": "completed",
            "config": {"lr": 0.01},
        }
        
        result = auto_track_returns(return_value, "test_exp")
        
        # Numbers should be metrics
        assert result["metrics"]["accuracy"] == 0.95
        assert result["metrics"]["loss"] == 0.05
        
        # Strings that look like paths should be artifacts
        assert result["artifacts"]["model_path"] == "/tmp/model.pkl"
        
        # Other values should be results
        assert result["results"]["status"] == "completed"
        assert result["results"]["config"] == {"lr": 0.01}

    def test_none_return(self):
        """Test None return value."""
        result = auto_track_returns(None, "test_exp")
        
        assert result["metrics"] == {}
        assert result["results"] == {}
        assert result["artifacts"] == {}


class TestRunSingleExperiment:
    """Test running single experiments."""

    def test_basic_single_run(self, unique_db_path):
        """Test basic single experiment run."""
        from rexf.run import ExperimentRunner
        
        runner = ExperimentRunner(storage_path=unique_db_path)
        old_runner = run._default_runner
        run._default_runner = runner
        
        try:
            @experiment
            def test_math_experiment(x=1.0, y=2.0):
                return {"result": x * y, "sum": x + y}
            
            run_id = run.single(test_math_experiment, x=3.0, y=4.0)
            
            assert isinstance(run_id, str)
            assert len(run_id) > 10  # Should be a UUID
            
            # Check experiment was stored
            experiments = run.recent(hours=1)
            assert len(experiments) == 1
            assert experiments[0].run_id == run_id
            
        finally:
            run._default_runner = old_runner
            runner.close()

    def test_multiple_runs(self, unique_db_path, sample_ml_experiment):
        """Test running multiple experiments."""
        from rexf.run import ExperimentRunner
        
        runner = ExperimentRunner(storage_path=unique_db_path)
        old_runner = run._default_runner
        run._default_runner = runner
        
        try:
            # Run multiple experiments
            run_id1 = run.single(sample_ml_experiment, learning_rate=0.001, batch_size=32)
            run_id2 = run.single(sample_ml_experiment, learning_rate=0.01, batch_size=64)
            run_id3 = run.single(sample_ml_experiment, learning_rate=0.1, batch_size=16)
            
            assert run_id1 != run_id2 != run_id3
            
            # Check all experiments were stored
            experiments = run.recent(hours=1)
            assert len(experiments) == 3
            
        finally:
            run._default_runner = old_runner
            runner.close()

    def test_experiment_with_failure(self, unique_db_path, sample_failing_experiment):
        """Test handling of failed experiments."""
        from rexf.run import ExperimentRunner
        
        runner = ExperimentRunner(storage_path=unique_db_path)
        old_runner = run._default_runner
        run._default_runner = runner
        
        try:
            # Should succeed
            run_id1 = run.single(sample_failing_experiment, should_fail=False)
            
            # Should fail but still be recorded
            run_id2 = run.single(sample_failing_experiment, should_fail=True)
            
            assert run_id1 != run_id2
            
            # Check experiments
            experiments = run.recent(hours=1)
            assert len(experiments) == 2
            
            # Find the failed experiment
            failed_exp = next(exp for exp in experiments if exp.run_id == run_id2)
            assert failed_exp.status == "failed"
            
        finally:
            run._default_runner = old_runner
            runner.close()


class TestRunBasicMethods:
    """Test basic run module methods."""

    def test_best_experiments(self, unique_db_path, sample_ml_experiment):
        """Test finding best experiments."""
        from rexf.run import ExperimentRunner
        
        runner = ExperimentRunner(storage_path=unique_db_path)
        old_runner = run._default_runner
        run._default_runner = runner
        
        try:
            # Run experiments with different performance
            run.single(sample_ml_experiment, learning_rate=0.001, epochs=5)  # Good
            run.single(sample_ml_experiment, learning_rate=0.1, epochs=5)    # Bad
            run.single(sample_ml_experiment, learning_rate=0.002, epochs=5)  # Good
            
            # Test best experiments
            best = run.best(top=2)
            assert len(best) >= 1  # At least one should have metrics
            
            # Test with specific metric
            best_accuracy = run.best(metric="accuracy", top=1)
            assert len(best_accuracy) >= 1
            
        finally:
            run._default_runner = old_runner
            runner.close()

    def test_failed_experiments(self, unique_db_path, sample_failing_experiment):
        """Test getting failed experiments."""
        from rexf.run import ExperimentRunner
        
        runner = ExperimentRunner(storage_path=unique_db_path)
        old_runner = run._default_runner
        run._default_runner = runner
        
        try:
            # Run some experiments
            run.single(sample_failing_experiment, should_fail=False)
            run.single(sample_failing_experiment, should_fail=True)
            run.single(sample_failing_experiment, should_fail=True)
            
            # Check failed experiments
            failed = run.failed()
            assert len(failed) == 2
            assert all(exp.status == "failed" for exp in failed)
            
        finally:
            run._default_runner = old_runner
            runner.close()

    def test_recent_experiments(self, unique_db_path, sample_math_experiment):
        """Test getting recent experiments."""
        from rexf.run import ExperimentRunner
        
        runner = ExperimentRunner(storage_path=unique_db_path)
        old_runner = run._default_runner
        run._default_runner = runner
        
        try:
            # Run some experiments
            run.single(sample_math_experiment, x=1)
            run.single(sample_math_experiment, x=2)
            
            # Test recent experiments
            recent = run.recent(hours=1)
            assert len(recent) == 2
            
            recent_24h = run.recent(hours=24)
            assert len(recent_24h) == 2
            
        finally:
            run._default_runner = old_runner
            runner.close()


@pytest.mark.unit
class TestErrorHandling:
    """Test error handling in core API."""

    def test_non_experiment_function(self):
        """Test error when function is not decorated."""
        def regular_function(x=1):
            return x + 1
        
        assert not is_experiment(regular_function)
        
        with pytest.raises(ValueError, match="not decorated with @experiment"):
            get_experiment_metadata(regular_function)

    def test_invalid_runner_usage(self):
        """Test invalid usage patterns."""
        # Test with None as experiment function
        with pytest.raises((TypeError, AttributeError)):
            run.single(None, x=1)


if __name__ == "__main__":
    pytest.main([__file__])
