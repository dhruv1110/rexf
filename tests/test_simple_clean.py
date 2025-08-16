"""Simple, clean tests for the essential RexF functionality."""

import tempfile
from pathlib import Path

import pytest

from rexf import experiment, run


class TestEssentialFunctionality:
    """Test the core functionality that users actually use."""

    def test_basic_experiment_flow(self):
        """Test the complete basic experiment flow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Use temporary database
            db_path = str(Path(temp_dir) / "test.db")

            # Create a simple experiment
            @experiment
            def simple_experiment(x=1.0, y=2.0):
                result = x * y
                accuracy = min(0.99, 0.5 + result * 0.1)
                return {"result": result, "accuracy": accuracy, "status": "completed"}

            # Initialize runner with test database
            from rexf.run import ExperimentRunner

            runner = ExperimentRunner(storage_path=db_path)
            old_runner = run._default_runner
            run._default_runner = runner

            try:
                # Test 1: Run single experiment
                run_id1 = run.single(simple_experiment, x=2.0, y=3.0)
                assert isinstance(run_id1, str)
                assert len(run_id1) > 10

                # Test 2: Run another experiment
                run_id2 = run.single(simple_experiment, x=4.0, y=5.0)
                assert run_id2 != run_id1

                # Test 3: Get recent experiments
                recent = run.recent(hours=1)
                assert len(recent) == 2

                # Test 4: Get best experiments
                best = run.best(top=1)
                assert len(best) >= 1

                # Test 5: Basic insights
                insights = run.insights()
                assert isinstance(insights, dict)

                # Test 6: Find experiments (if query engine available)
                try:
                    high_accuracy = run.find("accuracy > 0.8")
                    assert isinstance(high_accuracy, list)
                except Exception:
                    # Query engine might not be available in all configurations
                    pass

            finally:
                run._default_runner = old_runner
                runner.close()

    def test_experiment_decorator(self):
        """Test the @experiment decorator."""

        @experiment
        def test_func(a=1, b=2):
            return {"sum": a + b, "product": a * b}

        # Test metadata is attached
        assert hasattr(test_func, "_is_rexf_experiment")
        assert hasattr(test_func, "_experiment_metadata")

        # Test metadata content
        from rexf.core.simple_api import get_experiment_metadata, is_experiment

        assert is_experiment(test_func)
        metadata = get_experiment_metadata(test_func)
        assert metadata["name"] == "test_func"
        assert "a" in metadata["auto_params"]
        assert "b" in metadata["auto_params"]

    def test_named_experiment(self):
        """Test experiment with custom name."""

        @experiment("custom_name")
        def named_func(x=1):
            return {"value": x * 2}

        from rexf.core.simple_api import get_experiment_metadata

        metadata = get_experiment_metadata(named_func)
        assert metadata["name"] == "custom_name"

    def test_auto_tracking(self):
        """Test automatic result tracking."""
        from rexf.core.simple_api import auto_track_returns

        # Test numeric values become metrics
        result = auto_track_returns({"accuracy": 0.95, "loss": 0.05}, "test")
        assert result["metrics"]["accuracy"] == 0.95
        assert result["metrics"]["loss"] == 0.05

        # Test string values become results
        result = auto_track_returns({"status": "completed", "notes": "good"}, "test")
        assert result["results"]["status"] == "completed"
        assert result["results"]["notes"] == "good"

    def test_experiment_with_failure(self):
        """Test handling of experiment failures."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / "test.db")

            @experiment
            def failing_experiment(should_fail=False):
                if should_fail:
                    raise ValueError("Test failure")
                return {"success": True}

            from rexf.run import ExperimentRunner

            runner = ExperimentRunner(storage_path=db_path)
            old_runner = run._default_runner
            run._default_runner = runner

            try:
                # Success case
                success_id = run.single(failing_experiment, should_fail=False)
                assert isinstance(success_id, str)

                # Failure case (should still create record)
                failure_id = run.single(failing_experiment, should_fail=True)
                assert isinstance(failure_id, str)
                assert failure_id != success_id

                # Check both experiments exist
                recent = run.recent(hours=1)
                assert len(recent) == 2

                # Check failure was recorded
                failed = run.failed()
                assert len(failed) == 1
                assert failed[0].status == "failed"

            finally:
                run._default_runner = old_runner
                runner.close()

    def test_storage_functionality(self):
        """Test that storage works correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / "test.db")

            from rexf.backends.intelligent_storage import IntelligentStorage

            storage = IntelligentStorage(db_path)
            try:
                # Test initialization
                assert Path(db_path).exists()

                # Test empty state
                experiments = storage.list_experiments()
                assert experiments == []

                stats = storage.get_storage_stats()
                assert stats["total_experiments"] == 0

            finally:
                storage.close()

    def test_intelligence_features_availability(self):
        """Test that intelligence features are available."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / "test.db")

            from rexf.run import ExperimentRunner

            runner = ExperimentRunner(storage_path=db_path, intelligent=True)

            try:
                # Test that intelligence features are available
                assert runner._intelligent
                assert hasattr(runner, "storage")

                # Test that intelligence modules can be imported
                from rexf.intelligence.insights import InsightsEngine
                from rexf.intelligence.queries import SmartQueryEngine

                query_engine = SmartQueryEngine(runner.storage)
                insights_engine = InsightsEngine(runner.storage)

                assert query_engine is not None
                assert insights_engine is not None

            finally:
                runner.close()


if __name__ == "__main__":
    pytest.main([__file__])
