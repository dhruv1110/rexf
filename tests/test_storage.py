"""Test storage backends and data persistence."""

import pytest
from datetime import datetime, timedelta

from rexf.backends.intelligent_storage import IntelligentStorage
from rexf.core.models import ExperimentRun


class TestIntelligentStorage:
    """Test enhanced intelligent storage functionality."""

    def test_intelligent_storage_initialization(self, temp_db_path):
        """Test intelligent storage initialization."""
        storage = IntelligentStorage(temp_db_path)
        
        try:
            # Should create database with enhanced schema
            experiments = storage.list_experiments()
            assert experiments == []
            
            # Should have stats method
            stats = storage.get_storage_stats()
            assert "total_experiments" in stats
            assert stats["total_experiments"] == 0
            
        finally:
            storage.close()

    def test_enhanced_experiment_storage(self, temp_db_path):
        """Test enhanced experiment storage with normalized schema."""
        storage = IntelligentStorage(temp_db_path)
        
        try:
            # Create experiment with various parameter types
            experiment = ExperimentRun(
                run_id="test-enhanced",
                experiment_name="enhanced_test",
                parameters={
                    "learning_rate": 0.01,     # float
                    "batch_size": 32,          # int
                    "model_type": "neural_net", # string
                    "use_dropout": True,       # boolean
                },
                metrics={
                    "accuracy": 0.95,
                    "loss": 0.05,
                    "f1_score": 0.93,
                },
                status="completed",
                start_time=datetime.now(),
                end_time=datetime.now(),
                duration=2.5,
            )
            
            storage.save_experiment(experiment)
            
            # Load and verify
            loaded = storage.load_experiment("test-enhanced")
            assert loaded is not None
            assert loaded.parameters["learning_rate"] == 0.01
            assert loaded.parameters["batch_size"] == 32
            assert loaded.parameters["model_type"] == "neural_net"
            assert loaded.parameters["use_dropout"] is True
            
        finally:
            storage.close()

    def test_storage_statistics(self, temp_db_path):
        """Test storage statistics functionality."""
        storage = IntelligentStorage(temp_db_path)
        
        try:
            # Add some experiments
            for i in range(3):
                exp = ExperimentRun(
                    run_id=f"stats-test-{i}",
                    experiment_name="stats_experiment",
                    parameters={"x": i},
                    metrics={"score": i * 0.2},
                    status="completed" if i < 2 else "failed",
                    start_time=datetime.now(),
                    duration=i * 0.5,
                )
                storage.save_experiment(exp)
            
            # Get statistics
            stats = storage.get_storage_stats()
            
            assert stats["total_experiments"] == 3
            assert stats["by_status"]["completed"] == 2
            assert stats["by_status"]["failed"] == 1
            assert stats["by_experiment"]["stats_experiment"] == 3
            
        finally:
            storage.close()

    def test_parameter_space_summary(self, temp_db_path):
        """Test parameter space analysis."""
        storage = IntelligentStorage(temp_db_path)
        
        try:
            # Add experiments with varied parameters
            parameters_list = [
                {"lr": 0.01, "batch_size": 32, "model": "a"},
                {"lr": 0.1, "batch_size": 64, "model": "a"},
                {"lr": 0.001, "batch_size": 32, "model": "b"},
                {"lr": 0.01, "batch_size": 128, "model": "b"},
            ]
            
            for i, params in enumerate(parameters_list):
                exp = ExperimentRun(
                    run_id=f"param-{i}",
                    experiment_name="param_test",
                    parameters=params,
                    metrics={"accuracy": 0.8 + i * 0.05},
                    status="completed",
                    start_time=datetime.now(),
                )
                storage.save_experiment(exp)
            
            # Get parameter space summary
            summary = storage.get_parameter_space_summary("param_test")
            
            assert "lr" in summary
            assert "batch_size" in summary
            assert "model" in summary
            
            # Check lr parameter analysis
            lr_info = summary["lr"]
            assert lr_info["unique_values"] == 3  # 0.01, 0.1, 0.001
            assert lr_info["min_value"] == 0.001
            assert lr_info["max_value"] == 0.1
            
        finally:
            storage.close()

    def test_metric_trends(self, temp_db_path):
        """Test metric trends analysis."""
        storage = IntelligentStorage(temp_db_path)
        
        try:
            # Add experiments with increasing accuracy over time
            base_time = datetime.now() - timedelta(hours=5)
            
            for i in range(5):
                exp = ExperimentRun(
                    run_id=f"trend-{i}",
                    experiment_name="trend_test",
                    parameters={"iteration": i},
                    metrics={"accuracy": 0.7 + i * 0.05},  # Increasing trend
                    status="completed",
                    start_time=base_time + timedelta(hours=i),
                )
                storage.save_experiment(exp)
            
            # Get metric trends
            trends = storage.get_metric_trends("accuracy", "trend_test")
            
            assert "values" in trends
            assert "trend_direction" in trends
            assert len(trends["values"]) == 5
            
            # Values should be in chronological order
            values = [point["value"] for point in trends["values"]]
            assert values == [0.7, 0.75, 0.8, 0.85, 0.9]
            
        finally:
            storage.close()

    @pytest.mark.slow
    def test_complex_queries(self, temp_db_path):
        """Test complex query functionality."""
        storage = IntelligentStorage(temp_db_path)
        
        try:
            # Create diverse experiments
            experiments_data = [
                {"lr": 0.01, "bs": 32, "acc": 0.9, "loss": 0.1, "status": "completed"},
                {"lr": 0.1, "bs": 64, "acc": 0.85, "loss": 0.15, "status": "completed"},
                {"lr": 0.001, "bs": 32, "acc": 0.95, "loss": 0.05, "status": "completed"},
                {"lr": 0.01, "bs": 128, "acc": 0.8, "loss": 0.2, "status": "failed"},
                {"lr": 0.05, "bs": 64, "acc": 0.88, "loss": 0.12, "status": "completed"},
            ]
            
            for i, data in enumerate(experiments_data):
                exp = ExperimentRun(
                    run_id=f"query-{i}",
                    experiment_name="query_test",
                    parameters={"learning_rate": data["lr"], "batch_size": data["bs"]},
                    metrics={"accuracy": data["acc"], "loss": data["loss"]},
                    status=data["status"],
                    start_time=datetime.now(),
                )
                storage.save_experiment(exp)
            
            # Test complex queries
            high_acc = storage.query_experiments(
                metric_filters={"accuracy": {">": 0.9}}
            )
            assert len(high_acc) == 2  # acc 0.9 and 0.95
            
            small_batch = storage.query_experiments(
                parameter_filters={"batch_size": {"<=": 32}}
            )
            assert len(small_batch) == 2  # batch_size 32
            
            completed_only = storage.query_experiments(
                conditions={"status": "completed"}
            )
            assert len(completed_only) == 4  # All except the failed one
            
        finally:
            storage.close()


if __name__ == "__main__":
    pytest.main([__file__])
