"""Test intelligence features - exploration, insights, suggestions, and queries."""

import pytest

from rexf import run
from rexf.intelligence.exploration import ExplorationEngine
from rexf.intelligence.insights import InsightsEngine
from rexf.intelligence.queries import SmartQueryEngine
from rexf.intelligence.suggestions import SuggestionEngine


@pytest.mark.slow
class TestExplorationEngine:
    """Test parameter space exploration functionality."""

    def test_exploration_engine_initialization(self, unique_db_path):
        """Test exploration engine initialization."""
        from rexf.backends.intelligent_storage import IntelligentStorage

        storage = IntelligentStorage(unique_db_path)
        try:
            engine = ExplorationEngine(storage)
            assert engine.storage == storage
        finally:
            storage.close()

    def test_auto_detect_ranges(self, unique_db_path, sample_ml_experiment):
        """Test automatic parameter range detection."""
        from rexf.backends.intelligent_storage import IntelligentStorage
        from rexf.core.simple_api import get_experiment_metadata

        storage = IntelligentStorage(unique_db_path)
        try:
            engine = ExplorationEngine(storage)
            metadata = get_experiment_metadata(sample_ml_experiment)

            ranges = engine._auto_detect_ranges(metadata["auto_params"])

            # Should detect ranges for all parameters
            assert "learning_rate" in ranges
            assert "batch_size" in ranges
            assert "epochs" in ranges

            # Learning rate should be float range
            lr_range = ranges["learning_rate"]
            assert isinstance(lr_range, list)
            assert all(isinstance(x, float) for x in lr_range)

            # Batch size should be int range
            bs_range = ranges["batch_size"]
            assert isinstance(bs_range, list)
            assert all(isinstance(x, int) for x in bs_range)

        finally:
            storage.close()

    def test_grid_search_strategy(self, unique_db_path, sample_math_experiment):
        """Test grid search parameter combination generation."""
        from rexf.backends.intelligent_storage import IntelligentStorage

        storage = IntelligentStorage(unique_db_path)
        try:
            engine = ExplorationEngine(storage)

            parameter_ranges = {
                "x": [1.0, 2.0, 3.0],
                "y": [10.0, 20.0],
            }

            combinations = engine._grid_search(parameter_ranges, budget=10)

            # Should generate all combinations (3 * 2 = 6)
            assert len(combinations) == 6

            # Check all combinations are present
            expected = [
                {"x": 1.0, "y": 10.0},
                {"x": 1.0, "y": 20.0},
                {"x": 2.0, "y": 10.0},
                {"x": 2.0, "y": 20.0},
                {"x": 3.0, "y": 10.0},
                {"x": 3.0, "y": 20.0},
            ]

            assert len(combinations) == len(expected)
            for combo in combinations:
                assert combo in expected

        finally:
            storage.close()

    def test_random_search_strategy(self, unique_db_path, sample_math_experiment):
        """Test random search parameter combination generation."""
        from rexf.backends.intelligent_storage import IntelligentStorage

        storage = IntelligentStorage(unique_db_path)
        try:
            engine = ExplorationEngine(storage)

            parameter_ranges = {
                "x": [1.0, 2.0, 3.0, 4.0, 5.0],
                "y": [10.0, 20.0, 30.0],
            }

            combinations = engine._random_search(parameter_ranges, budget=8)

            # Should generate exactly 8 combinations
            assert len(combinations) == 8

            # All combinations should be valid
            for combo in combinations:
                assert combo["x"] in parameter_ranges["x"]
                assert combo["y"] in parameter_ranges["y"]

        finally:
            storage.close()

    def test_auto_explore_integration(self, unique_db_path, sample_math_experiment):
        """Test full auto-exploration integration."""
        from rexf.run import ExperimentRunner

        runner = ExperimentRunner(storage_path=unique_db_path, intelligent=True)
        old_runner = run._default_runner
        run._default_runner = runner

        try:
            if not runner._intelligent:
                pytest.skip("Intelligent features not available")

            # Run auto-exploration
            run_ids = run.auto_explore(
                sample_math_experiment,
                strategy="random",
                budget=3,
                optimization_target="result",
            )

            assert len(run_ids) == 3
            assert all(isinstance(run_id, str) for run_id in run_ids)

            # Check experiments were actually run
            experiments = run.recent(hours=1)
            assert len(experiments) >= 3

        finally:
            run._default_runner = old_runner
            runner.close()


class TestInsightsEngine:
    """Test insights generation functionality."""

    def test_insights_engine_initialization(self, unique_db_path):
        """Test insights engine initialization."""
        from rexf.backends.intelligent_storage import IntelligentStorage

        storage = IntelligentStorage(unique_db_path)
        try:
            engine = InsightsEngine(storage)
            assert engine.storage == storage
        finally:
            storage.close()

    def test_empty_insights(self, unique_db_path):
        """Test insights with no experiments."""
        from rexf.backends.intelligent_storage import IntelligentStorage

        storage = IntelligentStorage(unique_db_path)
        try:
            engine = InsightsEngine(storage)
            insights = engine.generate_insights()

            assert "message" in insights
            assert "No experiments found" in insights["message"]

        finally:
            storage.close()

    def test_basic_insights_generation(self, unique_db_path, sample_ml_experiment):
        """Test basic insights generation with real experiments."""
        from rexf.run import ExperimentRunner

        runner = ExperimentRunner(storage_path=unique_db_path, intelligent=True)
        old_runner = run._default_runner
        run._default_runner = runner

        try:
            if not runner._intelligent:
                pytest.skip("Intelligent features not available")

            # Run several experiments with different outcomes
            run.single(
                sample_ml_experiment, learning_rate=0.001, batch_size=32, epochs=5
            )
            run.single(
                sample_ml_experiment, learning_rate=0.01, batch_size=64, epochs=5
            )
            run.single(
                sample_ml_experiment, learning_rate=0.1, batch_size=32, epochs=10
            )

            # Generate insights
            insights = run.insights()

            assert "summary" in insights
            assert "parameter_insights" in insights
            assert "performance_insights" in insights
            assert "recommendations" in insights

            # Check summary
            summary = insights["summary"]
            assert summary["total_experiments"] == 3
            assert summary["success_rate"] > 0

            # Check recommendations
            recommendations = insights["recommendations"]
            assert isinstance(recommendations, list)

        finally:
            run._default_runner = old_runner
            runner.close()

    def test_parameter_impact_analysis(self, unique_db_path):
        """Test parameter impact analysis."""
        from datetime import datetime

        from rexf.backends.intelligent_storage import IntelligentStorage
        from rexf.core.models import ExperimentRun

        storage = IntelligentStorage(unique_db_path)
        try:
            engine = InsightsEngine(storage)

            # Create experiments with clear parameter impact
            experiments = [
                # High learning rate = low accuracy
                ExperimentRun(
                    run_id="impact-1",
                    experiment_name="impact_test",
                    parameters={"learning_rate": 0.1},
                    status="completed",
                    start_time=datetime.now(),
                ),
                # Low learning rate = high accuracy
                ExperimentRun(
                    run_id="impact-2",
                    experiment_name="impact_test",
                    parameters={"learning_rate": 0.001},
                    status="completed",
                    start_time=datetime.now(),
                ),
                # Medium learning rate = medium accuracy
                ExperimentRun(
                    run_id="impact-3",
                    experiment_name="impact_test",
                    parameters={"learning_rate": 0.01},
                    status="completed",
                    start_time=datetime.now(),
                ),
            ]

            # Add metrics to experiments
            experiments[0].metrics = {"accuracy": 0.6}
            experiments[1].metrics = {"accuracy": 0.9}
            experiments[2].metrics = {"accuracy": 0.75}

            for exp in experiments:
                storage.save_experiment(exp)

            insights = engine.generate_insights("impact_test")

            # Should detect parameter impact
            param_insights = insights["parameter_insights"]
            assert "individual_parameters" in param_insights
            assert "learning_rate" in param_insights["individual_parameters"]

            lr_analysis = param_insights["individual_parameters"]["learning_rate"]
            assert "metric_impact" in lr_analysis
            assert "accuracy" in lr_analysis["metric_impact"]

        finally:
            storage.close()


class TestSuggestionEngine:
    """Test experiment suggestion functionality."""

    def test_suggestion_engine_initialization(self, unique_db_path):
        """Test suggestion engine initialization."""
        from rexf.backends.intelligent_storage import IntelligentStorage

        storage = IntelligentStorage(unique_db_path)
        try:
            engine = SuggestionEngine(storage)
            assert engine.storage == storage
        finally:
            storage.close()

    def test_initial_experiments_suggestions(
        self, unique_db_path, sample_ml_experiment
    ):
        """Test suggestions when no experiments exist."""
        from rexf.backends.intelligent_storage import IntelligentStorage

        storage = IntelligentStorage(unique_db_path)
        try:
            engine = SuggestionEngine(storage)

            suggestions = engine.suggest_next_experiments(
                sample_ml_experiment, count=3, strategy="balanced"
            )

            assert len(suggestions) == 3

            for suggestion in suggestions:
                assert "parameters" in suggestion
                assert "reasoning" in suggestion
                assert "strategy" in suggestion
                assert "confidence" in suggestion
                assert suggestion["strategy"] == "initial_exploration"

        finally:
            storage.close()

    def test_exploit_strategy_suggestions(self, unique_db_path, sample_ml_experiment):
        """Test exploit strategy suggestions."""
        from rexf.run import ExperimentRunner

        runner = ExperimentRunner(storage_path=unique_db_path, intelligent=True)
        old_runner = run._default_runner
        run._default_runner = runner

        try:
            if not runner._intelligent:
                pytest.skip("Intelligent features not available")

            # Run some experiments to build history
            run.single(sample_ml_experiment, learning_rate=0.001, batch_size=32)
            run.single(sample_ml_experiment, learning_rate=0.01, batch_size=64)

            # Get exploit suggestions
            suggestions = run.suggest(
                sample_ml_experiment,
                count=2,
                strategy="exploit",
                optimization_target="accuracy",
            )

            assert "suggestions" in suggestions
            assert len(suggestions["suggestions"]) <= 2

            for suggestion in suggestions["suggestions"]:
                assert suggestion["strategy"] == "exploit"
                assert "base_experiment" in suggestion

        finally:
            run._default_runner = old_runner
            runner.close()

    def test_optimization_target_suggestions(
        self, unique_db_path, sample_ml_experiment
    ):
        """Test optimization target suggestions."""
        from rexf.run import ExperimentRunner

        runner = ExperimentRunner(storage_path=unique_db_path, intelligent=True)
        old_runner = run._default_runner
        run._default_runner = runner

        try:
            if not runner._intelligent:
                pytest.skip("Intelligent features not available")

            # Run experiments with multiple metrics
            run.single(sample_ml_experiment, learning_rate=0.001, batch_size=32)
            run.single(sample_ml_experiment, learning_rate=0.01, batch_size=64)
            run.single(sample_ml_experiment, learning_rate=0.1, batch_size=32)

            engine = SuggestionEngine(runner.storage)
            targets = engine.suggest_optimization_targets(sample_ml_experiment)

            assert isinstance(targets, list)
            assert len(targets) > 0

            for target in targets:
                assert "metric" in target
                assert "score" in target
                assert "reason" in target

        finally:
            run._default_runner = old_runner
            runner.close()


class TestSmartQueryEngine:
    """Test query engine functionality."""

    def test_query_engine_initialization(self, unique_db_path):
        """Test query engine initialization."""
        from rexf.backends.intelligent_storage import IntelligentStorage

        storage = IntelligentStorage(unique_db_path)
        try:
            engine = SmartQueryEngine(storage)
            assert engine.storage == storage
        finally:
            storage.close()

    def test_basic_query_parsing(self, unique_db_path):
        """Test basic query expression parsing."""
        from rexf.backends.intelligent_storage import IntelligentStorage

        storage = IntelligentStorage(unique_db_path)
        try:
            engine = SmartQueryEngine(storage)

            # Test simple queries
            queries = [
                "accuracy > 0.9",
                "loss < 0.1",
                "learning_rate between 0.001 and 0.01",
                "status == completed",
            ]

            for query_str in queries:
                # Should not raise an exception
                explanation = engine.explain_query(query_str)
                assert isinstance(explanation, str)
                assert len(explanation) > 0

        finally:
            storage.close()

    def test_query_execution(self, unique_db_path, sample_ml_experiment):
        """Test query execution with real data."""
        from rexf.run import ExperimentRunner

        runner = ExperimentRunner(storage_path=unique_db_path, intelligent=True)
        old_runner = run._default_runner
        run._default_runner = runner

        try:
            if not runner._intelligent:
                pytest.skip("Intelligent features not available")

            # Run experiments with known values
            run.single(
                sample_ml_experiment, learning_rate=0.0001, epochs=5
            )  # Should have high accuracy (0.89)
            run.single(
                sample_ml_experiment, learning_rate=0.1, epochs=5
            )  # Should have low accuracy
            run.single(
                sample_ml_experiment, learning_rate=0.01, epochs=10
            )  # Medium accuracy

            # Test queries
            high_acc = run.find("accuracy > 0.85")
            assert len(high_acc) >= 1

            low_lr = run.find("param_learning_rate < 0.005")
            assert len(low_lr) >= 1

        finally:
            run._default_runner = old_runner
            runner.close()

    def test_query_suggestions(self, unique_db_path, sample_ml_experiment):
        """Test query suggestions generation."""
        from rexf.run import ExperimentRunner

        runner = ExperimentRunner(storage_path=unique_db_path, intelligent=True)
        old_runner = run._default_runner
        run._default_runner = runner

        try:
            if not runner._intelligent:
                pytest.skip("Intelligent features not available")

            # Run some experiments
            run.single(sample_ml_experiment, learning_rate=0.001)
            run.single(sample_ml_experiment, learning_rate=0.01)

            engine = SmartQueryEngine(runner.storage)
            suggestions = engine.get_query_suggestions()

            assert isinstance(suggestions, list)
            assert len(suggestions) > 0

            # Should suggest queries based on available data
            for suggestion in suggestions:
                assert isinstance(suggestion, str)
                assert len(suggestion) > 0

        finally:
            run._default_runner = old_runner
            runner.close()


@pytest.mark.integration
class TestIntelligenceIntegration:
    """Test integration between different intelligence components."""

    def test_full_intelligence_workflow(
        self, unique_db_path, sample_optimization_experiment
    ):
        """Test complete intelligence workflow."""
        from rexf.run import ExperimentRunner

        runner = ExperimentRunner(storage_path=unique_db_path, intelligent=True)
        old_runner = run._default_runner
        run._default_runner = runner

        try:
            if not runner._intelligent:
                pytest.skip("Intelligent features not available")

            # 1. Run initial experiments
            run.single(sample_optimization_experiment, algorithm="gd", step_size=0.01)
            run.single(sample_optimization_experiment, algorithm="adam", step_size=0.1)

            # 2. Get insights
            insights = run.insights()
            assert "summary" in insights

            # 3. Get suggestions
            suggestions = run.suggest(
                sample_optimization_experiment, count=2, strategy="balanced"
            )
            assert "suggestions" in suggestions

            # 4. Query experiments
            good_convergence = run.find("final_value < 2.0")
            assert isinstance(good_convergence, list)

            # 5. Auto-explore based on insights
            if suggestions["suggestions"]:
                exploration_ids = run.auto_explore(
                    sample_optimization_experiment,
                    strategy="adaptive",
                    budget=2,
                    optimization_target="final_value",
                )
                assert len(exploration_ids) == 2

        finally:
            run._default_runner = old_runner
            runner.close()


if __name__ == "__main__":
    pytest.main([__file__])
