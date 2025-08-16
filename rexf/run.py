"""Main user interface for running experiments.

This module provides the simple, intelligent interface that users interact with.
No complex configuration needed - just run experiments and get insights.
"""

import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

from .core.models import ExperimentData, ExperimentRun
from .core.simple_api import (
    auto_track_returns,
    extract_parameter_values,
    get_experiment_metadata,
    is_experiment,
)


class ExperimentRunner:
    """
    Intelligent experiment runner with zero configuration.

    This replaces the complex ExperimentRunner with a much simpler interface
    focused on user experience rather than architectural purity.
    """

    def __init__(self, storage_path: str = "experiments.db"):
        """Initialize with minimal configuration."""
        # Use the existing storage backend for now
        from .backends.sqlite_storage import SQLiteStorage

        self.storage = SQLiteStorage(storage_path)
        self._current_experiments = []

    def single(self, experiment_func: Callable, **params) -> str:
        """
        Run a single experiment with given parameters.

        Args:
            experiment_func: Function decorated with @experiment
            **params: Parameter values to use

        Returns:
            Experiment run ID
        """
        if not is_experiment(experiment_func):
            raise ValueError(
                f"Function {experiment_func.__name__} must be decorated with @experiment"
            )

        metadata = get_experiment_metadata(experiment_func)
        experiment_name = metadata["name"]

        # Generate run ID
        run_id = str(uuid.uuid4())

        # Extract actual parameter values (including defaults)
        param_values = extract_parameter_values(experiment_func, **params)

        # Create experiment record
        start_time = datetime.now()
        experiment = ExperimentRun(
            run_id=run_id,
            experiment_name=experiment_name,
            parameters=param_values,
            start_time=start_time,
            status="running",
        )

        try:
            # Run the experiment
            print(f"🧪 Running experiment '{experiment_name}' (ID: {run_id[:8]}...)")
            result = experiment_func(**param_values)

            # Auto-categorize results
            categorized = auto_track_returns(result, experiment_name)
            experiment.metrics = categorized["metrics"]
            experiment.results = categorized["results"]
            # TODO: Handle artifacts in Phase 1.2

            # Mark as completed
            experiment.end_time = datetime.now()
            experiment.status = "completed"

            # Save to storage
            self.storage.save_experiment(ExperimentData(experiment))

            duration = experiment.duration or 0
            print(f"✅ Completed in {duration:.2f}s")

            # Show key results
            if experiment.metrics:
                print(f"📊 Metrics: {experiment.metrics}")
            if experiment.results:
                print(f"📋 Results: {experiment.results}")

            return run_id

        except Exception as e:
            # Mark as failed
            experiment.end_time = datetime.now()
            experiment.status = "failed"
            experiment.metadata = {"error": str(e)}

            self.storage.save_experiment(ExperimentData(experiment))
            print(f"❌ Experiment failed: {e}")
            raise

    def best(self, metric: Optional[str] = None, top: int = 5) -> List[ExperimentData]:
        """
        Get the best experiments by a metric.

        Args:
            metric: Metric to optimize for (auto-detected if None)
            top: Number of top experiments to return

        Returns:
            List of best experiments
        """
        all_experiments = self.storage.list_experiments()

        if not all_experiments:
            return []

        # Auto-detect metric if not specified
        if metric is None:
            metric = self._auto_detect_metric(all_experiments)
            if metric is None:
                print("⚠️ No numeric metrics found to optimize")
                return all_experiments[:top]

        # Filter experiments that have the metric
        with_metric = [exp for exp in all_experiments if metric in exp.metrics]

        if not with_metric:
            print(f"⚠️ No experiments found with metric '{metric}'")
            return all_experiments[:top]

        # Sort by metric (descending - assuming higher is better)
        sorted_experiments = sorted(
            with_metric, key=lambda x: x.metrics[metric], reverse=True
        )

        return sorted_experiments[:top]

    def failed(self) -> List[ExperimentData]:
        """Get all failed experiments."""
        all_experiments = self.storage.list_experiments()
        return [exp for exp in all_experiments if exp.status == "failed"]

    def recent(self, hours: int = 24) -> List[ExperimentData]:
        """Get experiments from the last N hours."""
        from datetime import timedelta

        cutoff = datetime.now() - timedelta(hours=hours)
        all_experiments = self.storage.list_experiments()

        return [exp for exp in all_experiments if exp.start_time >= cutoff]

    def find(self, query: str) -> List[ExperimentData]:
        """
        Find experiments using natural language queries.

        Supports queries like:
        - "accuracy > 0.9"
        - "status == 'completed'"
        - "runtime < 60"

        Args:
            query: Natural language query string

        Returns:
            List of matching experiments
        """
        # TODO: Implement proper query parsing in Phase 1.3
        # For now, just return all experiments with a warning
        print(f"⚠️ Query parsing not yet implemented: '{query}'")
        print("📋 Returning all experiments for now")
        return self.storage.list_experiments()

    def compare(
        self, experiments: Optional[List[Union[str, ExperimentData]]] = None
    ) -> None:
        """
        Smart comparison of experiments with auto-visualization.

        Args:
            experiments: List of experiment IDs or ExperimentData objects.
                        If None, compares recent experiments.
        """
        if experiments is None:
            # Compare recent experiments
            experiments = self.recent(hours=24)
            if len(experiments) < 2:
                experiments = self.storage.list_experiments()[:5]
        else:
            # Convert IDs to ExperimentData if needed
            resolved_experiments = []
            for exp in experiments:
                if isinstance(exp, str):
                    resolved_experiments.append(self.storage.load_experiment(exp))
                else:
                    resolved_experiments.append(exp)
            experiments = [exp for exp in resolved_experiments if exp is not None]

        if len(experiments) < 2:
            print("⚠️ Need at least 2 experiments to compare")
            return

        print(f"📊 Comparing {len(experiments)} experiments:")
        print("-" * 50)

        for i, exp in enumerate(experiments, 1):
            duration = exp.duration or 0
            print(f"{i}. {exp.experiment_name} ({exp.run_id[:8]}...)")
            print(f"   Status: {exp.status}, Duration: {duration:.2f}s")
            print(f"   Parameters: {exp.parameters}")
            print(f"   Metrics: {exp.metrics}")
            print()

        # TODO: Add visual comparison in Phase 3

    def insights(self) -> Dict[str, Any]:
        """
        Generate AI-powered insights from all experiments.

        Returns:
            Dictionary of insights and recommendations
        """
        # TODO: Implement proper insights in Phase 2.2
        # For now, return basic statistics
        all_experiments = self.storage.list_experiments()

        if not all_experiments:
            return {"message": "No experiments found. Run some experiments first!"}

        total = len(all_experiments)
        completed = len([exp for exp in all_experiments if exp.status == "completed"])
        failed = total - completed

        # Find most common parameters
        all_params = {}
        for exp in all_experiments:
            for param, value in exp.parameters.items():
                if param not in all_params:
                    all_params[param] = []
                all_params[param].append(value)

        return {
            "total_experiments": total,
            "completed": completed,
            "failed": failed,
            "success_rate": completed / total if total > 0 else 0,
            "common_parameters": list(all_params.keys()),
            "message": "🔍 Basic insights ready. Advanced AI insights coming in Phase 2!",
        }

    def suggest(self) -> Dict[str, Any]:
        """
        Suggest next experiments to run.

        Returns:
            Dictionary with parameter suggestions and reasoning
        """
        # TODO: Implement intelligent suggestions in Phase 2.3
        return {
            "message": "🤖 Intelligent experiment suggestions coming in Phase 2!",
            "current_status": "Use run.best() to see your top experiments for now",
        }

    def _auto_detect_metric(self, experiments: List[ExperimentData]) -> Optional[str]:
        """Auto-detect the most likely metric to optimize."""
        metric_counts = {}

        for exp in experiments:
            for metric_name in exp.metrics.keys():
                if isinstance(exp.metrics[metric_name], (int, float)):
                    metric_counts[metric_name] = metric_counts.get(metric_name, 0) + 1

        if not metric_counts:
            return None

        # Return the most common numeric metric
        return max(metric_counts.items(), key=lambda x: x[1])[0]

    def close(self):
        """Clean up resources."""
        if hasattr(self.storage, "close"):
            self.storage.close()


# Global instance for easy access
_default_runner = None


def _get_runner() -> ExperimentRunner:
    """Get or create the default runner instance."""
    global _default_runner
    if _default_runner is None:
        _default_runner = ExperimentRunner()
    return _default_runner


# Convenient module-level functions
def single(experiment_func: Callable, **params) -> str:
    """Run a single experiment. See ExperimentRunner.single()."""
    return _get_runner().single(experiment_func, **params)


def best(metric: Optional[str] = None, top: int = 5) -> List[ExperimentData]:
    """Get best experiments. See ExperimentRunner.best()."""
    return _get_runner().best(metric, top)


def failed() -> List[ExperimentData]:
    """Get failed experiments. See ExperimentRunner.failed()."""
    return _get_runner().failed()


def recent(hours: int = 24) -> List[ExperimentData]:
    """Get recent experiments. See ExperimentRunner.recent()."""
    return _get_runner().recent(hours)


def find(query: str) -> List[ExperimentData]:
    """Find experiments by query. See ExperimentRunner.find()."""
    return _get_runner().find(query)


def compare(experiments: Optional[List[Union[str, ExperimentData]]] = None) -> None:
    """Compare experiments. See ExperimentRunner.compare()."""
    return _get_runner().compare(experiments)


def insights() -> Dict[str, Any]:
    """Get experiment insights. See ExperimentRunner.insights()."""
    return _get_runner().insights()


def suggest() -> Dict[str, Any]:
    """Get experiment suggestions. See ExperimentRunner.suggest()."""
    return _get_runner().suggest()
