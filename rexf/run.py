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

    def __init__(self, storage_path: str = "experiments.db", intelligent: bool = True):
        """Initialize with minimal configuration."""
        # Always use intelligent storage - it's the only storage we have now
        from .backends.intelligent_storage import IntelligentStorage

        self.storage = IntelligentStorage(storage_path)
        self._intelligent = intelligent

        self._current_experiments = []

        # Initialize intelligence modules if available
        if self._intelligent:
            try:
                from .intelligence.exploration import ExplorationEngine
                from .intelligence.insights import InsightsEngine
                from .intelligence.queries import SmartQueryEngine
                from .intelligence.smart_compare import SmartComparer
                from .intelligence.suggestions import SuggestionEngine

                self.query_engine = SmartQueryEngine(self.storage)
                self.smart_comparer = SmartComparer()
                self.exploration_engine = ExplorationEngine(self.storage)
                self.insights_engine = InsightsEngine(self.storage)
                self.suggestion_engine = SuggestionEngine(self.storage)
            except ImportError:
                self._intelligent = False

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

            # Return the run_id even for failed experiments
            # This allows users to analyze failures without handling exceptions
            return run_id

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
        Find experiments using query expressions.

        Supports queries like:
        - "accuracy > 0.9"
        - "status == 'completed'"
        - "runtime < 60"
        - "param_learning_rate between 0.001 and 0.01"

        Args:
            query: Query expression string

        Returns:
            List of matching experiments
        """
        if self._intelligent and hasattr(self, "query_engine"):
            print(f"🔍 Query: {query}")
            explanation = self.query_engine.explain_query(query)
            print(f"💭 Interpretation: {explanation}")
            return self.query_engine.query(query)
        else:
            # Fallback to basic search
            print(f"⚠️ Basic search mode for: '{query}'")
            print("📋 Use enhanced storage for advanced queries")
            return self.storage.list_experiments()

    def compare(
        self, experiments: Optional[List[Union[str, ExperimentData]]] = None
    ) -> None:
        """
        Smart comparison of experiments with detailed analysis.

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

        # Use smart comparison if available
        if self._intelligent and hasattr(self, "smart_comparer"):
            comparison = self.smart_comparer.compare_experiments(experiments)
            report = self.smart_comparer.format_comparison_report(comparison)
            print(report)
        else:
            # Fallback to basic comparison
            print(f"📊 Comparing {len(experiments)} experiments:")
            print("-" * 50)

            for i, exp in enumerate(experiments, 1):
                duration = exp.duration or 0
                print(f"{i}. {exp.experiment_name} ({exp.run_id[:8]}...)")
                print(f"   Status: {exp.status}, Duration: {duration:.2f}s")
                print(f"   Parameters: {exp.parameters}")
                print(f"   Metrics: {exp.metrics}")
                print()

    def insights(self, experiment_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive insights from experiments.

        Args:
            experiment_name: Optional experiment name to filter by

        Returns:
            Dictionary of insights and recommendations
        """
        if self._intelligent and hasattr(self, "insights_engine"):
            return self.insights_engine.generate_insights(experiment_name)
        else:
            # Fallback to basic statistics
            all_experiments = self.storage.list_experiments(experiment_name)

            if not all_experiments:
                return {"message": "No experiments found. Run some experiments first!"}

            total = len(all_experiments)
            completed = len(
                [exp for exp in all_experiments if exp.status == "completed"]
            )
            failed = total - completed

            return {
                "total_experiments": total,
                "completed": completed,
                "failed": failed,
                "success_rate": completed / total if total > 0 else 0,
                "message": "🔍 Basic insights. Use enhanced storage for advanced analytics.",
            }

    def suggest(
        self,
        experiment_func=None,
        count: int = 3,
        strategy: str = "balanced",
        optimization_target: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Suggest next experiments to run.

        Args:
            experiment_func: Function decorated with @experiment
            count: Number of suggestions to generate
            strategy: Suggestion strategy ("exploit", "explore", "balanced")
            optimization_target: Metric to optimize for

        Returns:
            Dictionary with parameter suggestions and reasoning
        """
        if self._intelligent and hasattr(self, "suggestion_engine") and experiment_func:
            suggestions = self.suggestion_engine.suggest_next_experiments(
                experiment_func, count, strategy, optimization_target
            )
            return {
                "suggestions": suggestions,
                "count": len(suggestions),
                "strategy": strategy,
                "optimization_target": optimization_target,
            }
        else:
            # Fallback suggestions
            fallback = {
                "message": "🤖 For experiment-specific suggestions, provide experiment_func parameter",
                "current_status": "Use run.best() to see your top experiments for now",
            }

            # Add query suggestions if available
            if self._intelligent and hasattr(self, "query_engine"):
                query_suggestions = self.query_engine.get_query_suggestions()
                fallback["example_queries"] = query_suggestions[:5]
                fallback["tip"] = "Try these queries with run.find()"

            return fallback

    def auto_explore(
        self,
        experiment_func,
        strategy: str = "random",
        budget: int = 10,
        parameter_ranges: Optional[Dict[str, Any]] = None,
        optimization_target: Optional[str] = None,
    ) -> List[str]:
        """
        Automatically explore parameter space for an experiment.

        Args:
            experiment_func: Function decorated with @experiment
            strategy: Exploration strategy ("random", "grid", "adaptive")
            budget: Number of experiments to run
            parameter_ranges: Dict of parameter name -> range specification
            optimization_target: Metric to optimize for

        Returns:
            List of experiment run IDs
        """
        if self._intelligent and hasattr(self, "exploration_engine"):
            return self.exploration_engine.auto_explore(
                experiment_func, strategy, budget, parameter_ranges, optimization_target
            )
        else:
            print("⚠️ Auto-exploration requires enhanced storage backend")
            return []

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

    def parameter_space(self, experiment_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get parameter space summary for experiments.

        Args:
            experiment_name: Optional experiment name to filter by

        Returns:
            Dictionary with parameter space analysis
        """
        if self._intelligent and hasattr(self.storage, "get_parameter_space_summary"):
            return self.storage.get_parameter_space_summary(experiment_name)
        else:
            return {"message": "Parameter space analysis requires enhanced storage"}

    def metric_trends(
        self, metric_name: str, experiment_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get trends for a specific metric over time.

        Args:
            metric_name: Name of the metric to analyze
            experiment_name: Optional experiment name to filter by

        Returns:
            Dictionary with trend analysis
        """
        if self._intelligent and hasattr(self.storage, "get_metric_trends"):
            return self.storage.get_metric_trends(metric_name, experiment_name)
        else:
            return {"message": "Metric trends analysis requires enhanced storage"}

    def query_help(self) -> None:
        """Show help for query expressions."""
        print("🔍 QUERY EXPRESSION HELP")
        print("=" * 40)
        print("Examples:")
        print("  run.find('accuracy > 0.9')")
        print("  run.find('learning_rate between 0.001 and 0.01')")
        print("  run.find('status == completed and duration < 60')")
        print("  run.find('param_batch_size > 32')")
        print("")
        print("Supported operators: >, >=, <, <=, ==, !=")
        print("Supported keywords: between, and")
        print("")
        print("Field types:")
        print("  - Metrics: accuracy, loss, error, f1_score, etc.")
        print("  - Parameters: prefix with 'param_' (e.g., param_learning_rate)")
        print("  - General: status, duration, experiment_name")
        print("")

        if self._intelligent and hasattr(self, "query_engine"):
            suggestions = self.query_engine.get_query_suggestions()
            if suggestions:
                print("Try these queries:")
                for suggestion in suggestions[:5]:
                    print(f"  run.find('{suggestion}')")

    def dashboard(
        self,
        host: str = "localhost",
        port: int = 8080,
        open_browser: bool = True,
    ) -> None:
        """
        Launch web dashboard for experiment visualization.

        Args:
            host: Host to bind the server to
            port: Port to bind the server to
            open_browser: Whether to automatically open the browser

        Returns:
            None
        """
        try:
            from ..dashboard.app import run_dashboard

            # Get the storage path
            storage_path = getattr(self.storage, "db_path", "experiments.db")

            run_dashboard(
                storage_path=str(storage_path),
                host=host,
                port=port,
                open_browser=open_browser,
            )
        except ImportError:
            print("⚠️ Dashboard requires additional dependencies")
            print("📝 Run: pip install rexf[dashboard]")

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


def insights(experiment_name: Optional[str] = None) -> Dict[str, Any]:
    """Get experiment insights. See ExperimentRunner.insights()."""
    return _get_runner().insights(experiment_name)


def suggest(
    experiment_func=None,
    count: int = 3,
    strategy: str = "balanced",
    optimization_target: Optional[str] = None,
) -> Dict[str, Any]:
    """Get experiment suggestions. See ExperimentRunner.suggest()."""
    return _get_runner().suggest(experiment_func, count, strategy, optimization_target)


def parameter_space(experiment_name: Optional[str] = None) -> Dict[str, Any]:
    """Get parameter space analysis. See ExperimentRunner.parameter_space()."""
    return _get_runner().parameter_space(experiment_name)


def metric_trends(
    metric_name: str, experiment_name: Optional[str] = None
) -> Dict[str, Any]:
    """Get metric trends. See ExperimentRunner.metric_trends()."""
    return _get_runner().metric_trends(metric_name, experiment_name)


def query_help() -> None:
    """Show query help. See ExperimentRunner.query_help()."""
    return _get_runner().query_help()


def auto_explore(
    experiment_func,
    strategy: str = "random",
    budget: int = 10,
    parameter_ranges: Optional[Dict[str, Any]] = None,
    optimization_target: Optional[str] = None,
) -> List[str]:
    """Auto-explore parameter space. See ExperimentRunner.auto_explore()."""
    return _get_runner().auto_explore(
        experiment_func, strategy, budget, parameter_ranges, optimization_target
    )


def dashboard(
    host: str = "localhost",
    port: int = 8080,
    open_browser: bool = True,
) -> None:
    """Launch web dashboard. See ExperimentRunner.dashboard()."""
    return _get_runner().dashboard(host, port, open_browser)
