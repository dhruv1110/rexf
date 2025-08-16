"""Experiment runner that orchestrates experiment execution."""

import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from .interfaces import ArtifactManagerInterface, StorageInterface
from .models import ExperimentData, ExperimentRun, ReproducibilityTracker


class ExperimentRunner:
    """
    Orchestrates experiment execution with pluggable storage and artifact management.

    This is the main class for running experiments. It uses the plugin architecture
    to allow different storage backends and artifact managers.
    """

    def __init__(
        self,
        storage: Optional[StorageInterface] = None,
        artifact_manager: Optional[ArtifactManagerInterface] = None,
        **kwargs,
    ):
        """
        Initialize experiment runner.

        Args:
            storage: Storage implementation (will use default SQLite if None)
            artifact_manager: Artifact manager implementation (will use default FileSystem if None)
            **kwargs: Additional arguments passed to default implementations
        """
        # Import default implementations if not provided
        if storage is None:
            from ..backends.sqlite_storage import SQLiteStorage

            storage_path = kwargs.get("storage_path", "experiments.db")
            storage = SQLiteStorage(storage_path)

        if artifact_manager is None:
            from ..backends.filesystem_artifacts import FileSystemArtifactManager

            artifacts_path = kwargs.get("artifacts_path", "artifacts")
            artifact_manager = FileSystemArtifactManager(artifacts_path)

        self.storage = storage
        self.artifact_manager = artifact_manager
        self.tracker = ReproducibilityTracker()

    def run(self, experiment_func: Callable, **parameters) -> str:
        """
        Run an experiment function.

        Args:
            experiment_func: Function decorated with @experiment_config
            **parameters: Experiment parameters

        Returns:
            Run ID of the executed experiment
        """
        # Get experiment configuration from decorated function
        if not hasattr(experiment_func, "_experiment_config"):
            raise ValueError(
                "Function must be decorated with @experiment_config or @ExperimentBuilder.build"
            )

        config = experiment_func._experiment_config
        run_id = str(uuid.uuid4())

        # Set up seed if specified
        seed_param = config.get("seed")
        if seed_param and seed_param in parameters:
            seed_value = parameters[seed_param]
            self._set_random_seed(seed_value)
            print(f"Set random seed: {seed_param} = {seed_value}")

        # Create experiment run
        start_time = datetime.now()
        experiment = ExperimentRun(
            run_id=run_id,
            experiment_name=config["name"],
            parameters=parameters,
            start_time=start_time,
            git_commit=self.tracker.get_git_commit(),
            git_status=self.tracker.get_git_status(),
            environment_info=self.tracker.get_environment_info(),
            random_seed=parameters.get(seed_param) if seed_param else None,
        )

        print(f"Running experiment '{config['name']}' (run_id: {run_id})")

        try:
            # Execute the experiment
            result = experiment_func(**parameters)

            # Update experiment with results
            experiment.end_time = datetime.now()
            experiment.status = "completed"

            # Process results
            if isinstance(result, dict):
                # Separate metrics, results, and artifacts based on configuration
                for key, value in result.items():
                    if key in config["metrics"]:
                        experiment.metrics[key] = value
                    elif key in config["results"]:
                        experiment.results[key] = value
                    else:
                        # Default to results if not specified
                        experiment.results[key] = value

            # Handle artifacts (files that may have been created)
            for artifact_name, (filename, description) in config["artifacts"].items():
                try:
                    # Check if artifact file exists and store it
                    artifact_path = self.artifact_manager.store_artifact(
                        run_id, artifact_name, filename, {"description": description}
                    )
                    experiment.artifacts[artifact_name] = artifact_path
                except Exception as e:
                    print(f"Warning: Could not store artifact '{artifact_name}': {e}")

            # Save experiment to storage
            self.storage.save_experiment(ExperimentData(experiment))

            total_time = (experiment.end_time - experiment.start_time).total_seconds()
            print(f"Experiment completed in {total_time:.2f} seconds")

            return run_id

        except Exception as e:
            # Mark experiment as failed
            experiment.end_time = datetime.now()
            experiment.status = "failed"
            experiment.metadata["error"] = str(e)

            # Still save the failed experiment for debugging
            try:
                self.storage.save_experiment(ExperimentData(experiment))
            except Exception:
                pass

            print(f"Experiment failed: {e}")
            raise

    def get_experiment(self, run_id: str) -> Optional[ExperimentData]:
        """Get experiment by run ID."""
        return self.storage.load_experiment(run_id)

    def list_experiments(
        self, experiment_name: Optional[str] = None, limit: Optional[int] = None
    ) -> List[ExperimentData]:
        """List experiments."""
        return self.storage.list_experiments(experiment_name, limit)

    def compare_runs(self, run_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple experiment runs."""
        experiments = []
        for run_id in run_ids:
            exp = self.get_experiment(run_id)
            if exp:
                experiments.append(exp)

        if not experiments:
            return {}

        return {
            "experiments": experiments,
            "parameter_comparison": self._compare_parameters(experiments),
            "metric_comparison": self._compare_metrics(experiments),
            "result_comparison": self._compare_results(experiments),
            "performance_comparison": self._compare_performance(experiments),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get storage and system statistics."""
        storage_stats = self.storage.get_storage_stats()
        return {
            "storage": storage_stats,
            "tracker": {
                "git_available": self.tracker._git_repo is not None,
                "current_commit": self.tracker.get_git_commit(),
            },
        }

    def close(self) -> None:
        """Close all resources and clean up connections."""
        if hasattr(self.storage, "close"):
            self.storage.close()

    def _set_random_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        import random

        random.seed(seed)

        try:
            import numpy as np

            np.random.seed(seed)
        except ImportError:
            pass

    def _compare_parameters(self, experiments: List[ExperimentData]) -> Dict[str, Any]:
        """Compare parameters across experiments."""
        all_params = set()
        for exp in experiments:
            all_params.update(exp.parameters.keys())

        comparison = {}
        for param in all_params:
            values = []
            for exp in experiments:
                values.append(exp.parameters.get(param, None))
            comparison[param] = {
                "values": values,
                "unique_values": list(set(v for v in values if v is not None)),
                "varies": len(set(v for v in values if v is not None)) > 1,
            }
        return comparison

    def _compare_metrics(self, experiments: List[ExperimentData]) -> Dict[str, Any]:
        """Compare metrics across experiments."""
        all_metrics = set()
        for exp in experiments:
            all_metrics.update(exp.metrics.keys())

        comparison = {}
        for metric in all_metrics:
            values = []
            for exp in experiments:
                values.append(exp.metrics.get(metric, None))

            numeric_values = [v for v in values if isinstance(v, (int, float))]
            comparison[metric] = {
                "values": values,
                "min": min(numeric_values) if numeric_values else None,
                "max": max(numeric_values) if numeric_values else None,
                "mean": (
                    sum(numeric_values) / len(numeric_values)
                    if numeric_values
                    else None
                ),
            }
        return comparison

    def _compare_results(self, experiments: List[ExperimentData]) -> Dict[str, Any]:
        """Compare results across experiments."""
        all_results = set()
        for exp in experiments:
            all_results.update(exp.results.keys())

        comparison = {}
        for result in all_results:
            values = []
            for exp in experiments:
                values.append(exp.results.get(result, None))
            comparison[result] = {"values": values}
        return comparison

    def _compare_performance(self, experiments: List[ExperimentData]) -> Dict[str, Any]:
        """Compare performance metrics across experiments."""
        durations = []
        for exp in experiments:
            if exp.duration:
                durations.append(exp.duration)

        return {
            "durations": durations,
            "avg_duration": sum(durations) / len(durations) if durations else None,
            "min_duration": min(durations) if durations else None,
            "max_duration": max(durations) if durations else None,
        }
