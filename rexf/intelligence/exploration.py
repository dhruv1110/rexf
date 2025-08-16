"""Auto-exploration engine for parameter space exploration.

This module provides automated parameter space exploration strategies
including grid search, random search, and basic optimization.
"""

import itertools
import random
from typing import Any, Dict, List, Optional, Tuple, Union

from ..core.models import ExperimentData
from ..core.simple_api import extract_parameter_values, get_experiment_metadata


class ExplorationEngine:
    """
    Parameter space exploration engine.

    Provides various strategies for automatically exploring parameter spaces
    to find optimal configurations.
    """

    def __init__(self, storage):
        """Initialize with storage backend."""
        self.storage = storage

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
        metadata = get_experiment_metadata(experiment_func)
        auto_params = metadata["auto_params"]

        # Auto-detect parameter ranges if not provided
        if parameter_ranges is None:
            parameter_ranges = self._auto_detect_ranges(auto_params)

        # Generate parameter combinations based on strategy
        if strategy == "grid":
            param_combinations = self._grid_search(parameter_ranges, budget)
        elif strategy == "random":
            param_combinations = self._random_search(parameter_ranges, budget)
        elif strategy == "adaptive":
            param_combinations = self._adaptive_search(
                experiment_func, parameter_ranges, budget, optimization_target
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Run experiments
        run_ids = []
        for params in param_combinations:
            # Use the simple run interface
            from ..run import single

            run_id = single(experiment_func, **params)
            run_ids.append(run_id)

            print(f"🔬 Exploration {len(run_ids)}/{len(param_combinations)}: {params}")

        return run_ids

    def _auto_detect_ranges(self, auto_params: Dict[str, Any]) -> Dict[str, Any]:
        """Auto-detect reasonable parameter ranges."""
        ranges = {}

        for param_name, param_info in auto_params.items():
            param_type = param_info.get("type", str)
            default_value = param_info.get("default")

            if param_type == int or param_type == "int":
                if default_value is not None:
                    # Create range around default value
                    base = max(1, default_value)
                    ranges[param_name] = list(range(max(1, base // 2), base * 2 + 1))
                else:
                    ranges[param_name] = [1, 2, 4, 8, 16, 32]

            elif param_type == float or param_type == "float":
                if default_value is not None:
                    # Create range around default value
                    base = default_value
                    ranges[param_name] = [
                        base * 0.1,
                        base * 0.5,
                        base,
                        base * 2.0,
                        base * 5.0,
                    ]
                else:
                    ranges[param_name] = [0.001, 0.01, 0.1, 1.0, 10.0]

            elif param_type == bool or param_type == "bool":
                ranges[param_name] = [True, False]

            elif param_type == str or param_type == "str":
                # For strings, use default or common values
                if default_value is not None:
                    ranges[param_name] = [default_value]
                else:
                    ranges[param_name] = ["default"]
            else:
                # For unknown types, use default if available
                if default_value is not None:
                    ranges[param_name] = [default_value]
                else:
                    ranges[param_name] = [None]

        return ranges

    def _grid_search(
        self, parameter_ranges: Dict[str, Any], budget: int
    ) -> List[Dict[str, Any]]:
        """Generate parameter combinations using grid search."""
        param_names = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())

        # Generate all combinations
        all_combinations = list(itertools.product(*param_values))

        # Limit to budget
        if len(all_combinations) > budget:
            # Take evenly spaced combinations
            step = len(all_combinations) / budget
            selected_indices = [int(i * step) for i in range(budget)]
            all_combinations = [all_combinations[i] for i in selected_indices]

        # Convert to dictionaries
        combinations = []
        for combo in all_combinations:
            param_dict = dict(zip(param_names, combo))
            combinations.append(param_dict)

        return combinations

    def _random_search(
        self, parameter_ranges: Dict[str, Any], budget: int
    ) -> List[Dict[str, Any]]:
        """Generate parameter combinations using random search."""
        combinations = []

        for _ in range(budget):
            param_dict = {}
            for param_name, param_range in parameter_ranges.items():
                if isinstance(param_range, list):
                    param_dict[param_name] = random.choice(param_range)
                elif isinstance(param_range, tuple) and len(param_range) == 2:
                    # Assume (min, max) tuple for numeric ranges
                    min_val, max_val = param_range
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        param_dict[param_name] = random.randint(min_val, max_val)
                    else:
                        param_dict[param_name] = random.uniform(min_val, max_val)
                else:
                    # Single value
                    param_dict[param_name] = param_range

            combinations.append(param_dict)

        return combinations

    def _adaptive_search(
        self,
        experiment_func,
        parameter_ranges: Dict[str, Any],
        budget: int,
        optimization_target: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Generate parameter combinations using adaptive search."""
        # Start with random search for initial exploration
        initial_budget = min(budget // 2, 5)
        combinations = self._random_search(parameter_ranges, initial_budget)

        # For the remaining budget, use simple hill climbing
        remaining_budget = budget - initial_budget

        if remaining_budget > 0 and optimization_target:
            # Run initial experiments to get feedback
            from ..run import single

            initial_results = []
            for params in combinations:
                run_id = single(experiment_func, **params)
                experiment = self.storage.load_experiment(run_id)
                if experiment and optimization_target in experiment.metrics:
                    initial_results.append(
                        (params, experiment.metrics[optimization_target])
                    )

            # Find best performing parameters
            if initial_results:
                initial_results.sort(
                    key=lambda x: x[1], reverse=True
                )  # Assume higher is better
                best_params = initial_results[0][0]

                # Generate variations around best parameters
                for _ in range(remaining_budget):
                    variation = self._create_variation(best_params, parameter_ranges)
                    combinations.append(variation)

        return combinations

    def _create_variation(
        self, base_params: Dict[str, Any], parameter_ranges: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a variation of base parameters."""
        variation = base_params.copy()

        # Randomly modify 1-2 parameters
        params_to_modify = random.sample(
            list(base_params.keys()), min(2, len(base_params))
        )

        for param_name in params_to_modify:
            param_range = parameter_ranges[param_name]
            current_value = base_params[param_name]

            if isinstance(param_range, list):
                # Choose a nearby value
                current_index = (
                    param_range.index(current_value)
                    if current_value in param_range
                    else 0
                )
                # Choose index within +/- 1 of current
                new_index = max(
                    0,
                    min(
                        len(param_range) - 1, current_index + random.choice([-1, 0, 1])
                    ),
                )
                variation[param_name] = param_range[new_index]
            elif isinstance(param_range, tuple) and len(param_range) == 2:
                # Create small perturbation
                min_val, max_val = param_range
                if isinstance(current_value, (int, float)):
                    range_size = max_val - min_val
                    perturbation = random.uniform(-0.1 * range_size, 0.1 * range_size)
                    new_value = max(min_val, min(max_val, current_value + perturbation))
                    variation[param_name] = type(current_value)(new_value)

        return variation

    def get_exploration_summary(self, run_ids: List[str]) -> Dict[str, Any]:
        """Get summary of exploration results."""
        if not run_ids:
            return {"message": "No experiments to analyze"}

        experiments = []
        for run_id in run_ids:
            exp = self.storage.load_experiment(run_id)
            if exp:
                experiments.append(exp)

        if not experiments:
            return {"message": "No experiments found"}

        # Analyze results
        all_metrics = {}
        all_params = {}

        for exp in experiments:
            for metric_name, metric_value in exp.metrics.items():
                if isinstance(metric_value, (int, float)):
                    if metric_name not in all_metrics:
                        all_metrics[metric_name] = []
                    all_metrics[metric_name].append(metric_value)

            for param_name, param_value in exp.parameters.items():
                if param_name not in all_params:
                    all_params[param_name] = []
                all_params[param_name].append(param_value)

        # Find best experiment for each metric
        best_experiments = {}
        for metric_name, values in all_metrics.items():
            if values:
                best_value = max(values)  # Assume higher is better
                for exp in experiments:
                    if exp.metrics.get(metric_name) == best_value:
                        best_experiments[metric_name] = {
                            "run_id": exp.run_id,
                            "value": best_value,
                            "parameters": exp.parameters,
                        }
                        break

        return {
            "total_experiments": len(experiments),
            "metrics_explored": list(all_metrics.keys()),
            "parameters_explored": list(all_params.keys()),
            "best_by_metric": best_experiments,
            "metric_ranges": {
                name: {
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values),
                }
                for name, values in all_metrics.items()
            },
        }
