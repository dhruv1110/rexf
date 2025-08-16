"""Suggestion system for next experiment recommendations.

This module provides intelligent suggestions for next experiments to run
based on historical data, patterns, and optimization strategies.
"""

import random
from typing import Any, Dict, List, Optional, Tuple

from ..core.models import ExperimentData


class SuggestionEngine:
    """
    Experiment suggestion engine.

    Analyzes experiment history to suggest promising parameter combinations
    and experiment strategies.
    """

    def __init__(self, storage):
        """Initialize with storage backend."""
        self.storage = storage

    def suggest_next_experiments(
        self,
        experiment_func,
        count: int = 3,
        strategy: str = "balanced",
        optimization_target: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Suggest next experiments to run.

        Args:
            experiment_func: Function decorated with @experiment
            count: Number of suggestions to generate
            strategy: Suggestion strategy ("exploit", "explore", "balanced")
            optimization_target: Metric to optimize for

        Returns:
            List of suggested parameter combinations with reasoning
        """
        from ..core.simple_api import get_experiment_metadata

        metadata = get_experiment_metadata(experiment_func)
        experiment_name = metadata["name"]

        # Get historical experiments
        historical_experiments = self.storage.list_experiments(experiment_name)

        if not historical_experiments:
            return self._suggest_initial_experiments(experiment_func, count)

        # Generate suggestions based on strategy
        if strategy == "exploit":
            suggestions = self._exploit_strategy(
                historical_experiments, count, optimization_target
            )
        elif strategy == "explore":
            suggestions = self._explore_strategy(
                historical_experiments, experiment_func, count
            )
        else:  # balanced
            exploit_count = count // 2
            explore_count = count - exploit_count

            exploit_suggestions = self._exploit_strategy(
                historical_experiments, exploit_count, optimization_target
            )
            explore_suggestions = self._explore_strategy(
                historical_experiments, experiment_func, explore_count
            )

            suggestions = exploit_suggestions + explore_suggestions

        return suggestions[:count]

    def suggest_parameter_ranges(self, experiment_func) -> Dict[str, Any]:
        """
        Suggest parameter ranges based on historical data.

        Args:
            experiment_func: Function decorated with @experiment

        Returns:
            Dictionary of parameter ranges with reasoning
        """
        from ..core.simple_api import get_experiment_metadata

        metadata = get_experiment_metadata(experiment_func)
        experiment_name = metadata["name"]

        # Get historical experiments
        historical_experiments = self.storage.list_experiments(experiment_name)

        if not historical_experiments:
            return self._suggest_default_ranges(metadata["auto_params"])

        # Analyze parameter usage
        parameter_usage = {}
        for exp in historical_experiments:
            for param_name, param_value in exp.parameters.items():
                if param_name not in parameter_usage:
                    parameter_usage[param_name] = []
                parameter_usage[param_name].append(param_value)

        # Generate range suggestions
        range_suggestions = {}
        for param_name, values in parameter_usage.items():
            range_suggestions[param_name] = self._analyze_parameter_range(
                param_name, values, historical_experiments
            )

        return range_suggestions

    def suggest_optimization_targets(self, experiment_func) -> List[Dict[str, Any]]:
        """
        Suggest metrics to optimize for based on historical data.

        Args:
            experiment_func: Function decorated with @experiment

        Returns:
            List of suggested optimization targets with reasoning
        """
        from ..core.simple_api import get_experiment_metadata

        metadata = get_experiment_metadata(experiment_func)
        experiment_name = metadata["name"]

        # Get historical experiments
        historical_experiments = self.storage.list_experiments(experiment_name)
        completed_experiments = [
            exp for exp in historical_experiments if exp.status == "completed"
        ]

        if not completed_experiments:
            return [{"metric": "result", "reason": "Default optimization target"}]

        # Analyze metrics
        metric_analysis = {}
        for exp in completed_experiments:
            for metric_name, metric_value in exp.metrics.items():
                if isinstance(metric_value, (int, float)):
                    if metric_name not in metric_analysis:
                        metric_analysis[metric_name] = []
                    metric_analysis[metric_name].append(metric_value)

        # Score metrics for optimization potential
        suggestions = []
        for metric_name, values in metric_analysis.items():
            if len(values) >= 3:
                score = self._score_optimization_potential(metric_name, values)
                suggestions.append(score)

        # Sort by score
        suggestions.sort(key=lambda x: x["score"], reverse=True)
        return suggestions[:5]

    def suggest_unexplored_regions(self, experiment_func) -> List[Dict[str, Any]]:
        """
        Suggest unexplored regions of parameter space.

        Args:
            experiment_func: Function decorated with @experiment

        Returns:
            List of unexplored parameter combinations
        """
        from ..core.simple_api import get_experiment_metadata

        metadata = get_experiment_metadata(experiment_func)
        experiment_name = metadata["name"]

        # Get historical experiments
        historical_experiments = self.storage.list_experiments(experiment_name)

        if not historical_experiments:
            return self._suggest_initial_experiments(experiment_func, 5)

        # Analyze parameter space coverage
        parameter_coverage = self._analyze_parameter_coverage(historical_experiments)

        # Generate unexplored combinations
        unexplored_suggestions = []
        for _ in range(5):  # Generate 5 suggestions
            suggestion = self._generate_unexplored_combination(
                parameter_coverage, historical_experiments
            )
            if suggestion:
                unexplored_suggestions.append(suggestion)

        return unexplored_suggestions

    def _suggest_initial_experiments(
        self, experiment_func, count: int
    ) -> List[Dict[str, Any]]:
        """Suggest initial experiments when no history exists."""
        from ..core.simple_api import get_experiment_metadata

        metadata = get_experiment_metadata(experiment_func)
        auto_params = metadata["auto_params"]

        suggestions = []
        for i in range(count):
            params = {}
            reasoning = "Initial exploration experiment"

            for param_name, param_info in auto_params.items():
                param_type = param_info.get("type", str)
                default_value = param_info.get("default")

                if param_type == int or param_type == "int":
                    if default_value is not None:
                        params[param_name] = default_value * (2 ** (i - count // 2))
                    else:
                        params[param_name] = 2**i
                elif param_type == float or param_type == "float":
                    if default_value is not None:
                        params[param_name] = default_value * (
                            10 ** ((i - count // 2) * 0.5)
                        )
                    else:
                        params[param_name] = 10 ** (i - 2)
                elif param_type == bool or param_type == "bool":
                    params[param_name] = i % 2 == 0
                else:
                    params[param_name] = default_value

            suggestions.append(
                {
                    "parameters": params,
                    "reasoning": reasoning,
                    "strategy": "initial_exploration",
                    "confidence": 0.5,
                }
            )

        return suggestions

    def _exploit_strategy(
        self,
        experiments: List[ExperimentData],
        count: int,
        optimization_target: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Generate suggestions that exploit known good regions."""
        completed_experiments = [
            exp for exp in experiments if exp.status == "completed"
        ]

        if not completed_experiments:
            return []

        # Find best experiments
        if optimization_target:
            # Sort by specific metric
            metric_experiments = [
                exp
                for exp in completed_experiments
                if optimization_target in exp.metrics
                and isinstance(exp.metrics[optimization_target], (int, float))
            ]
            if metric_experiments:
                best_experiments = sorted(
                    metric_experiments,
                    key=lambda x: x.metrics[optimization_target],
                    reverse=True,
                )[
                    : count * 2
                ]  # Get more than needed for variation
            else:
                best_experiments = completed_experiments[: count * 2]
        else:
            # Use general success criteria (no errors, reasonable duration)
            best_experiments = [
                exp
                for exp in completed_experiments
                if exp.duration and exp.duration < 300
            ]
            if not best_experiments:
                best_experiments = completed_experiments
            best_experiments = best_experiments[: count * 2]

        # Generate variations of best experiments
        suggestions = []
        for i in range(count):
            if best_experiments:
                base_experiment = best_experiments[i % len(best_experiments)]
                variation = self._create_parameter_variation(base_experiment.parameters)

                suggestions.append(
                    {
                        "parameters": variation,
                        "reasoning": f"Variation of successful experiment {base_experiment.run_id[:8]}",
                        "strategy": "exploit",
                        "confidence": 0.8,
                        "base_experiment": base_experiment.run_id,
                    }
                )

        return suggestions

    def _explore_strategy(
        self, experiments: List[ExperimentData], experiment_func, count: int
    ) -> List[Dict[str, Any]]:
        """Generate suggestions that explore new regions."""
        # Analyze what has been tried
        tried_parameters = set()
        parameter_ranges = {}

        for exp in experiments:
            param_tuple = tuple(sorted(exp.parameters.items()))
            tried_parameters.add(param_tuple)

            for param_name, param_value in exp.parameters.items():
                if param_name not in parameter_ranges:
                    parameter_ranges[param_name] = []
                parameter_ranges[param_name].append(param_value)

        # Generate new combinations
        suggestions = []
        attempts = 0
        max_attempts = count * 10  # Avoid infinite loops

        while len(suggestions) < count and attempts < max_attempts:
            new_params = self._generate_exploration_parameters(
                parameter_ranges, experiment_func
            )
            param_tuple = tuple(sorted(new_params.items()))

            if param_tuple not in tried_parameters:
                suggestions.append(
                    {
                        "parameters": new_params,
                        "reasoning": "Exploring new parameter region",
                        "strategy": "explore",
                        "confidence": 0.6,
                    }
                )
                tried_parameters.add(param_tuple)

            attempts += 1

        return suggestions

    def _suggest_default_ranges(self, auto_params: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest default parameter ranges when no history exists."""
        ranges = {}

        for param_name, param_info in auto_params.items():
            param_type = param_info.get("type", str)
            default_value = param_info.get("default")

            if param_type == int or param_type == "int":
                if default_value is not None:
                    base = max(1, default_value)
                    ranges[param_name] = {
                        "suggested_range": [base // 2, base, base * 2],
                        "reasoning": f"Range around default value {default_value}",
                    }
                else:
                    ranges[param_name] = {
                        "suggested_range": [1, 2, 4, 8, 16],
                        "reasoning": "Common powers of 2",
                    }
            elif param_type == float or param_type == "float":
                if default_value is not None:
                    ranges[param_name] = {
                        "suggested_range": [
                            default_value * 0.1,
                            default_value,
                            default_value * 10,
                        ],
                        "reasoning": f"Log scale around default value {default_value}",
                    }
                else:
                    ranges[param_name] = {
                        "suggested_range": [0.001, 0.01, 0.1, 1.0],
                        "reasoning": "Common log scale values",
                    }
            elif param_type == bool or param_type == "bool":
                ranges[param_name] = {
                    "suggested_range": [True, False],
                    "reasoning": "Boolean parameter",
                }
            else:
                ranges[param_name] = {
                    "suggested_range": (
                        [default_value] if default_value is not None else ["default"]
                    ),
                    "reasoning": "Single value parameter",
                }

        return ranges

    def _analyze_parameter_range(
        self, param_name: str, values: List[Any], experiments: List[ExperimentData]
    ) -> Dict[str, Any]:
        """Analyze parameter range based on historical usage."""
        unique_values = list(
            set(str(v) for v in values)
        )  # Convert to string for comparison

        # Analyze performance by parameter value
        if len(unique_values) > 1:
            performance_by_value = {}
            for exp in experiments:
                if exp.status == "completed" and param_name in exp.parameters:
                    param_value = str(exp.parameters[param_name])
                    if param_value not in performance_by_value:
                        performance_by_value[param_value] = []

                    # Use duration as a simple performance metric
                    if exp.duration:
                        performance_by_value[param_value].append(exp.duration)

            # Find best performing values
            best_values = []
            for value, durations in performance_by_value.items():
                if durations:
                    avg_duration = sum(durations) / len(durations)
                    best_values.append((value, avg_duration))

            best_values.sort(key=lambda x: x[1])  # Sort by duration (lower is better)

            if best_values:
                return {
                    "current_range": unique_values,
                    "best_values": [v[0] for v in best_values[:3]],
                    "suggested_focus": best_values[0][0],
                    "reasoning": f"Based on {len(experiments)} experiments, best performance with {best_values[0][0]}",
                }

        return {
            "current_range": unique_values,
            "suggested_focus": unique_values[0] if unique_values else None,
            "reasoning": f"Limited data from {len(experiments)} experiments",
        }

    def _score_optimization_potential(
        self, metric_name: str, values: List[float]
    ) -> Dict[str, Any]:
        """Score a metric for its optimization potential."""
        if len(values) < 2:
            return {"metric": metric_name, "score": 0, "reason": "Insufficient data"}

        # Calculate variance (higher variance = more optimization potential)
        mean_val = sum(values) / len(values)
        variance = sum((v - mean_val) ** 2 for v in values) / len(values)
        std_dev = variance**0.5

        # Calculate coefficient of variation
        cv = std_dev / abs(mean_val) if mean_val != 0 else float("inf")

        # Score based on variance and range
        value_range = max(values) - min(values)
        score = cv * value_range

        reason = f"High variance (CV={cv:.3f}) suggests optimization potential"
        if cv < 0.1:
            reason = (
                f"Low variance (CV={cv:.3f}) suggests parameter space well-explored"
            )

        return {
            "metric": metric_name,
            "score": score,
            "reason": reason,
            "variance": variance,
            "range": value_range,
        }

    def _analyze_parameter_coverage(
        self, experiments: List[ExperimentData]
    ) -> Dict[str, Any]:
        """Analyze how well parameter space has been covered."""
        parameter_coverage = {}

        for exp in experiments:
            for param_name, param_value in exp.parameters.items():
                if param_name not in parameter_coverage:
                    parameter_coverage[param_name] = {
                        "values_tried": set(),
                        "value_counts": {},
                        "type": type(param_value).__name__,
                    }

                param_coverage = parameter_coverage[param_name]
                param_coverage["values_tried"].add(str(param_value))

                value_str = str(param_value)
                param_coverage["value_counts"][value_str] = (
                    param_coverage["value_counts"].get(value_str, 0) + 1
                )

        # Convert sets to lists for JSON serialization
        for param_name, coverage in parameter_coverage.items():
            coverage["values_tried"] = list(coverage["values_tried"])

        return parameter_coverage

    def _generate_unexplored_combination(
        self, parameter_coverage: Dict[str, Any], experiments: List[ExperimentData]
    ) -> Optional[Dict[str, Any]]:
        """Generate a parameter combination in unexplored regions."""
        if not parameter_coverage:
            return None

        # Generate new parameter values
        new_params = {}
        for param_name, coverage in parameter_coverage.items():
            param_type = coverage["type"]
            tried_values = coverage["values_tried"]

            if param_type in ["int", "float"]:
                # For numeric parameters, try values between or outside tried values
                numeric_values = []
                for value_str in tried_values:
                    try:
                        if param_type == "int":
                            numeric_values.append(int(value_str))
                        else:
                            numeric_values.append(float(value_str))
                    except ValueError:
                        continue

                if numeric_values:
                    numeric_values.sort()
                    # Try value between min and max
                    if len(numeric_values) > 1:
                        min_val, max_val = min(numeric_values), max(numeric_values)
                        if param_type == "int":
                            new_value = (min_val + max_val) // 2
                            if str(new_value) not in tried_values:
                                new_params[param_name] = new_value
                            else:
                                new_params[param_name] = max_val + 1
                        else:
                            new_value = (min_val + max_val) / 2
                            new_params[param_name] = new_value
                    else:
                        # Only one value tried, try a different scale
                        base_value = numeric_values[0]
                        if param_type == "int":
                            new_params[param_name] = base_value * 2
                        else:
                            new_params[param_name] = base_value * 1.5
                else:
                    # No valid numeric values found
                    new_params[param_name] = 1 if param_type == "int" else 1.0
            elif param_type == "bool":
                # For boolean, try the opposite if only one value tried
                if len(tried_values) == 1:
                    tried_bool = tried_values[0].lower() == "true"
                    new_params[param_name] = not tried_bool
                else:
                    new_params[param_name] = random.choice([True, False])
            else:
                # For other types, use a default
                new_params[param_name] = f"unexplored_{random.randint(1, 100)}"

        if new_params:
            return {
                "parameters": new_params,
                "reasoning": "Exploring completely unexplored parameter region",
                "strategy": "unexplored",
                "confidence": 0.4,
            }

        return None

    def _create_parameter_variation(
        self, base_parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a small variation of base parameters."""
        variation = base_parameters.copy()

        # Randomly modify 1-2 parameters
        param_names = list(base_parameters.keys())
        params_to_modify = random.sample(param_names, min(2, len(param_names)))

        for param_name in params_to_modify:
            current_value = base_parameters[param_name]

            if isinstance(current_value, int):
                # Small integer variation
                variation[param_name] = max(
                    1, current_value + random.choice([-1, 0, 1])
                )
            elif isinstance(current_value, float):
                # Small float variation (±20%)
                perturbation = current_value * random.uniform(-0.2, 0.2)
                variation[param_name] = current_value + perturbation
            elif isinstance(current_value, bool):
                # Sometimes flip boolean
                if random.random() < 0.3:  # 30% chance to flip
                    variation[param_name] = not current_value
            # For other types, keep the same value

        return variation

    def _generate_exploration_parameters(
        self, parameter_ranges: Dict[str, List[Any]], experiment_func
    ) -> Dict[str, Any]:
        """Generate parameters for exploration strategy."""
        from ..core.simple_api import get_experiment_metadata

        metadata = get_experiment_metadata(experiment_func)
        auto_params = metadata["auto_params"]

        new_params = {}

        for param_name, param_info in auto_params.items():
            param_type = param_info.get("type", str)

            if param_name in parameter_ranges:
                # Use information from historical data
                tried_values = parameter_ranges[param_name]

                if param_type == int or param_type == "int":
                    numeric_values = [v for v in tried_values if isinstance(v, int)]
                    if numeric_values:
                        # Try value outside current range
                        min_val, max_val = min(numeric_values), max(numeric_values)
                        new_params[param_name] = random.choice(
                            [
                                max(1, min_val - 1),
                                max_val + 1,
                                (min_val + max_val) // 2,
                            ]
                        )
                    else:
                        new_params[param_name] = random.randint(1, 10)
                elif param_type == float or param_type == "float":
                    numeric_values = [
                        v for v in tried_values if isinstance(v, (int, float))
                    ]
                    if numeric_values:
                        min_val, max_val = min(numeric_values), max(numeric_values)
                        range_size = max_val - min_val
                        new_params[param_name] = random.choice(
                            [
                                min_val - range_size * 0.5,
                                max_val + range_size * 0.5,
                                min_val + range_size * random.random(),
                            ]
                        )
                    else:
                        new_params[param_name] = random.uniform(0.001, 10.0)
                elif param_type == bool or param_type == "bool":
                    new_params[param_name] = random.choice([True, False])
                else:
                    new_params[param_name] = f"explore_{random.randint(1, 100)}"
            else:
                # No historical data, use defaults
                default_value = param_info.get("default")
                if default_value is not None:
                    new_params[param_name] = default_value
                else:
                    if param_type == int or param_type == "int":
                        new_params[param_name] = random.randint(1, 10)
                    elif param_type == float or param_type == "float":
                        new_params[param_name] = random.uniform(0.001, 10.0)
                    elif param_type == bool or param_type == "bool":
                        new_params[param_name] = random.choice([True, False])
                    else:
                        new_params[param_name] = "default"

        return new_params
