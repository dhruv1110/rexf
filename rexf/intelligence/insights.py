"""Insights engine for pattern detection and analysis.

This module provides automated pattern detection and insight generation
from experiment data using statistical analysis and rule-based approaches.
"""

import statistics
from typing import Any, Dict, List, Optional, Tuple

from ..core.models import ExperimentData


class InsightsEngine:
    """
    Pattern detection and insights generation engine.

    Analyzes experiment data to find patterns, correlations,
    and generate actionable insights.
    """

    def __init__(self, storage):
        """Initialize with storage backend."""
        self.storage = storage

    def generate_insights(
        self, experiment_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive insights from experiment data.

        Args:
            experiment_name: Optional experiment name to filter by

        Returns:
            Dictionary containing various insights and patterns
        """
        experiments = self.storage.list_experiments(experiment_name)

        if not experiments:
            return {"message": "No experiments found for analysis"}

        insights = {
            "summary": self._generate_summary_insights(experiments),
            "parameter_insights": self._analyze_parameters(experiments),
            "performance_insights": self._analyze_performance(experiments),
            "correlation_insights": self._analyze_correlations(experiments),
            "anomaly_insights": self._detect_anomalies(experiments),
            "recommendations": self._generate_recommendations(experiments),
        }

        return insights

    def _generate_summary_insights(
        self, experiments: List[ExperimentData]
    ) -> Dict[str, Any]:
        """Generate high-level summary insights."""
        total = len(experiments)
        completed = len([exp for exp in experiments if exp.status == "completed"])
        failed = total - completed

        # Calculate success patterns
        success_rate = completed / total if total > 0 else 0

        # Time analysis
        durations = [
            exp.duration for exp in experiments if exp.duration and exp.duration > 0
        ]
        avg_duration = sum(durations) / len(durations) if durations else 0

        # Recent trend
        recent_experiments = sorted(experiments, key=lambda x: x.start_time)[-5:]
        recent_success_rate = len(
            [exp for exp in recent_experiments if exp.status == "completed"]
        ) / len(recent_experiments)

        trend = (
            "improving"
            if recent_success_rate > success_rate
            else "stable" if recent_success_rate == success_rate else "declining"
        )

        return {
            "total_experiments": total,
            "success_rate": success_rate,
            "avg_duration": avg_duration,
            "recent_trend": trend,
            "completion_insights": self._analyze_completion_patterns(experiments),
        }

    def _analyze_parameters(self, experiments: List[ExperimentData]) -> Dict[str, Any]:
        """Analyze parameter patterns and effectiveness."""
        parameter_analysis = {}

        # Collect all parameter data
        param_metrics = {}
        for exp in experiments:
            if exp.status == "completed":
                for param_name, param_value in exp.parameters.items():
                    if param_name not in param_metrics:
                        param_metrics[param_name] = []

                    # Store parameter value with associated metrics
                    param_metrics[param_name].append(
                        {
                            "value": param_value,
                            "metrics": exp.metrics,
                            "duration": exp.duration,
                        }
                    )

        # Analyze each parameter
        for param_name, param_data in param_metrics.items():
            analysis = self._analyze_single_parameter(param_name, param_data)
            parameter_analysis[param_name] = analysis

        # Find most impactful parameters
        most_impactful = self._find_most_impactful_parameters(parameter_analysis)

        return {
            "individual_parameters": parameter_analysis,
            "most_impactful": most_impactful,
            "parameter_stability": self._analyze_parameter_stability(param_metrics),
        }

    def _analyze_single_parameter(
        self, param_name: str, param_data: List[Dict]
    ) -> Dict[str, Any]:
        """Analyze a single parameter's impact."""
        if not param_data:
            return {"message": "No data available"}

        # Group by parameter value
        value_groups = {}
        for entry in param_data:
            value = entry["value"]
            if value not in value_groups:
                value_groups[value] = []
            value_groups[value].append(entry)

        # Analyze impact on metrics
        metric_impact = {}
        for metric_name in param_data[0]["metrics"].keys():
            metric_values_by_param = {}
            for value, entries in value_groups.items():
                metric_values = [entry["metrics"].get(metric_name) for entry in entries]
                metric_values = [
                    v for v in metric_values if isinstance(v, (int, float))
                ]
                if metric_values:
                    metric_values_by_param[value] = {
                        "avg": sum(metric_values) / len(metric_values),
                        "min": min(metric_values),
                        "max": max(metric_values),
                        "count": len(metric_values),
                    }

            if len(metric_values_by_param) > 1:
                # Find best and worst performing values
                best_value = max(
                    metric_values_by_param.items(), key=lambda x: x[1]["avg"]
                )
                worst_value = min(
                    metric_values_by_param.items(), key=lambda x: x[1]["avg"]
                )

                improvement = best_value[1]["avg"] - worst_value[1]["avg"]

                metric_impact[metric_name] = {
                    "best_value": best_value[0],
                    "best_avg": best_value[1]["avg"],
                    "worst_value": worst_value[0],
                    "worst_avg": worst_value[1]["avg"],
                    "improvement": improvement,
                    "all_values": metric_values_by_param,
                }

        # Analyze impact on runtime
        runtime_impact = self._analyze_runtime_impact(value_groups)

        return {
            "total_experiments": len(param_data),
            "unique_values": len(value_groups),
            "metric_impact": metric_impact,
            "runtime_impact": runtime_impact,
            "recommendations": self._generate_parameter_recommendations(
                param_name, metric_impact, runtime_impact
            ),
        }

    def _analyze_performance(self, experiments: List[ExperimentData]) -> Dict[str, Any]:
        """Analyze performance patterns."""
        completed_experiments = [
            exp for exp in experiments if exp.status == "completed"
        ]

        if not completed_experiments:
            return {"message": "No completed experiments to analyze"}

        # Runtime analysis
        durations = [exp.duration for exp in completed_experiments if exp.duration]

        runtime_analysis = {}
        if durations:
            runtime_analysis = {
                "avg_duration": sum(durations) / len(durations),
                "min_duration": min(durations),
                "max_duration": max(durations),
                "median_duration": statistics.median(durations),
                "duration_trend": self._analyze_duration_trend(completed_experiments),
            }

        # Success rate analysis
        success_analysis = self._analyze_success_patterns(experiments)

        # Resource efficiency
        efficiency_analysis = self._analyze_efficiency_patterns(completed_experiments)

        return {
            "runtime_analysis": runtime_analysis,
            "success_analysis": success_analysis,
            "efficiency_analysis": efficiency_analysis,
        }

    def _analyze_correlations(
        self, experiments: List[ExperimentData]
    ) -> Dict[str, Any]:
        """Analyze correlations between parameters and metrics."""
        completed_experiments = [
            exp for exp in experiments if exp.status == "completed"
        ]

        if len(completed_experiments) < 3:
            return {
                "message": "Need at least 3 completed experiments for correlation analysis"
            }

        correlations = {}

        # Get all numeric parameters and metrics
        all_data = []
        for exp in completed_experiments:
            data_point = {}

            # Add numeric parameters
            for param_name, param_value in exp.parameters.items():
                if isinstance(param_value, (int, float)):
                    data_point[f"param_{param_name}"] = param_value

            # Add metrics
            for metric_name, metric_value in exp.metrics.items():
                if isinstance(metric_value, (int, float)):
                    data_point[f"metric_{metric_name}"] = metric_value

            # Add duration
            if exp.duration:
                data_point["duration"] = exp.duration

            all_data.append(data_point)

        # Calculate correlations
        if all_data:
            correlations = self._calculate_correlations(all_data)

        return {
            "correlations": correlations,
            "strong_correlations": self._find_strong_correlations(correlations),
        }

    def _detect_anomalies(self, experiments: List[ExperimentData]) -> Dict[str, Any]:
        """Detect anomalous experiments."""
        completed_experiments = [
            exp for exp in experiments if exp.status == "completed"
        ]

        if len(completed_experiments) < 5:
            return {
                "message": "Need at least 5 completed experiments for anomaly detection"
            }

        anomalies = {
            "runtime_anomalies": self._detect_runtime_anomalies(completed_experiments),
            "metric_anomalies": self._detect_metric_anomalies(completed_experiments),
            "failure_patterns": self._analyze_failure_patterns(experiments),
        }

        return anomalies

    def _generate_recommendations(self, experiments: List[ExperimentData]) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        completed = [exp for exp in experiments if exp.status == "completed"]
        failed = [exp for exp in experiments if exp.status == "failed"]

        # Success rate recommendations
        if len(failed) > len(completed):
            recommendations.append(
                "High failure rate detected. Review parameter ranges and experiment setup."
            )

        # Duration recommendations
        durations = [exp.duration for exp in completed if exp.duration]
        if durations:
            avg_duration = sum(durations) / len(durations)
            if avg_duration > 300:  # 5 minutes
                recommendations.append(
                    "Experiments are taking long to run. Consider optimizing or using smaller parameter ranges."
                )

        # Parameter space recommendations
        if len(experiments) < 10:
            recommendations.append(
                "Run more experiments to get better insights and patterns."
            )

        # Metric-specific recommendations
        all_metrics = {}
        for exp in completed:
            for metric_name, metric_value in exp.metrics.items():
                if isinstance(metric_value, (int, float)):
                    if metric_name not in all_metrics:
                        all_metrics[metric_name] = []
                    all_metrics[metric_name].append(metric_value)

        for metric_name, values in all_metrics.items():
            if len(values) > 3:
                std_dev = statistics.stdev(values)
                mean_val = statistics.mean(values)
                if std_dev / mean_val < 0.1:  # Low variance
                    recommendations.append(
                        f"Low variance in {metric_name}. Consider exploring different parameter ranges."
                    )

        return recommendations

    def _analyze_completion_patterns(
        self, experiments: List[ExperimentData]
    ) -> Dict[str, Any]:
        """Analyze patterns in experiment completion."""
        if not experiments:
            return {}

        # Group by status
        status_counts = {}
        for exp in experiments:
            status = exp.status
            status_counts[status] = status_counts.get(status, 0) + 1

        # Analyze timing patterns
        completed_experiments = [
            exp for exp in experiments if exp.status == "completed"
        ]
        if completed_experiments:
            recent_completions = sorted(
                completed_experiments, key=lambda x: x.start_time
            )[-5:]
            avg_recent_duration = sum(
                exp.duration for exp in recent_completions if exp.duration
            ) / len(recent_completions)
        else:
            avg_recent_duration = 0

        return {
            "status_distribution": status_counts,
            "recent_avg_duration": avg_recent_duration,
        }

    def _find_most_impactful_parameters(
        self, parameter_analysis: Dict[str, Any]
    ) -> List[Tuple[str, float]]:
        """Find parameters with the most impact on metrics."""
        impact_scores = []

        for param_name, analysis in parameter_analysis.items():
            if "metric_impact" in analysis:
                total_impact = 0
                metric_count = 0

                for metric_name, impact_data in analysis["metric_impact"].items():
                    if "improvement" in impact_data:
                        total_impact += abs(impact_data["improvement"])
                        metric_count += 1

                if metric_count > 0:
                    avg_impact = total_impact / metric_count
                    impact_scores.append((param_name, avg_impact))

        # Sort by impact score
        impact_scores.sort(key=lambda x: x[1], reverse=True)
        return impact_scores[:5]  # Top 5

    def _analyze_parameter_stability(
        self, param_metrics: Dict[str, List[Dict]]
    ) -> Dict[str, Any]:
        """Analyze stability of parameter values across experiments."""
        stability = {}

        for param_name, param_data in param_metrics.items():
            values = [entry["value"] for entry in param_data]
            unique_values = len(
                set(str(v) for v in values)
            )  # Convert to string for comparison
            total_values = len(values)

            stability_score = unique_values / total_values if total_values > 0 else 0

            stability[param_name] = {
                "unique_values": unique_values,
                "total_experiments": total_values,
                "stability_score": 1 - stability_score,  # Higher score = more stable
            }

        return stability

    def _analyze_runtime_impact(
        self, value_groups: Dict[Any, List[Dict]]
    ) -> Dict[str, Any]:
        """Analyze impact of parameter values on runtime."""
        runtime_by_value = {}

        for value, entries in value_groups.items():
            durations = [entry["duration"] for entry in entries if entry["duration"]]
            if durations:
                runtime_by_value[value] = {
                    "avg_duration": sum(durations) / len(durations),
                    "min_duration": min(durations),
                    "max_duration": max(durations),
                    "count": len(durations),
                }

        # Find fastest and slowest values
        if len(runtime_by_value) > 1:
            fastest = min(runtime_by_value.items(), key=lambda x: x[1]["avg_duration"])
            slowest = max(runtime_by_value.items(), key=lambda x: x[1]["avg_duration"])

            return {
                "fastest_value": fastest[0],
                "fastest_avg": fastest[1]["avg_duration"],
                "slowest_value": slowest[0],
                "slowest_avg": slowest[1]["avg_duration"],
                "speedup": slowest[1]["avg_duration"] / fastest[1]["avg_duration"],
                "all_values": runtime_by_value,
            }

        return {}

    def _generate_parameter_recommendations(
        self, param_name: str, metric_impact: Dict, runtime_impact: Dict
    ) -> List[str]:
        """Generate recommendations for a specific parameter."""
        recommendations = []

        # Metric-based recommendations
        if metric_impact:
            best_metrics = []
            for metric_name, impact_data in metric_impact.items():
                if "improvement" in impact_data and impact_data["improvement"] > 0:
                    best_metrics.append((metric_name, impact_data["best_value"]))

            if best_metrics:
                most_improved_metric = max(
                    best_metrics, key=lambda x: metric_impact[x[0]]["improvement"]
                )
                recommendations.append(
                    f"For best {most_improved_metric[0]}, use {param_name}={most_improved_metric[1]}"
                )

        # Runtime-based recommendations
        if runtime_impact and "speedup" in runtime_impact:
            if runtime_impact["speedup"] > 2:  # Significant speedup
                recommendations.append(
                    f"Use {param_name}={runtime_impact['fastest_value']} for {runtime_impact['speedup']:.1f}x speedup"
                )

        return recommendations

    def _analyze_duration_trend(self, experiments: List[ExperimentData]) -> str:
        """Analyze trend in experiment duration over time."""
        sorted_experiments = sorted(experiments, key=lambda x: x.start_time)
        durations = [exp.duration for exp in sorted_experiments if exp.duration]

        if len(durations) < 3:
            return "insufficient_data"

        # Simple trend analysis - compare first half vs second half
        mid_point = len(durations) // 2
        first_half_avg = sum(durations[:mid_point]) / mid_point
        second_half_avg = sum(durations[mid_point:]) / (len(durations) - mid_point)

        if second_half_avg < first_half_avg * 0.9:
            return "improving"
        elif second_half_avg > first_half_avg * 1.1:
            return "degrading"
        else:
            return "stable"

    def _analyze_success_patterns(
        self, experiments: List[ExperimentData]
    ) -> Dict[str, Any]:
        """Analyze patterns in experiment success/failure."""
        total = len(experiments)
        completed = len([exp for exp in experiments if exp.status == "completed"])
        failed = len([exp for exp in experiments if exp.status == "failed"])

        return {
            "total_experiments": total,
            "completed": completed,
            "failed": failed,
            "success_rate": completed / total if total > 0 else 0,
            "failure_rate": failed / total if total > 0 else 0,
        }

    def _analyze_efficiency_patterns(
        self, experiments: List[ExperimentData]
    ) -> Dict[str, Any]:
        """Analyze efficiency patterns in experiments."""
        if not experiments:
            return {}

        # Calculate efficiency metrics
        durations = [exp.duration for exp in experiments if exp.duration]
        metrics_collected = [len(exp.metrics) for exp in experiments]

        efficiency = {}
        if durations and metrics_collected:
            avg_duration = sum(durations) / len(durations)
            avg_metrics = sum(metrics_collected) / len(metrics_collected)

            efficiency = {
                "avg_duration": avg_duration,
                "avg_metrics_per_experiment": avg_metrics,
                "metrics_per_second": (
                    avg_metrics / avg_duration if avg_duration > 0 else 0
                ),
            }

        return efficiency

    def _calculate_correlations(
        self, data_points: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """Calculate correlations between all numeric variables."""
        if len(data_points) < 3:
            return {}

        # Get all variable names
        all_vars = set()
        for point in data_points:
            all_vars.update(point.keys())

        correlations = {}

        for var1 in all_vars:
            for var2 in all_vars:
                if var1 < var2:  # Avoid duplicate pairs
                    # Extract values for both variables
                    pairs = []
                    for point in data_points:
                        if var1 in point and var2 in point:
                            pairs.append((point[var1], point[var2]))

                    if len(pairs) >= 3:
                        correlation = self._pearson_correlation(pairs)
                        if correlation is not None:
                            correlations[f"{var1}_vs_{var2}"] = correlation

        return correlations

    def _pearson_correlation(self, pairs: List[Tuple[float, float]]) -> Optional[float]:
        """Calculate Pearson correlation coefficient."""
        if len(pairs) < 2:
            return None

        n = len(pairs)
        sum_x = sum(x for x, y in pairs)
        sum_y = sum(y for x, y in pairs)
        sum_xx = sum(x * x for x, y in pairs)
        sum_yy = sum(y * y for x, y in pairs)
        sum_xy = sum(x * y for x, y in pairs)

        denominator = (
            (n * sum_xx - sum_x * sum_x) * (n * sum_yy - sum_y * sum_y)
        ) ** 0.5

        if denominator == 0:
            return None

        correlation = (n * sum_xy - sum_x * sum_y) / denominator
        return correlation

    def _find_strong_correlations(
        self, correlations: Dict[str, float]
    ) -> List[Tuple[str, float]]:
        """Find correlations with absolute value > 0.7."""
        strong_correlations = []

        for pair, correlation in correlations.items():
            if abs(correlation) > 0.7:
                strong_correlations.append((pair, correlation))

        # Sort by absolute correlation strength
        strong_correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        return strong_correlations

    def _detect_runtime_anomalies(
        self, experiments: List[ExperimentData]
    ) -> List[Dict[str, Any]]:
        """Detect experiments with anomalous runtimes."""
        durations = [exp.duration for exp in experiments if exp.duration]

        if len(durations) < 5:
            return []

        mean_duration = statistics.mean(durations)
        std_duration = statistics.stdev(durations)

        # Find outliers (more than 2 standard deviations from mean)
        threshold = 2 * std_duration
        anomalies = []

        for exp in experiments:
            if exp.duration and abs(exp.duration - mean_duration) > threshold:
                anomalies.append(
                    {
                        "run_id": exp.run_id,
                        "duration": exp.duration,
                        "deviation": abs(exp.duration - mean_duration),
                        "type": "long" if exp.duration > mean_duration else "short",
                    }
                )

        return anomalies

    def _detect_metric_anomalies(
        self, experiments: List[ExperimentData]
    ) -> Dict[str, List[Dict]]:
        """Detect experiments with anomalous metric values."""
        metric_anomalies = {}

        # Group metrics by name
        metrics_by_name = {}
        for exp in experiments:
            for metric_name, metric_value in exp.metrics.items():
                if isinstance(metric_value, (int, float)):
                    if metric_name not in metrics_by_name:
                        metrics_by_name[metric_name] = []
                    metrics_by_name[metric_name].append((exp.run_id, metric_value))

        # Find anomalies for each metric
        for metric_name, values in metrics_by_name.items():
            if len(values) >= 5:
                metric_values = [v[1] for v in values]
                mean_value = statistics.mean(metric_values)
                std_value = statistics.stdev(metric_values)

                threshold = 2 * std_value
                anomalies = []

                for run_id, value in values:
                    if abs(value - mean_value) > threshold:
                        anomalies.append(
                            {
                                "run_id": run_id,
                                "value": value,
                                "deviation": abs(value - mean_value),
                                "type": "high" if value > mean_value else "low",
                            }
                        )

                if anomalies:
                    metric_anomalies[metric_name] = anomalies

        return metric_anomalies

    def _analyze_failure_patterns(
        self, experiments: List[ExperimentData]
    ) -> Dict[str, Any]:
        """Analyze patterns in failed experiments."""
        failed_experiments = [exp for exp in experiments if exp.status == "failed"]

        if not failed_experiments:
            return {"message": "No failed experiments to analyze"}

        # Analyze parameter patterns in failures
        failure_params = {}
        for exp in failed_experiments:
            for param_name, param_value in exp.parameters.items():
                if param_name not in failure_params:
                    failure_params[param_name] = []
                failure_params[param_name].append(param_value)

        # Find common failure parameter values
        common_failure_values = {}
        for param_name, values in failure_params.items():
            value_counts = {}
            for value in values:
                value_counts[value] = value_counts.get(value, 0) + 1

            # Find values that appear in >50% of failures
            total_failures = len(failed_experiments)
            common_values = {
                value: count
                for value, count in value_counts.items()
                if count / total_failures > 0.5
            }

            if common_values:
                common_failure_values[param_name] = common_values

        return {
            "total_failures": len(failed_experiments),
            "common_failure_parameters": common_failure_values,
            "failure_rate": len(failed_experiments) / len(experiments),
        }
