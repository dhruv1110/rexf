"""Smart experiment comparison with automatic analysis.

This module provides detailed comparison of experiments with automatic
selection of the most relevant insights and recommendations.
"""

from typing import Any, Dict, List

from ..core.models import ExperimentData


class SmartComparer:
    """
    Smart experiment comparison engine.

    Automatically analyzes experiments and provides insights
    based on the data available in experiments.
    """

    def __init__(self):
        """Initialize the smart comparer."""
        pass

    def compare_experiments(self, experiments: List[ExperimentData]) -> Dict[str, Any]:
        """
        Perform comprehensive comparison of experiments.

        Args:
            experiments: List of experiments to compare

        Returns:
            Dictionary with comparison results and insights
        """
        if len(experiments) < 2:
            return {"error": "Need at least 2 experiments to compare"}

        comparison = {
            "summary": self._generate_summary(experiments),
            "parameter_analysis": self._analyze_parameters(experiments),
            "metric_analysis": self._analyze_metrics(experiments),
            "performance_analysis": self._analyze_performance(experiments),
            "insights": self._generate_insights(experiments),
            "recommendations": self._generate_recommendations(experiments),
        }

        return comparison

    def _generate_summary(self, experiments: List[ExperimentData]) -> Dict[str, Any]:
        """Generate high-level summary of experiments."""
        total = len(experiments)
        completed = len([exp for exp in experiments if exp.status == "completed"])
        failed = total - completed

        # Find time range
        start_times = [exp.start_time for exp in experiments]
        end_times = [exp.end_time for exp in experiments if exp.end_time]

        # Calculate durations
        durations = [exp.duration for exp in experiments if exp.duration]

        summary = {
            "total_experiments": total,
            "completed": completed,
            "failed": failed,
            "success_rate": completed / total if total > 0 else 0,
            "time_span": {
                "earliest": min(start_times).isoformat() if start_times else None,
                "latest": max(end_times).isoformat() if end_times else None,
            },
            "duration_stats": {
                "avg": sum(durations) / len(durations) if durations else None,
                "min": min(durations) if durations else None,
                "max": max(durations) if durations else None,
            },
            "experiment_names": list(set(exp.experiment_name for exp in experiments)),
        }

        return summary

    def _analyze_parameters(self, experiments: List[ExperimentData]) -> Dict[str, Any]:
        """Analyze parameter variations and their impact."""
        # Collect all parameters
        all_params = {}
        for exp in experiments:
            for param, value in exp.parameters.items():
                if param not in all_params:
                    all_params[param] = []
                all_params[param].append(value)

        param_analysis = {}
        for param, values in all_params.items():
            unique_values = list(set(values))
            varies = len(unique_values) > 1

            # Numeric analysis for numeric parameters
            numeric_values = [v for v in values if isinstance(v, (int, float))]
            numeric_stats = None
            if numeric_values:
                numeric_stats = {
                    "min": min(numeric_values),
                    "max": max(numeric_values),
                    "avg": sum(numeric_values) / len(numeric_values),
                    "range": max(numeric_values) - min(numeric_values),
                }

            param_analysis[param] = {
                "varies": varies,
                "unique_values": unique_values,
                "value_count": len(unique_values),
                "numeric_stats": numeric_stats,
                "type": type(values[0]).__name__ if values else "unknown",
            }

        # Find most important varying parameters
        varying_params = {k: v for k, v in param_analysis.items() if v["varies"]}
        most_varying = sorted(
            varying_params.items(), key=lambda x: x[1]["value_count"], reverse=True
        )[:5]

        return {
            "all_parameters": param_analysis,
            "varying_parameters": dict(varying_params),
            "most_varying": dict(most_varying),
            "constant_parameters": {
                k: v for k, v in param_analysis.items() if not v["varies"]
            },
        }

    def _analyze_metrics(self, experiments: List[ExperimentData]) -> Dict[str, Any]:
        """Analyze metric variations and performance."""
        # Collect all metrics
        all_metrics = {}
        for exp in experiments:
            for metric, value in exp.metrics.items():
                if isinstance(value, (int, float)):
                    if metric not in all_metrics:
                        all_metrics[metric] = []
                    all_metrics[metric].append(value)

        metric_analysis = {}
        for metric, values in all_metrics.items():
            if values:
                metric_analysis[metric] = {
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values),
                    "std": self._calculate_std(values),
                    "range": max(values) - min(values),
                    "values": values,
                    "best_value": max(values),  # Assume higher is better by default
                    "worst_value": min(values),
                    "improvement": max(values) - min(values),
                }

        # Find best and worst experiments by each metric
        best_experiments = {}
        worst_experiments = {}

        for metric, stats in metric_analysis.items():
            best_idx = None
            worst_idx = None
            best_val = float("-inf")
            worst_val = float("inf")

            for i, exp in enumerate(experiments):
                if metric in exp.metrics:
                    val = exp.metrics[metric]
                    if isinstance(val, (int, float)):
                        if val > best_val:
                            best_val = val
                            best_idx = i
                        if val < worst_val:
                            worst_val = val
                            worst_idx = i

            if best_idx is not None:
                best_experiments[metric] = {
                    "experiment": experiments[best_idx],
                    "value": best_val,
                    "run_id": experiments[best_idx].run_id,
                }

            if worst_idx is not None:
                worst_experiments[metric] = {
                    "experiment": experiments[worst_idx],
                    "value": worst_val,
                    "run_id": experiments[worst_idx].run_id,
                }

        return {
            "metric_statistics": metric_analysis,
            "best_by_metric": best_experiments,
            "worst_by_metric": worst_experiments,
            "available_metrics": list(all_metrics.keys()),
        }

    def _analyze_performance(self, experiments: List[ExperimentData]) -> Dict[str, Any]:
        """Analyze experiment performance (runtime, efficiency)."""
        durations = []
        success_rates = {}

        for exp in experiments:
            if exp.duration:
                durations.append(exp.duration)

            exp_name = exp.experiment_name
            if exp_name not in success_rates:
                success_rates[exp_name] = {"completed": 0, "total": 0}

            success_rates[exp_name]["total"] += 1
            if exp.status == "completed":
                success_rates[exp_name]["completed"] += 1

        # Calculate success rates
        for exp_name in success_rates:
            stats = success_rates[exp_name]
            stats["rate"] = (
                stats["completed"] / stats["total"] if stats["total"] > 0 else 0
            )

        performance_analysis = {
            "duration_stats": {
                "avg": sum(durations) / len(durations) if durations else None,
                "min": min(durations) if durations else None,
                "max": max(durations) if durations else None,
                "total": sum(durations) if durations else None,
            },
            "success_rates_by_experiment": success_rates,
            "fastest_experiment": None,
            "slowest_experiment": None,
        }

        # Find fastest and slowest
        if durations:
            min_duration = min(durations)
            max_duration = max(durations)

            for exp in experiments:
                if exp.duration == min_duration:
                    performance_analysis["fastest_experiment"] = {
                        "run_id": exp.run_id,
                        "duration": min_duration,
                        "experiment_name": exp.experiment_name,
                    }
                if exp.duration == max_duration:
                    performance_analysis["slowest_experiment"] = {
                        "run_id": exp.run_id,
                        "duration": max_duration,
                        "experiment_name": exp.experiment_name,
                    }

        return performance_analysis

    def _generate_insights(self, experiments: List[ExperimentData]) -> List[str]:
        """Generate human-readable insights from the comparison."""
        insights = []

        # Parameter insights
        param_analysis = self._analyze_parameters(experiments)
        varying_params = param_analysis["varying_parameters"]

        if varying_params:
            most_varying = list(param_analysis["most_varying"].keys())[:3]
            insights.append(f"📊 Most varied parameters: {', '.join(most_varying)}")

        # Metric insights
        metric_analysis = self._analyze_metrics(experiments)
        metrics = metric_analysis["metric_statistics"]

        if metrics:
            best_metric = max(metrics.items(), key=lambda x: x[1]["improvement"])[0]
            improvement = metrics[best_metric]["improvement"]
            insights.append(
                f"📈 Largest improvement in {best_metric}: {improvement:.4f}"
            )

            # Find most stable metric (lowest std)
            most_stable = min(metrics.items(), key=lambda x: x[1]["std"])[0]
            insights.append(f"🎯 Most consistent metric: {most_stable}")

        # Performance insights
        perf_analysis = self._analyze_performance(experiments)
        duration_stats = perf_analysis["duration_stats"]

        if duration_stats["avg"]:
            avg_duration = duration_stats["avg"]
            insights.append(f"⏱️ Average runtime: {avg_duration:.2f} seconds")

            if duration_stats["max"] and duration_stats["min"]:
                speed_ratio = duration_stats["max"] / duration_stats["min"]
                insights.append(f"🚀 Fastest vs slowest: {speed_ratio:.1f}x difference")

        # Success rate insights
        success_rates = perf_analysis["success_rates_by_experiment"]
        if success_rates:
            avg_success = sum(stats["rate"] for stats in success_rates.values()) / len(
                success_rates
            )
            insights.append(f"✅ Overall success rate: {avg_success:.1%}")

        return insights

    def _generate_recommendations(self, experiments: List[ExperimentData]) -> List[str]:
        """Generate actionable recommendations based on comparison."""
        recommendations = []

        # Analyze patterns
        metric_analysis = self._analyze_metrics(experiments)
        param_analysis = self._analyze_parameters(experiments)

        # Find best performing experiment
        if metric_analysis["metric_statistics"]:
            # Get the first metric as a proxy for "primary" metric
            primary_metric = list(metric_analysis["metric_statistics"].keys())[0]
            best_exp_info = metric_analysis["best_by_metric"].get(primary_metric)

            if best_exp_info:
                best_exp = best_exp_info["experiment"]
                recommendations.append(
                    f"🏆 Best {primary_metric}: {best_exp_info['value']:.4f} "
                    f"(run {best_exp.run_id[:8]}...)"
                )

                # Recommend similar parameters
                key_params = []
                for param, value in best_exp.parameters.items():
                    if param in param_analysis["varying_parameters"]:
                        key_params.append(f"{param}={value}")

                if key_params:
                    recommendations.append(
                        f"🔧 Try similar parameters: {', '.join(key_params[:3])}"
                    )

        # Parameter recommendations
        varying_params = param_analysis["varying_parameters"]
        for param, info in list(varying_params.items())[:2]:  # Top 2 varying params
            if info["numeric_stats"]:
                stats = info["numeric_stats"]
                if stats["range"] > 0:
                    recommendations.append(
                        f"🎛️ Explore {param} range: {stats['min']:.3f} to {stats['max']:.3f}"
                    )

        # Experiment strategy recommendations
        completed_count = len([exp for exp in experiments if exp.status == "completed"])
        total_count = len(experiments)

        if completed_count < total_count:
            failed_count = total_count - completed_count
            recommendations.append(
                f"⚠️ {failed_count} experiments failed - check error logs"
            )

        if total_count < 10:
            recommendations.append("🔄 Run more experiments to identify patterns")

        return recommendations

    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance**0.5

    def format_comparison_report(self, comparison: Dict[str, Any]) -> str:
        """Format comparison results into a readable report."""
        report = []

        # Summary
        summary = comparison["summary"]
        report.append("📊 EXPERIMENT COMPARISON REPORT")
        report.append("=" * 50)
        report.append(f"Total experiments: {summary['total_experiments']}")
        report.append(f"Success rate: {summary['success_rate']:.1%}")

        if summary["duration_stats"]["avg"]:
            report.append(f"Average runtime: {summary['duration_stats']['avg']:.2f}s")

        report.append("")

        # Insights
        insights = comparison.get("insights", [])
        if insights:
            report.append("🔍 KEY INSIGHTS:")
            for insight in insights:
                report.append(f"  {insight}")
            report.append("")

        # Recommendations
        recommendations = comparison.get("recommendations", [])
        if recommendations:
            report.append("💡 RECOMMENDATIONS:")
            for rec in recommendations:
                report.append(f"  {rec}")
            report.append("")

        # Parameter analysis
        param_analysis = comparison["parameter_analysis"]
        varying_params = param_analysis["varying_parameters"]

        if varying_params:
            report.append("🎛️ VARYING PARAMETERS:")
            for param, info in list(varying_params.items())[:5]:
                report.append(f"  {param}: {info['value_count']} unique values")
                if info["numeric_stats"]:
                    stats = info["numeric_stats"]
                    report.append(
                        f"    Range: {stats['min']:.3f} to {stats['max']:.3f}"
                    )
            report.append("")

        # Metric analysis
        metric_analysis = comparison["metric_analysis"]
        metrics = metric_analysis["metric_statistics"]

        if metrics:
            report.append("📈 METRIC PERFORMANCE:")
            for metric, stats in list(metrics.items())[:5]:
                report.append(f"  {metric}: {stats['min']:.4f} to {stats['max']:.4f}")
                report.append(
                    f"    Average: {stats['avg']:.4f}, Std: {stats['std']:.4f}"
                )
            report.append("")

        return "\n".join(report)
