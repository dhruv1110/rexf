"""Visualization functionality for experiment comparison."""

from typing import Any, List, Optional

from ..core.interfaces import VisualizationInterface
from ..core.models import ExperimentData

# Optional dependencies
try:
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    plt = None
    patches = None
    HAS_MATPLOTLIB = False

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    pd = None
    HAS_PANDAS = False

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False


class ExperimentVisualizer(VisualizationInterface):
    """Visualization for experiment comparison and analysis."""

    def __init__(self):
        """Initialize experiment visualizer."""
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for visualization functionality")

    def plot_metrics(
        self,
        experiments: List[ExperimentData],
        metric_names: Optional[List[str]] = None,
    ) -> Any:
        """Plot metrics comparison across experiments."""
        if not experiments:
            return None

        # Get all available metrics if not specified
        if metric_names is None:
            all_metrics = set()
            for exp in experiments:
                all_metrics.update(exp.metrics.keys())
            metric_names = sorted(all_metrics)

        if not metric_names:
            return None

        # Create subplots
        n_metrics = len(metric_names)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)

        # Plot each metric
        for i, metric_name in enumerate(metric_names):
            row, col = divmod(i, n_cols)
            ax = axes[row, col] if n_rows > 1 else axes[col]

            values = []
            labels = []
            for j, exp in enumerate(experiments):
                if metric_name in exp.metrics:
                    values.append(exp.metrics[metric_name])
                    labels.append(f"Run {j+1}")

            if values:
                ax.bar(range(len(values)), values)
                ax.set_title(f"Metric: {metric_name}")
                ax.set_xlabel("Experiment Run")
                ax.set_ylabel("Value")
                ax.set_xticks(range(len(values)))
                ax.set_xticklabels(labels, rotation=45)

        # Hide empty subplots
        for i in range(n_metrics, n_rows * n_cols):
            row, col = divmod(i, n_cols)
            ax = axes[row, col] if n_rows > 1 else axes[col]
            ax.set_visible(False)

        plt.tight_layout()
        return fig

    def plot_parameter_space(
        self,
        experiments: List[ExperimentData],
        param_names: Optional[List[str]] = None,
    ) -> Any:
        """Visualize parameter space exploration."""
        if not experiments:
            return None

        # Get all parameters if not specified
        if param_names is None:
            all_params = set()
            for exp in experiments:
                all_params.update(exp.parameters.keys())
            param_names = sorted(all_params)

        if len(param_names) < 2:
            return None

        fig, ax = plt.subplots(figsize=(10, 8))

        # Take first two parameters for 2D visualization
        param_x, param_y = param_names[0], param_names[1]

        x_values = []
        y_values = []
        colors = []

        for exp in experiments:
            if param_x in exp.parameters and param_y in exp.parameters:
                x_values.append(exp.parameters[param_x])
                y_values.append(exp.parameters[param_y])
                # Color by experiment status
                colors.append("green" if exp.status == "completed" else "red")

        if x_values and y_values:
            ax.scatter(x_values, y_values, c=colors, alpha=0.7, s=100)
            ax.set_xlabel(param_x)
            ax.set_ylabel(param_y)
            ax.set_title(f"Parameter Space: {param_x} vs {param_y}")
            ax.grid(True, alpha=0.3)

            # Add legend
            from matplotlib.lines import Line2D

            legend_elements = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="green",
                    markersize=10,
                    label="Completed",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="red",
                    markersize=10,
                    label="Failed",
                ),
            ]
            ax.legend(handles=legend_elements)

        return fig

    def create_comparison_table(self, experiments: List[ExperimentData]) -> Any:
        """Create a comparison table of experiments."""
        if not experiments:
            return None

        # Collect all data
        data = []
        for i, exp in enumerate(experiments):
            row = {
                "Run": i + 1,
                "Name": exp.experiment_name,
                "Status": exp.status,
                "Duration (s)": exp.duration,
                "Start Time": exp.start_time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            # Add parameters
            for key, value in exp.parameters.items():
                # Handle numpy arrays
                if HAS_NUMPY and np and isinstance(value, np.ndarray):
                    value = value.tolist()
                row[f"param_{key}"] = value

            # Add metrics
            for key, value in exp.metrics.items():
                # Handle numpy arrays
                if HAS_NUMPY and np and isinstance(value, np.ndarray):
                    value = value.tolist()
                row[f"metric_{key}"] = value

            # Add results
            for key, value in exp.results.items():
                # Handle numpy arrays
                if HAS_NUMPY and np and isinstance(value, np.ndarray):
                    value = value.tolist()
                row[f"result_{key}"] = value

            data.append(row)

        # Create DataFrame if pandas is available
        if HAS_PANDAS:
            try:
                df = pd.DataFrame(data)
                return df
            except (TypeError, ValueError):
                # Fall back to dict format if pandas can't handle the data
                return data
        else:
            return data

    def plot_experiment_timeline(self, experiments: List[ExperimentData]) -> Any:
        """Plot experiment execution timeline."""
        if not experiments:
            return None

        fig, ax = plt.subplots(figsize=(12, 6))

        # Sort experiments by start time
        sorted_experiments = sorted(experiments, key=lambda x: x.start_time)

        y_positions = range(len(sorted_experiments))

        for i, exp in enumerate(sorted_experiments):
            if exp.end_time:
                duration = exp.duration or 0
                color = "green" if exp.status == "completed" else "red"

                # Draw bar representing experiment duration
                ax.barh(
                    i,
                    duration,
                    left=0,
                    height=0.6,
                    color=color,
                    alpha=0.7,
                    label=exp.status if i == 0 else "",
                )

                # Add experiment name
                ax.text(
                    duration / 2,
                    i,
                    f"{exp.experiment_name} (Run {i+1})",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

        ax.set_yticks(y_positions)
        ax.set_yticklabels([f"Run {i+1}" for i in range(len(sorted_experiments))])
        ax.set_xlabel("Duration (seconds)")
        ax.set_ylabel("Experiments")
        ax.set_title("Experiment Timeline")
        ax.grid(True, alpha=0.3)

        # Add legend
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels)

        plt.tight_layout()
        return fig
