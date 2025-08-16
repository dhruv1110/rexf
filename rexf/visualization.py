"""Visualization tools for comparing experiment runs and metrics.

This module provides built-in visualization capabilities for experiment
comparison and analysis.
"""

try:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None
    mdates = None

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None
from pathlib import Path
from typing import Any, List, Optional, Union

from .interfaces import ExperimentMetadata, VisualizationInterface


class ExperimentVisualizer(VisualizationInterface):
    """Built-in visualization tools for experiments."""

    def __init__(self, style: str = "seaborn-v0_8", figsize: tuple = (12, 8)):
        """Initialize visualizer.

        Args:
            style: Matplotlib style
            figsize: Default figure size
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib is required for visualization")

        try:
            plt.style.use(style)
        except OSError:
            # Fallback to default style if seaborn not available
            plt.style.use("default")

        self.figsize = figsize
        if HAS_NUMPY:
            self.colors = plt.cm.Set1(np.linspace(0, 1, 10))
        else:
            self.colors = [
                "blue",
                "red",
                "green",
                "orange",
                "purple",
                "brown",
                "pink",
                "gray",
                "olive",
                "cyan",
            ]

    def plot_metrics(
        self,
        experiments: List[ExperimentMetadata],
        metric_names: Optional[List[str]] = None,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Any:
        """Plot metrics comparison across experiments."""
        if not experiments:
            raise ValueError("No experiments provided")

        # Collect all metrics if none specified
        if metric_names is None:
            all_metrics = set()
            for exp in experiments:
                all_metrics.update(exp.metrics.keys())
            metric_names = sorted(list(all_metrics))

        if not metric_names:
            raise ValueError("No metrics found in experiments")

        # Create subplots
        n_metrics = len(metric_names)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(self.figsize[0], self.figsize[1] * n_rows / 2)
        )
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle("Experiment Metrics Comparison", fontsize=16, fontweight="bold")

        for i, metric_name in enumerate(metric_names):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]

            values = []
            labels = []

            for j, exp in enumerate(experiments):
                if metric_name in exp.metrics:
                    value = exp.metrics[metric_name]
                    if isinstance(value, (int, float)):
                        values.append(value)
                        labels.append(f"{exp.name}\n{exp.run_id[:8]}")

            if values:
                bars = ax.bar(
                    range(len(values)), values, color=self.colors[: len(values)]
                )
                ax.set_title(f"Metric: {metric_name}")
                ax.set_ylabel("Value")
                ax.set_xticks(range(len(values)))
                ax.set_xticklabels(labels, rotation=45, ha="right")

                # Add value labels on bars
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"{value:.3f}",
                        ha="center",
                        va="bottom",
                    )
            else:
                ax.text(
                    0.5,
                    0.5,
                    f"No numeric data\nfor {metric_name}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(f"Metric: {metric_name}")

        # Hide empty subplots
        for i in range(n_metrics, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            if n_rows > 1:
                axes[row, col].set_visible(False)
            else:
                axes[col].set_visible(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_parameter_space(
        self,
        experiments: List[ExperimentMetadata],
        param_names: Optional[List[str]] = None,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Any:
        """Visualize parameter space exploration."""
        if not experiments:
            raise ValueError("No experiments provided")

        # Collect all parameters if none specified
        if param_names is None:
            all_params = set()
            for exp in experiments:
                all_params.update(exp.parameters.keys())
            param_names = sorted(list(all_params))

        if len(param_names) < 2:
            raise ValueError("Need at least 2 parameters for parameter space plot")

        # Create parameter matrix plot
        n_params = len(param_names)
        if n_params == 1:
            fig, axes = plt.subplots(1, 1, figsize=self.figsize)
            axes = [[axes]]
        else:
            fig, axes = plt.subplots(
                n_params, n_params, figsize=(self.figsize[0], self.figsize[1])
            )
        fig.suptitle("Parameter Space Exploration", fontsize=16, fontweight="bold")

        # Collect parameter data
        param_data = {param: [] for param in param_names}
        for exp in experiments:
            for param in param_names:
                value = exp.parameters.get(param, None)
                param_data[param].append(value)

        # Create scatter plots
        for i, param_x in enumerate(param_names):
            for j, param_y in enumerate(param_names):
                ax = axes[i, j] if n_params > 1 else axes

                if i == j:
                    # Diagonal: histogram of parameter values
                    values = [
                        v
                        for v in param_data[param_x]
                        if v is not None and isinstance(v, (int, float))
                    ]
                    if values:
                        ax.hist(
                            values,
                            bins=min(10, len(set(values))),
                            alpha=0.7,
                            color=self.colors[0],
                        )
                    ax.set_title(param_x)
                else:
                    # Off-diagonal: scatter plot
                    x_values = param_data[param_x]
                    y_values = param_data[param_y]

                    # Filter numeric values
                    valid_pairs = [
                        (x, y)
                        for x, y in zip(x_values, y_values)
                        if x is not None
                        and y is not None
                        and isinstance(x, (int, float))
                        and isinstance(y, (int, float))
                    ]

                    if valid_pairs:
                        x_vals, y_vals = zip(*valid_pairs)
                        ax.scatter(x_vals, y_vals, alpha=0.7, color=self.colors[0])

                    if j == 0:
                        ax.set_ylabel(param_y)
                    if i == n_params - 1:
                        ax.set_xlabel(param_x)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_experiment_timeline(
        self,
        experiments: List[ExperimentMetadata],
        save_path: Optional[Union[str, Path]] = None,
    ) -> Any:
        """Plot experiment execution timeline."""
        if not experiments:
            raise ValueError("No experiments provided")

        fig, ax = plt.subplots(figsize=self.figsize)

        # Sort experiments by start time
        sorted_experiments = sorted(experiments, key=lambda x: x.start_time)

        y_pos = range(len(sorted_experiments))
        colors = []
        durations = []

        for exp in sorted_experiments:
            # Color by status
            if exp.status == "completed":
                colors.append("green")
            elif exp.status == "failed":
                colors.append("red")
            elif exp.status == "running":
                colors.append("orange")
            else:
                colors.append("gray")

            # Calculate duration
            if exp.end_time:
                duration = (exp.end_time - exp.start_time).total_seconds()
            else:
                duration = 0
            durations.append(duration)

        # Create horizontal bar chart
        bars = ax.barh(y_pos, durations, color=colors, alpha=0.7)

        # Format labels
        labels = [f"{exp.name}\n{exp.run_id[:8]}" for exp in sorted_experiments]
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Duration (seconds)")
        ax.set_title("Experiment Timeline")

        # Add duration labels
        for bar, duration in zip(bars, durations):
            width = bar.get_width()
            if width > 0:
                ax.text(
                    width / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{duration:.1f}s",
                    ha="center",
                    va="center",
                )

        # Add legend
        try:
            from matplotlib.patches import Rectangle
        except ImportError:
            Rectangle = None
        if Rectangle:
            legend_elements = [
                Rectangle(
                    (0, 0), 1, 1, facecolor="green", alpha=0.7, label="Completed"
                ),
                Rectangle((0, 0), 1, 1, facecolor="red", alpha=0.7, label="Failed"),
                Rectangle((0, 0), 1, 1, facecolor="orange", alpha=0.7, label="Running"),
                Rectangle((0, 0), 1, 1, facecolor="gray", alpha=0.7, label="Other"),
            ]
            ax.legend(handles=legend_elements, loc="upper right")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def create_comparison_table(
        self,
        experiments: List[ExperimentMetadata],
        save_path: Optional[Union[str, Path]] = None,
    ) -> Any:
        """Create a comparison table of experiments."""
        if not experiments:
            raise ValueError("No experiments provided")

        # Collect all columns
        all_params = set()
        all_metrics = set()
        all_results = set()

        for exp in experiments:
            all_params.update(exp.parameters.keys())
            all_metrics.update(exp.metrics.keys())
            all_results.update(exp.results.keys())

        # Create DataFrame
        data = []
        for exp in experiments:
            row = {
                "run_id": exp.run_id,
                "experiment_name": exp.name,
                "start_time": exp.start_time,
                "end_time": exp.end_time,
                "status": exp.status,
                "duration_seconds": (
                    (exp.end_time - exp.start_time).total_seconds()
                    if exp.end_time
                    else None
                ),
                "git_commit": exp.git_commit[:8] if exp.git_commit else None,
                "random_seed": exp.random_seed,
            }

            # Add parameters (ensure serializable)
            for param in sorted(all_params):
                value = exp.parameters.get(param)
                # Convert numpy arrays to lists for pandas compatibility
                if HAS_NUMPY and np and isinstance(value, np.ndarray):
                    value = value.tolist()
                row[f"param_{param}"] = value

            # Add metrics (ensure serializable)
            for metric in sorted(all_metrics):
                value = exp.metrics.get(metric)
                # Convert numpy arrays to lists for pandas compatibility
                if HAS_NUMPY and np and isinstance(value, np.ndarray):
                    value = value.tolist()
                row[f"metric_{metric}"] = value

            # Add results (ensure serializable)
            for result in sorted(all_results):
                value = exp.results.get(result)
                # Convert numpy arrays to lists for pandas compatibility
                if HAS_NUMPY and np and isinstance(value, np.ndarray):
                    value = value.tolist()
                row[f"result_{result}"] = value

            data.append(row)

        if HAS_PANDAS:
            try:
                df = pd.DataFrame(data)

                if save_path:
                    df.to_csv(save_path, index=False)

                return df
            except (TypeError, ValueError) as e:
                # Fall back to dict format if pandas can't handle the data
                if save_path:
                    import csv
                    with open(save_path, "w", newline="") as f:
                        if data:
                            writer = csv.DictWriter(f, fieldnames=data[0].keys())
                            writer.writeheader()
                            writer.writerows(data)
                return data
        else:
            # Return dict if pandas not available
            if save_path:
                import csv

                with open(save_path, "w", newline="") as f:
                    if data:
                        writer = csv.DictWriter(f, fieldnames=data[0].keys())
                        writer.writeheader()
                        writer.writerows(data)
            return data

    def plot_metric_correlation(
        self,
        experiments: List[ExperimentMetadata],
        save_path: Optional[Union[str, Path]] = None,
    ) -> Any:
        """Plot correlation matrix of metrics."""
        if not experiments:
            raise ValueError("No experiments provided")

        # Collect numeric metrics
        metric_data = {}
        for exp in experiments:
            for metric_name, metric_value in exp.metrics.items():
                if isinstance(metric_value, (int, float)):
                    if metric_name not in metric_data:
                        metric_data[metric_name] = []
                    metric_data[metric_name].append(metric_value)

        if len(metric_data) < 2:
            raise ValueError("Need at least 2 numeric metrics for correlation plot")

        # Create DataFrame and compute correlation
        if not HAS_PANDAS:
            raise ImportError("Pandas is required for correlation analysis")

        df = pd.DataFrame(metric_data)
        correlation_matrix = df.corr()

        # Create heatmap
        fig, ax = plt.subplots(figsize=self.figsize)
        im = ax.imshow(
            correlation_matrix, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1
        )

        # Add labels
        ax.set_xticks(range(len(correlation_matrix.columns)))
        ax.set_yticks(range(len(correlation_matrix.columns)))
        ax.set_xticklabels(correlation_matrix.columns, rotation=45, ha="right")
        ax.set_yticklabels(correlation_matrix.columns)

        # Add correlation values
        for i in range(len(correlation_matrix.columns)):
            for j in range(len(correlation_matrix.columns)):
                ax.text(
                    j,
                    i,
                    f"{correlation_matrix.iloc[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                )

        ax.set_title("Metric Correlation Matrix")
        plt.colorbar(im, ax=ax, label="Correlation")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def create_dashboard(
        self,
        experiments: List[ExperimentMetadata],
        save_path: Optional[Union[str, Path]] = None,
    ) -> Any:
        """Create a comprehensive dashboard of experiments."""
        if not experiments:
            raise ValueError("No experiments provided")

        # Create a large figure with multiple subplots
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Title
        fig.suptitle("Experiment Dashboard", fontsize=20, fontweight="bold")

        # 1. Experiment timeline
        ax1 = fig.add_subplot(gs[0, :])
        sorted_experiments = sorted(experiments, key=lambda x: x.start_time)
        y_pos = range(len(sorted_experiments))
        durations = [
            (exp.end_time - exp.start_time).total_seconds() if exp.end_time else 0
            for exp in sorted_experiments
        ]
        colors = [
            (
                "green"
                if exp.status == "completed"
                else "red" if exp.status == "failed" else "orange"
            )
            for exp in sorted_experiments
        ]

        ax1.barh(y_pos, durations, color=colors, alpha=0.7)
        labels = [
            f"{exp.name[:20]}..." if len(exp.name) > 20 else exp.name
            for exp in sorted_experiments
        ]
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(labels)
        ax1.set_xlabel("Duration (seconds)")
        ax1.set_title("Experiment Timeline")

        # 2. Status distribution
        ax2 = fig.add_subplot(gs[1, 0])
        status_counts = {}
        for exp in experiments:
            status_counts[exp.status] = status_counts.get(exp.status, 0) + 1

        ax2.pie(status_counts.values(), labels=status_counts.keys(), autopct="%1.1f%%")
        ax2.set_title("Experiment Status Distribution")

        # 3. Metrics comparison (if available)
        ax3 = fig.add_subplot(gs[1, 1])
        all_metrics = set()
        for exp in experiments:
            all_metrics.update(exp.metrics.keys())

        if all_metrics:
            metric_name = list(all_metrics)[0]  # Use first metric
            values = [
                exp.metrics.get(metric_name, 0)
                for exp in experiments
                if metric_name in exp.metrics
                and isinstance(exp.metrics[metric_name], (int, float))
            ]
            if values:
                ax3.hist(values, bins=min(10, len(set(values))), alpha=0.7)
                ax3.set_title(f"Distribution of {metric_name}")
                ax3.set_xlabel("Value")
                ax3.set_ylabel("Frequency")

        # 4. Parameter space (if available)
        ax4 = fig.add_subplot(gs[1, 2])
        all_params = set()
        for exp in experiments:
            all_params.update(exp.parameters.keys())

        if len(all_params) >= 2:
            param_names = list(all_params)[:2]
            x_values = [exp.parameters.get(param_names[0]) for exp in experiments]
            y_values = [exp.parameters.get(param_names[1]) for exp in experiments]

            # Filter numeric values
            valid_pairs = [
                (x, y)
                for x, y in zip(x_values, y_values)
                if x is not None
                and y is not None
                and isinstance(x, (int, float))
                and isinstance(y, (int, float))
            ]

            if valid_pairs:
                x_vals, y_vals = zip(*valid_pairs)
                ax4.scatter(x_vals, y_vals, alpha=0.7)
                ax4.set_xlabel(param_names[0])
                ax4.set_ylabel(param_names[1])
                ax4.set_title("Parameter Space")

        # 5. Experiment count over time
        ax5 = fig.add_subplot(gs[2, :])
        dates = [exp.start_time.date() for exp in experiments]
        date_counts = {}
        for date in dates:
            date_counts[date] = date_counts.get(date, 0) + 1

        sorted_dates = sorted(date_counts.keys())
        counts = [date_counts[date] for date in sorted_dates]

        ax5.plot(sorted_dates, counts, marker="o", linewidth=2, markersize=6)
        ax5.set_xlabel("Date")
        ax5.set_ylabel("Number of Experiments")
        ax5.set_title("Experiments Over Time")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig
