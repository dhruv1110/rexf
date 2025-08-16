#!/usr/bin/env python3
"""
Monte Carlo π Estimation - Clean API Demo

This example demonstrates the clean experiment_config decorator API for
reproducible computational experiments using Monte Carlo methods.
"""

import random
import time

import matplotlib.pyplot as plt
import numpy as np

from rexf import ExperimentRunner, experiment_config


@experiment_config(
    name="monte_carlo_pi_estimation",
    params={
        "n_samples": (int, "Number of random samples to generate"),
        "batch_size": (int, 1000, "Size of each processing batch"),
        "method": (str, "vectorized", "Sampling method: 'vectorized' or 'iterative'"),
    },
    metrics={
        "pi_estimate": (float, "Estimated value of π"),
        "error": (float, "Absolute error from true π"),
        "convergence_rate": (float, "Rate of convergence over batches"),
        "efficiency": (float, "Samples processed per second"),
    },
    results={
        "final_pi": (float, "Final π estimate"),
        "samples_inside_circle": (int, "Number of samples inside unit circle"),
        "total_samples": (int, "Total number of samples processed"),
        "runtime_seconds": (float, "Total runtime in seconds"),
    },
    artifacts={
        "convergence_plot": ("pi_convergence.png", "Plot showing convergence to π"),
        "sample_plot": ("sample_distribution.png", "Plot of random samples"),
        "error_plot": ("error_evolution.png", "Plot showing error evolution"),
    },
)
def estimate_pi(n_samples, batch_size=1000, method="vectorized", random_seed=42):
    """
    Estimate π using Monte Carlo method.

    This function demonstrates how to use the clean experiment_config decorator
    instead of stacking 8+ individual decorators. Much cleaner and more readable!
    """
    start_time = time.time()

    # Set random seeds for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)

    inside_circle = 0
    pi_estimates = []
    errors = []
    sample_counts = []

    print(f"🎯 Estimating π using {method} method with {n_samples:,} samples...")

    # Process in batches for convergence tracking
    for batch_start in range(0, n_samples, batch_size):
        batch_end = min(batch_start + batch_size, n_samples)
        batch_samples = batch_end - batch_start

        if method == "vectorized":
            # Vectorized approach (faster)
            x = np.random.uniform(-1, 1, batch_samples)
            y = np.random.uniform(-1, 1, batch_samples)
            distances = x**2 + y**2
            inside_circle += np.sum(distances <= 1)
        else:
            # Iterative approach (more educational)
            for _ in range(batch_samples):
                x, y = random.uniform(-1, 1), random.uniform(-1, 1)
                if x**2 + y**2 <= 1:
                    inside_circle += 1

        # Calculate running π estimate
        total_samples_so_far = batch_end
        pi_estimate = 4 * inside_circle / total_samples_so_far
        error = abs(pi_estimate - np.pi)

        pi_estimates.append(pi_estimate)
        errors.append(error)
        sample_counts.append(total_samples_so_far)

        # Progress update
        if batch_end % (5 * batch_size) == 0 or batch_end == n_samples:
            print(
                f"   📊 {batch_end:,} samples: π ≈ {pi_estimate:.6f} (error: {error:.6f})"
            )

    runtime = time.time() - start_time
    final_pi = pi_estimates[-1]
    final_error = errors[-1]
    convergence_rate = (
        abs(errors[0] - errors[-1]) / len(errors) if len(errors) > 1 else 0
    )
    efficiency = n_samples / runtime

    # Create visualizations
    create_convergence_plot(sample_counts, pi_estimates, errors)
    create_sample_distribution_plot(random_seed, min(2000, n_samples))
    create_error_evolution_plot(sample_counts, errors)

    print(f"✅ Final estimate: π ≈ {final_pi:.8f}")
    print(f"📏 True π:        π = {np.pi:.8f}")
    print(f"❌ Error:         {final_error:.8f}")
    print(f"⚡ Efficiency:    {efficiency:.1f} samples/second")

    return {
        "final_pi": final_pi,
        "samples_inside_circle": inside_circle,
        "total_samples": n_samples,
        "runtime_seconds": runtime,
        "pi_estimate": final_pi,
        "error": final_error,
        "convergence_rate": convergence_rate,
        "efficiency": efficiency,
    }


def create_convergence_plot(sample_counts, pi_estimates, errors):
    """Create convergence plot showing π estimation over time."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot π estimates
    ax1.plot(sample_counts, pi_estimates, "b-", linewidth=2, label="π estimate")
    ax1.axhline(
        y=np.pi, color="r", linestyle="--", linewidth=2, label=f"True π = {np.pi:.6f}"
    )
    ax1.set_xlabel("Number of Samples")
    ax1.set_ylabel("π Estimate")
    ax1.set_title("Monte Carlo π Estimation Convergence")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot error evolution
    ax2.semilogy(sample_counts, errors, "r-", linewidth=2, label="Absolute Error")
    ax2.set_xlabel("Number of Samples")
    ax2.set_ylabel("Absolute Error (log scale)")
    ax2.set_title("Error Evolution")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("pi_convergence.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_sample_distribution_plot(random_seed, n_plot_samples):
    """Create visualization of sample distribution."""
    # Generate fresh samples for visualization
    np.random.seed(random_seed)
    x = np.random.uniform(-1, 1, n_plot_samples)
    y = np.random.uniform(-1, 1, n_plot_samples)
    distances = x**2 + y**2

    plt.figure(figsize=(10, 10))

    # Plot samples
    inside_mask = distances <= 1
    plt.scatter(
        x[inside_mask],
        y[inside_mask],
        c="red",
        s=2,
        alpha=0.6,
        label=f"Inside circle ({np.sum(inside_mask):,} points)",
    )
    plt.scatter(
        x[~inside_mask],
        y[~inside_mask],
        c="blue",
        s=2,
        alpha=0.6,
        label=f"Outside circle ({np.sum(~inside_mask):,} points)",
    )

    # Draw unit circle
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)
    plt.plot(circle_x, circle_y, "k-", linewidth=3)

    # Draw square boundary
    plt.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], "k-", linewidth=2)

    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.axis("equal")
    plt.legend()
    plt.title(f"Monte Carlo Sampling Distribution (n={n_plot_samples:,})")
    plt.xlabel("x")
    plt.ylabel("y")

    # Add ratio annotation
    ratio = np.sum(inside_mask) / len(inside_mask)
    pi_estimate = 4 * ratio
    plt.text(
        0.02,
        0.98,
        f"Inside/Total = {ratio:.4f}\n4 × ratio = {pi_estimate:.4f}",
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.savefig("sample_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_error_evolution_plot(sample_counts, errors):
    """Create detailed error evolution plot."""
    plt.figure(figsize=(12, 6))

    plt.loglog(sample_counts, errors, "g-", linewidth=2, marker="o", markersize=3)

    # Add theoretical 1/sqrt(n) convergence line for comparison
    if len(sample_counts) > 1:
        theoretical_errors = errors[0] * np.sqrt(
            sample_counts[0] / np.array(sample_counts)
        )
        plt.loglog(
            sample_counts,
            theoretical_errors,
            "k--",
            alpha=0.5,
            label="Theoretical 1/√n convergence",
        )

    plt.xlabel("Number of Samples")
    plt.ylabel("Absolute Error")
    plt.title("Error Evolution (Log-Log Scale)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Add annotations
    if len(errors) > 1:
        plt.annotate(
            f"Final error: {errors[-1]:.2e}",
            xy=(sample_counts[-1], errors[-1]),
            xytext=(sample_counts[-1] / 3, errors[-1] * 3),
            arrowprops=dict(arrowstyle="->", color="red"),
        )

    plt.savefig("error_evolution.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    """Run comprehensive Monte Carlo π estimation experiments."""
    print("🧪 REXF DEMO: Monte Carlo π Estimation")
    print("=" * 60)
    print("This demo shows the clean experiment_config API in action!")
    print()

    # Initialize experiment runner with default backends
    runner = ExperimentRunner(
        # Uses SQLite storage and filesystem artifacts by default
        storage_path="monte_carlo_experiments.db",
        artifacts_path="monte_carlo_artifacts",
    )

    # Run different experimental configurations
    configs = [
        {
            "n_samples": 10000,
            "batch_size": 1000,
            "method": "vectorized",
            "random_seed": 42,
        },
        {
            "n_samples": 10000,
            "batch_size": 1000,
            "method": "iterative",
            "random_seed": 42,
        },
        {
            "n_samples": 50000,
            "batch_size": 2000,
            "method": "vectorized",
            "random_seed": 123,
        },
        {
            "n_samples": 100000,
            "batch_size": 5000,
            "method": "vectorized",
            "random_seed": 999,
        },
    ]

    results = []
    for i, config in enumerate(configs, 1):
        print(
            f"\n🔬 Experiment {i}/{len(configs)}: {config['method']} method, "
            f"{config['n_samples']:,} samples"
        )

        # Run experiment
        run_id = runner.run(estimate_pi, **config)
        experiment = runner.get_experiment(run_id)
        results.append(experiment)

        print(f"   π estimate: {experiment.results['final_pi']:.6f}")
        print(f"   Error: {experiment.metrics['error']:.6f}")
        print(f"   Runtime: {experiment.results['runtime_seconds']:.2f}s")
        print(f"   Efficiency: {experiment.metrics['efficiency']:.0f} samples/s")

    # Compare all results
    print(f"\n📊 EXPERIMENT COMPARISON ({len(results)} runs):")
    print("-" * 60)

    for i, exp in enumerate(results, 1):
        method = exp.parameters["method"]
        n_samples = exp.parameters["n_samples"]
        pi_est = exp.results["final_pi"]
        error = exp.metrics["error"]
        efficiency = exp.metrics["efficiency"]
        print(
            f"   {i}. {method:>10} ({n_samples:>6,} samples): "
            f"π≈{pi_est:.6f} (err: {error:.6f}, {efficiency:.0f} samp/s)"
        )

    # Find best estimate
    best_exp = min(results, key=lambda x: x.metrics["error"])
    print(
        f"\n🏆 Best estimate: {best_exp.results['final_pi']:.8f} "
        f"(error: {best_exp.metrics['error']:.8f})"
    )
    print(
        f"   Method: {best_exp.parameters['method']}, "
        f"Samples: {best_exp.parameters['n_samples']:,}"
    )

    # Show storage info
    stats = runner.get_stats()
    print("\n📈 STORAGE STATISTICS:")
    print(f"   Total experiments: {stats['storage']['total_experiments']}")
    print(f"   Database: {stats['storage']['database_path']}")
    print(f"   Database size: {stats['storage']['database_size_bytes']:,} bytes")

    # Demonstrate experiment comparison
    run_ids = [exp.run_id for exp in results]
    comparison = runner.compare_runs(run_ids)

    print("\n🔄 PARAMETER ANALYSIS:")
    for param, info in comparison["parameter_comparison"].items():
        if info["varies"]:
            print(f"   {param}: varies across runs {info['unique_values']}")
        else:
            print(f"   {param}: constant = {info['unique_values'][0]}")

    print("\n📏 METRIC ANALYSIS:")
    for metric, info in comparison["metric_comparison"].items():
        if info["min"] is not None and info["max"] is not None:
            print(
                f"   {metric}: min={info['min']:.6f}, max={info['max']:.6f}, "
                f"mean={info['mean']:.6f}"
            )

    # Try export functionality if available
    try:
        from rexf.plugins.export import ExperimentExporter

        exporter = ExperimentExporter()

        # Export to JSON
        exporter.export_to_json(results)
        exporter.export_to_file(results, "monte_carlo_results.json")
        print(f"\n💾 Exported {len(results)} experiments to monte_carlo_results.json")

        # Try YAML export
        try:
            exporter.export_to_file(results, "monte_carlo_results.yaml")
            print(f"💾 Exported {len(results)} experiments to monte_carlo_results.yaml")
        except ImportError:
            print("⚠️  YAML export not available (PyYAML not installed)")

    except ImportError:
        print("⚠️  Export functionality not available")

    # Try visualization if available
    try:
        from rexf.plugins.visualization import ExperimentVisualizer

        viz = ExperimentVisualizer()

        # Create metric comparison plot
        fig = viz.plot_metrics(results)
        if fig:
            fig.savefig("experiment_metrics.png", dpi=300, bbox_inches="tight")
            plt.close(fig)
            print("📊 Created experiment_metrics.png")

        # Create timeline plot
        timeline_fig = viz.plot_experiment_timeline(results)
        if timeline_fig:
            timeline_fig.savefig(
                "experiment_timeline.png", dpi=300, bbox_inches="tight"
            )
            plt.close(timeline_fig)
            print("📊 Created experiment_timeline.png")

    except ImportError:
        print("⚠️  Visualization functionality not available")

    print("\n✅ Demo completed! All data saved to:")
    print("   📁 Database: monte_carlo_experiments.db")
    print("   📁 Artifacts: monte_carlo_artifacts/")
    print("   📁 Plots: *.png files")

    # Clean up
    runner.close()


if __name__ == "__main__":
    main()
