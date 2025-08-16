"""Monte Carlo π Estimation Demo for rexf.

This demo showcases how to use rexf for reproducible computational experiments.
It demonstrates:
- Defining experiments using decorators
- Running multiple experiments reproducibly
- Comparing results and artifacts visually
- Retrieving metrics and logs programmatically
- Exporting experiment data
"""

import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path to import rexf
sys.path.insert(0, str(Path(__file__).parent.parent))

from rexf import (
    experiment, param, result, metric, artifact, seed,
    ExperimentRunner, ExperimentVisualizer, ExperimentExporter
)


@experiment("monte_carlo_pi")
@param("n_samples", int, description="Number of random samples to generate")
@param("batch_size", int, default=1000, description="Size of each processing batch")
@seed("random_seed")
@metric("pi_estimate", float, description="Estimated value of π")
@metric("error", float, description="Absolute error from true π")
@metric("convergence_rate", float, description="Rate of convergence")
@artifact("convergence_plot", "pi_convergence.png", description="Plot showing convergence to π")
@artifact("sample_plot", "sample_distribution.png", description="Plot of random samples")
@result("final_pi", float, description="Final π estimate")
@result("samples_inside_circle", int, description="Number of samples inside unit circle")
def estimate_pi(n_samples, batch_size=1000, random_seed=42):
    """Estimate π using Monte Carlo method.
    
    This function generates random points in a unit square and counts
    how many fall inside a quarter circle to estimate π.
    """
    print(f"Starting Monte Carlo π estimation with {n_samples} samples")
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    samples_inside = 0
    total_samples = 0
    
    # Track convergence
    convergence_points = []
    sample_x = []
    sample_y = []
    
    # Process in batches
    for batch_start in range(0, n_samples, batch_size):
        batch_end = min(batch_start + batch_size, n_samples)
        current_batch_size = batch_end - batch_start
        
        # Generate random points in unit square
        x = np.random.uniform(-1, 1, current_batch_size)
        y = np.random.uniform(-1, 1, current_batch_size)
        
        # Count points inside unit circle
        inside_circle = (x**2 + y**2) <= 1
        samples_inside += np.sum(inside_circle)
        total_samples += current_batch_size
        
        # Store some samples for visualization (first batch only)
        if batch_start == 0:
            sample_x = x[:min(1000, len(x))]  # Limit for performance
            sample_y = y[:min(1000, len(y))]
        
        # Record convergence
        pi_estimate = 4 * samples_inside / total_samples
        convergence_points.append((total_samples, pi_estimate))
        
        # Progress update
        if total_samples % (n_samples // 10) == 0 or total_samples == n_samples:
            progress = total_samples / n_samples * 100
            print(f"Progress: {progress:.1f}% - Current π estimate: {pi_estimate:.6f}")
    
    # Final calculations
    final_pi_estimate = 4 * samples_inside / n_samples
    error = abs(final_pi_estimate - np.pi)
    
    # Calculate convergence rate (slope of last half of convergence)
    if len(convergence_points) > 10:
        last_half = convergence_points[len(convergence_points)//2:]
        x_vals = [p[0] for p in last_half]
        y_vals = [p[1] for p in last_half]
        convergence_rate = np.abs(np.polyfit(x_vals, y_vals, 1)[0])
    else:
        convergence_rate = 0.0
    
    print(f"Final π estimate: {final_pi_estimate:.6f} (error: {error:.6f})")
    
    # Create convergence plot
    fig, ax = plt.subplots(figsize=(10, 6))
    samples_counts, pi_estimates = zip(*convergence_points)
    ax.plot(samples_counts, pi_estimates, 'b-', linewidth=2, label='π estimate')
    ax.axhline(y=np.pi, color='r', linestyle='--', linewidth=2, label='True π')
    ax.set_xlabel('Number of samples')
    ax.set_ylabel('π estimate')
    ax.set_title('Monte Carlo π Estimation Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Create sample distribution plot
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    inside_mask = (sample_x**2 + sample_y**2) <= 1
    ax2.scatter(sample_x[inside_mask], sample_y[inside_mask], c='red', s=1, alpha=0.6, label='Inside circle')
    ax2.scatter(sample_x[~inside_mask], sample_y[~inside_mask], c='blue', s=1, alpha=0.6, label='Outside circle')
    
    # Draw unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)
    ax2.plot(circle_x, circle_y, 'k-', linewidth=2)
    
    ax2.set_xlim(-1.1, 1.1)
    ax2.set_ylim(-1.1, 1.1)
    ax2.set_aspect('equal')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Monte Carlo Sample Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    return {
        "final_pi": final_pi_estimate,
        "samples_inside_circle": int(samples_inside),
        "pi_estimate": final_pi_estimate,
        "error": error,
        "convergence_rate": convergence_rate,
        "convergence_plot": fig,
        "sample_plot": fig2
    }


def run_demo():
    """Run the complete Monte Carlo π estimation demo."""
    print("=" * 60)
    print("REXF MONTE CARLO π ESTIMATION DEMO")
    print("=" * 60)
    
    # Initialize experiment runner
    runner = ExperimentRunner(
        storage_path="demo_experiments.db",
        artifacts_path="demo_artifacts"
    )
    
    # Define experiment parameters for multiple runs
    experiment_configs = [
        {"n_samples": 10000, "batch_size": 1000, "random_seed": 42},
        {"n_samples": 50000, "batch_size": 5000, "random_seed": 42},
        {"n_samples": 100000, "batch_size": 10000, "random_seed": 42},
        {"n_samples": 10000, "batch_size": 1000, "random_seed": 123},
        {"n_samples": 50000, "batch_size": 5000, "random_seed": 456},
    ]
    
    print(f"\n1. Running {len(experiment_configs)} experiments...")
    run_ids = []
    
    for i, config in enumerate(experiment_configs, 1):
        print(f"\n--- Experiment {i}/{len(experiment_configs)} ---")
        run_id = runner.run(estimate_pi, **config)
        run_ids.append(run_id)
    
    print(f"\n2. Retrieving and analyzing results...")
    
    # Retrieve all experiments
    experiments = [runner.get_experiment(run_id) for run_id in run_ids]
    experiments = [exp for exp in experiments if exp is not None]
    
    # Print results summary
    print("\nExperiment Results Summary:")
    print("-" * 80)
    print(f"{'Run ID':<12} {'Samples':<10} {'Seed':<6} {'π Estimate':<12} {'Error':<10} {'Status':<10}")
    print("-" * 80)
    
    for exp in experiments:
        run_id_short = exp.run_id[:8]
        n_samples = exp.parameters.get('n_samples', 'N/A')
        seed = exp.random_seed or 'N/A'
        pi_est = exp.metrics.get('pi_estimate', 'N/A')
        error = exp.metrics.get('error', 'N/A')
        status = exp.status
        
        if isinstance(pi_est, float):
            pi_est = f"{pi_est:.6f}"
        if isinstance(error, float):
            error = f"{error:.6f}"
        
        print(f"{run_id_short:<12} {n_samples:<10} {seed:<6} {pi_est:<12} {error:<10} {status:<10}")
    
    print("\n3. Comparing experiments...")
    comparison = runner.compare_runs(run_ids)
    
    # Print parameter comparison
    print("\nParameter Variations:")
    for param_name, param_data in comparison['parameter_comparison'].items():
        if param_data['varies']:
            print(f"  {param_name}: {param_data['unique_values']}")
    
    # Print metric statistics
    print("\nMetric Statistics:")
    for metric_name, metric_data in comparison['metric_comparison'].items():
        if metric_data['min'] is not None:
            print(f"  {metric_name}: min={metric_data['min']:.6f}, "
                  f"max={metric_data['max']:.6f}, mean={metric_data['mean']:.6f}")
    
    print("\n4. Creating visualizations...")
    
    # Create visualizer
    visualizer = ExperimentVisualizer()
    
    # Create metrics comparison plot
    metrics_fig = visualizer.plot_metrics(experiments, save_path="demo_metrics_comparison.png")
    print("  ✓ Metrics comparison plot saved")
    
    # Create timeline plot
    timeline_fig = visualizer.plot_experiment_timeline(experiments, save_path="demo_timeline.png")
    print("  ✓ Timeline plot saved")
    
    # Create dashboard
    dashboard_fig = visualizer.create_dashboard(experiments, save_path="demo_dashboard.png")
    print("  ✓ Dashboard saved")
    
    # Create comparison table
    comparison_df = visualizer.create_comparison_table(experiments, save_path="demo_comparison.csv")
    print("  ✓ Comparison table saved")
    print(f"    Table shape: {comparison_df.shape}")
    
    print("\n5. Exporting experiment data...")
    
    # Export to JSON and YAML
    exporter = ExperimentExporter()
    
    # Export all experiments
    exporter.export_to_file(experiments, "demo_all_experiments.json", format="json")
    exporter.export_to_file(experiments, "demo_all_experiments.yaml", format="yaml")
    print("  ✓ All experiments exported to JSON and YAML")
    
    # Export comparison
    exporter.export_comparison(experiments, "demo_comparison.json")
    print("  ✓ Comparison exported to JSON")
    
    # Export single experiment (best result)
    best_experiment = min(experiments, key=lambda x: x.metrics.get('error', float('inf')))
    exporter.export_to_file(best_experiment, "demo_best_experiment.json", include_artifacts=True)
    print(f"  ✓ Best experiment (run_id: {best_experiment.run_id[:8]}) exported")
    
    print("\n6. Programmatic data access...")
    
    # Show how to access data programmatically
    print("\nAccessing experiment data programmatically:")
    
    # Get best and worst experiments
    best_exp = min(experiments, key=lambda x: x.metrics.get('error', float('inf')))
    worst_exp = max(experiments, key=lambda x: x.metrics.get('error', 0))
    
    print(f"\nBest experiment (lowest error):")
    print(f"  Run ID: {best_exp.run_id}")
    print(f"  Parameters: {best_exp.parameters}")
    print(f"  π estimate: {best_exp.metrics['pi_estimate']:.6f}")
    print(f"  Error: {best_exp.metrics['error']:.6f}")
    print(f"  Artifacts: {list(best_exp.artifacts.keys())}")
    
    print(f"\nWorst experiment (highest error):")
    print(f"  Run ID: {worst_exp.run_id}")
    print(f"  π estimate: {worst_exp.metrics['pi_estimate']:.6f}")
    print(f"  Error: {worst_exp.metrics['error']:.6f}")
    
    # Show artifact access
    print(f"\nArtifact paths for best experiment:")
    for artifact_name, artifact_path in best_exp.artifacts.items():
        print(f"  {artifact_name}: {artifact_path}")
    
    # Get storage statistics
    print("\n7. Storage statistics...")
    stats = runner.get_stats()
    print(f"  Database experiments: {stats['storage']['total_experiments']}")
    print(f"  Database size: {stats['storage']['db_size_bytes']} bytes")
    print(f"  Artifact runs: {stats['artifacts']['total_runs']}")
    print(f"  Total artifacts: {stats['artifacts']['total_artifacts']}")
    print(f"  Artifacts size: {stats['artifacts']['total_size_bytes']} bytes")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  • demo_experiments.db - SQLite database with experiment metadata")
    print("  • demo_artifacts/ - Directory with experiment artifacts")
    print("  • demo_*.png - Visualization plots")
    print("  • demo_*.json/yaml - Exported experiment data")
    print("  • demo_comparison.csv - Comparison table")
    print("\nYou can now:")
    print("  1. Examine the artifacts in the demo_artifacts/ directory")
    print("  2. View the visualization plots")
    print("  3. Import the exported data into other tools")
    print("  4. Query the SQLite database directly if needed")


if __name__ == "__main__":
    # Set up matplotlib for non-interactive backend if needed
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    
    run_demo()
