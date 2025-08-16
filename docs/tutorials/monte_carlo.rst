🎯 Tutorial: Monte Carlo π Estimation
=====================================

This tutorial demonstrates RexF's capabilities using a Monte Carlo method to estimate π. You'll learn the complete workflow from experiment design to analysis.

Overview
--------

We'll use Monte Carlo simulation to estimate π by randomly sampling points in a unit square and counting how many fall inside a unit circle. This is a perfect example for RexF because:

- It has clear parameters (number of samples, sampling method)
- It produces measurable metrics (accuracy, error, performance)
- It benefits from parameter exploration
- Results are easy to analyze and compare

The Mathematical Foundation
--------------------------

The Monte Carlo method for π estimation:

1. Generate random points (x, y) in the square [-1, 1] × [-1, 1]
2. Check if each point is inside the unit circle: x² + y² ≤ 1
3. The ratio of points inside the circle approximates π/4
4. Therefore: π ≈ 4 × (points inside circle) / (total points)

Setting Up the Experiment
-------------------------

First, let's create our basic π estimation experiment:

.. code-block:: python

   import math
   import random
   import time
   from rexf import experiment, run

   @experiment
   def estimate_pi(num_samples=10000, method="uniform", random_seed=None):
       """
       Estimate π using Monte Carlo method.
       
       Args:
           num_samples: Number of random points to generate
           method: Sampling method ('uniform', 'gaussian', 'stratified')
           random_seed: Random seed for reproducibility
       """
       if random_seed:
           random.seed(random_seed)
       
       start_time = time.time()
       inside_circle = 0
       
       for _ in range(num_samples):
           # Generate random point based on method
           if method == "uniform":
               x, y = random.uniform(-1, 1), random.uniform(-1, 1)
           elif method == "gaussian":
               # Gaussian sampling (truncated to [-1, 1])
               x = max(-1, min(1, random.gauss(0, 0.5)))
               y = max(-1, min(1, random.gauss(0, 0.5)))
           elif method == "stratified":
               # Stratified sampling for better coverage
               i = _ % int(math.sqrt(num_samples))
               j = _ // int(math.sqrt(num_samples))
               grid_size = int(math.sqrt(num_samples))
               x = -1 + (2 * i + random.random()) / grid_size
               y = -1 + (2 * j + random.random()) / grid_size
           
           # Check if point is inside unit circle
           if x*x + y*y <= 1:
               inside_circle += 1
       
       computation_time = time.time() - start_time
       
       # Calculate results
       pi_estimate = 4 * inside_circle / num_samples
       absolute_error = abs(pi_estimate - math.pi)
       relative_error_percent = (absolute_error / math.pi) * 100
       
       # Calculate performance metrics
       accuracy_score = max(0, 1 - (absolute_error / math.pi))
       efficiency_score = accuracy_score / computation_time
       
       return {
           "pi_estimate": pi_estimate,
           "absolute_error": absolute_error,
           "relative_error_percent": relative_error_percent,
           "accuracy_score": accuracy_score,
           "efficiency_score": efficiency_score,
           "computation_time": computation_time,
           "samples_per_second": num_samples / computation_time,
           "inside_circle_count": inside_circle
       }

Running Your First Experiments
-----------------------------

Let's start with some basic experiments:

.. code-block:: python

   # Run a single experiment
   run_id = run.single(estimate_pi, num_samples=50000)
   print(f"Experiment completed: {run_id}")

   # Run with different parameters
   run.single(estimate_pi, num_samples=100000, method="gaussian")
   run.single(estimate_pi, num_samples=25000, method="stratified")

   # Get recent results
   recent_experiments = run.recent(hours=1)
   for exp in recent_experiments:
       pi_est = exp.metrics.get("pi_estimate", 0)
       error = exp.metrics.get("absolute_error", 0)
       print(f"π estimate: {pi_est:.6f}, error: {error:.6f}")

Exploring the Parameter Space
----------------------------

Now let's systematically explore different parameter combinations:

.. code-block:: python

   # Auto-explore with random strategy
   print("🔍 Random exploration...")
   random_run_ids = run.auto_explore(
       estimate_pi,
       strategy="random",
       budget=15,
       optimization_target="accuracy_score",
       parameter_ranges={
           "num_samples": (10000, 100000),
           "method": ["uniform", "gaussian", "stratified"]
       }
   )

   # Grid search over specific values
   print("📊 Grid search...")
   grid_run_ids = run.auto_explore(
       estimate_pi,
       strategy="grid",
       budget=12,
       optimization_target="efficiency_score",
       parameter_ranges={
           "num_samples": [25000, 50000, 100000],
           "method": ["uniform", "gaussian", "stratified"]
       }
   )

   # Adaptive exploration (learns from results)
   print("🧠 Adaptive exploration...")
   adaptive_run_ids = run.auto_explore(
       estimate_pi,
       strategy="adaptive",
       budget=20,
       optimization_target="accuracy_score"
   )

   print(f"Total experiments run: {len(random_run_ids + grid_run_ids + adaptive_run_ids)}")

Analyzing Results
----------------

Let's analyze our experiment results:

.. code-block:: python

   # Get overall insights
   insights = run.insights()
   
   print("📈 Experiment Insights:")
   print(f"Total experiments: {insights['summary']['total_experiments']}")
   print(f"Success rate: {insights['summary']['success_rate']:.1%}")
   
   # Parameter impact analysis
   param_insights = insights["parameter_insights"]
   for param, analysis in param_insights.items():
       if "impact_score" in analysis:
           print(f"{param} impact: {analysis['impact_score']:.3f}")

   # Find the best experiments
   best_accuracy = run.best(metric="accuracy_score", top=5)
   print("\n🏆 Top 5 by accuracy:")
   for i, exp in enumerate(best_accuracy, 1):
       acc = exp.metrics["accuracy_score"]
       method = exp.parameters["method"]
       samples = exp.parameters["num_samples"]
       print(f"{i}. Accuracy: {acc:.6f}, Method: {method}, Samples: {samples}")

   # Find the most efficient experiments
   best_efficiency = run.best(metric="efficiency_score", top=3)
   print("\n⚡ Top 3 by efficiency:")
   for i, exp in enumerate(best_efficiency, 1):
       eff = exp.metrics["efficiency_score"]
       time = exp.metrics["computation_time"]
       print(f"{i}. Efficiency: {eff:.3f}, Time: {time:.2f}s")

Advanced Analysis
----------------

Let's dive deeper into the results:

.. code-block:: python

   # Compare different methods
   uniform_experiments = run.find("param_method == 'uniform'")
   gaussian_experiments = run.find("param_method == 'gaussian'")
   stratified_experiments = run.find("param_method == 'stratified'")

   print(f"\nMethod comparison:")
   print(f"Uniform: {len(uniform_experiments)} experiments")
   print(f"Gaussian: {len(gaussian_experiments)} experiments")
   print(f"Stratified: {len(stratified_experiments)} experiments")

   # Statistical analysis
   def analyze_method(experiments, method_name):
       if not experiments:
           return
       
       accuracies = [exp.metrics["accuracy_score"] for exp in experiments]
       times = [exp.metrics["computation_time"] for exp in experiments]
       
       avg_accuracy = sum(accuracies) / len(accuracies)
       avg_time = sum(times) / len(times)
       
       print(f"{method_name}:")
       print(f"  Average accuracy: {avg_accuracy:.6f}")
       print(f"  Average time: {avg_time:.3f}s")
       print(f"  Best accuracy: {max(accuracies):.6f}")

   analyze_method(uniform_experiments, "Uniform")
   analyze_method(gaussian_experiments, "Gaussian")
   analyze_method(stratified_experiments, "Stratified")

   # Find experiments with high sample counts
   high_sample_experiments = run.find("param_num_samples > 75000")
   print(f"\nHigh sample experiments: {len(high_sample_experiments)}")

   # Find highly accurate experiments
   accurate_experiments = run.find("accuracy_score > 0.999")
   print(f"Highly accurate experiments: {len(accurate_experiments)}")

Visualizing Results
------------------

Launch the web dashboard to visualize your results:

.. code-block:: python

   # Launch interactive dashboard
   run.dashboard()

In the dashboard, you can:

- View accuracy vs. number of samples scatter plots
- Compare different methods visually
- See computation time trends
- Explore parameter space interactively

Getting Intelligent Suggestions
-------------------------------

Let RexF suggest optimal next experiments:

.. code-block:: python

   # Get suggestions for next experiments
   suggestions = run.suggest(
       estimate_pi,
       count=5,
       strategy="balanced",  # Balance exploration and exploitation
       optimization_target="accuracy_score"
   )

   print("🎯 Suggested next experiments:")
   for i, suggestion in enumerate(suggestions["suggestions"], 1):
       params = suggestion["parameters"]
       reasoning = suggestion["reasoning"]
       expected_improvement = suggestion.get("expected_improvement", 0)
       
       print(f"{i}. Samples: {params['num_samples']}, Method: {params['method']}")
       print(f"   Reason: {reasoning}")
       print(f"   Expected improvement: {expected_improvement:.4f}")

   # Run the top suggestion
   if suggestions["suggestions"]:
       top_suggestion = suggestions["suggestions"][0]
       print(f"\n🚀 Running top suggestion...")
       run_id = run.single(estimate_pi, **top_suggestion["parameters"])
       
       # Check results
       result = run.get_by_id(run_id)
       new_accuracy = result.metrics["accuracy_score"]
       print(f"New experiment accuracy: {new_accuracy:.6f}")

Reproducibility and Error Analysis
----------------------------------

Let's examine reproducibility and analyze any failures:

.. code-block:: python

   # Run reproducible experiments with fixed seeds
   print("🔬 Reproducibility test...")
   reproducible_runs = []
   for i in range(3):
       run_id = run.single(
           estimate_pi,
           num_samples=50000,
           method="uniform",
           random_seed=42  # Fixed seed
       )
       reproducible_runs.append(run_id)

   # Check if results are identical
   results = [run.get_by_id(rid) for rid in reproducible_runs]
   pi_estimates = [r.metrics["pi_estimate"] for r in results]
   
   print(f"Reproducible results: {all(p == pi_estimates[0] for p in pi_estimates)}")
   print(f"π estimates: {pi_estimates}")

   # Check for any failed experiments
   failed_experiments = run.find("status == 'failed'")
   if failed_experiments:
       print(f"\n⚠️ Found {len(failed_experiments)} failed experiments:")
       for exp in failed_experiments:
           error_msg = exp.metadata.get("error", "Unknown error")
           print(f"Run {exp.run_id[:8]}: {error_msg}")

Comparative Analysis
-------------------

Compare your best results with theoretical expectations:

.. code-block:: python

   # Theoretical analysis
   def theoretical_error(num_samples):
       """Theoretical standard error for Monte Carlo π estimation."""
       return math.sqrt(math.pi * (4 - math.pi) / num_samples)

   # Compare with best experiments
   best_experiments = run.best(metric="accuracy_score", top=10)
   
   print("\n📊 Theoretical vs Actual Performance:")
   print("Samples\tActual Error\tTheoretical Error\tRatio")
   print("-" * 50)
   
   for exp in best_experiments:
       samples = exp.parameters["num_samples"]
       actual_error = exp.metrics["absolute_error"]
       theoretical = theoretical_error(samples)
       ratio = actual_error / theoretical if theoretical > 0 else float('inf')
       
       print(f"{samples}\t{actual_error:.6f}\t{theoretical:.6f}\t\t{ratio:.2f}")

Export Results for Publication
-----------------------------

Export your results for use in papers or reports:

.. code-block:: python

   import pandas as pd

   # Get all successful experiments
   successful_experiments = run.find("status == 'completed'")

   # Convert to DataFrame for analysis
   data = []
   for exp in successful_experiments:
       row = {
           'run_id': exp.run_id,
           'method': exp.parameters['method'],
           'num_samples': exp.parameters['num_samples'],
           'pi_estimate': exp.metrics['pi_estimate'],
           'absolute_error': exp.metrics['absolute_error'],
           'relative_error_percent': exp.metrics['relative_error_percent'],
           'accuracy_score': exp.metrics['accuracy_score'],
           'computation_time': exp.metrics['computation_time'],
           'efficiency_score': exp.metrics['efficiency_score']
       }
       data.append(row)

   df = pd.DataFrame(data)
   
   # Save for publication
   df.to_csv('pi_estimation_results.csv', index=False)
   
   # Summary statistics by method
   summary = df.groupby('method').agg({
       'absolute_error': ['mean', 'std', 'min'],
       'computation_time': ['mean', 'std'],
       'efficiency_score': ['mean', 'std']
   }).round(6)
   
   print("\n📋 Summary by Method:")
   print(summary)

Command Line Analysis
--------------------

You can also analyze results from the command line:

.. code-block:: bash

   # Quick summary
   rexf-analytics --summary

   # Find best accuracy experiments
   rexf-analytics --query "accuracy_score > 0.999"

   # Compare different methods
   rexf-analytics --query "param_method == 'stratified'" --format csv

   # Export all results
   rexf-analytics --list --format csv --output pi_experiments.csv

   # Launch dashboard
   rexf-analytics --dashboard

Key Learnings
------------

From this tutorial, you've learned:

1. **Experiment Design**: How to structure experiments with clear parameters and metrics
2. **Parameter Exploration**: Using different strategies (random, grid, adaptive) to explore parameter space
3. **Result Analysis**: Getting insights, finding best experiments, and understanding patterns
4. **Intelligent Suggestions**: Leveraging RexF's intelligence to guide future experiments
5. **Reproducibility**: Ensuring experiments can be reproduced with proper seed management
6. **Visualization**: Using the dashboard for interactive exploration
7. **Export and Integration**: Getting data out for external analysis and publication

Best Practices Demonstrated
--------------------------

- **Meaningful Metrics**: Return multiple relevant metrics (accuracy, efficiency, performance)
- **Parameter Validation**: Handle different parameter types and ranges appropriately
- **Error Handling**: Robust experiment design that can handle edge cases
- **Reproducibility**: Use random seeds for reproducible results when needed
- **Performance Tracking**: Monitor computational efficiency alongside accuracy
- **Comprehensive Analysis**: Use multiple analysis approaches (insights, queries, comparisons)

Next Steps
---------

Try extending this tutorial:

1. **Add More Sampling Methods**: Implement quasi-Monte Carlo or importance sampling
2. **Multi-dimensional Analysis**: Extend to estimate other mathematical constants
3. **Parallel Processing**: Run multiple experiments in parallel
4. **Real-time Monitoring**: Use the dashboard to monitor long-running experiments
5. **Advanced Analytics**: Implement custom analysis functions for deeper insights

Continue with:

- :doc:`../advanced_features` - Advanced parameter exploration and analysis
- :doc:`../web_dashboard` - Interactive visualization and monitoring
- :doc:`machine_learning` - Apply RexF to machine learning experiments
