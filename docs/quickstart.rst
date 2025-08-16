🚀 Quick Start Guide
==================

This guide will get you up and running with RexF in under 5 minutes.

Installation
------------

Install RexF using pip:

.. code-block:: bash

   pip install rexf

That's it! No additional setup required.

Your First Experiment
---------------------

Let's create a simple experiment to estimate π using Monte Carlo methods:

.. code-block:: python

   import math
   import random
   from rexf import experiment, run

   @experiment
   def estimate_pi(num_samples=10000):
       """Estimate π using Monte Carlo methods."""
       inside_circle = 0
       
       for _ in range(num_samples):
           x, y = random.uniform(-1, 1), random.uniform(-1, 1)
           if x*x + y*y <= 1:
               inside_circle += 1
       
       pi_estimate = 4 * inside_circle / num_samples
       error = abs(pi_estimate - math.pi)
       
       return {
           "pi_estimate": pi_estimate,
           "error": error,
           "accuracy": 1 - (error / math.pi)
       }

   # Run a single experiment
   run_id = run.single(estimate_pi, num_samples=50000)
   print(f"Experiment completed with ID: {run_id}")

Running this code will:

1. Execute your experiment function
2. Automatically capture parameters (``num_samples=50000``)
3. Store the results (``pi_estimate``, ``error``, ``accuracy``)
4. Track execution time and environment info
5. Generate a unique run ID

Exploring Results
----------------

Now let's explore what RexF captured:

.. code-block:: python

   # Get recent experiments
   recent_runs = run.recent(hours=1)
   print(f"Found {len(recent_runs)} recent experiments")

   # Get the best experiments by accuracy
   best_runs = run.best(metric="accuracy", top=3)
   for exp in best_runs:
       print(f"Run {exp.run_id[:8]}: accuracy={exp.metrics['accuracy']:.4f}")

   # Generate insights
   insights = run.insights()
   print(f"Success rate: {insights['summary']['success_rate']:.1%}")

Auto-Exploration
---------------

Let RexF automatically explore different parameter values:

.. code-block:: python

   # Automatically explore parameter space
   run_ids = run.auto_explore(
       estimate_pi, 
       strategy="random",  # or "grid", "adaptive"
       budget=10,  # number of experiments to run
       optimization_target="accuracy"
   )

   print(f"Completed {len(run_ids)} experiments")

   # Find the best result
   best = run.best(metric="accuracy", top=1)[0]
   print(f"Best accuracy: {best.metrics['accuracy']:.4f}")
   print(f"With parameters: {best.parameters}")

Querying Experiments
-------------------

Find experiments using simple expressions:

.. code-block:: python

   # Find high-accuracy experiments
   high_acc = run.find("accuracy > 0.99")
   print(f"Found {len(high_acc)} high-accuracy experiments")

   # Find experiments with specific parameter ranges
   large_samples = run.find("param_num_samples > 25000")
   print(f"Found {len(large_samples)} experiments with large sample sizes")

   # Combine conditions
   recent_good = run.find("accuracy > 0.95 and num_samples > 10000")

Web Dashboard
------------

Launch the interactive web dashboard:

.. code-block:: python

   # This will open your browser to http://localhost:8080
   run.dashboard()

The dashboard provides:

- Real-time experiment monitoring
- Interactive charts and visualizations
- Experiment comparison tools
- Parameter space exploration
- Automated insights

CLI Analytics
------------

You can also analyze experiments from the command line:

.. code-block:: bash

   # Show experiment summary
   rexf-analytics --summary

   # Query experiments
   rexf-analytics --query "accuracy > 0.99"

   # Generate insights
   rexf-analytics --insights

   # Launch web dashboard
   rexf-analytics --dashboard

Next Steps
---------

Now that you've got the basics down, explore:

- :doc:`basic_usage` - Learn all core features
- :doc:`advanced_features` - Advanced exploration and insights
- :doc:`tutorials/monte_carlo` - Complete Monte Carlo tutorial
- :doc:`web_dashboard` - Dashboard features and customization

🎉 You're ready to accelerate your research with RexF!
