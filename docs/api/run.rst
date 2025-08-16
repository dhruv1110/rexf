🚀 Run Module API Reference
============================

The ``run`` module provides the main interface for executing and analyzing experiments.

.. automodule:: rexf.run
   :members:
   :undoc-members:
   :show-inheritance:

Quick Reference
--------------

.. code-block:: python

   from rexf import run

   # Execute experiments
   run_id = run.single(my_experiment, param1=value1)
   run_ids = run.auto_explore(my_experiment, strategy="random", budget=10)

   # Retrieve results
   experiments = run.all()
   recent = run.recent(hours=6)
   best = run.best(metric="accuracy", top=5)
   found = run.find("accuracy > 0.9")

   # Analysis and insights
   insights = run.insights()
   suggestions = run.suggest(my_experiment, count=5)
   run.compare(best[:3])

   # Interactive tools
   run.dashboard()

Execution Functions
------------------

single()
~~~~~~~~

Execute a single experiment with specified parameters.

.. autofunction:: rexf.run.single

.. code-block:: python

   # Basic usage
   run_id = run.single(my_experiment)

   # With parameters
   run_id = run.single(my_experiment, learning_rate=0.01, epochs=100)

   # With keyword arguments
   run_id = run.single(
       my_experiment,
       **{"learning_rate": 0.01, "batch_size": 32}
   )

auto_explore()
~~~~~~~~~~~~~

Automatically explore parameter space using different strategies.

.. autofunction:: rexf.run.auto_explore

.. code-block:: python

   # Random exploration
   run_ids = run.auto_explore(
       my_experiment,
       strategy="random",
       budget=20,
       optimization_target="accuracy"
   )

   # Grid search
   run_ids = run.auto_explore(
       my_experiment,
       strategy="grid",
       budget=15,
       parameter_ranges={
           "learning_rate": [0.001, 0.01, 0.1],
           "batch_size": [32, 64, 128]
       }
   )

   # Adaptive/Bayesian optimization
   run_ids = run.auto_explore(
       my_experiment,
       strategy="adaptive",
       budget=25,
       optimization_target="accuracy"
   )

Retrieval Functions
------------------

all()
~~~~~

Get all experiments from the database.

.. autofunction:: rexf.run.all

.. code-block:: python

   experiments = run.all()
   print(f"Total experiments: {len(experiments)}")

recent()
~~~~~~~~

Get experiments from recent time period.

.. autofunction:: rexf.run.recent

.. code-block:: python

   # Last 24 hours (default)
   recent_experiments = run.recent()

   # Last 6 hours
   last_6h = run.recent(hours=6)

   # Last 2 days
   last_2d = run.recent(hours=48)

best()
~~~~~~

Get top-performing experiments by a specific metric.

.. autofunction:: rexf.run.best

.. code-block:: python

   # Top 5 by accuracy (descending)
   top_accuracy = run.best(metric="accuracy", top=5)

   # Fastest experiments (ascending order)
   fastest = run.best(metric="training_time", top=3, ascending=True)

   # Best from specific experiment
   best_ml = run.best(
       metric="f1_score", 
       top=10, 
       experiment_name="ml_experiment"
   )

find()
~~~~~~

Query experiments using expression strings.

.. autofunction:: rexf.run.find

.. code-block:: python

   # Simple comparisons
   high_acc = run.find("accuracy > 0.9")
   fast_runs = run.find("training_time < 60")
   recent_good = run.find("accuracy > 0.8 and start_time > '2024-01-01'")

   # Parameter queries (use param_ prefix)
   low_lr = run.find("param_learning_rate < 0.01")
   large_batches = run.find("param_batch_size >= 64")

   # Range queries
   moderate_acc = run.find("accuracy between 0.7 and 0.9")

   # Status queries
   completed = run.find("status == 'completed'")
   failed = run.find("status == 'failed'")

Query Operators
~~~~~~~~~~~~~~

Supported operators in ``find()`` expressions:

- **Comparison**: ``>``, ``>=``, ``<``, ``<=``, ``==``, ``!=``
- **Range**: ``between`` (e.g., ``"value between 0.1 and 0.9"``)
- **Logical**: ``and`` (``or`` support coming soon)

Query Examples:

.. code-block:: python

   # Numeric comparisons
   run.find("accuracy > 0.95")
   run.find("loss <= 0.1") 
   run.find("epochs != 100")

   # String comparisons
   run.find("status == 'completed'")
   run.find("param_optimizer == 'adam'")

   # Range queries
   run.find("accuracy between 0.8 and 0.95")
   run.find("param_learning_rate between 0.001 and 0.1")

   # Combined conditions
   run.find("accuracy > 0.9 and training_time < 300")
   run.find("param_batch_size >= 32 and status == 'completed'")

by_name()
~~~~~~~~~

Get experiments by experiment function name.

.. autofunction:: rexf.run.by_name

.. code-block:: python

   # Get all runs of specific experiment
   ml_experiments = run.by_name("ml_hyperparameter_search")
   
   # Get recent runs of specific experiment
   recent_ml = run.by_name("ml_experiment", limit=10)

get_by_id()
~~~~~~~~~~

Get a specific experiment by its run ID.

.. autofunction:: rexf.run.get_by_id

.. code-block:: python

   # Get specific experiment
   experiment = run.get_by_id("abc123def456")
   
   if experiment:
       print(f"Status: {experiment.status}")
       print(f"Metrics: {experiment.metrics}")
       print(f"Parameters: {experiment.parameters}")

Analysis Functions
-----------------

insights()
~~~~~~~~~~

Generate intelligent insights from experiment data.

.. autofunction:: rexf.run.insights

.. code-block:: python

   # Overall insights
   insights = run.insights()

   # Insights for specific experiment
   ml_insights = run.insights(experiment_name="ml_experiment")

   # Access insight categories
   summary = insights["summary"]
   param_insights = insights["parameter_insights"]
   performance = insights["performance_insights"]
   correlations = insights["correlation_insights"]
   anomalies = insights["anomaly_insights"]
   recommendations = insights["recommendations"]

Example insight structure:

.. code-block:: python

   {
       "summary": {
           "total_experiments": 150,
           "success_rate": 0.92,
           "avg_accuracy": 0.847,
           "total_duration": 3600.5
       },
       "parameter_insights": {
           "learning_rate": {
               "impact_score": 0.85,
               "optimal_range": [0.001, 0.01],
               "correlation": 0.72
           }
       },
       "performance_insights": {
           "fastest_duration": 45.2,
           "slowest_duration": 320.1,
           "avg_duration": 120.5
       },
       "recommendations": [
           {
               "title": "Optimize learning rate",
               "description": "Try learning rates between 0.005 and 0.015",
               "priority": "high"
           }
       ]
   }

suggest()
~~~~~~~~~

Get intelligent suggestions for next experiments.

.. autofunction:: rexf.run.suggest

.. code-block:: python

   # Basic suggestions
   suggestions = run.suggest(my_experiment, count=5)

   # Exploitation strategy (focus on best regions)
   exploit_suggestions = run.suggest(
       my_experiment,
       count=3,
       strategy="exploit",
       optimization_target="accuracy"
   )

   # Exploration strategy (try new regions)
   explore_suggestions = run.suggest(
       my_experiment,
       count=5,
       strategy="explore"
   )

   # Balanced strategy (mix of both)
   balanced_suggestions = run.suggest(
       my_experiment,
       count=4,
       strategy="balanced",
       optimization_target="f1_score"
   )

   # Use suggestions
   for suggestion in suggestions["suggestions"]:
       print(f"Try: {suggestion['parameters']}")
       print(f"Reason: {suggestion['reasoning']}")
       print(f"Expected improvement: {suggestion['expected_improvement']}")

compare()
~~~~~~~~~

Compare multiple experiments side-by-side.

.. autofunction:: rexf.run.compare

.. code-block:: python

   # Compare best experiments
   best_experiments = run.best(metric="accuracy", top=3)
   run.compare(best_experiments)

   # Compare specific experiments by ID
   run.compare(["run_id_1", "run_id_2", "run_id_3"])

   # Compare from query results
   high_accuracy = run.find("accuracy > 0.9")
   run.compare(high_accuracy[:5])

The comparison displays:

- Parameter differences between experiments
- Metric comparisons with statistical analysis
- Performance analysis
- Recommendations based on differences

Interactive Functions
--------------------

dashboard()
~~~~~~~~~~~

Launch the interactive web dashboard.

.. autofunction:: rexf.run.dashboard

.. code-block:: python

   # Launch with defaults (localhost:8080)
   run.dashboard()

   # Custom host and port
   run.dashboard(host="0.0.0.0", port=9000)

   # Don't open browser automatically
   run.dashboard(open_browser=False)

query_help()
~~~~~~~~~~~~

Get help with query syntax and suggestions.

.. autofunction:: rexf.run.query_help

.. code-block:: python

   # Get query suggestions
   suggestions = run.query_help()
   
   print("Suggested queries:")
   for suggestion in suggestions:
       print(f"  {suggestion}")

   # Example output:
   # accuracy > 0.9
   # param_learning_rate < 0.01
   # training_time between 60 and 300

Utility Functions
----------------

summary()
~~~~~~~~~

Get a quick summary of all experiments.

.. autofunction:: rexf.run.summary

.. code-block:: python

   summary = run.summary()
   
   print(f"Total experiments: {summary['total']}")
   print(f"Successful: {summary['successful']}")
   print(f"Failed: {summary['failed']}")
   print(f"Average duration: {summary['avg_duration']:.2f}s")

stats()
~~~~~~~

Get detailed statistics about experiments.

.. autofunction:: rexf.run.stats

.. code-block:: python

   stats = run.stats()
   
   # Experiment counts by name
   by_name = stats["by_experiment_name"]
   
   # Success/failure rates
   success_rate = stats["success_rate"]
   
   # Performance statistics
   performance = stats["performance_stats"]

Advanced Usage
-------------

Working with ExperimentRunner
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For advanced use cases, you can work directly with the ExperimentRunner:

.. code-block:: python

   from rexf.run import ExperimentRunner

   # Create custom runner with specific storage path
   runner = ExperimentRunner(storage_path="custom_experiments.db")

   # Use custom runner
   run_id = runner.single(my_experiment, param1=value1)
   
   # Get insights from custom runner
   insights = runner.insights()
   
   # Clean up
   runner.close()

Batch Operations
~~~~~~~~~~~~~~~

For efficient batch processing:

.. code-block:: python

   # Process multiple parameter sets efficiently
   param_sets = [
       {"learning_rate": 0.01, "batch_size": 32},
       {"learning_rate": 0.005, "batch_size": 64},
       {"learning_rate": 0.001, "batch_size": 128},
   ]

   run_ids = []
   for params in param_sets:
       run_id = run.single(my_experiment, **params)
       run_ids.append(run_id)

   # Analyze batch results
   batch_experiments = [run.get_by_id(rid) for rid in run_ids]
   run.compare(batch_experiments)

Error Handling
-------------

The run module gracefully handles various error conditions:

.. code-block:: python

   # Experiment function that might fail
   @experiment
   def risky_experiment(fail_probability=0.1):
       import random
       if random.random() < fail_probability:
           raise ValueError("Simulated failure")
       return {"success": True}

   # Failed experiments still return run_ids
   run_id = run.single(risky_experiment, fail_probability=0.5)
   
   # Check if experiment failed
   experiment = run.get_by_id(run_id)
   if experiment.status == "failed":
       error_msg = experiment.metadata.get("error", "Unknown error")
       print(f"Experiment failed: {error_msg}")

   # Query failed experiments
   failed_experiments = run.find("status == 'failed'")
   print(f"Found {len(failed_experiments)} failed experiments")

Performance Tips
---------------

1. **Use batch operations** for multiple experiments
2. **Limit query results** with ``run.find()`` for large datasets
3. **Close runners** when using custom ExperimentRunner instances
4. **Use specific queries** instead of retrieving all experiments

.. code-block:: python

   # Good: Specific query
   recent_good = run.find("accuracy > 0.9 and start_time > '2024-01-01'")

   # Less efficient: Get all then filter
   all_experiments = run.all()
   recent_good = [exp for exp in all_experiments 
                  if exp.metrics.get("accuracy", 0) > 0.9]
