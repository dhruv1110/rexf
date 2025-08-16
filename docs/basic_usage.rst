📖 Basic Usage
=============

This guide covers all the essential RexF features you'll use in your daily research workflow.

The @experiment Decorator
-------------------------

The ``@experiment`` decorator is the heart of RexF. It automatically captures everything about your function execution:

.. code-block:: python

   from rexf import experiment, run

   @experiment
   def my_experiment(learning_rate=0.01, epochs=10, model_type="cnn"):
       """Example ML experiment."""
       # Your experiment code
       model = create_model(model_type)
       accuracy = train_model(model, learning_rate, epochs)
       
       # Return metrics (automatically detected)
       return {
           "accuracy": accuracy,
           "loss": 1 - accuracy,
           "training_time": get_training_time(),
           "model_size": get_model_size(model)
       }

What gets captured automatically:

- **Parameters**: All function arguments (``learning_rate``, ``epochs``, ``model_type``)
- **Results**: Everything in the return dictionary
- **Metadata**: Execution time, Git commit, Python environment
- **Reproducibility**: Random seeds, system info

Running Experiments
-------------------

Single Experiments
~~~~~~~~~~~~~~~~~

Run one experiment at a time:

.. code-block:: python

   # Run with default parameters
   run_id = run.single(my_experiment)

   # Run with custom parameters
   run_id = run.single(my_experiment, learning_rate=0.005, epochs=20)

   # Run with keyword arguments
   run_id = run.single(
       my_experiment,
       learning_rate=0.001,
       epochs=50,
       model_type="transformer"
   )

Batch Experiments
~~~~~~~~~~~~~~~~

Run multiple experiments efficiently:

.. code-block:: python

   # Define parameter combinations
   param_sets = [
       {"learning_rate": 0.01, "epochs": 10},
       {"learning_rate": 0.005, "epochs": 20},
       {"learning_rate": 0.001, "epochs": 50},
   ]

   # Run all combinations
   run_ids = []
   for params in param_sets:
       run_id = run.single(my_experiment, **params)
       run_ids.append(run_id)

   print(f"Completed {len(run_ids)} experiments")

Retrieving Results
-----------------

List Experiments
~~~~~~~~~~~~~~~

.. code-block:: python

   # Get all experiments
   all_experiments = run.all()

   # Get recent experiments (last 24 hours by default)
   recent = run.recent()

   # Get experiments from last 6 hours
   recent_6h = run.recent(hours=6)

   # Get experiments by name
   my_experiments = run.by_name("my_experiment")

Find Best Results
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Get top 5 experiments by accuracy
   top_5 = run.best(metric="accuracy", top=5)

   # Get best experiment by custom metric
   fastest = run.best(metric="training_time", top=1, ascending=True)

   # Print results
   for exp in top_5:
       print(f"Run {exp.run_id[:8]}: {exp.metrics['accuracy']:.4f}")

Query with Expressions
~~~~~~~~~~~~~~~~~~~~~

Use simple expressions to find specific experiments:

.. code-block:: python

   # Find high-accuracy experiments
   high_acc = run.find("accuracy > 0.9")

   # Find fast training experiments
   fast_training = run.find("training_time < 100")

   # Find experiments with specific parameters
   cnn_experiments = run.find("param_model_type == 'cnn'")

   # Combine multiple conditions
   good_cnns = run.find("accuracy > 0.85 and param_model_type == 'cnn'")

   # Use parameter queries (note the param_ prefix)
   low_lr = run.find("param_learning_rate < 0.005")

Supported query operators:

- Comparison: ``>``, ``>=``, ``<``, ``<=``, ``==``, ``!=``
- Range: ``between`` (e.g., ``"accuracy between 0.8 and 0.9"``)
- Logical: ``and`` (``or`` support coming soon)

Analyzing Results
----------------

Generate Insights
~~~~~~~~~~~~~~~~~

Get automated insights about your experiments:

.. code-block:: python

   insights = run.insights()

   # Summary statistics
   print(f"Total experiments: {insights['summary']['total_experiments']}")
   print(f"Success rate: {insights['summary']['success_rate']:.1%}")
   print(f"Average accuracy: {insights['summary']['avg_accuracy']:.3f}")

   # Parameter insights
   param_insights = insights['parameter_insights']
   for param, analysis in param_insights.items():
       if analysis['impact_score'] > 0.5:
           print(f"{param} has high impact on results")

   # Performance insights
   perf_insights = insights['performance_insights']
   print(f"Fastest experiment: {perf_insights['fastest_duration']:.2f}s")
   print(f"Average duration: {perf_insights['avg_duration']:.2f}s")

Compare Experiments
~~~~~~~~~~~~~~~~~~

Compare multiple experiments side-by-side:

.. code-block:: python

   # Compare best 3 experiments
   best_3 = run.best(metric="accuracy", top=3)
   run.compare(best_3)

   # Compare specific experiments by ID
   run.compare(["run_id_1", "run_id_2", "run_id_3"])

   # Compare experiments from a query
   high_acc_experiments = run.find("accuracy > 0.9")
   run.compare(high_acc_experiments[:5])  # Compare first 5

The comparison shows:

- Parameter differences
- Metric comparisons
- Performance analysis
- Statistical significance

Get Suggestions
~~~~~~~~~~~~~~

Get intelligent suggestions for next experiments:

.. code-block:: python

   suggestions = run.suggest(
       my_experiment,
       count=5,
       strategy="balanced",  # "exploit", "explore", or "balanced"
       optimization_target="accuracy"
   )

   print("Suggested experiments:")
   for i, suggestion in enumerate(suggestions["suggestions"]):
       print(f"{i+1}. {suggestion['parameters']}")
       print(f"   Reason: {suggestion['reasoning']}")
       print(f"   Expected improvement: {suggestion['expected_improvement']:.3f}")

Working with Data
----------------

Export Results
~~~~~~~~~~~~~

Export experiment data for external analysis:

.. code-block:: python

   import pandas as pd

   # Get experiments as list of dictionaries
   experiments = run.all()

   # Convert to DataFrame
   data = []
   for exp in experiments:
       row = {
           'run_id': exp.run_id,
           'experiment_name': exp.experiment_name,
           'start_time': exp.start_time,
           'duration': exp.duration,
           'status': exp.status,
           **exp.parameters,  # Flatten parameters
           **exp.metrics,     # Flatten metrics
       }
       data.append(row)

   df = pd.DataFrame(data)
   df.to_csv('experiments.csv', index=False)

Access Raw Data
~~~~~~~~~~~~~~

For advanced analysis, access the underlying data:

.. code-block:: python

   # Get a specific experiment
   experiment = run.get_by_id("your_run_id")

   # Access all attributes
   print(f"Parameters: {experiment.parameters}")
   print(f"Metrics: {experiment.metrics}")
   print(f"Duration: {experiment.duration}")
   print(f"Git commit: {experiment.git_commit}")
   print(f"Environment: {experiment.environment}")

Error Handling
-------------

RexF gracefully handles experiment failures:

.. code-block:: python

   @experiment
   def failing_experiment(fail_rate=0.5):
       import random
       if random.random() < fail_rate:
           raise ValueError("Simulated failure")
       return {"success": True}

   # Failed experiments are still recorded
   run_id = run.single(failing_experiment, fail_rate=0.8)

   # Check experiment status
   experiment = run.get_by_id(run_id)
   if experiment.status == "failed":
       print(f"Experiment failed: {experiment.metadata.get('error', 'Unknown error')}")

Failed experiments are stored with:

- Status marked as "failed"
- Error message in metadata
- All parameters preserved
- Execution time recorded

This allows you to analyze failure patterns and debug issues.

Next Steps
---------

You've learned the basics! Now explore:

- :doc:`advanced_features` - Parameter exploration and intelligent insights
- :doc:`web_dashboard` - Interactive visualization and monitoring
- :doc:`cli_tools` - Command-line analytics and automation
- :doc:`reproducibility` - Ensuring reproducible research
