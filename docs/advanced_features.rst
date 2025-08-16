🔍 Advanced Features
==================

This guide covers RexF's advanced capabilities for power users and complex research workflows.

Intelligent Parameter Exploration
---------------------------------

RexF provides automated parameter space exploration with multiple strategies.

Random Exploration
~~~~~~~~~~~~~~~~~

Best for initial exploration of large parameter spaces:

.. code-block:: python

   from rexf import experiment, run

   @experiment
   def hyperparameter_search(learning_rate, batch_size, dropout_rate=0.1):
       # Your model training code
       model = create_model(dropout_rate)
       accuracy = train_model(model, learning_rate, batch_size)
       return {"accuracy": accuracy, "training_time": get_time()}

   # Random exploration
   run_ids = run.auto_explore(
       hyperparameter_search,
       strategy="random",
       budget=50,  # Number of experiments
       optimization_target="accuracy",
       # Parameter ranges (optional - RexF can infer reasonable ranges)
       parameter_ranges={
           "learning_rate": (0.0001, 0.1),
           "batch_size": [16, 32, 64, 128],
           "dropout_rate": (0.0, 0.5)
       }
   )

Grid Search
~~~~~~~~~~

Systematic exploration of discrete parameter combinations:

.. code-block:: python

   # Grid search
   run_ids = run.auto_explore(
       hyperparameter_search,
       strategy="grid",
       budget=20,
       optimization_target="accuracy",
       parameter_ranges={
           "learning_rate": [0.001, 0.01, 0.1],
           "batch_size": [32, 64, 128],
           "dropout_rate": [0.1, 0.2, 0.3]
       }
   )

Adaptive Exploration
~~~~~~~~~~~~~~~~~~~

Learns from previous results to focus on promising regions:

.. code-block:: python

   # Adaptive exploration (Bayesian-style optimization)
   run_ids = run.auto_explore(
       hyperparameter_search,
       strategy="adaptive",
       budget=30,
       optimization_target="accuracy",
       # Adaptive strategy learns and doesn't need predefined ranges
   )

The adaptive strategy:

- Starts with random exploration
- Builds a model of the parameter-performance relationship
- Balances exploration of unknown regions with exploitation of good regions
- Recommends parameters likely to improve results

Advanced Querying
----------------

Complex Query Expressions
~~~~~~~~~~~~~~~~~~~~~~~~~

Build sophisticated queries to find specific experiments:

.. code-block:: python

   # Complex accuracy and timing queries
   efficient_models = run.find(
       "accuracy > 0.9 and training_time < 300 and param_batch_size >= 32"
   )

   # Range queries
   moderate_lr = run.find("param_learning_rate between 0.01 and 0.1")

   # Status and timing combinations
   recent_successes = run.find(
       "status == 'completed' and start_time > '2024-01-01'"
   )

Query Suggestions
~~~~~~~~~~~~~~~~

Get intelligent query suggestions based on your data:

.. code-block:: python

   # Get suggested queries
   suggestions = run.query_help()

   print("Suggested queries for your experiments:")
   for suggestion in suggestions:
       print(f"- {suggestion}")

   # Example output:
   # - accuracy > 0.9
   # - param_learning_rate < 0.01
   # - training_time between 100 and 500

Custom Query Functions
~~~~~~~~~~~~~~~~~~~~~

For complex analysis, use the underlying query engine:

.. code-block:: python

   from rexf.intelligence.queries import SmartQueryEngine
   from rexf.backends.intelligent_storage import IntelligentStorage

   # Direct access to query engine
   storage = IntelligentStorage("experiments.db")
   query_engine = SmartQueryEngine(storage)

   # Advanced filtering
   results = storage.query_experiments(
       parameter_filters={"learning_rate": {"lt": 0.01}},
       metric_filters={"accuracy": {"gte": 0.9}},
       order_by="start_time",
       limit=10
   )

Intelligent Insights
-------------------

Deep Pattern Analysis
~~~~~~~~~~~~~~~~~~~~

Get comprehensive insights about your experiment patterns:

.. code-block:: python

   # Generate detailed insights
   insights = run.insights(experiment_name="hyperparameter_search")

   # Parameter impact analysis
   param_insights = insights["parameter_insights"]
   for param_name, analysis in param_insights.items():
       print(f"\n{param_name}:")
       print(f"  Impact score: {analysis['impact_score']:.3f}")
       print(f"  Optimal range: {analysis['optimal_range']}")
       print(f"  Correlation with accuracy: {analysis['correlation']:.3f}")

   # Performance patterns
   perf_insights = insights["performance_insights"]
   print(f"\nPerformance Insights:")
   print(f"  Best configuration: {perf_insights['best_configuration']}")
   print(f"  Efficiency sweet spot: {perf_insights['efficiency_sweet_spot']}")

   # Correlation insights
   correlations = insights["correlation_insights"]
   for metric_pair, correlation in correlations.items():
       if abs(correlation) > 0.5:
           print(f"Strong correlation: {metric_pair} = {correlation:.3f}")

Anomaly Detection
~~~~~~~~~~~~~~~~

Identify unusual experiments or outliers:

.. code-block:: python

   insights = run.insights()
   anomalies = insights["anomaly_insights"]

   print("Detected anomalies:")
   for anomaly in anomalies["outliers"]:
       print(f"  Run {anomaly['run_id'][:8]}: {anomaly['reason']}")
       print(f"    {anomaly['details']}")

   # Performance anomalies
   perf_anomalies = anomalies["performance_anomalies"]
   for anomaly in perf_anomalies:
       print(f"  Unusually {anomaly['type']}: {anomaly['description']}")

Smart Recommendations
~~~~~~~~~~~~~~~~~~~~

Get actionable recommendations for improving your experiments:

.. code-block:: python

   insights = run.insights()
   recommendations = insights["recommendations"]

   print("Recommendations:")
   for rec in recommendations:
       print(f"  🎯 {rec['title']}")
       print(f"     {rec['description']}")
       print(f"     Priority: {rec['priority']}")
       if "action" in rec:
           print(f"     Action: {rec['action']}")

Advanced Experiment Management
-----------------------------

Experiment Lineage and Relationships
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Track relationships between experiments:

.. code-block:: python

   @experiment
   def data_preprocessing(dataset_size=1000, normalization="standard"):
       # Data preprocessing
       processed_data = preprocess(dataset_size, normalization)
       return {"data_quality": evaluate_quality(processed_data)}

   @experiment
   def model_training(data_run_id, model_type="cnn"):
       # Use data from previous experiment
       data_experiment = run.get_by_id(data_run_id)
       data_quality = data_experiment.metrics["data_quality"]
       
       # Train model
       accuracy = train_model(model_type, data_quality)
       return {
           "accuracy": accuracy,
           "parent_experiment": data_run_id  # Track lineage
       }

   # Run preprocessing
   data_run_id = run.single(data_preprocessing, dataset_size=5000)

   # Run training with reference to preprocessing
   model_run_id = run.single(model_training, 
                             data_run_id=data_run_id, 
                             model_type="transformer")

Batch Processing and Parallel Execution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For large-scale experiments, use batch processing:

.. code-block:: python

   import concurrent.futures
   from functools import partial

   def run_experiment_batch(param_combinations, experiment_func):
       """Run experiments in parallel."""
       run_ids = []
       
       # Create partial function with fixed experiment
       run_func = partial(run.single, experiment_func)
       
       # Run in parallel (be careful with resource usage)
       with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
           futures = []
           for params in param_combinations:
               future = executor.submit(run_func, **params)
               futures.append(future)
           
           # Collect results
           for future in concurrent.futures.as_completed(futures):
               try:
                   run_id = future.result()
                   run_ids.append(run_id)
               except Exception as e:
                   print(f"Experiment failed: {e}")
       
       return run_ids

   # Define parameter grid
   param_grid = [
       {"learning_rate": lr, "batch_size": bs}
       for lr in [0.001, 0.01, 0.1]
       for bs in [32, 64, 128]
   ]

   # Run batch
   run_ids = run_experiment_batch(param_grid, hyperparameter_search)
   print(f"Completed {len(run_ids)} experiments in parallel")

Advanced Analysis and Visualization
----------------------------------

Custom Metrics and Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Define custom analysis functions:

.. code-block:: python

   def analyze_learning_curves(experiment_runs):
       """Custom analysis of learning progression."""
       analysis = {}
       
       for exp in experiment_runs:
           # Extract learning curve data from metrics
           if "learning_curve" in exp.metrics:
               curve = exp.metrics["learning_curve"]
               analysis[exp.run_id] = {
                   "convergence_epoch": find_convergence(curve),
                   "overfitting_detected": detect_overfitting(curve),
                   "final_accuracy": curve[-1]
               }
       
       return analysis

   # Get experiments and analyze
   recent_experiments = run.recent(hours=24)
   learning_analysis = analyze_learning_curves(recent_experiments)

   # Use analysis for recommendations
   for run_id, analysis in learning_analysis.items():
       if analysis["overfitting_detected"]:
           print(f"Run {run_id[:8]}: Consider adding regularization")

Statistical Analysis
~~~~~~~~~~~~~~~~~~~

Perform statistical tests on experiment results:

.. code-block:: python

   import scipy.stats as stats

   def compare_experiment_groups(group1_query, group2_query, metric="accuracy"):
       """Compare two groups of experiments statistically."""
       group1 = run.find(group1_query)
       group2 = run.find(group2_query)
       
       values1 = [exp.metrics[metric] for exp in group1 if metric in exp.metrics]
       values2 = [exp.metrics[metric] for exp in group2 if metric in exp.metrics]
       
       # Perform t-test
       t_stat, p_value = stats.ttest_ind(values1, values2)
       
       return {
           "group1_mean": np.mean(values1),
           "group2_mean": np.mean(values2),
           "t_statistic": t_stat,
           "p_value": p_value,
           "significant": p_value < 0.05
       }

   # Compare high vs low learning rates
   comparison = compare_experiment_groups(
       "param_learning_rate > 0.01",
       "param_learning_rate <= 0.01",
       metric="accuracy"
   )

   print(f"High LR mean: {comparison['group1_mean']:.4f}")
   print(f"Low LR mean: {comparison['group2_mean']:.4f}")
   print(f"Statistically significant: {comparison['significant']}")

Performance Optimization
------------------------

Database Optimization
~~~~~~~~~~~~~~~~~~~~

For large numbers of experiments, optimize database performance:

.. code-block:: python

   from rexf.backends.intelligent_storage import IntelligentStorage

   # Create storage with optimizations
   storage = IntelligentStorage(
       "experiments.db",
       optimize_for_analytics=True  # Enables additional indexing
   )

   # Batch operations for efficiency
   experiments_batch = []
   for params in large_param_list:
       exp_data = create_experiment_data(params)
       experiments_batch.append(exp_data)
   
   # Bulk insert (more efficient than individual saves)
   storage.save_experiments_batch(experiments_batch)

Memory Management
~~~~~~~~~~~~~~~~

For memory-intensive experiments:

.. code-block:: python

   @experiment
   def memory_intensive_experiment(dataset_size=1000000):
       # Process data in chunks to manage memory
       results = []
       chunk_size = 10000
       
       for i in range(0, dataset_size, chunk_size):
           chunk_result = process_data_chunk(i, i + chunk_size)
           results.append(chunk_result)
           
           # Clear intermediate data
           del chunk_result
       
       # Return aggregated metrics only
       return {
           "accuracy": aggregate_accuracy(results),
           "throughput": dataset_size / get_elapsed_time(),
           "memory_peak": get_peak_memory_usage()
       }

Integration with External Tools
------------------------------

Export to External Analysis Tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Export data for analysis in R, MATLAB, or other tools:

.. code-block:: python

   def export_for_r_analysis():
       """Export experiment data for R analysis."""
       experiments = run.all()
       
       # Create R-friendly data structure
       data_for_r = {
           "run_id": [],
           "parameters": [],
           "metrics": [],
           "metadata": []
       }
       
       for exp in experiments:
           data_for_r["run_id"].append(exp.run_id)
           data_for_r["parameters"].append(exp.parameters)
           data_for_r["metrics"].append(exp.metrics)
           data_for_r["metadata"].append({
               "duration": exp.duration,
               "status": exp.status,
               "start_time": exp.start_time.isoformat()
           })
       
       # Save as R data file
       import rpy2.robjects as robjects
       r_data = robjects.conversion.py2rpy(data_for_r)
       robjects.r.assign("experiment_data", r_data)
       robjects.r("save(experiment_data, file='experiments.RData')")

Integration with MLflow/Weights & Biases
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use RexF alongside other experiment tracking tools:

.. code-block:: python

   import mlflow

   @experiment
   def dual_tracking_experiment(learning_rate=0.01):
       # Start MLflow run
       with mlflow.start_run():
           # Your experiment code
           accuracy = train_model(learning_rate)
           
           # Log to MLflow
           mlflow.log_param("learning_rate", learning_rate)
           mlflow.log_metric("accuracy", accuracy)
           
           # RexF automatically captures everything
           return {"accuracy": accuracy}

   # Both RexF and MLflow will track this experiment
   run_id = run.single(dual_tracking_experiment, learning_rate=0.005)

This allows you to:

- Use RexF for quick analysis and exploration
- Use MLflow/W&B for detailed logging and team collaboration
- Compare and validate results across both platforms

Next Steps
---------

You've mastered RexF's advanced features! Continue with:

- :doc:`web_dashboard` - Interactive visualization and real-time monitoring
- :doc:`cli_tools` - Powerful command-line analytics
- :doc:`tutorials/machine_learning` - Complete ML workflow tutorial
- :doc:`api/intelligence` - Detailed API reference for advanced features
