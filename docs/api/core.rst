🧪 Core API Reference
=====================

This page documents the core RexF API that you'll use in your experiments.

Main Module
-----------

.. automodule:: rexf
   :members:
   :undoc-members:
   :show-inheritance:

The ``experiment`` Decorator
---------------------------

.. autofunction:: rexf.experiment

The decorator automatically captures:

- **Parameters**: All function arguments with their values
- **Results**: Everything returned by the function
- **Metadata**: Execution time, Git commit, environment info
- **Reproducibility**: Random seeds, system information

Example:

.. code-block:: python

   from rexf import experiment

   @experiment
   def my_research_function(param1, param2=42):
       # Your experiment code here
       result = perform_calculation(param1, param2)
       return {"metric": result, "quality": assess_quality(result)}

Simple API Module
-----------------

.. automodule:: rexf.core.simple_api
   :members:
   :undoc-members:
   :show-inheritance:

Models
------

Data structures used throughout RexF.

.. automodule:: rexf.core.models
   :members:
   :undoc-members:
   :show-inheritance:

ExperimentRun
~~~~~~~~~~~~~

.. autoclass:: rexf.core.models.ExperimentRun
   :members:
   :undoc-members:
   :show-inheritance:

   The ``ExperimentRun`` class represents a single experiment execution with all its metadata:

   .. code-block:: python

      # Access experiment data
      experiment = run.get_by_id("your_run_id")
      
      print(f"Parameters: {experiment.parameters}")
      print(f"Metrics: {experiment.metrics}")
      print(f"Duration: {experiment.duration} seconds")
      print(f"Status: {experiment.status}")
      print(f"Git commit: {experiment.git_commit}")

ExperimentData
~~~~~~~~~~~~~~

.. autoclass:: rexf.core.models.ExperimentData
   :members:
   :undoc-members:
   :show-inheritance:

   Container for structured experiment data used in storage and analysis.

Function Signature Detection
---------------------------

RexF automatically analyzes your experiment functions to understand their parameters and return values:

Parameter Detection
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   @experiment
   def my_experiment(required_param, optional_param=42, keyword_only=None):
       return {"result": required_param * optional_param}

   # RexF automatically detects:
   # - required_param: required parameter
   # - optional_param: optional with default value 42
   # - keyword_only: optional parameter

Return Value Processing
~~~~~~~~~~~~~~~~~~~~~~

RexF intelligently processes return values:

.. code-block:: python

   @experiment
   def different_return_types():
       # Dictionary (recommended) - becomes metrics
       return {"accuracy": 0.95, "loss": 0.05}

   @experiment  
   def single_value():
       # Single value - becomes "result" metric
       return 0.95

   @experiment
   def multiple_values():
       # Tuple/list - becomes indexed metrics
       return [0.95, 0.05, 0.98]  # result_0, result_1, result_2

Error Handling
--------------

RexF gracefully handles experiment failures:

.. code-block:: python

   @experiment
   def potentially_failing_experiment(error_probability=0.1):
       import random
       if random.random() < error_probability:
           raise ValueError("Simulated experiment failure")
       return {"success_metric": 1.0}

   # Failed experiments are still recorded with:
   # - status: "failed"
   # - error message in metadata
   # - all parameters preserved
   # - execution time captured

Environment Capture
------------------

RexF automatically captures environment information for reproducibility:

Git Integration
~~~~~~~~~~~~~~

- Current commit hash
- Repository status (clean/dirty)
- Branch name
- Remote URL

Python Environment
~~~~~~~~~~~~~~~~~

- Python version
- Installed packages and versions
- Virtual environment info

System Information
~~~~~~~~~~~~~~~~~

- Operating system
- Hardware details
- Execution timestamp

Random Seed Management
~~~~~~~~~~~~~~~~~~~~~

- Automatic seed capture
- Seed restoration for reproducibility
- Support for NumPy, Python random, and other libraries

Customization
-------------

Advanced Decorator Usage
~~~~~~~~~~~~~~~~~~~~~~~

The ``@experiment`` decorator supports customization options:

.. code-block:: python

   # Basic usage (recommended)
   @experiment
   def simple_experiment(param1, param2=42):
       return {"metric": param1 * param2}

   # The decorator handles everything automatically:
   # - Parameter extraction
   # - Result processing  
   # - Metadata collection
   # - Storage management

Integration Notes
----------------

Threading and Multiprocessing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RexF is thread-safe and supports concurrent experiment execution:

.. code-block:: python

   import concurrent.futures

   @experiment
   def thread_safe_experiment(worker_id, data_size=1000):
       # Each thread gets its own experiment context
       result = process_data(worker_id, data_size)
       return {"worker_result": result}

   # Safe to run in parallel
   with concurrent.futures.ThreadPoolExecutor() as executor:
       futures = [
           executor.submit(run.single, thread_safe_experiment, worker_id=i)
           for i in range(4)
       ]
       
       run_ids = [f.result() for f in futures]

Context Managers
~~~~~~~~~~~~~~~

For advanced use cases, you can access the experiment context:

.. code-block:: python

   from rexf.core.simple_api import get_current_experiment_context

   @experiment
   def context_aware_experiment(param1):
       # Access current experiment context
       context = get_current_experiment_context()
       
       if context:
           print(f"Running experiment: {context.experiment_name}")
           print(f"Run ID: {context.run_id}")
       
       return {"result": param1 * 2}

Best Practices
-------------

Function Design
~~~~~~~~~~~~~~

- Use descriptive parameter names
- Provide sensible default values
- Return dictionaries with meaningful key names
- Keep functions focused on single experiments

Parameter Naming
~~~~~~~~~~~~~~~

- Use lowercase with underscores: ``learning_rate``
- Be descriptive: ``num_samples`` not ``n``
- Group related parameters: ``model_type``, ``model_layers``

Return Value Structure
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   @experiment
   def well_structured_experiment(param1, param2=42):
       # Good return structure
       return {
           # Primary metrics
           "accuracy": 0.95,
           "precision": 0.92,
           "recall": 0.94,
           
           # Performance metrics
           "training_time": 120.5,
           "memory_usage": 256.7,
           
           # Quality metrics  
           "convergence_epoch": 45,
           "final_loss": 0.023
       }

Error Reporting
~~~~~~~~~~~~~~

.. code-block:: python

   @experiment
   def robust_experiment(param1, validate=True):
       try:
           if validate and param1 < 0:
               raise ValueError(f"Invalid parameter: {param1} must be >= 0")
           
           result = complex_computation(param1)
           return {"result": result, "validation_passed": True}
           
       except Exception as e:
           # Let RexF handle the error - it will:
           # 1. Record the experiment as failed
           # 2. Store the error message
           # 3. Preserve all parameters
           # 4. Return a valid run_id for analysis
           raise  # Re-raise to let RexF handle it
