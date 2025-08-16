🔄 Reproducibility
=================

RexF is designed from the ground up to ensure your experiments are reproducible. This guide covers all the ways RexF helps you achieve reliable, repeatable results.

Automatic Reproducibility Tracking
----------------------------------

RexF automatically captures all information needed to reproduce your experiments:

Code Version Control
~~~~~~~~~~~~~~~~~~~

Every experiment automatically records:

.. code-block:: python

   from rexf import experiment, run

   @experiment
   def my_experiment(param1=42):
       return {"result": param1 * 2}

   # RexF automatically captures:
   # - Git commit hash
   # - Repository status (clean/dirty)  
   # - Branch name
   # - Uncommitted changes (if any)

   run_id = run.single(my_experiment, param1=100)
   
   # View captured git info
   experiment = run.get_by_id(run_id)
   print(f"Git commit: {experiment.git_commit}")
   print(f"Repository status: {experiment.git_status}")

Environment Capture
~~~~~~~~~~~~~~~~~~

Python environment details are automatically recorded:

.. code-block:: python

   # Automatically captured for each experiment:
   # - Python version
   # - Installed packages and versions
   # - Virtual environment info
   # - Operating system details

   experiment = run.get_by_id(run_id)
   env_info = experiment.environment
   
   print(f"Python version: {env_info['python_version']}")
   print(f"Platform: {env_info['platform']}")
   print(f"Installed packages: {len(env_info['packages'])} packages")

Random Seed Management
---------------------

RexF provides comprehensive random seed handling for true reproducibility:

Automatic Seed Capture
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import random
   import numpy as np
   from rexf import experiment, run

   @experiment
   def random_experiment(n_samples=1000):
       # RexF automatically captures seeds for:
       # - Python's random module
       # - NumPy random state
       # - Other supported libraries
       
       random_values = [random.random() for _ in range(n_samples)]
       numpy_values = np.random.rand(n_samples)
       
       return {
           "mean_random": sum(random_values) / len(random_values),
           "mean_numpy": np.mean(numpy_values)
       }

   # Seeds are automatically captured and stored
   run_id = run.single(random_experiment, n_samples=500)

Manual Seed Control
~~~~~~~~~~~~~~~~~~

For explicit control over randomness:

.. code-block:: python

   @experiment
   def seeded_experiment(data_size=1000, random_seed=None):
       if random_seed is not None:
           random.seed(random_seed)
           np.random.seed(random_seed)
       
       # Your experiment code using random numbers
       data = np.random.normal(0, 1, data_size)
       result = np.mean(data)
       
       return {"mean_value": result, "std_value": np.std(data)}

   # Reproducible run with fixed seed
   run_id1 = run.single(seeded_experiment, data_size=500, random_seed=42)
   run_id2 = run.single(seeded_experiment, data_size=500, random_seed=42)

   # Results will be identical
   exp1 = run.get_by_id(run_id1)
   exp2 = run.get_by_id(run_id2)
   
   assert exp1.metrics["mean_value"] == exp2.metrics["mean_value"]
   print("✅ Experiments are perfectly reproducible!")

Supported Random Libraries
~~~~~~~~~~~~~~~~~~~~~~~~~

RexF automatically handles seeds for:

- **Python's random module**: ``random.seed()``
- **NumPy**: ``np.random.seed()``
- **PyTorch**: ``torch.manual_seed()`` (if available)
- **TensorFlow**: ``tf.random.set_seed()`` (if available)
- **Scikit-learn**: Via ``random_state`` parameters

.. code-block:: python

   @experiment
   def ml_experiment(random_seed=None):
       if random_seed is not None:
           # Set all seeds for reproducibility
           random.seed(random_seed)
           np.random.seed(random_seed)
           
           # If using PyTorch
           try:
               import torch
               torch.manual_seed(random_seed)
           except ImportError:
               pass
           
           # If using TensorFlow
           try:
               import tensorflow as tf
               tf.random.set_seed(random_seed)
           except ImportError:
               pass
       
       # Your ML experiment code
       return {"accuracy": train_model()}

Parameter Tracking
-----------------

All function parameters are automatically captured:

Default Parameter Values
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   @experiment
   def experiment_with_defaults(required_param, optional_param=42, another_param="default"):
       return {
           "result": required_param * optional_param,
           "param_length": len(str(another_param))
       }

   # All parameters are recorded, including defaults
   run_id = run.single(experiment_with_defaults, required_param=10)
   
   experiment = run.get_by_id(run_id)
   print(experiment.parameters)
   # Output: {"required_param": 10, "optional_param": 42, "another_param": "default"}

Complex Parameter Types
~~~~~~~~~~~~~~~~~~~~~~

RexF handles various parameter types:

.. code-block:: python

   @experiment
   def complex_experiment(
       numeric_param=3.14,
       string_param="test",
       list_param=[1, 2, 3],
       dict_param={"key": "value"},
       bool_param=True
   ):
       # All parameter types are properly serialized and stored
       return {"processed": True}

   run_id = run.single(
       complex_experiment,
       list_param=[10, 20, 30],
       dict_param={"model": "cnn", "layers": 5}
   )

Verification and Validation
--------------------------

RexF provides tools to verify reproducibility:

Reproducibility Testing
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def test_reproducibility(experiment_func, params, num_runs=3):
       """Test if an experiment is reproducible."""
       results = []
       
       for i in range(num_runs):
           # Use same seed for all runs
           run_id = run.single(experiment_func, random_seed=42, **params)
           experiment = run.get_by_id(run_id)
           results.append(experiment.metrics)
       
       # Check if all results are identical
       first_result = results[0]
       is_reproducible = all(
           result == first_result for result in results[1:]
       )
       
       return is_reproducible, results

   # Test your experiment
   is_repro, results = test_reproducibility(
       seeded_experiment, 
       {"data_size": 1000}
   )
   
   print(f"Experiment is reproducible: {is_repro}")

Environment Comparison
~~~~~~~~~~~~~~~~~~~~~

Compare environments between experiments:

.. code-block:: python

   def compare_environments(run_id1, run_id2):
       """Compare environments between two experiments."""
       exp1 = run.get_by_id(run_id1)
       exp2 = run.get_by_id(run_id2)
       
       env1 = exp1.environment
       env2 = exp2.environment
       
       differences = {}
       
       # Compare Python versions
       if env1["python_version"] != env2["python_version"]:
           differences["python_version"] = (env1["python_version"], env2["python_version"])
       
       # Compare packages
       packages1 = set(env1["packages"].items())
       packages2 = set(env2["packages"].items())
       
       different_packages = packages1.symmetric_difference(packages2)
       if different_packages:
           differences["packages"] = different_packages
       
       return differences

   # Compare two experiments
   differences = compare_environments(run_id1, run_id2)
   if differences:
       print("Environment differences found:")
       for key, diff in differences.items():
           print(f"  {key}: {diff}")
   else:
       print("✅ Environments are identical")

Best Practices for Reproducibility
----------------------------------

Experiment Design
~~~~~~~~~~~~~~~~

1. **Use explicit seeds** when reproducibility is critical:

.. code-block:: python

   @experiment
   def reproducible_experiment(data_size=1000, random_seed=42):
       # Always accept and use random_seed parameter
       if random_seed is not None:
           random.seed(random_seed)
           np.random.seed(random_seed)
       
       # Your experiment code
       return {"result": generate_results()}

2. **Document stochastic processes**:

.. code-block:: python

   @experiment
   def documented_experiment(iterations=1000):
       """
       Experiment with stochastic optimization.
       
       Note: This experiment uses random initialization and may
       produce different results on each run unless random_seed is set.
       """
       # Your stochastic code
       return {"final_value": optimize_randomly()}

3. **Separate deterministic and stochastic parts**:

.. code-block:: python

   @experiment
   def hybrid_experiment(deterministic_param=10, random_seed=None):
       # Deterministic computation
       deterministic_result = deterministic_param ** 2
       
       # Stochastic computation (controlled by seed)
       if random_seed is not None:
           random.seed(random_seed)
       stochastic_result = random.random()
       
       return {
           "deterministic": deterministic_result,
           "stochastic": stochastic_result,
           "combined": deterministic_result + stochastic_result
       }

Version Control Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Commit code before important experiments**:

.. code-block:: bash

   # Good practice: commit your experiment code
   git add experiment.py
   git commit -m "Add new hyperparameter search experiment"
   
   # Run experiments
   python run_experiments.py

2. **Tag important experiment runs**:

.. code-block:: bash

   # Tag significant results
   git tag -a "v1.0-baseline" -m "Baseline experiment results"

3. **Use branches for experimental features**:

.. code-block:: bash

   # Create branch for new experiment variants
   git checkout -b "experiment/new-optimization"

Data Dependencies
~~~~~~~~~~~~~~~~

1. **Record data versions and sources**:

.. code-block:: python

   @experiment
   def data_dependent_experiment(data_path="data/v1.0/dataset.csv"):
       # Record data version in results
       data_info = get_data_info(data_path)
       
       # Your experiment
       results = process_data(data_path)
       
       return {
           **results,
           "data_version": data_info["version"],
           "data_checksum": data_info["checksum"],
           "data_size": data_info["size"]
       }

2. **Use content hashing for data integrity**:

.. code-block:: python

   import hashlib

   def get_data_checksum(file_path):
       """Calculate checksum of data file."""
       hash_sha256 = hashlib.sha256()
       with open(file_path, "rb") as f:
           for chunk in iter(lambda: f.read(4096), b""):
               hash_sha256.update(chunk)
       return hash_sha256.hexdigest()

   @experiment
   def checksum_verified_experiment(data_path="dataset.csv"):
       checksum = get_data_checksum(data_path)
       
       # Your experiment
       results = process_data(data_path)
       
       return {
           **results,
           "data_checksum": checksum
       }

Reproducing Experiments
-----------------------

From Experiment IDs
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def reproduce_experiment(original_run_id):
       """Reproduce an experiment from its run ID."""
       original = run.get_by_id(original_run_id)
       
       if not original:
           raise ValueError(f"Experiment {original_run_id} not found")
       
       # Get the original experiment function (you need to import it)
       experiment_name = original.experiment_name
       experiment_func = globals()[experiment_name]  # Or import appropriately
       
       # Reproduce with same parameters
       new_run_id = run.single(experiment_func, **original.parameters)
       
       return new_run_id

   # Reproduce a specific experiment
   original_run_id = "your_run_id_here"
   reproduced_run_id = reproduce_experiment(original_run_id)

   # Compare results
   run.compare([original_run_id, reproduced_run_id])

From Saved Configurations
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Save experiment configuration
   def save_experiment_config(run_id, config_file):
       """Save experiment configuration for later reproduction."""
       experiment = run.get_by_id(run_id)
       
       config = {
           "experiment_name": experiment.experiment_name,
           "parameters": experiment.parameters,
           "git_commit": experiment.git_commit,
           "environment": experiment.environment
       }
       
       import json
       with open(config_file, "w") as f:
           json.dump(config, f, indent=2)

   # Load and reproduce from configuration
   def reproduce_from_config(config_file):
       """Reproduce experiment from saved configuration."""
       import json
       with open(config_file, "r") as f:
           config = json.load(f)
       
       # Check environment compatibility
       current_env = get_current_environment()  # You'd implement this
       if current_env != config["environment"]:
           print("⚠️ Warning: Current environment differs from original")
       
       # Reproduce experiment
       experiment_func = globals()[config["experiment_name"]]
       return run.single(experiment_func, **config["parameters"])

Cross-Platform Reproducibility
------------------------------

Handling Platform Differences
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import platform

   @experiment
   def platform_aware_experiment(param1=42):
       # Record platform information
       platform_info = {
           "system": platform.system(),
           "machine": platform.machine(),
           "processor": platform.processor()
       }
       
       # Adjust behavior for platform differences if needed
       if platform.system() == "Windows":
           # Windows-specific handling
           result = windows_specific_computation(param1)
       else:
           # Unix-like systems
           result = unix_specific_computation(param1)
       
       return {
           "result": result,
           "platform": platform_info
       }

Numerical Precision
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np

   @experiment
   def precision_controlled_experiment(data_size=1000, dtype="float64"):
       # Control numerical precision explicitly
       np_dtype = getattr(np, dtype)
       
       data = np.random.rand(data_size).astype(np_dtype)
       result = np.sum(data)
       
       return {
           "result": float(result),  # Convert to Python float for JSON
           "dtype_used": dtype,
           "precision_bits": np.finfo(np_dtype).bits
       }

Troubleshooting Reproducibility Issues
-------------------------------------

Common Issues and Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Different random number sequences**:

.. code-block:: python

   # Problem: Random sequences differ between runs
   # Solution: Always set and use seeds explicitly
   
   @experiment
   def fixed_random_experiment(n_samples=100, random_seed=42):
       # Set seed at the beginning
       np.random.seed(random_seed)
       random.seed(random_seed)
       
       # Your random computations
       return {"result": np.random.rand(n_samples).mean()}

2. **Environment dependency issues**:

.. code-block:: bash

   # Create reproducible environment with exact versions
   pip freeze > requirements.txt
   
   # Or use conda
   conda env export > environment.yml

3. **Floating point precision differences**:

.. code-block:: python

   @experiment
   def precision_robust_experiment(tolerance=1e-10):
       # Use appropriate tolerances for comparisons
       result = complex_computation()
       
       return {
           "result": round(result, 10),  # Round to avoid precision issues
           "tolerance_used": tolerance
       }

Debugging Non-Reproducible Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def debug_reproducibility(experiment_func, params, num_runs=5):
       """Debug why an experiment isn't reproducible."""
       results = []
       
       for i in range(num_runs):
           print(f"Run {i+1}...")
           
           # Use same seed but capture more info
           run_id = run.single(
               experiment_func, 
               random_seed=42,  # Same seed
               **params
           )
           
           experiment = run.get_by_id(run_id)
           results.append({
               "run_id": run_id,
               "metrics": experiment.metrics,
               "start_time": experiment.start_time,
               "environment": experiment.environment
           })
       
       # Analyze differences
       print("\nAnalyzing differences...")
       
       # Check metrics
       first_metrics = results[0]["metrics"]
       for i, result in enumerate(results[1:], 1):
           if result["metrics"] != first_metrics:
               print(f"Run {i+1} differs from run 1:")
               for key in first_metrics:
                   if key in result["metrics"]:
                       if first_metrics[key] != result["metrics"][key]:
                           print(f"  {key}: {first_metrics[key]} vs {result['metrics'][key]}")
       
       return results

   # Debug your experiment
   debug_results = debug_reproducibility(my_experiment, {"param1": 100})

Next Steps
---------

- :doc:`advanced_features` - Advanced analysis techniques
- :doc:`tutorials/monte_carlo` - Complete reproducibility example
- :doc:`api/core` - Core API for experiment control
