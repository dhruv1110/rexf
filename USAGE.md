# rexf Usage Guide

## Quick Start

### 1. Installation

```bash
pip install -e .  # For development
# or
pip install rexf  # When published
```

### 2. Basic Experiment Definition

```python
from rexf import experiment, param, result, metric, artifact, seed, ExperimentRunner

@experiment("my_experiment")
@param("input_size", int, description="Size of input data")
@param("learning_rate", float, default=0.001, description="Learning rate")
@seed("random_seed")
@metric("accuracy", float, description="Model accuracy")
@artifact("model_plot", "model_performance.png", description="Performance plot")
@result("final_score", float, description="Final evaluation score")
def my_experiment(input_size, learning_rate=0.001, random_seed=42):
    # Your experiment code here
    import matplotlib.pyplot as plt
    
    # Simulate some work
    score = input_size * learning_rate + random_seed / 1000
    accuracy = min(1.0, score)
    
    # Create a plot
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3, 4], [0.1, 0.4, 0.7, accuracy])
    ax.set_title("Model Performance")
    
    return {
        "final_score": score,
        "accuracy": accuracy,
        "model_plot": fig
    }
```

### 3. Running Experiments

```python
# Initialize runner
runner = ExperimentRunner()

# Run single experiment
run_id = runner.run(my_experiment, input_size=100, learning_rate=0.01)

# Run multiple experiments
configs = [
    {"input_size": 50, "learning_rate": 0.001},
    {"input_size": 100, "learning_rate": 0.01},
    {"input_size": 200, "learning_rate": 0.1},
]

run_ids = []
for config in configs:
    run_id = runner.run(my_experiment, **config)
    run_ids.append(run_id)
```

### 4. Analyzing Results

```python
# Get experiment data
experiment = runner.get_experiment(run_id)
print(f"Parameters: {experiment.parameters}")
print(f"Metrics: {experiment.metrics}")
print(f"Results: {experiment.results}")

# Compare multiple runs
comparison = runner.compare_runs(run_ids)
print("Parameter variations:", comparison['parameter_comparison'])
print("Metric statistics:", comparison['metric_comparison'])

# List all experiments
all_experiments = runner.list_experiments()
```

### 5. Visualization

```python
from rexf import ExperimentVisualizer

visualizer = ExperimentVisualizer()

# Plot metrics comparison
experiments = [runner.get_experiment(rid) for rid in run_ids]
fig = visualizer.plot_metrics(experiments)

# Create dashboard
dashboard = visualizer.create_dashboard(experiments)

# Generate comparison table
table = visualizer.create_comparison_table(experiments, save_path="comparison.csv")
```

### 6. Export and Sharing

```python
from rexf import ExperimentExporter

exporter = ExperimentExporter()

# Export to JSON
exporter.export_to_file(experiments, "my_experiments.json")

# Export to YAML
exporter.export_to_file(experiments, "my_experiments.yaml")

# Export comparison
exporter.export_comparison(experiments, "comparison.json")
```

## Advanced Features

### Custom Storage Backend

```python
from rexf.interfaces import StorageInterface
from rexf import ExperimentRunner

class MyCustomStorage(StorageInterface):
    # Implement required methods
    pass

# Use custom storage
runner = ExperimentRunner()
runner.storage = MyCustomStorage()
```

### Artifact Management

```python
# Store different types of artifacts
def my_experiment():
    import numpy as np
    import pandas as pd
    
    # NumPy arrays are automatically detected and saved
    data_array = np.random.random((100, 100))
    
    # Pandas DataFrames are saved as CSV
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    
    # Any object can be pickled
    custom_object = {"key": "value", "nested": {"data": [1, 2, 3]}}
    
    return {
        "data_array": data_array,
        "dataframe": df,
        "custom_object": custom_object
    }
```

### Environment Tracking

The library automatically tracks:
- Git commit hash and repository status
- Python version and environment
- System information (OS, architecture, etc.)
- Random seeds for reproducibility
- Execution timestamps

### Error Handling

```python
try:
    run_id = runner.run(my_experiment, **params)
except Exception as e:
    # Experiment failures are automatically recorded
    failed_experiment = runner.get_experiment(run_id)
    print(f"Error: {failed_experiment.results.get('error')}")
    print(f"Traceback: {failed_experiment.results.get('traceback')}")
```

## File Organization

```
your_project/
├── experiments.db          # SQLite database with metadata
├── artifacts/              # Artifact storage directory
│   ├── run_id_1/
│   │   ├── plot.png
│   │   ├── plot.meta.json
│   │   ├── data.npy
│   │   └── data.meta.json
│   └── run_id_2/
│       └── ...
└── your_experiment.py      # Your experiment code
```

## Best Practices

1. **Use descriptive names**: Choose meaningful names for experiments, parameters, and metrics
2. **Version your code**: Use Git to track code changes alongside experiments
3. **Document parameters**: Add descriptions to all parameters
4. **Seed everything**: Use the `@seed` decorator for reproducible results
5. **Regular cleanup**: Periodically clean up old experiments and artifacts
6. **Export important results**: Export key experiments for long-term storage

## Examples

See the `examples/` directory for complete working examples:
- `monte_carlo_pi_demo.py`: Comprehensive Monte Carlo π estimation
- `simple_demo.py`: Basic usage example
