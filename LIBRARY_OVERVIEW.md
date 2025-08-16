# rexf - Reproducible Experiments Framework

## Overview

rexf is a lightweight Python library for reproducible computational experiments that addresses the limitations of existing libraries like Sacred, MLflow, Trackio, and Meticulous. It provides a decorator-based API for marking experiment metadata, automatic reproducibility tracking, and built-in visualization tools.

## Key Features ✅

### 1. Decorator-based API
- `@experiment(name)` - Mark a function as an experiment
- `@param(name, type, default, description)` - Define parameters
- `@result(name, type, description)` - Define expected results
- `@metric(name, type, description)` - Define metrics to track
- `@artifact(name, filename, description)` - Define artifacts to store
- `@seed(name)` - Mark random seed parameters

### 2. Automatic Reproducibility Tracking
- ✅ Random seeds management (Python, NumPy, PyTorch)
- ✅ Git commit hash and repository status
- ✅ Environment information (Python version, platform, etc.)
- ✅ Data versioning support
- ✅ Execution timestamps

### 3. File-system Based Storage
- ✅ SQLite database for experiment metadata
- ✅ Separate artifact storage in organized folders
- ✅ No external database dependencies
- ✅ Local-first approach

### 4. Artifact Management
- ✅ Automatic detection of data types (NumPy arrays, Pandas DataFrames, Matplotlib figures, etc.)
- ✅ Intelligent storage (CSV for DataFrames, PNG for figures, NPY for arrays, JSON for structured data)
- ✅ File integrity checking with SHA256 hashes
- ✅ Metadata storage for each artifact

### 5. Built-in Visualization
- ✅ Metrics comparison plots
- ✅ Parameter space exploration
- ✅ Experiment timeline visualization
- ✅ Metric correlation matrices
- ✅ Comprehensive dashboards
- ✅ Comparison tables (CSV export)

### 6. Export Capabilities
- ✅ JSON export for single experiments or all runs
- ✅ YAML export for human-readable sharing
- ✅ Comparison export for analysis
- ✅ Optional artifact path inclusion

### 7. Extensible Design
- ✅ Clear interfaces for storage, artifacts, export, and visualization
- ✅ Swappable storage backends
- ✅ Modular component architecture
- ✅ Easy to extend with new features

### 8. Error Handling and Robustness
- ✅ Graceful handling of missing dependencies
- ✅ Experiment failure tracking with traceback storage
- ✅ Validation of experiment function signatures
- ✅ Safe JSON serialization with fallbacks

## Architecture

```
rexf/
├── interfaces.py          # Core interfaces and base classes
├── decorators.py          # Decorator-based API
├── storage.py             # SQLite storage backend
├── artifacts.py           # File-system artifact management
├── runner.py              # Experiment execution engine
├── visualization.py       # Built-in visualization tools
├── export.py              # JSON/YAML export functionality
└── __init__.py           # Public API
```

## Example Usage

```python
from rexf import experiment, param, result, metric, artifact, seed, ExperimentRunner

@experiment("monte_carlo_pi")
@param("n_samples", int)
@seed("random_seed")
@metric("pi_estimate", float)
@artifact("plot", "convergence.png")
@result("final_pi", float)
def estimate_pi(n_samples, random_seed=42):
    # Your experiment implementation
    return {"final_pi": 3.14159, "pi_estimate": 3.14159, "plot": figure}

# Run experiments
runner = ExperimentRunner()
run_id = runner.run(estimate_pi, n_samples=100000)

# Analyze results
experiment = runner.get_experiment(run_id)
comparison = runner.compare_runs([run_id1, run_id2])

# Visualize
from rexf import ExperimentVisualizer
visualizer = ExperimentVisualizer()
visualizer.plot_metrics([experiment])
```

## File Organization

```
your_project/
├── experiments.db          # SQLite metadata database
├── artifacts/              # Artifact storage
│   ├── run_id_1/
│   │   ├── plot.png
│   │   ├── plot.meta.json
│   │   └── data.npy
│   └── run_id_2/
└── your_experiments.py     # Your experiment code
```

## Dependencies

**Core (minimal):**
- Python 3.8+
- None (uses only standard library for core functionality)

**Full features:**
- numpy >= 1.20.0 (for NumPy array support)
- matplotlib >= 3.3.0 (for visualization)
- pandas >= 1.3.0 (for DataFrame support)
- pyyaml >= 6.0 (for YAML export)
- gitpython >= 3.1.0 (for Git integration)

## Advantages over Existing Libraries

### vs Sacred
- ✅ No complex configuration files
- ✅ Simple decorator-based API
- ✅ Built-in visualization
- ✅ Local storage by default

### vs MLflow
- ✅ No external server required
- ✅ Lightweight and minimal dependencies
- ✅ Not ML-specific, works for any computational research
- ✅ File-system based storage

### vs Trackio
- ✅ More comprehensive artifact management
- ✅ Built-in visualization tools
- ✅ Better reproducibility tracking
- ✅ Extensible architecture

### vs Meticulous
- ✅ Simpler setup process
- ✅ No dependency on external databases
- ✅ Built-in comparison and visualization
- ✅ More general-purpose (not just ML)

## Testing Status

✅ **Basic functionality tests passed**
- Experiment definition and execution
- Parameter handling and validation
- Storage and retrieval
- Comparison functionality
- Export capabilities

✅ **Comprehensive demo completed**
- Monte Carlo π estimation with 5 experiments
- Artifact storage (plots, data)
- Metrics tracking and comparison
- Visualization generation
- Export functionality

## Installation and Usage

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run basic test:**
   ```bash
   python test_basic.py
   ```

3. **Try demos:**
   ```bash
   python examples/simple_demo.py
   python examples/monte_carlo_pi_demo.py
   ```

4. **Use in your project:**
   ```bash
   pip install -e .  # For development
   ```

The library is production-ready and provides a clean, intuitive API for reproducible computational experiments.
