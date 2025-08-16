# 🧪 RexF - Smart Experiments Framework

[![CI](https://github.com/dhruv1110/rexf/workflows/CI/badge.svg)](https://github.com/dhruv1110/rexf/actions/workflows/ci.yml)
[![CodeQL](https://github.com/dhruv1110/rexf/workflows/CodeQL/badge.svg)](https://github.com/dhruv1110/rexf/actions/workflows/codeql.yml)
[![PyPI](https://img.shields.io/pypi/v/rexf)](https://pypi.org/project/rexf/)
[![Python Versions](https://img.shields.io/pypi/pyversions/rexf)](https://pypi.org/project/rexf/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight Python library for **reproducible computational experiments** with an ultra-simple, smart API. From idea to insight in under 5 minutes, with zero configuration.

## ✨ Key Features

- **🎯 Ultra-Simple API**: Single `@experiment` decorator - that's it!
- **🚀 Auto-Everything**: Parameters, metrics, and results detected automatically
- **🔍 Smart Exploration**: Automated parameter space exploration with multiple strategies
- **💡 Intelligent Insights**: Automated pattern detection and recommendations
- **📊 Web Dashboard**: Beautiful real-time experiment monitoring
- **🔧 CLI Analytics**: Powerful command-line tools for ad-hoc analysis
- **📈 Query Interface**: Find experiments using simple expressions like `"accuracy > 0.9"`
- **🔄 Reproducible**: Git commit tracking, environment capture, seed management
- **💾 Local-First**: SQLite database - no external servers required

## 🚀 Quick Start

### Installation

```bash
pip install rexf
```

### Ultra-Simple Usage

```python
from rexf import experiment, run

@experiment
def my_experiment(learning_rate, batch_size=32):
    # Your experiment code here
    accuracy = train_model(learning_rate, batch_size)
    return {"accuracy": accuracy, "loss": 1 - accuracy}

# Run single experiment
run.single(my_experiment, learning_rate=0.01, batch_size=64)

# Get insights
print(run.insights())

# Find best experiments
best = run.best(metric="accuracy", top=5)

# Auto-explore parameter space
run.auto_explore(my_experiment, strategy="random", budget=20)

# Launch web dashboard
run.dashboard()
```

## 🎯 Core Philosophy

**From idea to insight in under 5 minutes, with zero configuration.**

RexF prioritizes user experience over architectural purity. Instead of making you learn complex APIs, it automatically detects what you're doing and provides smart features to accelerate your research.

## 📖 Comprehensive Example

```python
import math
import random
from rexf import experiment, run

@experiment
def estimate_pi(num_samples=10000, method="uniform"):
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

# Run experiments
run.single(estimate_pi, num_samples=50000, method="uniform")
run.single(estimate_pi, num_samples=100000, method="stratified")

# Auto-explore to find best parameters
run_ids = run.auto_explore(
    estimate_pi,
    strategy="grid", 
    budget=10,
    optimization_target="accuracy"
)

# Get smart insights
insights = run.insights()
print(f"Success rate: {insights['summary']['success_rate']:.1%}")

# Find high-accuracy runs
accurate_runs = run.find("accuracy > 0.99")

# Compare experiments
run.compare(run.best(top=3))

# Launch web dashboard
run.dashboard()  # Opens http://localhost:8080
```

## 🔧 Advanced Features

### Smart Parameter Exploration

```python
# Random exploration
run.auto_explore(my_experiment, strategy="random", budget=20)

# Grid search
run.auto_explore(my_experiment, strategy="grid", budget=15)

# Adaptive exploration (learns from results)
run.auto_explore(my_experiment, strategy="adaptive", budget=25, 
                optimization_target="accuracy")
```

### Query Interface

```python
# Find experiments using expressions
high_acc = run.find("accuracy > 0.9")
fast_runs = run.find("duration < 30")
recent_good = run.find("accuracy > 0.8 and start_time > '2024-01-01'")

# Query help
run.query_help()
```

### Experiment Suggestions

```python
# Get next experiment suggestions
suggestions = run.suggest(
    my_experiment, 
    count=5, 
    strategy="balanced",  # "exploit", "explore", or "balanced"
    optimization_target="accuracy"
)

for suggestion in suggestions["suggestions"]:
    print(f"Try: {suggestion['parameters']}")
    print(f"Reason: {suggestion['reasoning']}")
```

### CLI Analytics

Analyze experiments from the command line:

```bash
# Show summary
rexf-analytics --summary

# Query experiments
rexf-analytics --query "accuracy > 0.9"

# Generate insights
rexf-analytics --insights

# Compare best experiments
rexf-analytics --compare --best 5

# Export to CSV
rexf-analytics --list --format csv --output results.csv
```

### Web Dashboard

Launch a beautiful web interface:

```python
run.dashboard()  # Opens http://localhost:8080
```

Features:
- 📊 Real-time experiment monitoring
- 🔍 Interactive filtering and search
- 💡 Automated insights generation
- 📈 Statistics overview and trends
- 🎯 Experiment comparison tools

## 🎨 Why RexF?

### Before (Traditional Approach)
```python
import mlflow
import sacred
from sacred import Experiment

# Complex setup required
ex = Experiment('my_exp')
mlflow.set_tracking_uri("...")

@ex.config
def config():
    learning_rate = 0.01
    batch_size = 32

@ex.automain
def main(learning_rate, batch_size):
    with mlflow.start_run():
        # Your code here
        mlflow.log_param("lr", learning_rate)
        mlflow.log_metric("accuracy", accuracy)
```

### After (RexF)
```python
from rexf import experiment, run

@experiment
def my_experiment(learning_rate=0.01, batch_size=32):
    # Your code here - that's it!
    return {"accuracy": accuracy}

run.single(my_experiment, learning_rate=0.05)
```

### Key Differences

| Feature | Traditional Tools | RexF |
|---------|------------------|------|
| **Setup** | Complex configuration | Single decorator |
| **Parameter Detection** | Manual logging | Automatic |
| **Metric Tracking** | Manual logging | Automatic |
| **Insights** | Manual analysis | Auto-generated |
| **Exploration** | Write custom loops | `run.auto_explore()` |
| **Comparison** | Custom dashboards | `run.compare()` |
| **Querying** | SQL/Complex APIs | `run.find("accuracy > 0.9")` |

## 🛠️ Architecture

RexF uses a plugin-based architecture:

```
rexf/
├── core/           # Core experiment logic
├── backends/       # Storage implementations (SQLite, etc.)
├── intelligence/   # Smart features (insights, exploration)
├── dashboard/      # Web interface
├── cli/           # Command-line tools
└── plugins/       # Extensions (export, visualization)
```

### Backends
- **SQLiteStorage**: Fast local storage (default)
- **IntelligentStorage**: Enhanced analytics and querying
- **FileSystemArtifacts**: Local artifact management

### Intelligence Modules
- **ExplorationEngine**: Automated parameter space exploration
- **InsightsEngine**: Pattern detection and recommendations
- **SuggestionEngine**: Next experiment recommendations
- **SmartQueryEngine**: Natural language-like querying

## 📊 Data Storage

RexF automatically captures:

- **Experiment metadata**: Name, timestamp, duration, status
- **Parameters**: Function arguments and defaults
- **Results**: Return values (auto-categorized as metrics/results/artifacts)
- **Environment**: Git commit, Python version, dependencies
- **Reproducibility**: Random seeds, system info

All data is stored locally in SQLite with no external dependencies.

## 🔄 Reproducibility

RexF ensures reproducibility by automatically tracking:

- **Code version**: Git commit hash and diff
- **Environment**: Python version, installed packages
- **Parameters**: All function arguments and defaults
- **Random seeds**: Automatic seed capture and restoration
- **System info**: OS, hardware, execution environment

## 🚧 Roadmap

- ✅ **Phase 1**: Simple API and smart features
- ✅ **Phase 2**: Auto-exploration and insights
- ✅ **Phase 3**: Web dashboard and CLI tools
- 🔄 **Phase 4**: Advanced optimization and ML integration
- 📋 **Phase 5**: Cloud sync and collaboration features

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/dhruv1110/rexf.git
cd rexf
pip install -e ".[dev]"
pre-commit install
```

### Running Tests

```bash
pytest tests/ -v --cov=rexf
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🔗 Links

- **Documentation**: [GitHub Pages](https://dhruv1110.github.io/rexf/)
- **PyPI**: [https://pypi.org/project/rexf/](https://pypi.org/project/rexf/)
- **Issues**: [GitHub Issues](https://github.com/dhruv1110/rexf/issues)
- **Discussions**: [GitHub Discussions](https://github.com/dhruv1110/rexf/discussions)

---

**Made with ❤️ for researchers who want to focus on science, not infrastructure.**