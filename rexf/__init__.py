"""rexf - Smart Experiments Framework.

A lightweight Python library for reproducible computational experiments
with an ultra-simple, smart API.

Usage:
    from rexf import experiment, run

    @experiment
    def my_experiment(param1, param2):
        result = do_something(param1, param2)
        return {"score": result}

    # Run single experiment
    run.single(my_experiment, param1=1.0, param2=2.0)

    # Auto-explore parameter space
    run.auto_explore(my_experiment, strategy="random", budget=10)

    # Get insights
    print(run.insights())

    # Launch web dashboard
    run.dashboard()
"""

from . import run
from .core.models import ExperimentData, ExperimentRun
from .core.simple_api import experiment

__version__ = "0.1.0"

# Clean, simple API - this is all users need
__all__ = [
    "experiment",      # @experiment decorator
    "run",            # All experiment operations
    "ExperimentRun",  # Data model (for advanced users)
    "ExperimentData", # Alias for ExperimentRun
]