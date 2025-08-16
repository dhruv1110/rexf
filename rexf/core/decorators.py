"""Clean decorator API for rexf experiments.

This module provides the primary API for defining experiments using a single,
clean decorator instead of stacking multiple decorators.
"""

import functools
from typing import Any, Callable, Dict, Optional, Tuple, Union


def experiment_config(
    name: str,
    params: Optional[Dict[str, Union[Tuple[type, str], Tuple[type, Any, str]]]] = None,
    metrics: Optional[Dict[str, Tuple[type, str]]] = None,
    results: Optional[Dict[str, Tuple[type, str]]] = None,
    artifacts: Optional[Dict[str, Tuple[str, str]]] = None,
    seed: str = "random_seed",
):
    """
    Clean single decorator that defines the entire experiment configuration.

    This is the primary API for defining experiments in rexf. It replaces
    the need to stack multiple decorators and provides a clean, readable
    configuration in a single place.

    Args:
        name: Name of the experiment
        params: Dictionary of parameters {name: (type, description) or (type, default, description)}
        metrics: Dictionary of metrics {name: (type, description)}
        results: Dictionary of results {name: (type, description)}
        artifacts: Dictionary of artifacts {name: (filename, description)}
        seed: Name of the seed parameter (default: "random_seed")

    Example:
        @experiment_config(
            name="monte_carlo_pi",
            params={
                "n_samples": (int, "Number of random samples to generate"),
                "batch_size": (int, 1000, "Size of each processing batch")
            },
            metrics={
                "pi_estimate": (float, "Estimated value of π"),
                "error": (float, "Absolute error from true π"),
            },
            results={
                "final_pi": (float, "Final π estimate"),
                "samples_inside": (int, "Samples inside circle")
            },
            artifacts={
                "convergence_plot": ("pi_convergence.png", "Convergence plot"),
                "sample_plot": ("sample_distribution.png", "Sample distribution")
            }
        )
        def estimate_pi(n_samples, batch_size=1000, random_seed=42):
            # ... implementation ...
            return {"final_pi": pi_est, "samples_inside": count}
    """

    def decorator(func: Callable) -> Callable:
        # Store experiment configuration on the function
        func._experiment_config = {
            "name": name,
            "params": params or {},
            "metrics": metrics or {},
            "results": results or {},
            "artifacts": artifacts or {},
            "seed": seed,
        }

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        # Copy the configuration to the wrapper
        wrapper._experiment_config = func._experiment_config
        return wrapper

    return decorator


class ExperimentBuilder:
    """
    Fluent interface builder for experiment configuration.

    Provides a clean, chainable API for defining experiments.

    Example:
        exp = (ExperimentBuilder("monte_carlo_pi")
            .param("n_samples", int, description="Number of samples")
            .param("batch_size", int, default=1000, description="Batch size")
            .metric("pi_estimate", float, description="π estimate")
            .result("final_pi", float, description="Final π value")
            .artifact("plot", "convergence.png", description="Plot")
            .seed("random_seed")
        )

        @exp.build
        def estimate_pi(n_samples, batch_size=1000, random_seed=42):
            # ... implementation ...
            return {"final_pi": pi_est}
    """

    def __init__(self, name: str):
        """Initialize builder with experiment name."""
        self.name = name
        self.params = {}
        self.metrics = {}
        self.results = {}
        self.artifacts = {}
        self.seed_param = "random_seed"

    def param(self, name: str, type_: type, default: Any = None, description: str = ""):
        """Add a parameter definition."""
        if default is not None:
            self.params[name] = (type_, default, description)
        else:
            self.params[name] = (type_, description)
        return self

    def metric(self, name: str, type_: type, description: str = ""):
        """Add a metric definition."""
        self.metrics[name] = (type_, description)
        return self

    def result(self, name: str, type_: type, description: str = ""):
        """Add a result definition."""
        self.results[name] = (type_, description)
        return self

    def artifact(self, name: str, filename: str, description: str = ""):
        """Add an artifact definition."""
        self.artifacts[name] = (filename, description)
        return self

    def seed(self, param_name: str = "random_seed"):
        """Set the seed parameter name."""
        self.seed_param = param_name
        return self

    def build(self, func: Callable) -> Callable:
        """Apply the configuration to a function."""
        return experiment_config(
            name=self.name,
            params=self.params,
            metrics=self.metrics,
            results=self.results,
            artifacts=self.artifacts,
            seed=self.seed_param,
        )(func)


# Convenience aliases
configure_experiment = experiment_config  # Alternative name
experiment_builder = ExperimentBuilder  # Alternative name
