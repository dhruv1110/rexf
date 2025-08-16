"""Ultra-simple API for rexf experiments.

This module provides the zero-config experiment decorator that users actually want.
No complex configuration, just decorate your function and go.
"""

import functools
import inspect
from typing import Any, Callable, Dict, Optional


def experiment(
    name_or_func: Optional[str] = None, *, optimize_for: Optional[str] = None
):
    """
    Ultra-simple experiment decorator with zero configuration required.

    Usage:
        @experiment
        def my_experiment(param1, param2):
            result = do_something(param1, param2)
            return {"score": result}

        @experiment("custom_name")
        def my_experiment(param1, param2):
            return {"accuracy": 0.95}

        @experiment(optimize_for="accuracy")
        def my_experiment(param1, param2):
            return {"accuracy": 0.95, "loss": 0.1}

    Args:
        name_or_func: Optional experiment name or function (for @experiment usage)
        optimize_for: Optional metric name to optimize for in auto-exploration

    Returns:
        Decorated function with experiment metadata attached
    """

    def decorator(func: Callable) -> Callable:
        # Auto-detect experiment name
        if isinstance(name_or_func, str):
            experiment_name = name_or_func
        else:
            experiment_name = func.__name__

        # Auto-detect parameters from function signature
        sig = inspect.signature(func)
        auto_params = {}

        for param_name, param in sig.parameters.items():
            param_type = (
                param.annotation if param.annotation != inspect.Parameter.empty else Any
            )
            param_default = (
                param.default if param.default != inspect.Parameter.empty else None
            )

            auto_params[param_name] = {
                "type": param_type,
                "default": param_default,
                "description": f"Auto-detected parameter: {param_name}",
            }

        # Store simple experiment metadata
        experiment_metadata = {
            "name": experiment_name,
            "function": func,
            "auto_params": auto_params,
            "optimize_for": optimize_for,
            "original_signature": sig,
        }

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        # Attach metadata to function
        wrapper._experiment_metadata = experiment_metadata
        wrapper._is_rexf_experiment = True

        return wrapper

    # Handle both @experiment and @experiment("name") usage
    if callable(name_or_func):
        # @experiment (without parentheses)
        return decorator(name_or_func)
    else:
        # @experiment("name") or @experiment(optimize_for="metric")
        return decorator


def is_experiment(func: Callable) -> bool:
    """Check if a function is decorated with @experiment."""
    return hasattr(func, "_is_rexf_experiment") and func._is_rexf_experiment


def get_experiment_metadata(func: Callable) -> Dict[str, Any]:
    """Get experiment metadata from a decorated function."""
    if not is_experiment(func):
        raise ValueError(f"Function {func.__name__} is not decorated with @experiment")

    return func._experiment_metadata


def auto_track_returns(result: Any, experiment_name: str) -> Dict[str, Any]:
    """
    Automatically categorize return values into metrics, results, and artifacts.

    Rules:
    - Numbers (int, float) -> metrics
    - Strings, complex objects -> results
    - Files that exist -> artifacts
    - Dictionaries -> split based on content
    """
    if result is None:
        return {"metrics": {}, "results": {}, "artifacts": {}}

    if isinstance(result, dict):
        # Smart categorization of dictionary returns
        metrics = {}
        results = {}
        artifacts = {}

        for key, value in result.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                # Numbers go to metrics (except booleans)
                metrics[key] = value
            elif isinstance(value, str) and _looks_like_filepath(value):
                # File paths go to artifacts
                artifacts[key] = value
            else:
                # Everything else goes to results
                results[key] = value

        return {"metrics": metrics, "results": results, "artifacts": artifacts}

    elif isinstance(result, (int, float)) and not isinstance(result, bool):
        # Single number result
        return {"metrics": {"result": result}, "results": {}, "artifacts": {}}

    else:
        # Single non-numeric result
        return {"metrics": {}, "results": {"result": result}, "artifacts": {}}


def _looks_like_filepath(value: str) -> bool:
    """Heuristic to detect if a string looks like a file path."""
    import os

    # Check if it has file extension
    if "." in value and len(value.split(".")[-1]) <= 4:
        return True

    # Check if file actually exists
    if os.path.exists(value):
        return True

    # Check for common file patterns
    file_indicators = [
        ".png",
        ".jpg",
        ".pdf",
        ".csv",
        ".json",
        ".txt",
        ".pkl",
        ".model",
    ]
    return any(indicator in value.lower() for indicator in file_indicators)


def extract_parameter_values(func: Callable, *args, **kwargs) -> Dict[str, Any]:
    """Extract parameter values from function call arguments."""
    sig = inspect.signature(func)
    bound_args = sig.bind(*args, **kwargs)
    bound_args.apply_defaults()

    return dict(bound_args.arguments)
