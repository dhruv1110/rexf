"""Decorator-based API for marking experiment metadata.

This module provides decorators for marking parameters, results, metrics,
artifacts, and seeds on functions and classes.
"""

import functools
import inspect
from typing import Any, Callable, Dict, List, Optional, Type, Union


class ExperimentMeta:
    """Container for experiment metadata collected from decorators."""
    
    def __init__(self, name: str):
        self.name = name
        self.parameters: Dict[str, Dict[str, Any]] = {}
        self.results: Dict[str, Type] = {}
        self.metrics: Dict[str, Type] = {}
        self.artifacts: Dict[str, str] = {}
        self.seeds: List[str] = []
        self.function: Optional[Callable] = None


def experiment(name: str) -> Callable:
    """Decorator to mark a function as an experiment.
    
    Args:
        name: Name of the experiment
        
    Returns:
        Decorated function with experiment metadata
    """
    def decorator(func: Callable) -> Callable:
        if not hasattr(func, '_rexf_meta'):
            func._rexf_meta = ExperimentMeta(name)
        func._rexf_meta.name = name
        func._rexf_meta.function = func
        return func
    return decorator


def param(
    name: str, 
    param_type: Type, 
    default: Any = None, 
    description: str = ""
) -> Callable:
    """Decorator to mark a parameter for an experiment.
    
    Args:
        name: Parameter name
        param_type: Expected type of the parameter
        default: Default value (optional)
        description: Parameter description (optional)
        
    Returns:
        Decorated function with parameter metadata
    """
    def decorator(func: Callable) -> Callable:
        if not hasattr(func, '_rexf_meta'):
            func._rexf_meta = ExperimentMeta(getattr(func, '__name__', 'unknown'))
        
        func._rexf_meta.parameters[name] = {
            'type': param_type,
            'default': default,
            'description': description,
            'required': default is None
        }
        return func
    return decorator


def result(name: str, result_type: Type, description: str = "") -> Callable:
    """Decorator to mark a result for an experiment.
    
    Args:
        name: Result name
        result_type: Expected type of the result
        description: Result description (optional)
        
    Returns:
        Decorated function with result metadata
    """
    def decorator(func: Callable) -> Callable:
        if not hasattr(func, '_rexf_meta'):
            func._rexf_meta = ExperimentMeta(getattr(func, '__name__', 'unknown'))
        
        func._rexf_meta.results[name] = {
            'type': result_type,
            'description': description
        }
        return func
    return decorator


def metric(name: str, metric_type: Type, description: str = "") -> Callable:
    """Decorator to mark a metric for an experiment.
    
    Args:
        name: Metric name
        metric_type: Expected type of the metric
        description: Metric description (optional)
        
    Returns:
        Decorated function with metric metadata
    """
    def decorator(func: Callable) -> Callable:
        if not hasattr(func, '_rexf_meta'):
            func._rexf_meta = ExperimentMeta(getattr(func, '__name__', 'unknown'))
        
        func._rexf_meta.metrics[name] = {
            'type': metric_type,
            'description': description
        }
        return func
    return decorator


def artifact(name: str, filename: str, description: str = "") -> Callable:
    """Decorator to mark an artifact for an experiment.
    
    Args:
        name: Artifact name
        filename: Expected filename or pattern
        description: Artifact description (optional)
        
    Returns:
        Decorated function with artifact metadata
    """
    def decorator(func: Callable) -> Callable:
        if not hasattr(func, '_rexf_meta'):
            func._rexf_meta = ExperimentMeta(getattr(func, '__name__', 'unknown'))
        
        func._rexf_meta.artifacts[name] = {
            'filename': filename,
            'description': description
        }
        return func
    return decorator


def seed(name: str = "random_seed") -> Callable:
    """Decorator to mark a random seed parameter for reproducibility.
    
    Args:
        name: Name of the seed parameter
        
    Returns:
        Decorated function with seed metadata
    """
    def decorator(func: Callable) -> Callable:
        if not hasattr(func, '_rexf_meta'):
            func._rexf_meta = ExperimentMeta(getattr(func, '__name__', 'unknown'))
        
        func._rexf_meta.seeds.append(name)
        
        # Also add as a parameter if not already defined
        if name not in func._rexf_meta.parameters:
            func._rexf_meta.parameters[name] = {
                'type': int,
                'default': None,
                'description': f'Random seed for reproducibility',
                'required': False
            }
        return func
    return decorator


class ExperimentClass:
    """Base class for experiment classes with decorator support."""
    
    def __init__(self):
        self._rexf_meta = ExperimentMeta(self.__class__.__name__)
        self._collect_metadata()
    
    def _collect_metadata(self):
        """Collect metadata from decorated methods."""
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if hasattr(method, '_rexf_meta'):
                # Merge metadata from method into class metadata
                meta = method._rexf_meta
                self._rexf_meta.parameters.update(meta.parameters)
                self._rexf_meta.results.update(meta.results)
                self._rexf_meta.metrics.update(meta.metrics)
                self._rexf_meta.artifacts.update(meta.artifacts)
                self._rexf_meta.seeds.extend(meta.seeds)


def validate_experiment_function(func: Callable) -> bool:
    """Validate that a function has proper experiment metadata.
    
    Args:
        func: Function to validate
        
    Returns:
        True if valid experiment function
    """
    if not hasattr(func, '_rexf_meta'):
        return False
    
    meta = func._rexf_meta
    
    # Check that function signature matches declared parameters
    sig = inspect.signature(func)
    declared_params = set(meta.parameters.keys())
    function_params = set(sig.parameters.keys())
    
    # Allow extra function parameters (like *args, **kwargs)
    missing_params = declared_params - function_params
    if missing_params:
        raise ValueError(
            f"Function {func.__name__} is missing declared parameters: {missing_params}"
        )
    
    return True


def get_experiment_metadata(func: Callable) -> Optional[ExperimentMeta]:
    """Get experiment metadata from a decorated function.
    
    Args:
        func: Decorated function
        
    Returns:
        ExperimentMeta object or None if not an experiment function
    """
    return getattr(func, '_rexf_meta', None)
