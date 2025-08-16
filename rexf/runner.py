"""Experiment execution and management system.

This module provides the main ExperimentRunner class that orchestrates
experiment execution, metadata tracking, and result storage.
"""

import uuid
import time
import random
import traceback
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union
from pathlib import Path

from .interfaces import ExperimentMetadata, ReproducibilityTracker
from .decorators import get_experiment_metadata, validate_experiment_function
from .storage import SQLiteStorage
from .artifacts import FileSystemArtifactManager


class ExperimentRunner:
    """Main class for running and managing experiments."""
    
    def __init__(
        self,
        storage_path: Union[str, Path] = "experiments.db",
        artifacts_path: Union[str, Path] = "artifacts",
        auto_seed: bool = True
    ):
        """Initialize experiment runner.
        
        Args:
            storage_path: Path to SQLite database for metadata
            artifacts_path: Path to artifacts directory
            auto_seed: Whether to automatically set random seeds for reproducibility
        """
        self.storage = SQLiteStorage(storage_path)
        self.artifact_manager = FileSystemArtifactManager(artifacts_path)
        self.reproducibility_tracker = ReproducibilityTracker()
        self.auto_seed = auto_seed
    
    def run(
        self, 
        experiment_func: Callable,
        run_id: Optional[str] = None,
        **kwargs
    ) -> str:
        """Run an experiment function.
        
        Args:
            experiment_func: Decorated experiment function to run
            run_id: Optional run ID (generated if not provided)
            **kwargs: Parameters to pass to the experiment function
            
        Returns:
            Run ID of the executed experiment
        """
        # Validate experiment function
        if not validate_experiment_function(experiment_func):
            raise ValueError(f"Function {experiment_func.__name__} is not a valid experiment")
        
        meta = get_experiment_metadata(experiment_func)
        if not meta:
            raise ValueError(f"No experiment metadata found for {experiment_func.__name__}")
        
        # Generate run ID if not provided
        if not run_id:
            run_id = str(uuid.uuid4())
        
        # Create experiment metadata
        experiment_metadata = ExperimentMetadata(
            experiment_id=meta.name,
            run_id=run_id,
            name=meta.name,
            parameters={},
            start_time=datetime.now(),
            git_commit=self.reproducibility_tracker.get_git_commit(),
            environment_info=self.reproducibility_tracker.get_environment_info()
        )
        
        try:
            # Process parameters
            processed_params = self._process_parameters(meta, kwargs)
            experiment_metadata.parameters = processed_params
            
            # Set random seeds for reproducibility
            if self.auto_seed:
                self._set_random_seeds(meta, processed_params)
                if meta.seeds:
                    # Use the first seed as the main random seed
                    experiment_metadata.random_seed = processed_params.get(meta.seeds[0])
            
            # Save initial experiment state
            experiment_metadata.status = "running"
            self.storage.save_experiment(experiment_metadata)
            
            # Run the experiment
            print(f"Running experiment '{meta.name}' (run_id: {run_id})")
            start_time = time.time()
            
            result = experiment_func(**processed_params)
            
            end_time = time.time()
            experiment_metadata.end_time = datetime.now()
            experiment_metadata.status = "completed"
            
            # Process results
            if result:
                self._process_results(experiment_metadata, meta, result)
            
            # Save final experiment state
            self.storage.save_experiment(experiment_metadata)
            
            print(f"Experiment completed in {end_time - start_time:.2f} seconds")
            return run_id
            
        except Exception as e:
            # Mark experiment as failed
            experiment_metadata.end_time = datetime.now()
            experiment_metadata.status = "failed"
            experiment_metadata.results["error"] = str(e)
            experiment_metadata.results["traceback"] = traceback.format_exc()
            
            self.storage.save_experiment(experiment_metadata)
            
            print(f"Experiment failed: {e}")
            raise
    
    def _process_parameters(self, meta, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Process and validate experiment parameters."""
        processed = {}
        
        # Check required parameters
        for param_name, param_info in meta.parameters.items():
            if param_name in kwargs:
                processed[param_name] = kwargs[param_name]
            elif param_info.get('default') is not None:
                processed[param_name] = param_info['default']
            elif param_info.get('required', True):
                raise ValueError(f"Required parameter '{param_name}' not provided")
        
        return processed
    
    def _set_random_seeds(self, meta, parameters: Dict[str, Any]) -> None:
        """Set random seeds for reproducibility."""
        import numpy as np
        
        for seed_param in meta.seeds:
            if seed_param in parameters and parameters[seed_param] is not None:
                seed_value = parameters[seed_param]
                
                # Set Python random seed
                random.seed(seed_value)
                
                # Set NumPy random seed
                np.random.seed(seed_value)
                
                # Set other common random seeds if available
                try:
                    import torch
                    torch.manual_seed(seed_value)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed_value)
                except ImportError:
                    pass
                
                print(f"Set random seed: {seed_param} = {seed_value}")
    
    def _process_results(
        self, 
        experiment_metadata: ExperimentMetadata,
        meta,
        result: Any
    ) -> None:
        """Process experiment results and artifacts."""
        if isinstance(result, dict):
            # Process results and metrics
            for key, value in result.items():
                if key in meta.results:
                    experiment_metadata.results[key] = value
                elif key in meta.metrics:
                    experiment_metadata.metrics[key] = value
                else:
                    # Default to metrics for unknown keys
                    experiment_metadata.metrics[key] = value
            
            # Process artifacts
            for artifact_name, artifact_info in meta.artifacts.items():
                if artifact_name in result:
                    artifact_data = result[artifact_name]
                    try:
                        artifact_path = self.artifact_manager.store_artifact(
                            experiment_metadata.run_id,
                            artifact_name,
                            artifact_data
                        )
                        experiment_metadata.artifacts[artifact_name] = artifact_path
                        print(f"Stored artifact: {artifact_name} -> {artifact_path}")
                    except Exception as e:
                        print(f"Failed to store artifact {artifact_name}: {e}")
        else:
            # Single result value
            experiment_metadata.results["result"] = result
    
    def get_experiment(self, run_id: str) -> Optional[ExperimentMetadata]:
        """Get experiment metadata by run ID."""
        return self.storage.load_experiment(run_id)
    
    def list_experiments(
        self, 
        experiment_name: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[ExperimentMetadata]:
        """List experiments."""
        return self.storage.list_experiments(experiment_name, limit)
    
    def compare_runs(self, run_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple experiment runs."""
        experiments = []
        for run_id in run_ids:
            exp = self.get_experiment(run_id)
            if exp:
                experiments.append(exp)
        
        if not experiments:
            return {}
        
        comparison = {
            "runs": experiments,
            "parameter_comparison": self._compare_parameters(experiments),
            "metric_comparison": self._compare_metrics(experiments),
            "result_comparison": self._compare_results(experiments)
        }
        
        return comparison
    
    def _compare_parameters(self, experiments: List[ExperimentMetadata]) -> Dict[str, Any]:
        """Compare parameters across experiments."""
        all_params = set()
        for exp in experiments:
            all_params.update(exp.parameters.keys())
        
        comparison = {}
        for param in all_params:
            values = []
            for exp in experiments:
                values.append(exp.parameters.get(param, None))
            comparison[param] = {
                "values": values,
                "unique_values": list(set(v for v in values if v is not None)),
                "varies": len(set(v for v in values if v is not None)) > 1
            }
        
        return comparison
    
    def _compare_metrics(self, experiments: List[ExperimentMetadata]) -> Dict[str, Any]:
        """Compare metrics across experiments."""
        all_metrics = set()
        for exp in experiments:
            all_metrics.update(exp.metrics.keys())
        
        comparison = {}
        for metric in all_metrics:
            values = []
            for exp in experiments:
                values.append(exp.metrics.get(metric, None))
            
            numeric_values = [v for v in values if isinstance(v, (int, float))]
            
            comparison[metric] = {
                "values": values,
                "min": min(numeric_values) if numeric_values else None,
                "max": max(numeric_values) if numeric_values else None,
                "mean": sum(numeric_values) / len(numeric_values) if numeric_values else None
            }
        
        return comparison
    
    def _compare_results(self, experiments: List[ExperimentMetadata]) -> Dict[str, Any]:
        """Compare results across experiments."""
        all_results = set()
        for exp in experiments:
            all_results.update(exp.results.keys())
        
        comparison = {}
        for result in all_results:
            values = []
            for exp in experiments:
                values.append(exp.results.get(result, None))
            comparison[result] = {"values": values}
        
        return comparison
    
    def delete_experiment(self, run_id: str, delete_artifacts: bool = True) -> bool:
        """Delete an experiment and optionally its artifacts."""
        success = self.storage.delete_experiment(run_id)
        
        if delete_artifacts:
            self.artifact_manager.cleanup_run(run_id)
        
        return success
    
    def get_stats(self) -> Dict[str, Any]:
        """Get runner statistics."""
        storage_stats = self.storage.get_database_stats()
        artifact_stats = self.artifact_manager.get_storage_stats()
        
        return {
            "storage": storage_stats,
            "artifacts": artifact_stats,
            "git_status": self.reproducibility_tracker.get_git_status()
        }
