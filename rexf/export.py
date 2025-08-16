"""Export functionality for experiments to JSON/YAML formats.

This module provides exporters that can serialize experiment data
for sharing or paper submission.
"""

import json
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Union

from .interfaces import ExportInterface, ExperimentMetadata


class JSONExporter(ExportInterface):
    """JSON format exporter for experiments."""
    
    def __init__(self, indent: int = 2):
        """Initialize JSON exporter.
        
        Args:
            indent: JSON indentation level
        """
        self.indent = indent
    
    def export_experiment(
        self, 
        metadata: ExperimentMetadata,
        include_artifacts: bool = False
    ) -> str:
        """Export a single experiment to JSON string."""
        data = self._metadata_to_dict(metadata, include_artifacts)
        return json.dumps(data, indent=self.indent, default=self._json_serializer)
    
    def export_experiments(
        self, 
        experiments: List[ExperimentMetadata],
        include_artifacts: bool = False
    ) -> str:
        """Export multiple experiments to JSON string."""
        data = {
            "export_info": {
                "exported_at": datetime.now().isoformat(),
                "experiment_count": len(experiments),
                "format": "json",
                "version": "1.0"
            },
            "experiments": [
                self._metadata_to_dict(exp, include_artifacts) 
                for exp in experiments
            ]
        }
        return json.dumps(data, indent=self.indent, default=self._json_serializer)
    
    def save_export(self, data: str, filepath: Union[str, Path]) -> None:
        """Save exported data to JSON file."""
        filepath = Path(filepath)
        if not filepath.suffix:
            filepath = filepath.with_suffix('.json')
        
        with open(filepath, 'w') as f:
            f.write(data)
    
    def _json_serializer(self, obj: Any) -> Any:
        """Custom JSON serializer for non-standard types."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, Path):
            return str(obj)
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)
    
    def _metadata_to_dict(
        self, 
        metadata: ExperimentMetadata, 
        include_artifacts: bool = False
    ) -> Dict[str, Any]:
        """Convert experiment metadata to dictionary."""
        data = {
            "experiment_id": metadata.experiment_id,
            "run_id": metadata.run_id,
            "name": metadata.name,
            "parameters": metadata.parameters,
            "start_time": metadata.start_time.isoformat(),
            "end_time": metadata.end_time.isoformat() if metadata.end_time else None,
            "status": metadata.status,
            "git_commit": metadata.git_commit,
            "environment_info": metadata.environment_info,
            "random_seed": metadata.random_seed,
            "metrics": metadata.metrics,
            "results": metadata.results,
        }
        
        if include_artifacts:
            data["artifacts"] = metadata.artifacts
        else:
            data["artifact_names"] = list(metadata.artifacts.keys())
        
        return data


class YAMLExporter(ExportInterface):
    """YAML format exporter for experiments."""
    
    def __init__(self, default_flow_style: bool = False):
        """Initialize YAML exporter.
        
        Args:
            default_flow_style: YAML flow style setting
        """
        self.default_flow_style = default_flow_style
    
    def export_experiment(
        self, 
        metadata: ExperimentMetadata,
        include_artifacts: bool = False
    ) -> str:
        """Export a single experiment to YAML string."""
        data = self._metadata_to_dict(metadata, include_artifacts)
        return yaml.dump(
            data, 
            default_flow_style=self.default_flow_style,
            allow_unicode=True,
            sort_keys=False
        )
    
    def export_experiments(
        self, 
        experiments: List[ExperimentMetadata],
        include_artifacts: bool = False
    ) -> str:
        """Export multiple experiments to YAML string."""
        data = {
            "export_info": {
                "exported_at": datetime.now().isoformat(),
                "experiment_count": len(experiments),
                "format": "yaml",
                "version": "1.0"
            },
            "experiments": [
                self._metadata_to_dict(exp, include_artifacts) 
                for exp in experiments
            ]
        }
        return yaml.dump(
            data, 
            default_flow_style=self.default_flow_style,
            allow_unicode=True,
            sort_keys=False
        )
    
    def save_export(self, data: str, filepath: Union[str, Path]) -> None:
        """Save exported data to YAML file."""
        filepath = Path(filepath)
        if not filepath.suffix:
            filepath = filepath.with_suffix('.yaml')
        
        with open(filepath, 'w') as f:
            f.write(data)
    
    def _metadata_to_dict(
        self, 
        metadata: ExperimentMetadata, 
        include_artifacts: bool = False
    ) -> Dict[str, Any]:
        """Convert experiment metadata to dictionary."""
        data = {
            "experiment_id": metadata.experiment_id,
            "run_id": metadata.run_id,
            "name": metadata.name,
            "parameters": metadata.parameters,
            "start_time": metadata.start_time.isoformat(),
            "end_time": metadata.end_time.isoformat() if metadata.end_time else None,
            "status": metadata.status,
            "git_commit": metadata.git_commit,
            "environment_info": metadata.environment_info,
            "random_seed": metadata.random_seed,
            "metrics": metadata.metrics,
            "results": metadata.results,
        }
        
        if include_artifacts:
            data["artifacts"] = metadata.artifacts
        else:
            data["artifact_names"] = list(metadata.artifacts.keys())
        
        return data


class ExperimentExporter:
    """High-level exporter that combines JSON and YAML exporters."""
    
    def __init__(self):
        """Initialize experiment exporter."""
        self.json_exporter = JSONExporter()
        self.yaml_exporter = YAMLExporter()
    
    def export_to_file(
        self,
        experiments: Union[ExperimentMetadata, List[ExperimentMetadata]],
        filepath: Union[str, Path],
        format: str = "auto",
        include_artifacts: bool = False
    ) -> None:
        """Export experiments to file.
        
        Args:
            experiments: Single experiment or list of experiments
            filepath: Output file path
            format: Export format ("json", "yaml", or "auto")
            include_artifacts: Whether to include artifact paths
        """
        filepath = Path(filepath)
        
        # Auto-detect format from file extension
        if format == "auto":
            if filepath.suffix.lower() in ['.json']:
                format = "json"
            elif filepath.suffix.lower() in ['.yaml', '.yml']:
                format = "yaml"
            else:
                format = "json"  # Default to JSON
        
        # Ensure experiments is a list
        if isinstance(experiments, ExperimentMetadata):
            experiments = [experiments]
        
        # Export based on format
        if format == "json":
            if len(experiments) == 1:
                data = self.json_exporter.export_experiment(experiments[0], include_artifacts)
            else:
                data = self.json_exporter.export_experiments(experiments, include_artifacts)
            self.json_exporter.save_export(data, filepath)
        
        elif format == "yaml":
            if len(experiments) == 1:
                data = self.yaml_exporter.export_experiment(experiments[0], include_artifacts)
            else:
                data = self.yaml_exporter.export_experiments(experiments, include_artifacts)
            self.yaml_exporter.save_export(data, filepath)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def export_comparison(
        self,
        experiments: List[ExperimentMetadata],
        filepath: Union[str, Path],
        format: str = "auto"
    ) -> None:
        """Export experiment comparison to file.
        
        Args:
            experiments: List of experiments to compare
            filepath: Output file path
            format: Export format ("json", "yaml", or "auto")
        """
        from .runner import ExperimentRunner
        
        # Create a temporary runner to use comparison logic
        runner = ExperimentRunner()
        comparison = runner._compare_parameters(experiments)
        metrics_comparison = runner._compare_metrics(experiments)
        results_comparison = runner._compare_results(experiments)
        
        comparison_data = {
            "comparison_info": {
                "generated_at": datetime.now().isoformat(),
                "experiment_count": len(experiments),
                "experiment_names": [exp.name for exp in experiments],
                "run_ids": [exp.run_id for exp in experiments]
            },
            "parameter_comparison": comparison,
            "metric_comparison": metrics_comparison,
            "result_comparison": results_comparison,
            "experiments": [
                {
                    "run_id": exp.run_id,
                    "name": exp.name,
                    "start_time": exp.start_time.isoformat(),
                    "status": exp.status,
                    "parameters": exp.parameters,
                    "metrics": exp.metrics,
                    "results": exp.results
                }
                for exp in experiments
            ]
        }
        
        filepath = Path(filepath)
        
        # Auto-detect format
        if format == "auto":
            if filepath.suffix.lower() in ['.json']:
                format = "json"
            elif filepath.suffix.lower() in ['.yaml', '.yml']:
                format = "yaml"
            else:
                format = "json"
        
        # Save comparison
        if format == "json":
            with open(filepath, 'w') as f:
                json.dump(comparison_data, f, indent=2, default=str)
        elif format == "yaml":
            with open(filepath, 'w') as f:
                yaml.dump(comparison_data, f, default_flow_style=False, allow_unicode=True)
        else:
            raise ValueError(f"Unsupported export format: {format}")
