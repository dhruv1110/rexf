"""Export functionality for experiments to JSON/YAML formats."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Union

from ..core.interfaces import ExportInterface
from ..core.models import ExperimentData

# Optional dependency
try:
    import yaml

    HAS_YAML = True
except ImportError:
    yaml = None
    HAS_YAML = False


class JSONExporter(ExportInterface):
    """JSON format exporter for experiments."""

    def __init__(self, indent: int = 2):
        """
        Initialize JSON exporter.

        Args:
            indent: JSON indentation level
        """
        self.indent = indent

    def export_experiment(
        self, experiment: ExperimentData, include_artifacts: bool = False
    ) -> str:
        """Export a single experiment to JSON string."""
        data = self._experiment_to_dict(experiment, include_artifacts)
        return json.dumps(data, indent=self.indent, default=self._json_serializer)

    def export_experiments(
        self, experiments: List[ExperimentData], include_artifacts: bool = False
    ) -> str:
        """Export multiple experiments to JSON string."""
        data = {
            "export_info": {
                "exported_at": datetime.now().isoformat(),
                "experiment_count": len(experiments),
                "format": "json",
                "version": "1.0",
            },
            "experiments": [
                self._experiment_to_dict(exp, include_artifacts) for exp in experiments
            ],
        }
        return json.dumps(data, indent=self.indent, default=self._json_serializer)

    def save_export(self, data: str, filepath: Union[str, Path]) -> None:
        """Save exported data to file."""
        with open(filepath, "w") as f:
            f.write(data)

    def _experiment_to_dict(
        self, experiment: ExperimentData, include_artifacts: bool = False
    ) -> Dict[str, Any]:
        """Convert experiment to dictionary."""
        data = {
            "run_id": experiment.run_id,
            "experiment_name": experiment.experiment_name,
            "start_time": experiment.start_time.isoformat(),
            "end_time": (
                experiment.end_time.isoformat() if experiment.end_time else None
            ),
            "duration": experiment.duration,
            "status": experiment.status,
            "parameters": experiment.parameters,
            "metrics": experiment.metrics,
            "results": experiment.results,
            "git_commit": experiment.git_commit,
            "git_status": experiment.git_status,
            "environment_info": experiment.environment_info,
            "random_seed": experiment.random_seed,
            "metadata": experiment.metadata,
        }

        if include_artifacts:
            data["artifacts"] = experiment.artifacts

        return data

    def _json_serializer(self, obj: Any) -> Any:
        """Custom JSON serializer for non-standard types."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, "__dict__"):
            return obj.__dict__
        else:
            return str(obj)


class YAMLExporter(ExportInterface):
    """YAML format exporter for experiments."""

    def __init__(self):
        """Initialize YAML exporter."""
        if not HAS_YAML:
            raise ImportError("PyYAML is required for YAML export functionality")

    def export_experiment(
        self, experiment: ExperimentData, include_artifacts: bool = False
    ) -> str:
        """Export a single experiment to YAML string."""
        data = self._experiment_to_dict(experiment, include_artifacts)
        return yaml.dump(data, default_flow_style=False, sort_keys=False)

    def export_experiments(
        self, experiments: List[ExperimentData], include_artifacts: bool = False
    ) -> str:
        """Export multiple experiments to YAML string."""
        data = {
            "export_info": {
                "exported_at": datetime.now().isoformat(),
                "experiment_count": len(experiments),
                "format": "yaml",
                "version": "1.0",
            },
            "experiments": [
                self._experiment_to_dict(exp, include_artifacts) for exp in experiments
            ],
        }
        return yaml.dump(data, default_flow_style=False, sort_keys=False)

    def save_export(self, data: str, filepath: Union[str, Path]) -> None:
        """Save exported data to file."""
        with open(filepath, "w") as f:
            f.write(data)

    def _experiment_to_dict(
        self, experiment: ExperimentData, include_artifacts: bool = False
    ) -> Dict[str, Any]:
        """Convert experiment to dictionary."""
        data = {
            "run_id": experiment.run_id,
            "experiment_name": experiment.experiment_name,
            "start_time": experiment.start_time.isoformat(),
            "end_time": (
                experiment.end_time.isoformat() if experiment.end_time else None
            ),
            "duration": experiment.duration,
            "status": experiment.status,
            "parameters": experiment.parameters,
            "metrics": experiment.metrics,
            "results": experiment.results,
            "git_commit": experiment.git_commit,
            "git_status": experiment.git_status,
            "environment_info": experiment.environment_info,
            "random_seed": experiment.random_seed,
            "metadata": experiment.metadata,
        }

        if include_artifacts:
            data["artifacts"] = experiment.artifacts

        return data


class ExperimentExporter:
    """High-level experiment exporter that supports multiple formats."""

    def __init__(self):
        """Initialize experiment exporter."""
        self.json_exporter = JSONExporter()
        self.yaml_exporter = YAMLExporter() if HAS_YAML else None

    def export_to_json(
        self,
        experiments: Union[ExperimentData, List[ExperimentData]],
        include_artifacts: bool = False,
    ) -> str:
        """Export experiments to JSON format."""
        if isinstance(experiments, list):
            return self.json_exporter.export_experiments(experiments, include_artifacts)
        else:
            return self.json_exporter.export_experiment(experiments, include_artifacts)

    def export_to_yaml(
        self,
        experiments: Union[ExperimentData, List[ExperimentData]],
        include_artifacts: bool = False,
    ) -> str:
        """Export experiments to YAML format."""
        if not self.yaml_exporter:
            raise ImportError("PyYAML is required for YAML export")

        if isinstance(experiments, list):
            return self.yaml_exporter.export_experiments(experiments, include_artifacts)
        else:
            return self.yaml_exporter.export_experiment(experiments, include_artifacts)

    def export_to_file(
        self,
        experiments: Union[ExperimentData, List[ExperimentData]],
        filepath: Union[str, Path],
        format: str = "auto",
        include_artifacts: bool = False,
    ) -> None:
        """Export experiments to file with automatic format detection."""
        filepath = Path(filepath)

        if format == "auto":
            format = filepath.suffix.lower()
            if format == ".json":
                format = "json"
            elif format in [".yaml", ".yml"]:
                format = "yaml"
            else:
                format = "json"  # Default to JSON

        if format == "json":
            data = self.export_to_json(experiments, include_artifacts)
            self.json_exporter.save_export(data, filepath)
        elif format == "yaml":
            data = self.export_to_yaml(experiments, include_artifacts)
            if self.yaml_exporter:
                self.yaml_exporter.save_export(data, filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")

    @property
    def available_formats(self) -> List[str]:
        """Get list of available export formats."""
        formats = ["json"]
        if self.yaml_exporter:
            formats.append("yaml")
        return formats
