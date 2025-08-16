"""Core interfaces for rexf storage and export systems.

These interfaces provide clean abstractions that allow for easy extension
and swapping of storage backends, export formats, and other components.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .models import ExperimentData


class StorageInterface(ABC):
    """Abstract interface for experiment data storage."""

    @abstractmethod
    def save_experiment(self, experiment: ExperimentData) -> str:
        """Save experiment data and return run_id."""
        pass

    @abstractmethod
    def load_experiment(self, run_id: str) -> Optional[ExperimentData]:
        """Load experiment data by run_id."""
        pass

    @abstractmethod
    def list_experiments(
        self, experiment_name: Optional[str] = None, limit: Optional[int] = None
    ) -> List[ExperimentData]:
        """List experiments, optionally filtered by name."""
        pass

    @abstractmethod
    def update_experiment(self, experiment: ExperimentData) -> None:
        """Update existing experiment data."""
        pass

    @abstractmethod
    def delete_experiment(self, run_id: str) -> bool:
        """Delete experiment by run_id. Returns True if deleted."""
        pass

    @abstractmethod
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close storage connection and cleanup resources."""
        pass


class ArtifactManagerInterface(ABC):
    """Abstract interface for artifact storage and management."""

    @abstractmethod
    def store_artifact(
        self,
        run_id: str,
        artifact_name: str,
        data: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store an artifact and return its path."""
        pass

    @abstractmethod
    def load_artifact(self, run_id: str, artifact_name: str) -> Any:
        """Load an artifact by name."""
        pass

    @abstractmethod
    def get_artifact_path(self, run_id: str, artifact_name: str) -> Optional[Path]:
        """Get the file path for an artifact."""
        pass

    @abstractmethod
    def list_artifacts(self, run_id: str) -> List[str]:
        """List all artifact names for a run."""
        pass

    @abstractmethod
    def delete_artifact(self, run_id: str, artifact_name: str) -> bool:
        """Delete an artifact. Returns True if deleted."""
        pass


class ExportInterface(ABC):
    """Abstract interface for experiment data export."""

    @abstractmethod
    def export_experiment(
        self, experiment: ExperimentData, include_artifacts: bool = False
    ) -> str:
        """Export a single experiment to string format."""
        pass

    @abstractmethod
    def export_experiments(
        self, experiments: List[ExperimentData], include_artifacts: bool = False
    ) -> str:
        """Export multiple experiments to string format."""
        pass

    @abstractmethod
    def save_export(self, data: str, filepath: Union[str, Path]) -> None:
        """Save exported data to file."""
        pass


class VisualizationInterface(ABC):
    """Abstract interface for experiment visualization."""

    @abstractmethod
    def plot_metrics(
        self,
        experiments: List[ExperimentData],
        metric_names: Optional[List[str]] = None,
    ) -> Any:
        """Plot metrics comparison across experiments."""
        pass

    @abstractmethod
    def plot_parameter_space(
        self,
        experiments: List[ExperimentData],
        param_names: Optional[List[str]] = None,
    ) -> Any:
        """Visualize parameter space exploration."""
        pass

    @abstractmethod
    def create_comparison_table(self, experiments: List[ExperimentData]) -> Any:
        """Create a comparison table of experiments."""
        pass
