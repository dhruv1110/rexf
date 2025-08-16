"""Core interfaces for rexf storage and export systems.

These interfaces provide clean abstractions that allow for easy extension
and swapping of storage backends, export formats, and other components.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import uuid
from datetime import datetime


class ExperimentMetadata:
    """Container for experiment metadata."""
    
    def __init__(
        self,
        experiment_id: str,
        run_id: str,
        name: str,
        parameters: Dict[str, Any],
        start_time: datetime,
        end_time: Optional[datetime] = None,
        status: str = "running",
        git_commit: Optional[str] = None,
        environment_info: Optional[Dict[str, Any]] = None,
        random_seed: Optional[int] = None,
    ):
        self.experiment_id = experiment_id
        self.run_id = run_id
        self.name = name
        self.parameters = parameters
        self.start_time = start_time
        self.end_time = end_time
        self.status = status
        self.git_commit = git_commit
        self.environment_info = environment_info or {}
        self.random_seed = random_seed
        self.metrics: Dict[str, Any] = {}
        self.results: Dict[str, Any] = {}
        self.artifacts: Dict[str, str] = {}  # name -> path
        

class StorageInterface(ABC):
    """Abstract interface for experiment data storage."""
    
    @abstractmethod
    def save_experiment(self, metadata: ExperimentMetadata) -> str:
        """Save experiment metadata and return run_id."""
        pass
    
    @abstractmethod
    def load_experiment(self, run_id: str) -> Optional[ExperimentMetadata]:
        """Load experiment metadata by run_id."""
        pass
    
    @abstractmethod
    def list_experiments(
        self, 
        experiment_name: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[ExperimentMetadata]:
        """List experiments, optionally filtered by name."""
        pass
    
    @abstractmethod
    def update_experiment(self, metadata: ExperimentMetadata) -> None:
        """Update existing experiment metadata."""
        pass
    
    @abstractmethod
    def delete_experiment(self, run_id: str) -> bool:
        """Delete experiment by run_id. Returns True if deleted."""
        pass


class ArtifactManagerInterface(ABC):
    """Abstract interface for artifact storage and management."""
    
    @abstractmethod
    def store_artifact(
        self, 
        run_id: str, 
        artifact_name: str, 
        data: Any,
        metadata: Optional[Dict[str, Any]] = None
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
        self, 
        metadata: ExperimentMetadata,
        include_artifacts: bool = False
    ) -> str:
        """Export a single experiment to string format."""
        pass
    
    @abstractmethod
    def export_experiments(
        self, 
        experiments: List[ExperimentMetadata],
        include_artifacts: bool = False
    ) -> str:
        """Export multiple experiments to string format."""
        pass
    
    @abstractmethod
    def save_export(
        self, 
        data: str, 
        filepath: Union[str, Path]
    ) -> None:
        """Save exported data to file."""
        pass


class VisualizationInterface(ABC):
    """Abstract interface for experiment visualization."""
    
    @abstractmethod
    def plot_metrics(
        self, 
        experiments: List[ExperimentMetadata],
        metric_names: Optional[List[str]] = None
    ) -> Any:
        """Plot metrics comparison across experiments."""
        pass
    
    @abstractmethod
    def plot_parameter_space(
        self, 
        experiments: List[ExperimentMetadata],
        param_names: Optional[List[str]] = None
    ) -> Any:
        """Visualize parameter space exploration."""
        pass
    
    @abstractmethod
    def create_comparison_table(
        self, 
        experiments: List[ExperimentMetadata]
    ) -> Any:
        """Create a comparison table of experiments."""
        pass


class ReproducibilityTracker:
    """Tracks reproducibility information for experiments."""
    
    def __init__(self):
        self._git_repo = None
        try:
            import git
            self._git_repo = git.Repo(search_parent_directories=True)
        except (ImportError, Exception):
            pass
    
    def get_git_commit(self) -> Optional[str]:
        """Get current Git commit hash."""
        if self._git_repo:
            try:
                return self._git_repo.head.commit.hexsha
            except Exception:
                pass
        return None
    
    def get_git_status(self) -> Dict[str, Any]:
        """Get Git repository status."""
        if self._git_repo:
            try:
                return {
                    "commit": self._git_repo.head.commit.hexsha,
                    "branch": self._git_repo.active_branch.name,
                    "is_dirty": self._git_repo.is_dirty(),
                    "untracked_files": self._git_repo.untracked_files,
                }
            except Exception:
                pass
        return {}
    
    def get_environment_info(self) -> Dict[str, Any]:
        """Get environment information."""
        import sys
        import platform
        import os
        
        return {
            "python_version": sys.version,
            "platform": platform.platform(),
            "architecture": platform.architecture(),
            "processor": platform.processor(),
            "hostname": platform.node(),
            "user": os.getenv("USER", "unknown"),
            "cwd": os.getcwd(),
            "python_executable": sys.executable,
        }
