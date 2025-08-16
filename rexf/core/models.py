"""Data models and structures for rexf."""

from datetime import datetime
from typing import Any, Dict, Optional


class ExperimentRun:
    """Container for experiment run metadata and results."""

    def __init__(
        self,
        run_id: str,
        experiment_name: str,
        parameters: Dict[str, Any],
        start_time: datetime,
        end_time: Optional[datetime] = None,
        status: str = "running",
        git_commit: Optional[str] = None,
        git_status: Optional[Dict[str, Any]] = None,
        environment_info: Optional[Dict[str, Any]] = None,
        random_seed: Optional[int] = None,
    ):
        self.run_id = run_id
        self.experiment_name = experiment_name
        self.parameters = parameters
        self.start_time = start_time
        self.end_time = end_time
        self.status = status
        self.git_commit = git_commit
        self.git_status = git_status or {}
        self.environment_info = environment_info or {}
        self.random_seed = random_seed
        self.metrics: Dict[str, Any] = {}
        self.results: Dict[str, Any] = {}
        self.artifacts: Dict[str, str] = {}  # name -> path
        self.metadata: Dict[str, Any] = {}  # Additional metadata

    @property
    def duration(self) -> Optional[float]:
        """Get experiment duration in seconds."""
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "parameters": self.parameters,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "status": self.status,
            "git_commit": self.git_commit,
            "git_status": self.git_status,
            "environment_info": self.environment_info,
            "random_seed": self.random_seed,
            "metrics": self.metrics,
            "results": self.results,
            "artifacts": self.artifacts,
            "metadata": self.metadata,
            "duration": self.duration,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentRun":
        """Create from dictionary."""
        start_time = datetime.fromisoformat(data["start_time"])
        end_time = (
            datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None
        )

        run = cls(
            run_id=data["run_id"],
            experiment_name=data["experiment_name"],
            parameters=data["parameters"],
            start_time=start_time,
            end_time=end_time,
            status=data.get("status", "completed"),
            git_commit=data.get("git_commit"),
            git_status=data.get("git_status", {}),
            environment_info=data.get("environment_info", {}),
            random_seed=data.get("random_seed"),
        )

        run.metrics = data.get("metrics", {})
        run.results = data.get("results", {})
        run.artifacts = data.get("artifacts", {})
        run.metadata = data.get("metadata", {})

        return run


class ExperimentData:
    """Unified container for experiment data - alias for ExperimentRun for compatibility."""

    def __init__(self, run: ExperimentRun):
        """Initialize with ExperimentRun."""
        self._run = run

    def __getattr__(self, name):
        """Delegate attribute access to the underlying run."""
        return getattr(self._run, name)

    @classmethod
    def create(
        cls,
        run_id: str,
        experiment_name: str,
        parameters: Dict[str, Any],
        start_time: datetime,
        **kwargs,
    ) -> "ExperimentData":
        """Create new experiment data."""
        run = ExperimentRun(
            run_id=run_id,
            experiment_name=experiment_name,
            parameters=parameters,
            start_time=start_time,
            **kwargs,
        )
        return cls(run)


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
        import os
        import platform
        import sys

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
