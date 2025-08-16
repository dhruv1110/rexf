"""File-system based artifact storage and management."""

import hashlib
import json
import pickle
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ..core.interfaces import ArtifactManagerInterface

# Optional dependencies with fallbacks
try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    pd = None
    HAS_PANDAS = False

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    plt = None
    HAS_MATPLOTLIB = False


class FileSystemArtifactManager(ArtifactManagerInterface):
    """File-system based artifact manager."""

    def __init__(self, base_path: Union[str, Path] = "artifacts"):
        """
        Initialize artifact manager.

        Args:
            base_path: Base directory for storing artifacts
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_run_dir(self, run_id: str) -> Path:
        """Get the directory for a specific run."""
        run_dir = self.base_path / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _get_artifact_path(
        self, run_id: str, artifact_name: str, extension: str = ""
    ) -> Path:
        """Get the full path for an artifact."""
        run_dir = self._get_run_dir(run_id)
        filename = f"{artifact_name}{extension}"
        return run_dir / filename

    def _save_metadata(
        self, run_id: str, artifact_name: str, metadata: Dict[str, Any]
    ) -> None:
        """Save metadata for an artifact."""
        run_dir = self._get_run_dir(run_id)
        meta_path = run_dir / f"{artifact_name}.meta.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

    def _detect_file_type(self, data: Any) -> Tuple[str, str]:
        """Detect file type and appropriate extension."""
        if isinstance(data, str):
            # Check if it's a file path
            if Path(data).exists():
                return "file", Path(data).suffix
            else:
                return "text", ".txt"
        elif isinstance(data, bytes):
            return "binary", ".bin"
        elif HAS_NUMPY and isinstance(data, np.ndarray):
            return "numpy", ".npy"
        elif HAS_PANDAS and isinstance(data, pd.DataFrame):
            return "dataframe", ".csv"
        elif HAS_MATPLOTLIB and hasattr(data, "savefig"):
            return "matplotlib", ".png"
        elif isinstance(data, (dict, list)):
            return "json", ".json"
        else:
            return "pickle", ".pkl"

    def store_artifact(
        self,
        run_id: str,
        artifact_name: str,
        data: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store an artifact and return its path."""
        file_type, extension = self._detect_file_type(data)
        artifact_path = self._get_artifact_path(run_id, artifact_name, extension)

        # Prepare metadata
        artifact_metadata = {
            "name": artifact_name,
            "type": file_type,
            "run_id": run_id,
            "path": str(artifact_path),
            "size_bytes": 0,
            "hash": "",
            **(metadata or {}),
        }

        # Store the artifact based on its type
        if file_type == "file":
            # Copy existing file
            shutil.copy2(data, artifact_path)
        elif file_type == "text":
            with open(artifact_path, "w") as f:
                f.write(str(data))
        elif file_type == "binary":
            with open(artifact_path, "wb") as f:
                f.write(data)
        elif file_type == "numpy" and HAS_NUMPY:
            np.save(artifact_path, data)
        elif file_type == "dataframe" and HAS_PANDAS:
            data.to_csv(artifact_path, index=False)
        elif file_type == "matplotlib" and HAS_MATPLOTLIB:
            data.savefig(artifact_path, dpi=300, bbox_inches="tight")
        elif file_type == "json":
            with open(artifact_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        else:
            # Default to pickle
            with open(artifact_path, "wb") as f:
                pickle.dump(data, f)

        # Update metadata with file info
        if artifact_path.exists():
            artifact_metadata["size_bytes"] = artifact_path.stat().st_size
            artifact_metadata["hash"] = self._calculate_hash(artifact_path)

        # Save metadata
        self._save_metadata(run_id, artifact_name, artifact_metadata)

        return str(artifact_path)

    def load_artifact(self, run_id: str, artifact_name: str) -> Any:
        """Load an artifact by name."""
        # Try to find the artifact with any extension
        run_dir = self._get_run_dir(run_id)

        # Look for metadata first to get the exact filename
        meta_path = run_dir / f"{artifact_name}.meta.json"
        if meta_path.exists():
            with open(meta_path, "r") as f:
                metadata = json.load(f)
            artifact_path = Path(metadata["path"])
            file_type = metadata["type"]
        else:
            # Fall back to searching for the file
            matching_files = list(run_dir.glob(f"{artifact_name}.*"))
            if not matching_files:
                raise FileNotFoundError(
                    f"Artifact '{artifact_name}' not found for run {run_id}"
                )

            artifact_path = matching_files[0]
            file_type, _ = self._detect_file_type(str(artifact_path))

        if not artifact_path.exists():
            raise FileNotFoundError(f"Artifact file not found: {artifact_path}")

        # Load based on file type
        if file_type == "text":
            with open(artifact_path, "r") as f:
                return f.read()
        elif file_type == "binary":
            with open(artifact_path, "rb") as f:
                return f.read()
        elif file_type == "numpy" and HAS_NUMPY:
            return np.load(artifact_path)
        elif file_type == "dataframe" and HAS_PANDAS:
            return pd.read_csv(artifact_path)
        elif file_type == "json":
            with open(artifact_path, "r") as f:
                return json.load(f)
        elif file_type == "pickle":
            with open(artifact_path, "rb") as f:
                return pickle.load(f)
        else:
            # Return path for files we can't load directly
            return str(artifact_path)

    def get_artifact_path(self, run_id: str, artifact_name: str) -> Optional[Path]:
        """Get the file path for an artifact."""
        run_dir = self._get_run_dir(run_id)

        # Check metadata first
        meta_path = run_dir / f"{artifact_name}.meta.json"
        if meta_path.exists():
            with open(meta_path, "r") as f:
                metadata = json.load(f)
            return Path(metadata["path"])

        # Fall back to searching
        matching_files = list(run_dir.glob(f"{artifact_name}.*"))
        if matching_files:
            return matching_files[0]

        return None

    def list_artifacts(self, run_id: str) -> List[str]:
        """List all artifact names for a run."""
        run_dir = self._get_run_dir(run_id)
        if not run_dir.exists():
            return []

        # Get artifacts from metadata files
        artifacts = []
        for meta_file in run_dir.glob("*.meta.json"):
            artifact_name = meta_file.stem.replace(".meta", "")
            artifacts.append(artifact_name)

        # Also check for files without metadata
        for file_path in run_dir.iterdir():
            if file_path.is_file() and not file_path.name.endswith(".meta.json"):
                artifact_name = file_path.stem
                if artifact_name not in artifacts:
                    artifacts.append(artifact_name)

        return sorted(artifacts)

    def delete_artifact(self, run_id: str, artifact_name: str) -> bool:
        """Delete an artifact. Returns True if deleted."""
        artifact_path = self.get_artifact_path(run_id, artifact_name)
        if artifact_path and artifact_path.exists():
            try:
                artifact_path.unlink()

                # Also delete metadata if it exists
                run_dir = self._get_run_dir(run_id)
                meta_path = run_dir / f"{artifact_name}.meta.json"
                if meta_path.exists():
                    meta_path.unlink()

                return True
            except Exception:
                pass
        return False

    def _calculate_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of a file."""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception:
            return ""
