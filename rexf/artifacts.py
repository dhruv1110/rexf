"""File-system based artifact storage and management.

This module provides artifact storage capabilities that work with large files
like model weights, plots, CSV outputs, etc.
"""

import os
import pickle
import shutil
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json

from .interfaces import ArtifactManagerInterface


class FileSystemArtifactManager(ArtifactManagerInterface):
    """File-system based artifact manager."""
    
    def __init__(self, base_path: Union[str, Path] = "artifacts"):
        """Initialize artifact manager.
        
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
    
    def _get_artifact_path(self, run_id: str, artifact_name: str, extension: str = "") -> Path:
        """Get the full path for an artifact."""
        run_dir = self._get_run_dir(run_id)
        filename = f"{artifact_name}{extension}"
        return run_dir / filename
    
    def _save_metadata(self, run_id: str, artifact_name: str, metadata: Dict[str, Any]) -> None:
        """Save metadata for an artifact."""
        run_dir = self._get_run_dir(run_id)
        meta_path = run_dir / f"{artifact_name}.meta.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    def _load_metadata(self, run_id: str, artifact_name: str) -> Dict[str, Any]:
        """Load metadata for an artifact."""
        run_dir = self._get_run_dir(run_id)
        meta_path = run_dir / f"{artifact_name}.meta.json"
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _detect_file_type(self, data: Any) -> tuple[str, str]:
        """Detect file type and extension based on data type."""
        try:
            import numpy as np
        except ImportError:
            np = None
        
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            plt = None
        
        if plt and hasattr(plt, 'Figure') and isinstance(data, plt.Figure):
            return "matplotlib_figure", ".png"
        elif np and hasattr(np, 'ndarray') and isinstance(data, np.ndarray):
            return "numpy_array", ".npy"
        elif isinstance(data, (dict, list)):
            return "json", ".json"
        elif isinstance(data, str) and data.endswith(('.csv', '.txt', '.log')):
            return "text_file", ""
        elif hasattr(data, 'to_csv'):  # pandas DataFrame
            return "pandas_dataframe", ".csv"
        else:
            return "pickle", ".pkl"
    
    def store_artifact(
        self, 
        run_id: str, 
        artifact_name: str, 
        data: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store an artifact and return its path."""
        file_type, extension = self._detect_file_type(data)
        artifact_path = self._get_artifact_path(run_id, artifact_name, extension)
        
        # Store the artifact based on its type
        try:
            if file_type == "matplotlib_figure":
                data.savefig(artifact_path, dpi=300, bbox_inches='tight')
            elif file_type == "numpy_array":
                try:
                    import numpy as np
                    np.save(artifact_path, data)
                except ImportError:
                    raise RuntimeError("NumPy is required to store numpy arrays")
            elif file_type == "json":
                with open(artifact_path, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
            elif file_type == "text_file":
                # If data is a file path, copy it
                if isinstance(data, (str, Path)) and Path(data).exists():
                    shutil.copy2(data, artifact_path.parent / Path(data).name)
                    artifact_path = artifact_path.parent / Path(data).name
                else:
                    # If data is text content, write it
                    with open(artifact_path, 'w') as f:
                        f.write(str(data))
            elif file_type == "pandas_dataframe":
                try:
                    data.to_csv(artifact_path, index=False)
                except Exception:
                    raise RuntimeError("Failed to save pandas DataFrame")
            else:  # pickle
                with open(artifact_path, 'wb') as f:
                    pickle.dump(data, f)
            
            # Calculate file hash for integrity checking
            file_hash = self._calculate_file_hash(artifact_path)
            
            # Save metadata
            artifact_metadata = {
                "name": artifact_name,
                "file_type": file_type,
                "file_path": str(artifact_path),
                "file_size": artifact_path.stat().st_size,
                "file_hash": file_hash,
                "custom_metadata": metadata or {}
            }
            self._save_metadata(run_id, artifact_name, artifact_metadata)
            
            return str(artifact_path)
            
        except Exception as e:
            raise RuntimeError(f"Failed to store artifact {artifact_name}: {e}")
    
    def load_artifact(self, run_id: str, artifact_name: str) -> Any:
        """Load an artifact by name."""
        metadata = self._load_metadata(run_id, artifact_name)
        if not metadata:
            raise FileNotFoundError(f"Artifact {artifact_name} not found for run {run_id}")
        
        artifact_path = Path(metadata["file_path"])
        if not artifact_path.exists():
            raise FileNotFoundError(f"Artifact file not found: {artifact_path}")
        
        # Verify file integrity
        current_hash = self._calculate_file_hash(artifact_path)
        if current_hash != metadata.get("file_hash"):
            raise ValueError(f"Artifact {artifact_name} has been corrupted")
        
        file_type = metadata["file_type"]
        
        try:
            if file_type == "numpy_array":
                try:
                    import numpy as np
                    return np.load(artifact_path)
                except ImportError:
                    raise RuntimeError("NumPy is required to load numpy arrays")
            elif file_type == "json":
                with open(artifact_path, 'r') as f:
                    return json.load(f)
            elif file_type == "text_file":
                with open(artifact_path, 'r') as f:
                    return f.read()
            elif file_type == "pandas_dataframe":
                try:
                    import pandas as pd
                    return pd.read_csv(artifact_path)
                except ImportError:
                    raise RuntimeError("Pandas is required to load pandas DataFrames")
            elif file_type == "matplotlib_figure":
                # Return the path since we can't recreate the figure object
                return str(artifact_path)
            else:  # pickle
                with open(artifact_path, 'rb') as f:
                    return pickle.load(f)
                    
        except Exception as e:
            raise RuntimeError(f"Failed to load artifact {artifact_name}: {e}")
    
    def get_artifact_path(self, run_id: str, artifact_name: str) -> Optional[Path]:
        """Get the file path for an artifact."""
        metadata = self._load_metadata(run_id, artifact_name)
        if metadata:
            return Path(metadata["file_path"])
        return None
    
    def list_artifacts(self, run_id: str) -> List[str]:
        """List all artifact names for a run."""
        run_dir = self._get_run_dir(run_id)
        if not run_dir.exists():
            return []
        
        artifacts = []
        for meta_file in run_dir.glob("*.meta.json"):
            artifact_name = meta_file.stem.replace(".meta", "")
            artifacts.append(artifact_name)
        
        return sorted(artifacts)
    
    def delete_artifact(self, run_id: str, artifact_name: str) -> bool:
        """Delete an artifact. Returns True if deleted."""
        metadata = self._load_metadata(run_id, artifact_name)
        if not metadata:
            return False
        
        artifact_path = Path(metadata["file_path"])
        meta_path = self._get_run_dir(run_id) / f"{artifact_name}.meta.json"
        
        deleted = False
        if artifact_path.exists():
            artifact_path.unlink()
            deleted = True
        
        if meta_path.exists():
            meta_path.unlink()
            deleted = True
        
        return deleted
    
    def get_artifact_info(self, run_id: str, artifact_name: str) -> Dict[str, Any]:
        """Get detailed information about an artifact."""
        metadata = self._load_metadata(run_id, artifact_name)
        if not metadata:
            raise FileNotFoundError(f"Artifact {artifact_name} not found for run {run_id}")
        
        artifact_path = Path(metadata["file_path"])
        
        info = metadata.copy()
        info["exists"] = artifact_path.exists()
        if artifact_path.exists():
            stat = artifact_path.stat()
            info["current_size"] = stat.st_size
            info["modified_time"] = stat.st_mtime
        
        return info
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def cleanup_run(self, run_id: str) -> bool:
        """Remove all artifacts for a run."""
        run_dir = self._get_run_dir(run_id)
        if run_dir.exists():
            shutil.rmtree(run_dir)
            return True
        return False
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        stats = {
            "total_runs": 0,
            "total_artifacts": 0,
            "total_size_bytes": 0,
            "artifact_types": {}
        }
        
        if not self.base_path.exists():
            return stats
        
        for run_dir in self.base_path.iterdir():
            if run_dir.is_dir():
                stats["total_runs"] += 1
                
                for meta_file in run_dir.glob("*.meta.json"):
                    stats["total_artifacts"] += 1
                    
                    try:
                        with open(meta_file, 'r') as f:
                            metadata = json.load(f)
                        
                        file_type = metadata.get("file_type", "unknown")
                        file_size = metadata.get("file_size", 0)
                        
                        stats["total_size_bytes"] += file_size
                        stats["artifact_types"][file_type] = stats["artifact_types"].get(file_type, 0) + 1
                        
                    except Exception:
                        continue
        
        return stats
