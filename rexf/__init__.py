"""rexf - Reproducible Experiments Framework.

A lightweight Python library for reproducible computational experiments
with a clean, decorator-based API and pluggable architecture.
"""

# Default backend implementations
from .backends.filesystem_artifacts import FileSystemArtifactManager
from .backends.sqlite_storage import SQLiteStorage

# Primary API - Clean decorator approach
from .core.decorators import ExperimentBuilder, configure_experiment, experiment_config

# Core interfaces for plugin development
from .core.interfaces import (
    ArtifactManagerInterface,
    ExportInterface,
    StorageInterface,
    VisualizationInterface,
)

# Data models
from .core.models import ExperimentData, ExperimentRun
from .core.runner import ExperimentRunner

__version__ = "0.1.0"

# Primary API - recommended usage
__all__ = [
    # Main API
    "experiment_config",
    "ExperimentRunner",
    # Alternative approaches
    "ExperimentBuilder",
    "configure_experiment",
    # Core interfaces for extensibility
    "StorageInterface",
    "ArtifactManagerInterface",
    "ExportInterface",
    "VisualizationInterface",
    # Data models
    "ExperimentData",
    "ExperimentRun",
    # Default implementations
    "SQLiteStorage",
    "FileSystemArtifactManager",
]

# Optional plugins that may not be available
try:
    from .plugins.export import ExperimentExporter, JSONExporter, YAMLExporter  # noqa: F401
    __all__.extend(["JSONExporter", "YAMLExporter", "ExperimentExporter"])
except ImportError:
    pass

try:
    from .plugins.visualization import ExperimentVisualizer  # noqa: F401
    __all__.append("ExperimentVisualizer")
except ImportError:
    pass
