"""rexf - Reproducible Experiments Framework.

A lightweight Python library for reproducible computational experiments
with an ultra-simple, intelligent API.

Usage:
    from rexf import experiment, run

    @experiment
    def my_experiment(param1, param2):
        result = do_something(param1, param2)
        return {"score": result}

    # Run single experiment
    run.single(my_experiment, param1=1.0, param2=2.0)

    # Get insights
    print(run.insights())

    # Compare best experiments
    run.compare(run.best())
"""

from . import run

# Legacy API (for backward compatibility)
from .backends.filesystem_artifacts import FileSystemArtifactManager
from .backends.sqlite_storage import SQLiteStorage
from .core.decorators import ExperimentBuilder, configure_experiment, experiment_config
from .core.interfaces import (
    ArtifactManagerInterface,
    ExportInterface,
    StorageInterface,
    VisualizationInterface,
)
from .core.models import ExperimentData, ExperimentRun
from .core.runner import ExperimentRunner

# NEW SIMPLE API (Primary - what users should use)
from .core.simple_api import experiment

__version__ = "0.1.0"

# Primary API - what users should use
__all__ = [
    # 🎯 SIMPLE API (recommended for all users)
    "experiment",  # Ultra-simple decorator
    "run",  # All experiment management functions
    # 🔧 LEGACY API (for backward compatibility)
    "experiment_config",
    "ExperimentRunner",
    "ExperimentBuilder",
    "configure_experiment",
    "StorageInterface",
    "ArtifactManagerInterface",
    "ExportInterface",
    "VisualizationInterface",
    "ExperimentData",
    "ExperimentRun",
    "SQLiteStorage",
    "FileSystemArtifactManager",
]

# Optional plugins that may not be available
try:
    from .plugins.export import (  # noqa: F401
        ExperimentExporter,
        JSONExporter,
        YAMLExporter,
    )

    __all__.extend(["JSONExporter", "YAMLExporter", "ExperimentExporter"])
except ImportError:
    pass

try:
    from .plugins.visualization import ExperimentVisualizer  # noqa: F401

    __all__.append("ExperimentVisualizer")
except ImportError:
    pass
