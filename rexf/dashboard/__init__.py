"""Web dashboard for experiment visualization."""

try:
    from .app import create_dashboard_app, run_dashboard

    __all__ = ["create_dashboard_app", "run_dashboard"]
except ImportError as e:

    def create_dashboard_app(*args, **kwargs):
        raise ImportError(f"Dashboard requires additional dependencies: {e}")

    def run_dashboard(*args, **kwargs):
        raise ImportError(f"Dashboard requires additional dependencies: {e}")

    __all__ = ["create_dashboard_app", "run_dashboard"]
