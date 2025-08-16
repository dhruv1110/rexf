"""Web dashboard for experiment visualization."""

try:
    from .app import create_dashboard_app, run_dashboard

    __all__ = ["create_dashboard_app", "run_dashboard"]
except ImportError as e:
    # Store the error message to use in stub functions
    _import_error_msg = str(e)

    def create_dashboard_app(*args, **kwargs):
        raise ImportError(
            f"Dashboard requires additional dependencies: {_import_error_msg}"
        )

    def run_dashboard(*args, **kwargs):
        raise ImportError(
            f"Dashboard requires additional dependencies: {_import_error_msg}"
        )

    __all__ = ["create_dashboard_app", "run_dashboard"]
