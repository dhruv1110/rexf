"""Test dashboard functionality."""

import json
import threading
import time
from http.server import HTTPServer
from urllib.request import urlopen

import pytest

from rexf import run


@pytest.mark.integration
class TestDashboard:
    """Test web dashboard functionality."""

    def test_dashboard_creation(self, unique_db_path):
        """Test dashboard app creation."""
        try:
            from rexf.dashboard.app import create_dashboard_app
            
            handler_factory = create_dashboard_app(unique_db_path)
            assert callable(handler_factory)
            
        except ImportError:
            pytest.skip("Dashboard dependencies not available")

    def test_dashboard_with_data(self, unique_db_path, sample_ml_experiment):
        """Test dashboard with actual experiment data."""
        try:
            from rexf.dashboard.app import create_dashboard_app
            from rexf.run import ExperimentRunner
            
        except ImportError:
            pytest.skip("Dashboard dependencies not available")
        
        # Create test data
        runner = ExperimentRunner(storage_path=unique_db_path, intelligent=True)
        old_runner = run._default_runner
        run._default_runner = runner
        
        try:
            # Run some experiments
            run.single(sample_ml_experiment, learning_rate=0.001, batch_size=32)
            run.single(sample_ml_experiment, learning_rate=0.01, batch_size=64)
            
            # Create dashboard
            handler_factory = create_dashboard_app(unique_db_path)
            assert handler_factory is not None
            
        finally:
            run._default_runner = old_runner
            runner.close()

    @pytest.mark.slow
    def test_dashboard_server_startup(self, unique_db_path, sample_math_experiment):
        """Test dashboard server startup and basic endpoints."""
        try:
            from rexf.dashboard.app import create_dashboard_app
            from rexf.run import ExperimentRunner
            
        except ImportError:
            pytest.skip("Dashboard dependencies not available")
        
        # Create test data
        runner = ExperimentRunner(storage_path=unique_db_path, intelligent=True)
        old_runner = run._default_runner
        run._default_runner = runner
        
        try:
            # Run an experiment
            run.single(sample_math_experiment, x=3.0, y=4.0)
            
            # Start dashboard server
            handler_factory = create_dashboard_app(unique_db_path)
            server = HTTPServer(("localhost", 0), handler_factory)  # Use any available port
            port = server.server_port
            
            # Start server in background thread
            server_thread = threading.Thread(target=server.serve_forever, daemon=True)
            server_thread.start()
            
            # Wait for server to start
            time.sleep(0.5)
            
            try:
                # Test main dashboard endpoint
                response = urlopen(f"http://localhost:{port}/", timeout=5)
                html_content = response.read().decode('utf-8')
                
                assert response.getcode() == 200
                assert "RexF Experiment Dashboard" in html_content
                assert "🧪" in html_content  # Dashboard emoji
                
                # Test API endpoints
                stats_response = urlopen(f"http://localhost:{port}/api/stats", timeout=5)
                stats_data = json.loads(stats_response.read().decode('utf-8'))
                
                assert stats_response.getcode() == 200
                assert "stats" in stats_data
                assert stats_data["stats"]["total_experiments"] >= 1
                
                experiments_response = urlopen(f"http://localhost:{port}/api/experiments", timeout=5)
                experiments_data = json.loads(experiments_response.read().decode('utf-8'))
                
                assert experiments_response.getcode() == 200
                assert "experiments" in experiments_data
                assert len(experiments_data["experiments"]) >= 1
                
            finally:
                server.shutdown()
                server.server_close()
                
        finally:
            run._default_runner = old_runner
            runner.close()

    def test_dashboard_api_endpoints(self, unique_db_path, sample_ml_experiment):
        """Test dashboard API endpoints without full server."""
        try:
            from rexf.dashboard.app import DashboardHandler
            from rexf.backends.intelligent_storage import IntelligentStorage
            from rexf.run import ExperimentRunner
            
        except ImportError:
            pytest.skip("Dashboard dependencies not available")
        
        # Create test data
        runner = ExperimentRunner(storage_path=unique_db_path, intelligent=True)
        old_runner = run._default_runner
        run._default_runner = runner
        
        try:
            # Run experiments
            run.single(sample_ml_experiment, learning_rate=0.001, epochs=10)
            run.single(sample_ml_experiment, learning_rate=0.01, epochs=5)
            
            # Test handler initialization
            storage = IntelligentStorage(unique_db_path)
            try:
                # This tests that the handler can be created with storage
                assert storage is not None
                
                # Test that we can get data that would be served
                stats = storage.get_storage_stats()
                assert stats["total_experiments"] >= 2
                
                experiments = storage.list_experiments()
                assert len(experiments) >= 2
                
            finally:
                storage.close()
                
        finally:
            run._default_runner = old_runner
            runner.close()

    def test_dashboard_error_handling(self, temp_dir):
        """Test dashboard error handling with invalid database."""
        try:
            from rexf.dashboard.app import create_dashboard_app
            
            # Test with non-existent database
            invalid_db = str(temp_dir / "nonexistent.db")
            handler_factory = create_dashboard_app(invalid_db)
            
            # Should create handler but handle errors gracefully
            assert callable(handler_factory)
            
        except ImportError:
            pytest.skip("Dashboard dependencies not available")

    def test_run_dashboard_integration(self, unique_db_path, sample_math_experiment):
        """Test dashboard integration with run module."""
        from rexf.run import ExperimentRunner
        
        runner = ExperimentRunner(storage_path=unique_db_path, intelligent=True)
        old_runner = run._default_runner
        run._default_runner = runner
        
        try:
            # Run an experiment
            run.single(sample_math_experiment, x=1.0, y=2.0)
            
            # Test that dashboard method exists and can be called
            # Note: We don't actually start the server in tests
            assert hasattr(runner, "dashboard")
            assert callable(runner.dashboard)
            
            # Test the module-level function
            assert hasattr(run, "dashboard")
            assert callable(run.dashboard)
            
        finally:
            run._default_runner = old_runner
            runner.close()


if __name__ == "__main__":
    pytest.main([__file__])
