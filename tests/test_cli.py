"""Test CLI analytics functionality."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from rexf import run


class TestCLIAnalytics:
    """Test CLI analytics tool functionality."""

    def test_cli_help(self):
        """Test CLI help functionality."""
        result = subprocess.run(
            [sys.executable, "-m", "rexf.cli.analytics", "--help"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "rexf Analytics CLI" in result.stdout
        assert "--summary" in result.stdout
        assert "--query" in result.stdout

    def test_cli_with_no_database(self):
        """Test CLI behavior with non-existent database."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "rexf.cli.analytics",
                "--database",
                "nonexistent.db",
                "--summary",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1
        assert "not found" in result.stdout

    def test_cli_summary_with_data(self, unique_db_path, sample_ml_experiment):
        """Test CLI summary with actual experiment data."""
        from rexf.run import ExperimentRunner

        # Create some test data
        runner = ExperimentRunner(storage_path=unique_db_path, intelligent=True)
        old_runner = run._default_runner
        run._default_runner = runner

        try:
            # Run some experiments
            run.single(sample_ml_experiment, learning_rate=0.001)
            run.single(sample_ml_experiment, learning_rate=0.01)
            run.single(sample_ml_experiment, learning_rate=0.1)

            # Test CLI summary
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "rexf.cli.analytics",
                    "--database",
                    unique_db_path,
                    "--summary",
                ],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0
            assert "EXPERIMENT SUMMARY" in result.stdout
            assert "Total experiments:" in result.stdout

        finally:
            run._default_runner = old_runner
            runner.close()

    def test_cli_list_experiments(self, unique_db_path, sample_math_experiment):
        """Test CLI list functionality."""
        from rexf.run import ExperimentRunner

        runner = ExperimentRunner(storage_path=unique_db_path, intelligent=True)
        old_runner = run._default_runner
        run._default_runner = runner

        try:
            # Run some experiments
            run.single(sample_math_experiment, x=1.0, y=2.0)
            run.single(sample_math_experiment, x=3.0, y=4.0)

            # Test CLI list
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "rexf.cli.analytics",
                    "--database",
                    unique_db_path,
                    "--list",
                ],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0
            assert "EXPERIMENTS" in result.stdout
            assert "sample_math_experiment" in result.stdout

        finally:
            run._default_runner = old_runner
            runner.close()

    def test_cli_json_output(self, unique_db_path, sample_math_experiment):
        """Test CLI JSON output format."""
        from rexf.run import ExperimentRunner

        runner = ExperimentRunner(storage_path=unique_db_path, intelligent=True)
        old_runner = run._default_runner
        run._default_runner = runner

        try:
            # Run an experiment
            run.single(sample_math_experiment, x=5.0, y=6.0)

            # Test JSON format
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "rexf.cli.analytics",
                    "--database",
                    unique_db_path,
                    "--list",
                    "--format",
                    "json",
                ],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0

            # Parse JSON output
            output_data = json.loads(result.stdout)
            assert "operation" in output_data
            assert "experiments" in output_data
            assert len(output_data["experiments"]) == 1

        finally:
            run._default_runner = old_runner
            runner.close()

    def test_cli_csv_output(self, unique_db_path, sample_math_experiment):
        """Test CLI CSV output format."""
        from rexf.run import ExperimentRunner

        runner = ExperimentRunner(storage_path=unique_db_path, intelligent=True)
        old_runner = run._default_runner
        run._default_runner = runner

        try:
            # Run an experiment
            run.single(sample_math_experiment, x=2.0, y=3.0)

            # Test CSV format
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "rexf.cli.analytics",
                    "--database",
                    unique_db_path,
                    "--list",
                    "--format",
                    "csv",
                ],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0
            assert (
                "run_id,experiment_name,status,duration,parameters,metrics"
                in result.stdout
            )
            assert "sample_math_experiment" in result.stdout

        finally:
            run._default_runner = old_runner
            runner.close()

    @pytest.mark.slow
    def test_cli_insights(self, unique_db_path, sample_ml_experiment):
        """Test CLI insights generation."""
        from rexf.run import ExperimentRunner

        runner = ExperimentRunner(storage_path=unique_db_path, intelligent=True)
        old_runner = run._default_runner
        run._default_runner = runner

        try:
            if not runner._intelligent:
                pytest.skip("Intelligent features not available")

            # Run multiple experiments for insights
            run.single(sample_ml_experiment, learning_rate=0.001, batch_size=32)
            run.single(sample_ml_experiment, learning_rate=0.01, batch_size=64)
            run.single(sample_ml_experiment, learning_rate=0.1, batch_size=32)

            # Test insights
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "rexf.cli.analytics",
                    "--database",
                    unique_db_path,
                    "--insights",
                ],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0
            assert "EXPERIMENT INSIGHTS" in result.stdout

        finally:
            run._default_runner = old_runner
            runner.close()

    def test_cli_filtering(
        self, unique_db_path, sample_ml_experiment, sample_math_experiment
    ):
        """Test CLI filtering options."""
        from rexf.run import ExperimentRunner

        runner = ExperimentRunner(storage_path=unique_db_path, intelligent=True)
        old_runner = run._default_runner
        run._default_runner = runner

        try:
            # Run different types of experiments
            run.single(sample_ml_experiment, learning_rate=0.001)
            run.single(sample_math_experiment, x=1.0)

            # Test filtering by experiment name
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "rexf.cli.analytics",
                    "--database",
                    unique_db_path,
                    "--list",
                    "--experiment-name",
                    "_sample_ml_experiment",
                ],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0
            assert "_sample_ml_experiment" in result.stdout
            assert "sample_math_experiment" not in result.stdout

        finally:
            run._default_runner = old_runner
            runner.close()

    def test_cli_output_file(self, unique_db_path, sample_math_experiment, temp_dir):
        """Test CLI output to file."""
        from rexf.run import ExperimentRunner

        runner = ExperimentRunner(storage_path=unique_db_path, intelligent=True)
        old_runner = run._default_runner
        run._default_runner = runner

        try:
            # Run an experiment
            run.single(sample_math_experiment, x=1.0, y=2.0)

            # Test output to file
            output_file = temp_dir / "test_output.json"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "rexf.cli.analytics",
                    "--database",
                    unique_db_path,
                    "--summary",
                    "--output",
                    str(output_file),
                ],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0
            assert output_file.exists()

            # Verify file contents
            with open(output_file) as f:
                data = json.load(f)
            assert "operation" in data
            assert data["operation"] == "summary"

        finally:
            run._default_runner = old_runner
            runner.close()


@pytest.mark.integration
class TestCLIIntegration:
    """Test CLI integration with the full system."""

    def test_cli_query_integration(self, unique_db_path, sample_ml_experiment):
        """Test CLI query functionality integration."""
        from rexf.run import ExperimentRunner

        runner = ExperimentRunner(storage_path=unique_db_path, intelligent=True)
        old_runner = run._default_runner
        run._default_runner = runner

        try:
            if not runner._intelligent:
                pytest.skip("Intelligent features not available")

            # Run experiments with known values
            run.single(
                sample_ml_experiment, learning_rate=0.001, epochs=5
            )  # High accuracy
            run.single(
                sample_ml_experiment, learning_rate=0.1, epochs=5
            )  # Low accuracy

            # Test query via CLI
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "rexf.cli.analytics",
                    "--database",
                    unique_db_path,
                    "--query",
                    "accuracy > 0.8",
                ],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0
            assert "Query: accuracy > 0.8" in result.stdout

        finally:
            run._default_runner = old_runner
            runner.close()

    def test_cli_compare_integration(
        self, unique_db_path, sample_optimization_experiment
    ):
        """Test CLI compare functionality."""
        from rexf.run import ExperimentRunner

        runner = ExperimentRunner(storage_path=unique_db_path, intelligent=True)
        old_runner = run._default_runner
        run._default_runner = runner

        try:
            if not runner._intelligent:
                pytest.skip("Intelligent features not available")

            # Run multiple experiments
            run.single(sample_optimization_experiment, algorithm="gd", step_size=0.01)
            run.single(sample_optimization_experiment, algorithm="adam", step_size=0.1)
            run.single(
                sample_optimization_experiment, algorithm="newton", step_size=0.05
            )

            # Test compare via CLI
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "rexf.cli.analytics",
                    "--database",
                    unique_db_path,
                    "--compare",
                    "--best",
                    "3",
                ],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0
            assert "EXPERIMENT COMPARISON" in result.stdout

        finally:
            run._default_runner = old_runner
            runner.close()


if __name__ == "__main__":
    pytest.main([__file__])
