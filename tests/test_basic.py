"""Basic tests for rexf functionality."""

import os
import tempfile
from pathlib import Path

from rexf import ExperimentRunner, experiment_config


def windows_safe_cleanup(runner, temp_dir):
    """Windows-specific cleanup that handles file locking issues."""
    import gc
    import platform
    import time

    try:
        if runner:
            runner.close()

        # Force garbage collection multiple times
        for _ in range(3):
            gc.collect()
            time.sleep(0.1)

        # On Windows, try to manually close any remaining file handles
        if platform.system() == "Windows":
            temp_path = Path(temp_dir)
            for db_file in temp_path.glob("*.db"):
                try:
                    if db_file.exists():
                        for attempt in range(5):
                            try:
                                db_file.unlink()
                                break
                            except PermissionError:
                                time.sleep(0.2)
                                gc.collect()
                except Exception:
                    pass

            # Additional delay for Windows
            time.sleep(1.0)
        else:
            time.sleep(0.2)

    except Exception:
        pass


@experiment_config(
    name="test_experiment",
    params={
        "x": (float, "Input value x"),
        "multiplier": (float, 2.0, "Multiplier value"),
    },
    metrics={
        "output": (float, "Final computed output"),
    },
    results={
        "final_result": (float, "The final result"),
    },
    seed="random_seed",
)
def simple_test_experiment(x, multiplier=2.0, random_seed=42):
    """Simple test experiment."""
    import random

    random.seed(random_seed)

    # Add some randomness
    noise = random.uniform(-0.1, 0.1)
    result = x * multiplier + noise

    return {"final_result": result, "output": result}


def test_basic_functionality():
    """Test basic rexf functionality."""
    print("Testing basic rexf functionality...")

    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        os.chdir(temp_path)

        try:
            # Test 1: Initialize runner
            print("✓ Testing ExperimentRunner initialization...")
            runner = ExperimentRunner(
                storage_path="test_experiments.db", artifacts_path="test_artifacts"
            )

            # Test 2: Run experiment
            print("✓ Testing experiment execution...")
            run_id = runner.run(
                simple_test_experiment, x=5.0, multiplier=3.0, random_seed=123
            )
            assert run_id is not None

            # Test 3: Retrieve experiment
            print("✓ Testing experiment retrieval...")
            experiment = runner.get_experiment(run_id)
            assert experiment is not None
            assert experiment.experiment_name == "test_experiment"
            assert experiment.parameters["x"] == 5.0
            assert experiment.parameters["multiplier"] == 3.0
            assert "output" in experiment.metrics
            assert "final_result" in experiment.results

            # Test 4: Run multiple experiments
            print("✓ Testing multiple experiments...")
            run_ids = []
            for x in [1.0, 2.0, 3.0]:
                rid = runner.run(simple_test_experiment, x=x, random_seed=42)
                run_ids.append(rid)

            # Test 5: Compare experiments
            print("✓ Testing experiment comparison...")
            comparison = runner.compare_runs(run_ids)
            assert "parameter_comparison" in comparison
            assert "metric_comparison" in comparison

            # Test 6: List experiments
            print("✓ Testing experiment listing...")
            all_experiments = runner.list_experiments()
            assert len(all_experiments) >= 4  # 1 + 3 experiments

            # Test 7: Storage stats
            print("✓ Testing storage statistics...")
            stats = runner.get_stats()
            assert "storage" in stats
            assert stats["storage"]["total_experiments"] >= 4

            print("\n🎉 All basic tests passed!")

        except ImportError as e:
            print(f"⚠️  Skipping tests due to missing dependencies: {e}")
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback

            traceback.print_exc()
            raise  # Re-raise the exception to fail the test
        finally:
            # Clean up resources before temp directory deletion (Windows compatibility)
            windows_safe_cleanup(runner if "runner" in locals() else None, temp_dir)


def test_with_plugins():
    """Test optional plugin functionality."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # Use non-interactive backend

        print("\n✓ Testing plugin functionality...")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            os.chdir(temp_path)

            # Run some experiments
            runner = ExperimentRunner()
            run_ids = []
            for x in [1.0, 2.0, 3.0]:
                rid = runner.run(simple_test_experiment, x=x, random_seed=42)
                run_ids.append(rid)

            experiments = [runner.get_experiment(rid) for rid in run_ids]

            # Test export functionality if available
            try:
                from rexf.plugins.export import ExperimentExporter

                exporter = ExperimentExporter()
                json_data = exporter.export_to_json(experiments)
                assert "experiments" in json_data or "run_id" in json_data
                print("✓ Export functionality working")

            except ImportError:
                print("⚠️  Export plugin not available")

            # Test visualization functionality if available
            try:
                from rexf.plugins.visualization import ExperimentVisualizer

                visualizer = ExperimentVisualizer()

                # Test metrics plot
                fig = visualizer.plot_metrics(experiments)
                assert fig is not None

                # Test timeline plot
                timeline_fig = visualizer.plot_experiment_timeline(experiments)
                assert timeline_fig is not None

                print("✓ Visualization functionality working")

            except ImportError:
                print("⚠️  Visualization plugin not available")

    except ImportError:
        print("⚠️  Matplotlib not available, skipping plugin tests")
    finally:
        # Clean up resources before temp directory deletion (Windows compatibility)
        windows_safe_cleanup(runner if "runner" in locals() else None, temp_dir)


def test_pluggable_backends():
    """Test that backends are pluggable."""
    print("\n✓ Testing pluggable backend architecture...")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        os.chdir(temp_path)

        try:
            # Test with explicit backend specification
            from rexf.backends.filesystem_artifacts import FileSystemArtifactManager
            from rexf.backends.sqlite_storage import SQLiteStorage

            storage = SQLiteStorage("custom_experiments.db")
            artifacts = FileSystemArtifactManager("custom_artifacts")

            runner = ExperimentRunner(storage=storage, artifact_manager=artifacts)

            # Run a test experiment
            run_id = runner.run(simple_test_experiment, x=10.0, random_seed=456)
            experiment = runner.get_experiment(run_id)

            assert experiment is not None
            assert experiment.parameters["x"] == 10.0

            print("✓ Custom backend configuration working")

        except Exception as e:
            print(f"❌ Backend test failed: {e}")
            raise
        finally:
            windows_safe_cleanup(runner if "runner" in locals() else None, temp_dir)


if __name__ == "__main__":
    print("REXF CLEAN API FUNCTIONALITY TEST")
    print("=" * 50)

    try:
        test_basic_functionality()
        test_with_plugins()
        test_pluggable_backends()

        print("\n" + "=" * 50)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
    except Exception as e:
        print(f"\n❌ TESTS FAILED: {e}")
        raise
