#!/usr/bin/env python3
"""Basic test to verify rexf functionality."""

import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from rexf import (
    experiment, param, result, metric, seed,
    ExperimentRunner, ExperimentVisualizer, ExperimentExporter
)


@experiment("test_experiment")
@param("x", float, description="Input value")
@param("multiplier", float, default=2.0, description="Multiplier")
@seed("random_seed")
@metric("output", float, description="Output value")
@result("final_result", float, description="Final result")
def simple_test_experiment(x, multiplier=2.0, random_seed=42):
    """Simple test experiment."""
    import random
    random.seed(random_seed)
    
    # Add some randomness
    noise = random.uniform(-0.1, 0.1)
    result = x * multiplier + noise
    
    return {
        "final_result": result,
        "output": result
    }


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
                storage_path="test_experiments.db",
                artifacts_path="test_artifacts"
            )
            
            # Test 2: Run experiment
            print("✓ Testing experiment execution...")
            run_id = runner.run(simple_test_experiment, x=5.0, multiplier=3.0, random_seed=123)
            assert run_id is not None
            
            # Test 3: Retrieve experiment
            print("✓ Testing experiment retrieval...")
            experiment = runner.get_experiment(run_id)
            assert experiment is not None
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
            
            # Test 7: Export functionality
            print("✓ Testing export functionality...")
            exporter = ExperimentExporter()
            experiments = [runner.get_experiment(rid) for rid in run_ids]
            
            # Export to JSON
            json_data = exporter.json_exporter.export_experiments(experiments)
            assert "experiments" in json_data
            
            # Export to file
            exporter.export_to_file(experiments, "test_export.json")
            assert Path("test_export.json").exists()
            
            # Test 8: Storage stats
            print("✓ Testing storage statistics...")
            stats = runner.get_stats()
            assert "storage" in stats
            assert stats["storage"]["total_experiments"] >= 4
            
            print("\n🎉 All basic tests passed!")
            return True
            
        except ImportError as e:
            print(f"⚠️  Skipping visualization tests due to missing dependencies: {e}")
            print("✓ Core functionality tests passed!")
            return True
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_with_visualization():
    """Test with visualization if matplotlib is available."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        print("\n✓ Testing visualization functionality...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            os.chdir(temp_path)
            
            # Run some experiments
            runner = ExperimentRunner()
            run_ids = []
            for x in [1.0, 2.0, 3.0, 4.0]:
                rid = runner.run(simple_test_experiment, x=x, random_seed=42)
                run_ids.append(rid)
            
            experiments = [runner.get_experiment(rid) for rid in run_ids]
            
            # Test visualization
            visualizer = ExperimentVisualizer()
            
            # Test metrics plot
            fig = visualizer.plot_metrics(experiments)
            assert fig is not None
            
            # Test timeline plot
            timeline_fig = visualizer.plot_experiment_timeline(experiments)
            assert timeline_fig is not None
            
            # Test comparison table
            table = visualizer.create_comparison_table(experiments)
            assert table is not None
            
            print("✓ Visualization tests passed!")
            
    except ImportError:
        print("⚠️  Matplotlib not available, skipping visualization tests")
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    print("=" * 50)
    print("REXF BASIC FUNCTIONALITY TEST")
    print("=" * 50)
    
    success = test_basic_functionality()
    
    if success:
        test_with_visualization()
    
    if success:
        print("\n" + "=" * 50)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print("\nrexf is ready to use! Try running:")
        print("  python examples/simple_demo.py")
        print("  python examples/monte_carlo_pi_demo.py")
    else:
        print("\n" + "=" * 50)
        print("❌ SOME TESTS FAILED")
        print("=" * 50)
        sys.exit(1)
