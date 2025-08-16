#!/usr/bin/env python3
"""Quick test of the new simple API."""

import tempfile
import os
from pathlib import Path

# Test the new simple API
from rexf import experiment, run


@experiment
def simple_test(x, y=2.0):
    """Simple test experiment."""
    result = x * y + 0.1
    return {"score": result, "product": x * y}


@experiment("named_experiment")
def named_test(learning_rate, epochs=10):
    """Named experiment test."""
    accuracy = 0.5 + learning_rate * epochs * 0.01
    return {"accuracy": accuracy, "final_lr": learning_rate}


@experiment(optimize_for="accuracy")
def optimized_test(param1, param2=1.0):
    """Experiment with optimization target."""
    accuracy = param1 + param2 * 0.1
    loss = 2.0 - accuracy
    return {"accuracy": accuracy, "loss": loss}


def test_simple_api():
    """Test the simple API functionality."""
    print("🧪 Testing Simple API")
    print("=" * 40)
    
    # Change to temp directory
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Test 1: Simple experiment
            print("\n1️⃣ Testing simple experiment...")
            run_id1 = run.single(simple_test, x=3.0, y=4.0)
            print(f"   Run ID: {run_id1[:8]}...")
            
            # Test 2: Named experiment
            print("\n2️⃣ Testing named experiment...")
            run_id2 = run.single(named_test, learning_rate=0.01, epochs=20)
            print(f"   Run ID: {run_id2[:8]}...")
            
            # Test 3: Optimized experiment
            print("\n3️⃣ Testing optimized experiment...")
            run_id3 = run.single(optimized_test, param1=0.8, param2=2.0)
            print(f"   Run ID: {run_id3[:8]}...")
            
            # Test 4: Get best experiments
            print("\n4️⃣ Testing best experiments...")
            best_experiments = run.best(top=3)
            print(f"   Found {len(best_experiments)} experiments")
            
            # Test 5: Get insights
            print("\n5️⃣ Testing insights...")
            insights = run.insights()
            print(f"   Insights: {insights}")
            
            # Test 6: Compare experiments
            print("\n6️⃣ Testing comparison...")
            run.compare()
            
            # Test 7: Failed experiments
            print("\n7️⃣ Testing failed experiments...")
            failed_exps = run.failed()
            print(f"   Failed experiments: {len(failed_exps)}")
            
            # Test 8: Recent experiments
            print("\n8️⃣ Testing recent experiments...")
            recent_exps = run.recent(hours=1)
            print(f"   Recent experiments: {len(recent_exps)}")
            
            print("\n✅ All tests passed!")
            
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            os.chdir(original_cwd)


if __name__ == "__main__":
    test_simple_api()
