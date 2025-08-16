"""Simple demo showing basic rexf usage.

This is a minimal example showing how to use rexf decorators
and run experiments.
"""

import sys
from pathlib import Path

# Add parent directory to path to import rexf
sys.path.insert(0, str(Path(__file__).parent.parent))

from rexf import experiment, param, result, metric, seed, ExperimentRunner


@experiment("simple_math")
@param("a", float, description="First number")
@param("b", float, description="Second number")
@param("operation", str, default="add", description="Operation to perform")
@seed("random_seed")
@metric("result_value", float, description="Result of the operation")
@result("output", float, description="Final output")
def simple_math_experiment(a, b, operation="add", random_seed=42):
    """A simple mathematical experiment."""
    print(f"Performing {operation} on {a} and {b}")
    
    if operation == "add":
        result = a + b
    elif operation == "multiply":
        result = a * b
    elif operation == "divide":
        result = a / b if b != 0 else float('inf')
    else:
        result = 0
    
    return {
        "output": result,
        "result_value": result
    }


def main():
    """Run simple demo."""
    print("Simple rexf Demo")
    print("================")
    
    # Initialize runner
    runner = ExperimentRunner(
        storage_path="simple_demo.db",
        artifacts_path="simple_artifacts"
    )
    
    # Run experiments
    experiments = [
        {"a": 5.0, "b": 3.0, "operation": "add"},
        {"a": 5.0, "b": 3.0, "operation": "multiply"},
        {"a": 10.0, "b": 2.0, "operation": "divide"},
    ]
    
    run_ids = []
    for config in experiments:
        print(f"\nRunning experiment: {config}")
        run_id = runner.run(simple_math_experiment, **config)
        run_ids.append(run_id)
    
    # Show results
    print("\nResults:")
    for run_id in run_ids:
        exp = runner.get_experiment(run_id)
        if exp:
            print(f"  {exp.parameters} -> {exp.results}")
    
    print("\nDemo completed!")


if __name__ == "__main__":
    main()
