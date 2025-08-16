"""π Value Estimator - Monte Carlo Experiment Demo.

This demonstration shows how RexF can be used for computational research:
- Monte Carlo π estimation with varying parameters
- Automatic experiment tracking and analysis
- Performance optimization insights
- Statistical analysis and recommendations
- Interactive visualization of results
"""

import math
import random
import time

from rexf import experiment, run





@experiment
def pi_estimator_experiment(num_samples=10000, method="uniform", random_seed=None):
    """
    Estimate π using Monte Carlo methods.
    
    This is a classic computational experiment that estimates the value of π
    by randomly sampling points in a unit square and counting how many fall
    within the inscribed circle.
    
    Args:
        num_samples: Number of random points to sample
        method: Sampling method ('uniform', 'gaussian', 'stratified')
        random_seed: Random seed for reproducibility (None for random)
    """
    print(f"🎯 Estimating π with {num_samples:,} samples using {method} method")

    if random_seed:
        random.seed(random_seed)

    inside_circle = 0
    start_time = time.time()

    for _ in range(num_samples):
        if method == "uniform":
            x, y = random.uniform(-1, 1), random.uniform(-1, 1)
        elif method == "gaussian":
            # Use rejection sampling with Gaussian
            while True:
                x, y = random.gauss(0, 0.5), random.gauss(0, 0.5)
                if -1 <= x <= 1 and -1 <= y <= 1:
                    break
        else:  # stratified
            # Simple stratified sampling
            grid_size = int(math.sqrt(num_samples))
            i = _ % grid_size
            j = _ // grid_size
            if j >= grid_size:
                x, y = random.uniform(-1, 1), random.uniform(-1, 1)
            else:
                x = random.uniform(-1 + 2 * i / grid_size, -1 + 2 * (i + 1) / grid_size)
                y = random.uniform(-1 + 2 * j / grid_size, -1 + 2 * (j + 1) / grid_size)

        if x * x + y * y <= 1:
            inside_circle += 1

    computation_time = time.time() - start_time
    pi_estimate = 4 * inside_circle / num_samples
    error = abs(pi_estimate - math.pi)

    return {
        "pi_estimate": round(pi_estimate, 6),
        "absolute_error": round(error, 6),
        "relative_error_percent": round(error / math.pi * 100, 4),
        "accuracy_score": round(1 - (error / math.pi), 6),
        "samples_per_second": round(num_samples / computation_time, 0),
        "computation_time": round(computation_time, 4),
        "efficiency_score": round(num_samples / (computation_time * 1000), 2),  # samples per millisecond
    }


def generate_sample_sizes(count=100):
    """Generate diverse sample sizes for comprehensive testing."""
    sizes = []
    
    # Small sizes (for quick testing)
    sizes.extend([100, 500, 1000, 2000, 5000] * 4)
    
    # Medium sizes (for balanced accuracy/speed)
    sizes.extend([10000, 25000, 50000, 75000, 100000] * 8)
    
    # Large sizes (for high accuracy)
    sizes.extend([250000, 500000, 750000, 1000000] * 6)
    
    # Random sizes in different ranges
    sizes.extend([random.randint(1000, 10000) for _ in range(20)])
    sizes.extend([random.randint(10000, 100000) for _ in range(20)])
    sizes.extend([random.randint(100000, 500000) for _ in range(10)])
    
    # Ensure we have exactly the requested count
    return sizes[:count]


def analyze_results_and_recommend():
    """Analyze experiment results and provide intelligent recommendations."""
    print("\n🔍 Analyzing results and generating recommendations...")
    
    # Get all π estimation experiments
    all_experiments = run.find("pi_estimate > 0")
    
    if len(all_experiments) < 10:
        print("   ⚠️  Need at least 10 experiments for meaningful analysis")
        return
    
    print(f"   📊 Analyzing {len(all_experiments)} experiments...")
    
    # Statistical analysis
    accuracies = [exp.metrics.get('accuracy_score', 0) for exp in all_experiments]
    errors = [exp.metrics.get('absolute_error', 1) for exp in all_experiments]
    times = [exp.metrics.get('computation_time', 0) for exp in all_experiments]
    efficiencies = [exp.metrics.get('efficiency_score', 0) for exp in all_experiments]
    sample_sizes = [exp.parameters.get('num_samples', 0) for exp in all_experiments]
    
    # Find optimal ranges
    best_accuracy_idx = accuracies.index(max(accuracies))
    best_efficiency_idx = efficiencies.index(max(efficiencies))
    fastest_idx = times.index(min([t for t in times if t > 0]))
    
    best_accuracy_exp = all_experiments[best_accuracy_idx]
    best_efficiency_exp = all_experiments[best_efficiency_idx]
    fastest_exp = all_experiments[fastest_idx]
    
    print(f"\n   🏆 PERFORMANCE ANALYSIS:")
    print(f"   • Best Accuracy: {max(accuracies):.6f} (π estimate: {best_accuracy_exp.metrics['pi_estimate']:.6f})")
    print(f"     └─ Samples: {best_accuracy_exp.parameters['num_samples']:,}, Method: {best_accuracy_exp.parameters['method']}")
    
    print(f"   • Best Efficiency: {max(efficiencies):.2f} samples/ms")
    print(f"     └─ Samples: {best_efficiency_exp.parameters['num_samples']:,}, Time: {best_efficiency_exp.metrics['computation_time']:.4f}s")
    
    print(f"   • Fastest Execution: {min([t for t in times if t > 0]):.4f}s")
    print(f"     └─ Samples: {fastest_exp.parameters['num_samples']:,}, Accuracy: {fastest_exp.metrics['accuracy_score']:.6f}")
    
    # Sample size analysis
    sample_ranges = {
        'small': [exp for exp in all_experiments if exp.parameters.get('num_samples', 0) < 10000],
        'medium': [exp for exp in all_experiments if 10000 <= exp.parameters.get('num_samples', 0) < 100000],
        'large': [exp for exp in all_experiments if exp.parameters.get('num_samples', 0) >= 100000]
    }
    
    print(f"\n   📈 SAMPLE SIZE ANALYSIS:")
    for range_name, experiments in sample_ranges.items():
        if experiments:
            avg_accuracy = sum(exp.metrics.get('accuracy_score', 0) for exp in experiments) / len(experiments)
            avg_time = sum(exp.metrics.get('computation_time', 0) for exp in experiments) / len(experiments)
            print(f"   • {range_name.title()} samples ({len(experiments)} runs): Avg accuracy {avg_accuracy:.6f}, Avg time {avg_time:.4f}s")
    
    # Generate intelligent recommendations
    print(f"\n   💡 INTELLIGENT RECOMMENDATIONS:")
    
    # Accuracy vs Speed trade-off
    high_accuracy_exps = [exp for exp in all_experiments if exp.metrics.get('accuracy_score', 0) > 0.999]
    if high_accuracy_exps:
        min_samples_for_high_acc = min(exp.parameters.get('num_samples', 0) for exp in high_accuracy_exps)
        print(f"   1. For high accuracy (>99.9%): Use at least {min_samples_for_high_acc:,} samples")
    else:
        min_samples_for_high_acc = 100000  # default fallback
    
    # Efficiency recommendation
    efficient_exps = sorted(all_experiments, key=lambda x: x.metrics.get('efficiency_score', 0), reverse=True)[:10]
    avg_efficient_samples = sum(exp.parameters.get('num_samples', 0) for exp in efficient_exps) / len(efficient_exps)
    print(f"   2. For best efficiency: Sweet spot around {int(avg_efficient_samples):,} samples")
    
    # Method comparison
    methods = {}
    for exp in all_experiments:
        method = exp.parameters.get('method', 'uniform')
        if method not in methods:
            methods[method] = []
        methods[method].append(exp.metrics.get('accuracy_score', 0))
    
    if len(methods) > 1:
        best_method = max(methods.keys(), key=lambda m: sum(methods[m])/len(methods[m]))
        print(f"   3. Best sampling method: '{best_method}' (avg accuracy: {sum(methods[best_method])/len(methods[best_method]):.6f})")
    
    # Resource optimization
    quick_good_exps = [exp for exp in all_experiments 
                      if exp.metrics.get('computation_time', 1) < 0.1 and 
                         exp.metrics.get('accuracy_score', 0) > 0.995]
    if quick_good_exps:
        best_quick = max(quick_good_exps, key=lambda x: x.metrics.get('accuracy_score', 0))
        print(f"   4. For quick prototyping: {best_quick.parameters['num_samples']:,} samples gives {best_quick.metrics['accuracy_score']:.6f} accuracy in {best_quick.metrics['computation_time']:.4f}s")
    
    # Future experiment suggestions
    print(f"\n   🔮 SUGGESTED NEXT EXPERIMENTS:")
    print(f"   • Test {int(avg_efficient_samples * 1.5):,} samples with 'stratified' method")
    print(f"   • Compare performance with {min_samples_for_high_acc * 2:,} samples for ultra-high precision")
    print(f"   • Benchmark different methods with exactly {int(avg_efficient_samples):,} samples each")


def main():
    """π Value Estimator - Comprehensive Monte Carlo Analysis."""
    print("🥧 π Value Estimator - Monte Carlo Experiment Suite")
    print("=" * 60)
    print("This demo runs 100 experiments with varying sample sizes to find")
    print("optimal parameters for Monte Carlo π estimation.\n")

    # 1. Run comprehensive experiment suite
    print("1️⃣ Running 100 Monte Carlo experiments with varying sample sizes...")
    print("   This will take a few minutes - tracking accuracy vs performance trade-offs\n")
    
    sample_sizes = generate_sample_sizes(100)
    methods = ['uniform', 'gaussian', 'stratified']
    
    experiment_count = 0
    total_experiments = len(sample_sizes)
    
    for i, num_samples in enumerate(sample_sizes):
        method = methods[i % len(methods)]  # Cycle through methods
        random_seed = random.randint(1, 1000) if i % 5 == 0 else None  # Some with fixed seeds
        
        experiment_count += 1
        
        # Progress indicator
        if experiment_count % 10 == 0:
            print(f"   🔬 Progress: {experiment_count}/{total_experiments} experiments completed")
        
        try:
            run_id = run.single(
                pi_estimator_experiment,
                num_samples=num_samples,
                method=method,
                random_seed=random_seed
            )
        except Exception as e:
            print(f"   ⚠️  Experiment {experiment_count} failed: {e}")
            continue
    
    print(f"\n✅ Completed {experiment_count} experiments!")

    # 2. Generate comprehensive insights
    print("\n2️⃣ Generating comprehensive insights...")
    
    insights = run.insights()
    if "summary" in insights:
        summary = insights["summary"]
        print(f"   📊 Total experiments: {summary.get('total_experiments', 0)}")
        print(f"   📈 Success rate: {summary.get('success_rate', 0):.1%}")
        print(f"   ⏱️ Average duration: {summary.get('avg_duration', 0):.4f}s")
        print(f"   🎯 Average π accuracy: {1 - summary.get('avg_absolute_error', 0.1) / math.pi:.6f}")

    # 3. Find best performing experiments
    print("\n3️⃣ Finding top performing experiments...")
    
    # Most accurate estimates
    most_accurate = run.find("accuracy_score > 0.999")
    print(f"   🎯 {len(most_accurate)} experiments achieved >99.9% accuracy")
    
    # Most efficient experiments
    most_efficient = run.find("efficiency_score > 50")
    print(f"   ⚡ {len(most_efficient)} experiments achieved >50 samples/ms efficiency")
    
    # Quick and accurate
    quick_accurate = run.find("computation_time < 0.1 AND accuracy_score > 0.995")
    print(f"   🚀 {len(quick_accurate)} experiments were both quick (<0.1s) and accurate (>99.5%)")

    # 4. Detailed analysis and recommendations
    analyze_results_and_recommend()

    # 5. Comparison of best experiments
    print("\n4️⃣ Comparing top experiments...")
    
    recent_experiments = run.recent(hours=1)
    if len(recent_experiments) >= 3:
        # Get top 5 by accuracy
        top_by_accuracy = sorted(recent_experiments, 
                               key=lambda x: x.metrics.get('accuracy_score', 0), 
                               reverse=True)[:5]
        print("   📊 Comparing top 5 most accurate experiments:")
        run.compare(top_by_accuracy)
    
    # 6. Interactive exploration options
    print("\n5️⃣ Interactive Exploration:")
    print("   📊 Launch web dashboard to visualize results:")
    print("      rexf-analytics --dashboard")
    print()
    print("   🔍 Query specific results:")
    print("      rexf-analytics --query 'accuracy_score > 0.999'")
    print("      rexf-analytics --query 'num_samples > 100000'")
    print("      rexf-analytics --query 'efficiency_score > 30'")
    print()
    print("   📈 Compare best experiments:")
    print("      rexf-analytics --compare --best 10")
    print()
    print("   💡 Get more insights:")
    print("      rexf-analytics --insights")

    print(f"\n🎉 π Estimation Analysis Complete!")
    print(f"📝 All {experiment_count} experiments stored in: experiments.db")
    print(f"🥧 True π value: {math.pi:.10f}")
    print(f"🎯 Launch the dashboard to explore your results interactively!")


if __name__ == "__main__":
    main()
