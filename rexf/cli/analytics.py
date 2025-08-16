"""Command-line analytics tool for rexf experiments.

This tool allows users to perform ad-hoc analytics on experiment data
without writing Python code.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from ..backends.intelligent_storage import IntelligentStorage
from ..intelligence.insights import InsightsEngine
from ..intelligence.queries import SmartQueryEngine
from ..intelligence.smart_compare import SmartComparer


def main():
    """Main entry point for the CLI analytics tool."""
    parser = argparse.ArgumentParser(
        description="rexf Analytics CLI - Analyze experiment data from command line",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show experiment summary
  rexf-analytics --database experiments.db --summary

  # Find high-accuracy experiments
  rexf-analytics --database experiments.db --query "accuracy > 0.9"

  # Generate insights
  rexf-analytics --database experiments.db --insights

  # Compare best experiments
  rexf-analytics --database experiments.db --compare --best 5

  # Export results to JSON
  rexf-analytics --database experiments.db --summary --output results.json

  # Launch web dashboard
  rexf-analytics --database experiments.db --dashboard

  # Launch dashboard on custom host/port
  rexf-analytics --database experiments.db --dashboard --host 0.0.0.0 --port 9090
        """,
    )

    # Database options
    parser.add_argument(
        "--database",
        "-d",
        type=str,
        default="experiments.db",
        help="Path to experiment database (default: experiments.db)",
    )

    # Query operations
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        help="Query experiments using expressions (e.g., 'accuracy > 0.9')",
    )

    parser.add_argument(
        "--list", "-l", action="store_true", help="List all experiments"
    )

    parser.add_argument(
        "--summary",
        "-s",
        action="store_true",
        help="Show experiment summary statistics",
    )

    parser.add_argument(
        "--insights",
        "-i",
        action="store_true",
        help="Generate insights from experiment data",
    )

    parser.add_argument(
        "--compare", "-c", action="store_true", help="Compare experiments"
    )

    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Launch web dashboard for interactive analysis"
    )

    # Dashboard options
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Dashboard host (default: localhost)"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Dashboard port (default: 8080)"
    )

    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't automatically open browser"
    )

    # Filtering options
    parser.add_argument(
        "--experiment-name", "-n", type=str, help="Filter by experiment name"
    )

    parser.add_argument(
        "--status",
        type=str,
        choices=["completed", "failed", "running"],
        help="Filter by experiment status",
    )

    parser.add_argument("--best", type=int, help="Show top N best experiments")

    parser.add_argument("--recent", type=int, help="Show experiments from last N hours")

    # Output options
    parser.add_argument(
        "--output", "-o", type=str, help="Output file for results (JSON format)"
    )

    parser.add_argument(
        "--format",
        type=str,
        choices=["table", "json", "csv"],
        default="table",
        help="Output format (default: table)",
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Validate database exists
    if not Path(args.database).exists():
        print(f"Error: Database file '{args.database}' not found.")
        sys.exit(1)

    try:
        # Special handling for dashboard (it manages its own storage)
        if args.dashboard:
            result = execute_dashboard(None, args)
            output_results(result, args)
            return
        
        # Initialize storage for other operations
        storage = IntelligentStorage(args.database)

        # Execute requested operation
        result = execute_operation(storage, args)

        # Output results
        output_results(result, args)

    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)
    finally:
        if "storage" in locals():
            storage.close()


def execute_operation(storage: IntelligentStorage, args) -> dict:
    """Execute the requested analytics operation."""
    if args.dashboard:
        return execute_dashboard(storage, args)
    elif args.query:
        return execute_query(storage, args)
    elif args.list:
        return execute_list(storage, args)
    elif args.summary:
        return execute_summary(storage, args)
    elif args.insights:
        return execute_insights(storage, args)
    elif args.compare:
        return execute_compare(storage, args)
    else:
        # Default to summary if no operation specified
        return execute_summary(storage, args)


def execute_dashboard(storage, args) -> dict:
    """Launch the web dashboard."""
    try:
        from ..dashboard.app import run_dashboard
        
        print(f"🚀 Launching RexF Dashboard...")
        print(f"📊 Database: {args.database}")
        print(f"🌐 Host: {args.host}")
        print(f"🔌 Port: {args.port}")
        print(f"🌐 URL: http://{args.host}:{args.port}")
        
        if args.no_browser:
            print("📝 Browser auto-launch disabled")
        
        print("⏹️  Press Ctrl+C to stop the dashboard")
        print("-" * 50)
        
        # Launch dashboard (it will create its own storage)
        run_dashboard(
            storage_path=args.database,
            host=args.host,
            port=args.port,
            open_browser=not args.no_browser,
        )
        
        return {
            "operation": "dashboard",
            "message": "Dashboard launched successfully",
            "host": args.host,
            "port": args.port,
            "database": args.database,
        }
        
    except ImportError:
        return {
            "operation": "dashboard",
            "error": "Dashboard dependencies not available",
            "message": "Install dashboard dependencies with: pip install rexf[dashboard]",
        }
    except KeyboardInterrupt:
        return {
            "operation": "dashboard",
            "message": "Dashboard stopped by user",
        }
    except Exception as e:
        return {
            "operation": "dashboard",
            "error": str(e),
            "message": "Failed to launch dashboard",
        }


def execute_query(storage: IntelligentStorage, args) -> dict:
    """Execute a query operation."""
    query_engine = SmartQueryEngine(storage)
    experiments = query_engine.query(args.query)

    return {
        "operation": "query",
        "query": args.query,
        "count": len(experiments),
        "experiments": [format_experiment_summary(exp) for exp in experiments],
    }


def execute_list(storage: IntelligentStorage, args) -> dict:
    """Execute a list operation."""
    experiments = storage.list_experiments(args.experiment_name)

    # Apply filters
    if args.status:
        experiments = [exp for exp in experiments if exp.status == args.status]

    if args.recent:
        from datetime import datetime, timedelta

        cutoff = datetime.now() - timedelta(hours=args.recent)
        experiments = [exp for exp in experiments if exp.start_time >= cutoff]

    if args.best:
        # Sort by first available metric (descending)
        experiments_with_metrics = []
        for exp in experiments:
            if exp.metrics:
                first_metric = list(exp.metrics.values())[0]
                if isinstance(first_metric, (int, float)):
                    experiments_with_metrics.append((exp, first_metric))

        experiments_with_metrics.sort(key=lambda x: x[1], reverse=True)
        experiments = [exp for exp, _ in experiments_with_metrics[: args.best]]

    return {
        "operation": "list",
        "count": len(experiments),
        "filters": {
            "experiment_name": args.experiment_name,
            "status": args.status,
            "recent_hours": args.recent,
            "best_count": args.best,
        },
        "experiments": [format_experiment_summary(exp) for exp in experiments],
    }


def execute_summary(storage: IntelligentStorage, args) -> dict:
    """Execute a summary operation."""
    stats = storage.get_storage_stats()

    # Get parameter space summary
    param_space = storage.get_parameter_space_summary(args.experiment_name)

    # Get recent experiment trends
    experiments = storage.list_experiments(args.experiment_name)
    recent_experiments = sorted(experiments, key=lambda x: x.start_time)[-10:]

    return {
        "operation": "summary",
        "storage_stats": stats,
        "parameter_space": param_space,
        "recent_trends": {
            "total_recent": len(recent_experiments),
            "completed_recent": len(
                [exp for exp in recent_experiments if exp.status == "completed"]
            ),
            "avg_duration_recent": (
                sum(exp.duration or 0 for exp in recent_experiments)
                / len(recent_experiments)
                if recent_experiments
                else 0
            ),
        },
    }


def execute_insights(storage: IntelligentStorage, args) -> dict:
    """Execute an insights operation."""
    insights_engine = InsightsEngine(storage)
    insights = insights_engine.generate_insights(args.experiment_name)

    return {
        "operation": "insights",
        "experiment_name": args.experiment_name,
        "insights": insights,
    }


def execute_compare(storage: IntelligentStorage, args) -> dict:
    """Execute a compare operation."""
    # Get experiments to compare
    experiments = storage.list_experiments(args.experiment_name)

    if args.best:
        # Get best experiments
        experiments_with_metrics = []
        for exp in experiments:
            if exp.metrics and exp.status == "completed":
                # Use first available metric
                first_metric = list(exp.metrics.values())[0]
                if isinstance(first_metric, (int, float)):
                    experiments_with_metrics.append((exp, first_metric))

        experiments_with_metrics.sort(key=lambda x: x[1], reverse=True)
        experiments = [exp for exp, _ in experiments_with_metrics[: args.best]]
    else:
        # Use recent experiments
        experiments = sorted(experiments, key=lambda x: x.start_time)[-5:]

    if len(experiments) < 2:
        return {
            "operation": "compare",
            "error": "Need at least 2 experiments to compare",
        }

    # Compare experiments
    comparer = SmartComparer()
    comparison = comparer.compare_experiments(experiments)

    return {
        "operation": "compare",
        "experiment_count": len(experiments),
        "comparison": comparison,
    }


def format_experiment_summary(exp) -> dict:
    """Format experiment for summary display."""
    return {
        "run_id": exp.run_id[:8] + "...",
        "experiment_name": exp.experiment_name,
        "status": exp.status,
        "start_time": exp.start_time.isoformat(),
        "duration": exp.duration,
        "parameters": exp.parameters,
        "metrics": exp.metrics,
        "artifacts_count": len(exp.artifacts) if exp.artifacts else 0,
    }


def output_results(result: dict, args):
    """Output results in the requested format."""
    if args.output:
        # Write to file
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"Results written to {args.output}")
    elif args.format == "json":
        print(json.dumps(result, indent=2, default=str))
    elif args.format == "csv":
        output_csv(result, args)
    else:
        output_table(result, args)


def output_table(result: dict, args):
    """Output results in table format."""
    operation = result.get("operation", "unknown")

    if operation == "dashboard":
        # Dashboard output is handled in execute_dashboard
        # Just show any error or success messages
        if "error" in result:
            print(f"❌ Dashboard Error: {result['error']}")
            if "message" in result:
                print(f"💡 {result['message']}")
        elif "message" in result:
            print(f"✅ {result['message']}")
        return

    elif operation == "summary":
        print("📊 EXPERIMENT SUMMARY")
        print("=" * 50)

        stats = result.get("storage_stats", {})
        print(f"Total experiments: {stats.get('total_experiments', 0)}")
        print(f"Database size: {stats.get('database_size_bytes', 0)} bytes")

        if "by_status" in stats:
            print("\nStatus distribution:")
            for status, count in stats["by_status"].items():
                print(f"  {status}: {count}")

        if "by_experiment" in stats:
            print("\nBy experiment type:")
            for exp_name, count in stats["by_experiment"].items():
                print(f"  {exp_name}: {count}")

        param_space = result.get("parameter_space", {})
        if param_space:
            print(f"\nParameters explored: {len(param_space)}")
            for param, info in list(param_space.items())[:5]:
                print(f"  {param}: {info.get('unique_values', 0)} unique values")

    elif operation in ["query", "list"]:
        experiments = result.get("experiments", [])
        print(f"📋 EXPERIMENTS ({len(experiments)} found)")
        print("=" * 80)

        if operation == "query":
            print(f"Query: {result.get('query', '')}")
            print()

        for exp in experiments:
            print(f"🧪 {exp['run_id']} - {exp['experiment_name']}")
            print(
                f"   Status: {exp['status']}, Duration: {exp.get('duration', 0):.2f}s"
            )
            print(f"   Parameters: {exp['parameters']}")
            if exp["metrics"]:
                print(f"   Metrics: {exp['metrics']}")
            print()

    elif operation == "insights":
        insights = result.get("insights", {})
        print("🔍 EXPERIMENT INSIGHTS")
        print("=" * 50)

        summary = insights.get("summary", {})
        print(f"Total experiments: {summary.get('total_experiments', 0)}")
        print(f"Success rate: {summary.get('success_rate', 0):.1%}")
        print(f"Average duration: {summary.get('avg_duration', 0):.2f}s")
        print(f"Recent trend: {summary.get('recent_trend', 'unknown')}")

        recommendations = insights.get("recommendations", [])
        if recommendations:
            print("\n💡 Recommendations:")
            for i, rec in enumerate(recommendations[:5], 1):
                print(f"  {i}. {rec}")

    elif operation == "compare":
        comparison = result.get("comparison", {})
        if "error" in comparison:
            print(f"❌ {comparison['error']}")
            return

        print(
            f"📊 EXPERIMENT COMPARISON ({result.get('experiment_count', 0)} experiments)"
        )
        print("=" * 70)

        summary = comparison.get("summary", {})
        print(f"Success rate: {summary.get('success_rate', 0):.1%}")
        print(f"Average duration: {summary.get('avg_duration', 0):.2f}s")

        insights = comparison.get("insights", [])
        if insights:
            print("\n🔍 Key insights:")
            for insight in insights[:3]:
                print(f"  • {insight}")

        recommendations = comparison.get("recommendations", [])
        if recommendations:
            print("\n💡 Recommendations:")
            for rec in recommendations[:3]:
                print(f"  • {rec}")


def output_csv(result: dict, args):
    """Output results in CSV format."""
    operation = result.get("operation", "unknown")

    if operation in ["query", "list"]:
        experiments = result.get("experiments", [])

        if not experiments:
            print("run_id,experiment_name,status,duration,parameters,metrics")
            return

        print("run_id,experiment_name,status,duration,parameters,metrics")
        for exp in experiments:
            params_str = json.dumps(exp["parameters"]).replace('"', '""')
            metrics_str = json.dumps(exp["metrics"]).replace('"', '""')

            print(
                f"{exp['run_id']},{exp['experiment_name']},{exp['status']},{exp.get('duration', 0)},\"{params_str}\",\"{metrics_str}\""
            )
    else:
        print("CSV output not supported for this operation")


if __name__ == "__main__":
    main()
