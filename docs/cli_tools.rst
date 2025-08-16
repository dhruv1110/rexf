🔧 Command Line Tools
====================

RexF provides powerful command-line tools for experiment analysis, automation, and integration with other tools.

Installation and Setup
----------------------

The CLI tools are automatically installed with RexF:

.. code-block:: bash

   pip install rexf

   # Verify installation
   rexf-analytics --help

Main Command: rexf-analytics
---------------------------

The ``rexf-analytics`` command provides comprehensive experiment analysis capabilities.

Basic Usage
~~~~~~~~~~

.. code-block:: bash

   # Show help
   rexf-analytics --help

   # Quick summary of all experiments
   rexf-analytics --summary

   # List all experiments
   rexf-analytics --list

Common Options
~~~~~~~~~~~~~

All commands support these common options:

.. code-block:: bash

   # Specify database file (default: experiments.db)
   rexf-analytics --database custom.db --summary

   # Limit number of results
   rexf-analytics --list --limit 10

   # Verbose output for debugging
   rexf-analytics --list --verbose

   # Filter by experiment name
   rexf-analytics --list --experiment-name my_experiment

Core Commands
------------

List Experiments
~~~~~~~~~~~~~~~

Display experiment information in various formats:

.. code-block:: bash

   # Basic list (table format)
   rexf-analytics --list

   # JSON format
   rexf-analytics --list --format json

   # CSV format (great for Excel/analysis)
   rexf-analytics --list --format csv

   # Save to file
   rexf-analytics --list --format csv --output experiments.csv

   # Filter by experiment name
   rexf-analytics --list --experiment-name hyperparameter_search

   # Show only recent experiments
   rexf-analytics --list --limit 20

Query Experiments
~~~~~~~~~~~~~~~~

Use expressions to find specific experiments:

.. code-block:: bash

   # Find high-accuracy experiments
   rexf-analytics --query "accuracy > 0.9"

   # Find fast experiments
   rexf-analytics --query "training_time < 60"

   # Complex queries
   rexf-analytics --query "accuracy > 0.8 and param_learning_rate < 0.01"

   # Parameter queries (use param_ prefix)
   rexf-analytics --query "param_batch_size >= 64"

   # Save query results
   rexf-analytics --query "accuracy > 0.9" --output high_accuracy.csv

Generate Summary
~~~~~~~~~~~~~~~

Get quick overview statistics:

.. code-block:: bash

   # Overall summary
   rexf-analytics --summary

   # Summary for specific experiment
   rexf-analytics --summary --experiment-name ml_experiment

   # Summary in JSON format
   rexf-analytics --summary --format json

Example output:

.. code-block:: text

   📊 Experiment Summary
   =====================
   
   Total Experiments: 127
   Successful: 119 (93.7%)
   Failed: 8 (6.3%)
   
   Average Duration: 145.2 seconds
   Total Runtime: 5.1 hours
   
   Top Experiments by accuracy:
   1. Run a1b2c3d4: 0.9847
   2. Run e5f6g7h8: 0.9823
   3. Run i9j0k1l2: 0.9801

Generate Insights
~~~~~~~~~~~~~~~~

Get intelligent analysis and recommendations:

.. code-block:: bash

   # Generate insights
   rexf-analytics --insights

   # Insights for specific experiment
   rexf-analytics --insights --experiment-name optimization_experiment

   # Save insights to file
   rexf-analytics --insights --output insights_report.json

Compare Experiments
~~~~~~~~~~~~~~~~~

Compare multiple experiments side-by-side:

.. code-block:: bash

   # Compare best experiments
   rexf-analytics --compare --best 5

   # Compare specific experiments
   rexf-analytics --compare --run-ids abc123,def456,ghi789

   # Compare experiments from query
   rexf-analytics --query "accuracy > 0.95" --compare

Web Dashboard
~~~~~~~~~~~~

Launch the web dashboard from command line:

.. code-block:: bash

   # Launch dashboard
   rexf-analytics --dashboard

   # Custom host and port
   rexf-analytics --dashboard --host 0.0.0.0 --port 9090

   # Don't open browser automatically
   rexf-analytics --dashboard --no-browser

Advanced Usage
-------------

Batch Analysis
~~~~~~~~~~~~~

Process multiple databases or perform batch operations:

.. code-block:: bash

   # Analyze multiple databases
   for db in experiments_*.db; do
       echo "=== Analysis for $db ==="
       rexf-analytics --database "$db" --summary
   done

   # Export all data for external analysis
   rexf-analytics --list --format csv --output all_experiments.csv
   rexf-analytics --insights --output insights.json

Automation and Scripting
~~~~~~~~~~~~~~~~~~~~~~~

Use CLI tools in scripts and automation:

.. code-block:: bash

   #!/bin/bash
   
   # Automated experiment monitoring script
   
   # Check for new experiments
   RECENT_COUNT=$(rexf-analytics --query "start_time > '$(date -d '1 hour ago' --iso-8601)'" --format json | jq length)
   
   if [ "$RECENT_COUNT" -gt 0 ]; then
       echo "Found $RECENT_COUNT new experiments"
       
       # Generate insights for recent experiments
       rexf-analytics --insights --output "insights_$(date +%Y%m%d_%H%M%S).json"
       
       # Check for high-performing experiments
       HIGH_PERF=$(rexf-analytics --query "accuracy > 0.95" --format json | jq length)
       
       if [ "$HIGH_PERF" -gt 0 ]; then
           echo "🎉 Found $HIGH_PERF high-performing experiments!"
           # Send notification, update dashboard, etc.
       fi
   fi

Integration with Other Tools
---------------------------

Git Integration
~~~~~~~~~~~~~~

Track experiment results with Git:

.. code-block:: bash

   # Export current results
   rexf-analytics --list --format csv --output experiments_$(git rev-parse --short HEAD).csv

   # Add to Git for version tracking
   git add experiments_$(git rev-parse --short HEAD).csv
   git commit -m "Experiment results for commit $(git rev-parse --short HEAD)"

Continuous Integration
~~~~~~~~~~~~~~~~~~~~~

Use in CI/CD pipelines:

.. code-block:: yaml

   # GitHub Actions example
   - name: Run Experiments and Analyze
     run: |
       python run_experiments.py
       rexf-analytics --summary
       rexf-analytics --query "accuracy < 0.8" --format json > failing_experiments.json
       
   - name: Upload Results
     uses: actions/upload-artifact@v3
     with:
       name: experiment-results
       path: |
         experiments.db
         failing_experiments.json

External Analysis Tools
~~~~~~~~~~~~~~~~~~~~~~

Export data for analysis in R, Python, MATLAB:

.. code-block:: bash

   # For R analysis
   rexf-analytics --list --format csv --output experiments_for_r.csv

   # For Python/Pandas
   rexf-analytics --list --format json --output experiments.json

   # For MATLAB
   rexf-analytics --list --format csv --output experiments.csv

Output Formats
-------------

JSON Format
~~~~~~~~~~

Structured data perfect for programmatic processing:

.. code-block:: bash

   rexf-analytics --list --format json

.. code-block:: json

   [
     {
       "run_id": "abc123def456",
       "experiment_name": "ml_experiment",
       "status": "completed",
       "start_time": "2024-01-15T10:30:00",
       "duration": 145.2,
       "parameters": {
         "learning_rate": 0.01,
         "batch_size": 32
       },
       "metrics": {
         "accuracy": 0.9547,
         "loss": 0.0453
       }
     }
   ]

CSV Format
~~~~~~~~~

Tabular data ideal for Excel and data analysis:

.. code-block:: bash

   rexf-analytics --list --format csv

.. code-block:: csv

   run_id,experiment_name,status,start_time,duration,param_learning_rate,param_batch_size,metric_accuracy,metric_loss
   abc123def456,ml_experiment,completed,2024-01-15T10:30:00,145.2,0.01,32,0.9547,0.0453

Table Format
~~~~~~~~~~~

Human-readable tables for terminal display:

.. code-block:: bash

   rexf-analytics --list --format table

.. code-block:: text

   ┌──────────┬─────────────────┬───────────┬─────────────────────┬──────────┬──────────┐
   │ Run ID   │ Experiment      │ Status    │ Start Time          │ Duration │ Accuracy │
   ├──────────┼─────────────────┼───────────┼─────────────────────┼──────────┼──────────┤
   │ abc123de │ ml_experiment   │ completed │ 2024-01-15 10:30:00 │ 145.2s   │ 0.9547   │
   │ def456gh │ ml_experiment   │ completed │ 2024-01-15 11:15:00 │ 132.1s   │ 0.9423   │
   └──────────┴─────────────────┴───────────┴─────────────────────┴──────────┴──────────┘

Filtering and Search
-------------------

Experiment Name Filtering
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Show only specific experiments
   rexf-analytics --list --experiment-name hyperparameter_search

   # Works with all commands
   rexf-analytics --summary --experiment-name ml_experiment
   rexf-analytics --insights --experiment-name optimization_study

Status Filtering
~~~~~~~~~~~~~~~

.. code-block:: bash

   # Show only completed experiments
   rexf-analytics --query "status == 'completed'"

   # Show failed experiments for debugging
   rexf-analytics --query "status == 'failed'" --format table

Time-based Filtering
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Recent experiments (last 24 hours)
   rexf-analytics --query "start_time > '$(date -d '1 day ago' --iso-8601)'"

   # Experiments from specific date range
   rexf-analytics --query "start_time between '2024-01-01' and '2024-01-31'"

Performance and Optimization
---------------------------

Large Databases
~~~~~~~~~~~~~~

For databases with many experiments:

.. code-block:: bash

   # Use limits to avoid overwhelming output
   rexf-analytics --list --limit 100

   # Use specific queries instead of listing all
   rexf-analytics --query "start_time > '2024-01-01'" --limit 50

   # Export in chunks for very large datasets
   rexf-analytics --query "start_time between '2024-01-01' and '2024-01-07'" --output week1.csv

Caching and Speed
~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Use specific queries for faster results
   rexf-analytics --query "experiment_name == 'ml_experiment' and status == 'completed'"

   # Limit output fields for faster processing
   rexf-analytics --list --format csv --output minimal.csv

Common Use Cases
---------------

Daily Experiment Review
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   #!/bin/bash
   echo "=== Daily Experiment Review ==="
   echo ""
   
   echo "Recent experiments (last 24h):"
   rexf-analytics --query "start_time > '$(date -d '1 day ago' --iso-8601)'" --format table
   
   echo ""
   echo "Best performers today:"
   rexf-analytics --query "start_time > '$(date -d '1 day ago' --iso-8601)' and accuracy > 0.9"
   
   echo ""
   echo "Any failures to investigate:"
   rexf-analytics --query "start_time > '$(date -d '1 day ago' --iso-8601)' and status == 'failed'"

Research Paper Data Export
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Export specific experiment set for paper
   rexf-analytics --query "experiment_name == 'paper_experiments' and accuracy > 0.0" \
                  --format csv --output paper_results.csv

   # Generate insights for methodology section
   rexf-analytics --insights --experiment-name paper_experiments \
                  --output paper_insights.json

Hyperparameter Optimization Monitoring
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Monitor optimization progress
   rexf-analytics --query "experiment_name == 'hyperopt' and status == 'completed'" \
                  --format table

   # Find best parameters found so far
   rexf-analytics --query "experiment_name == 'hyperopt'" --compare --best 5

   # Export for external optimization tools
   rexf-analytics --query "experiment_name == 'hyperopt'" \
                  --format json --output hyperopt_results.json

Troubleshooting
--------------

Common Issues
~~~~~~~~~~~~

**Database not found**:

.. code-block:: bash

   # Check current directory
   ls -la *.db

   # Specify full path
   rexf-analytics --database /full/path/to/experiments.db --summary

**No experiments found**:

.. code-block:: bash

   # Verify database has data
   rexf-analytics --list --limit 1

   # Check experiment names
   rexf-analytics --summary

**Query syntax errors**:

.. code-block:: bash

   # Use proper syntax for parameter queries
   rexf-analytics --query "param_learning_rate > 0.01"  # Correct
   # rexf-analytics --query "learning_rate > 0.01"      # Wrong

   # Use quotes for string values
   rexf-analytics --query "status == 'completed'"       # Correct
   # rexf-analytics --query "status == completed"       # Wrong

Performance Issues
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # For large databases, use specific queries
   rexf-analytics --query "start_time > '2024-01-01'" --limit 100

   # Export data in smaller chunks
   rexf-analytics --query "experiment_name == 'specific_exp'" --output chunk.csv

Next Steps
---------

- :doc:`web_dashboard` - Interactive visualization and monitoring
- :doc:`advanced_features` - Advanced analysis and exploration  
- :doc:`api/cli` - Detailed CLI API reference
