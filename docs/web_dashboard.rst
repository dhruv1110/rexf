📊 Web Dashboard
===============

RexF provides a beautiful, interactive web dashboard for real-time experiment monitoring and analysis.

Getting Started
--------------

Launch the dashboard with a single command:

.. code-block:: python

   from rexf import run

   # Launch dashboard (opens browser automatically)
   run.dashboard()

   # Or from command line
   # rexf-analytics --dashboard

The dashboard will open in your browser at ``http://localhost:8080``.

Configuration Options
--------------------

Python API
~~~~~~~~~~

.. code-block:: python

   # Custom host and port
   run.dashboard(host="0.0.0.0", port=9000)

   # Don't open browser automatically
   run.dashboard(open_browser=False)

   # Specify custom database path
   run.dashboard(storage_path="custom_experiments.db")

Command Line
~~~~~~~~~~~

.. code-block:: bash

   # Default settings
   rexf-analytics --dashboard

   # Custom host and port
   rexf-analytics --dashboard --host 0.0.0.0 --port 9090

   # Don't open browser
   rexf-analytics --dashboard --no-browser

   # Custom database
   rexf-analytics --database custom.db --dashboard

Dashboard Features
-----------------

Overview Tab
~~~~~~~~~~~

The main dashboard shows:

- **Experiment Summary**: Total experiments, success rate, average metrics
- **Recent Activity**: Latest experiment runs with status and key metrics
- **Quick Stats**: Performance overview and trends
- **Status Distribution**: Visual breakdown of experiment statuses

Metrics Visualization
~~~~~~~~~~~~~~~~~~~~

Interactive charts showing:

- **Metric Trends**: How metrics change over time
- **Metric Distributions**: Histograms of metric values
- **Correlation Analysis**: Relationships between different metrics
- **Performance Over Time**: Execution time and efficiency trends

Parameter Space
~~~~~~~~~~~~~~

Explore parameter relationships:

- **Parameter vs Metric Scatter Plots**: Visualize parameter impact
- **Parameter Distributions**: See parameter value distributions
- **Multi-dimensional Views**: Explore complex parameter interactions
- **Optimization Landscapes**: Identify optimal parameter regions

Experiment Browser
~~~~~~~~~~~~~~~~~

Browse and filter experiments:

- **Sortable Tables**: Sort by any column (metrics, parameters, time)
- **Advanced Filtering**: Filter by status, metrics ranges, parameter values
- **Search Functionality**: Quick text search across experiments
- **Detailed Views**: Click experiments for full details

Interactive Features
------------------

Real-time Updates
~~~~~~~~~~~~~~~~

The dashboard automatically updates when new experiments are run:

.. code-block:: python

   # Run experiments while dashboard is open
   @experiment
   def live_experiment(param1, param2=42):
       return {"metric": param1 * param2}

   # Dashboard will show new results immediately
   run.single(live_experiment, param1=5)
   run.single(live_experiment, param1=10)

Filtering and Querying
~~~~~~~~~~~~~~~~~~~~~

Use the dashboard's query interface:

- **Metric Filters**: ``accuracy > 0.9``, ``loss < 0.1``
- **Parameter Filters**: ``param_learning_rate between 0.001 and 0.01``
- **Status Filters**: ``status == 'completed'``
- **Time Filters**: ``start_time > '2024-01-01'``

Chart Interactions
~~~~~~~~~~~~~~~~~

All charts are interactive:

- **Zoom and Pan**: Explore data at different scales
- **Hover Details**: Get detailed information on hover
- **Click to Filter**: Click chart elements to filter data
- **Export Options**: Save charts as images

Comparison Mode
~~~~~~~~~~~~~~

Compare multiple experiments side-by-side:

1. Select experiments from the browser
2. Click "Compare Selected"
3. View side-by-side parameter and metric comparison
4. See statistical analysis and recommendations

Advanced Usage
-------------

Custom Queries
~~~~~~~~~~~~~

Use the query bar for complex filtering:

.. code-block:: sql

   -- High-performing recent experiments
   accuracy > 0.9 and start_time > '2024-01-01'

   -- Parameter optimization
   param_learning_rate < 0.01 and training_time < 300

   -- Failed experiment analysis
   status == 'failed' and param_complexity > 0.5

Data Export
~~~~~~~~~~

Export filtered data directly from the dashboard:

- **CSV Export**: Download table data as CSV
- **JSON Export**: Get structured data for further analysis
- **Chart Export**: Save visualizations as PNG/SVG

Dashboard API Endpoints
----------------------

The dashboard exposes REST API endpoints for programmatic access:

Experiment Data
~~~~~~~~~~~~~~

.. code-block:: bash

   # Get all experiments
   GET /api/experiments

   # Get specific experiment
   GET /api/experiments/{run_id}

   # Get experiments with filters
   GET /api/experiments?status=completed&limit=10

Metrics and Analytics
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Get metric trends
   GET /api/metrics?metric=accuracy

   # Get parameter space data
   GET /api/parameter_space

   # Get experiment statistics
   GET /api/stats

Real-time Updates
~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Server-sent events for real-time updates
   GET /api/stream

   # WebSocket connection for live data
   WS /api/websocket

Customization
------------

Theme and Styling
~~~~~~~~~~~~~~~~~

The dashboard supports customization through CSS:

.. code-block:: python

   # Custom styling (future feature)
   run.dashboard(
       theme="dark",  # "light", "dark", "auto"
       custom_css="path/to/custom.css"
   )

Custom Metrics Display
~~~~~~~~~~~~~~~~~~~~~

Configure which metrics to highlight:

.. code-block:: python

   # Configure dashboard focus (future feature)
   run.dashboard(
       primary_metrics=["accuracy", "f1_score"],
       secondary_metrics=["training_time", "memory_usage"]
   )

Integration with External Tools
------------------------------

Jupyter Notebooks
~~~~~~~~~~~~~~~~~

Use the dashboard alongside Jupyter:

.. code-block:: python

   # In Jupyter cell
   from rexf import run

   # Launch dashboard in background
   import threading
   dashboard_thread = threading.Thread(
       target=run.dashboard, 
       kwargs={"open_browser": False}
   )
   dashboard_thread.daemon = True
   dashboard_thread.start()

   # Continue with experiments
   # Dashboard will update automatically

Remote Access
~~~~~~~~~~~~

For remote servers or cloud environments:

.. code-block:: bash

   # Make dashboard accessible from any IP
   rexf-analytics --dashboard --host 0.0.0.0 --port 8080

   # Access from remote machine
   # http://your-server-ip:8080

Security Considerations
~~~~~~~~~~~~~~~~~~~~~~

For production deployments:

- Run behind a reverse proxy (nginx, Apache)
- Use HTTPS for secure connections
- Implement authentication if needed
- Restrict access by IP/network

Performance Tips
---------------

Large Datasets
~~~~~~~~~~~~~

For experiments with many runs:

- Use filtering to limit displayed data
- Consider pagination for very large datasets
- Use time-based filtering for recent data focus

Resource Usage
~~~~~~~~~~~~~

The dashboard is lightweight but consider:

- Memory usage increases with experiment count
- Chart rendering performance depends on data size
- Use query limits for better responsiveness

Troubleshooting
--------------

Common Issues
~~~~~~~~~~~~

**Dashboard won't start**:

.. code-block:: bash

   # Check if port is already in use
   lsof -i :8080

   # Try different port
   rexf-analytics --dashboard --port 8081

**No data showing**:

- Verify database path is correct
- Check that experiments have been run
- Ensure database is not corrupted

**Performance issues**:

- Reduce data scope with filters
- Check available system memory
- Consider using a more powerful machine for large datasets

**Browser compatibility**:

- Use modern browsers (Chrome, Firefox, Safari, Edge)
- Enable JavaScript
- Clear browser cache if needed

Next Steps
---------

- :doc:`cli_tools` - Command-line analytics and automation
- :doc:`advanced_features` - Advanced analysis and exploration
- :doc:`api/dashboard` - Detailed API reference
