🧪 RexF - Smart Experiments Framework
========================================

.. image:: https://github.com/dhruv1110/rexf/workflows/CI/badge.svg
   :target: https://github.com/dhruv1110/rexf/actions/workflows/ci.yml
   :alt: CI Status

.. image:: https://img.shields.io/pypi/v/rexf
   :target: https://pypi.org/project/rexf/
   :alt: PyPI Version

.. image:: https://img.shields.io/pypi/pyversions/rexf
   :target: https://pypi.org/project/rexf/
   :alt: Python Versions

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

A lightweight Python library for **reproducible computational experiments** with an ultra-simple, smart API. From idea to insight in under 5 minutes, with zero configuration.

✨ Key Features
--------------

* **🎯 Ultra-Simple API**: Single ``@experiment`` decorator - that's it!
* **🚀 Auto-Everything**: Parameters, metrics, and results detected automatically
* **🔍 Smart Exploration**: Automated parameter space exploration with multiple strategies
* **💡 Intelligent Insights**: Automated pattern detection and recommendations
* **📊 Web Dashboard**: Beautiful real-time experiment monitoring
* **🔧 CLI Analytics**: Powerful command-line tools for ad-hoc analysis
* **📈 Query Interface**: Find experiments using simple expressions like ``"accuracy > 0.9"``
* **🔄 Reproducible**: Git commit tracking, environment capture, seed management
* **💾 Local-First**: SQLite database - no external servers required

🚀 Quick Start
--------------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   pip install rexf

Ultra-Simple Usage
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rexf import experiment, run

   @experiment
   def my_experiment(learning_rate, batch_size=32):
       # Your experiment code here
       accuracy = train_model(learning_rate, batch_size)
       return {"accuracy": accuracy, "loss": 1 - accuracy}

   # Run single experiment
   run.single(my_experiment, learning_rate=0.01, batch_size=64)

   # Get insights
   print(run.insights())

   # Find best experiments
   best = run.best(metric="accuracy", top=5)

   # Auto-explore parameter space
   run.auto_explore(my_experiment, strategy="random", budget=20)

   # Launch web dashboard
   run.dashboard()

📚 Documentation
----------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   quickstart
   basic_usage
   advanced_features
   web_dashboard
   cli_tools
   reproducibility

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/monte_carlo

.. toctree::
   :maxdepth: 2
   :caption: Advanced Usage

   advanced_features

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/core
   api/run

.. toctree::
   :maxdepth: 2
   :caption: Development

   contributing

🎯 Core Philosophy
-----------------

**From idea to insight in under 5 minutes, with zero configuration.**

RexF prioritizes user experience over architectural purity. Instead of making you learn complex APIs, it automatically detects what you're doing and provides smart features to accelerate your research.

🎨 Why RexF?
-----------

Traditional Approach
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import mlflow
   import sacred
   from sacred import Experiment

   # Complex setup required
   ex = Experiment('my_exp')
   mlflow.set_tracking_uri("...")

   @ex.config
   def config():
       learning_rate = 0.01
       batch_size = 32

   @ex.automain
   def main(learning_rate, batch_size):
       with mlflow.start_run():
           # Your code here
           mlflow.log_param("lr", learning_rate)
           mlflow.log_metric("accuracy", accuracy)

RexF Approach
~~~~~~~~~~~~

.. code-block:: python

   from rexf import experiment, run

   @experiment
   def my_experiment(learning_rate=0.01, batch_size=32):
       # Your code here - that's it!
       return {"accuracy": accuracy}

   run.single(my_experiment, learning_rate=0.05)

Comparison Table
~~~~~~~~~~~~~~~

.. list-table:: Feature Comparison
   :header-rows: 1
   :widths: 30 35 35

   * - Feature
     - Traditional Tools
     - RexF
   * - Setup
     - Complex configuration
     - Single decorator
   * - Parameter Detection
     - Manual logging
     - Automatic
   * - Metric Tracking
     - Manual logging
     - Automatic
   * - Insights
     - Manual analysis
     - Auto-generated
   * - Exploration
     - Write custom loops
     - ``run.auto_explore()``
   * - Comparison
     - Custom dashboards
     - ``run.compare()``
   * - Querying
     - SQL/Complex APIs
     - ``run.find("accuracy > 0.9")``

🔗 Links
--------

* **PyPI**: https://pypi.org/project/rexf/
* **GitHub**: https://github.com/dhruv1110/rexf
* **Issues**: https://github.com/dhruv1110/rexf/issues
* **Discussions**: https://github.com/dhruv1110/rexf/discussions

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
