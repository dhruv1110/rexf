🤝 Contributing to RexF
======================

We welcome contributions to RexF! This guide will help you get started with contributing to the project.

Quick Start for Contributors
----------------------------

1. **Fork and Clone**:

.. code-block:: bash

   # Fork the repository on GitHub, then clone your fork
   git clone https://github.com/YOUR_USERNAME/rexf.git
   cd rexf

2. **Set up Development Environment**:

.. code-block:: bash

   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install in development mode with all dependencies
   pip install -e ".[dev]"

   # Install pre-commit hooks
   pre-commit install

3. **Run Tests**:

.. code-block:: bash

   # Run the test suite
   pytest tests/ -v

   # Run with coverage
   pytest tests/ --cov=rexf --cov-report=html

4. **Make Changes and Submit**:

.. code-block:: bash

   # Create a feature branch
   git checkout -b feature/your-feature-name

   # Make your changes, add tests, update docs
   # ...

   # Run tests and quality checks
   pytest tests/
   black .
   isort .
   flake8 rexf

   # Commit and push
   git add .
   git commit -m "Add your feature description"
   git push origin feature/your-feature-name

   # Create a Pull Request on GitHub

Development Setup
----------------

Dependencies
~~~~~~~~~~~

RexF uses several development tools:

- **Testing**: ``pytest``, ``pytest-cov``
- **Code Quality**: ``black``, ``isort``, ``flake8``, ``mypy``
- **Pre-commit**: ``pre-commit``
- **Documentation**: ``sphinx``, ``sphinx-rtd-theme``

Install all development dependencies:

.. code-block:: bash

   pip install -e ".[dev]"

Pre-commit Hooks
~~~~~~~~~~~~~~~

We use pre-commit hooks to ensure code quality:

.. code-block:: bash

   # Install hooks
   pre-commit install

   # Run hooks manually
   pre-commit run --all-files

The hooks will automatically:

- Format code with Black
- Sort imports with isort
- Check for common issues with flake8
- Run basic tests

Testing
-------

Test Structure
~~~~~~~~~~~~~

Our test suite is organized as follows:

.. code-block:: text

   tests/
   ├── conftest.py              # Test fixtures and configuration
   ├── test_core_api.py         # Tests for @experiment decorator and core API
   ├── test_storage.py          # Tests for storage backends
   ├── test_intelligence.py     # Tests for smart features
   ├── test_cli.py              # Tests for command-line tools
   └── test_dashboard.py        # Tests for web dashboard

Running Tests
~~~~~~~~~~~~

.. code-block:: bash

   # Run all tests
   pytest tests/

   # Run specific test file
   pytest tests/test_core_api.py

   # Run with coverage
   pytest tests/ --cov=rexf --cov-report=html

   # Run tests in parallel (faster)
   pytest tests/ -n auto

   # Run only failed tests
   pytest tests/ --lf

Writing Tests
~~~~~~~~~~~~

We use pytest for testing. Here's how to write good tests:

.. code-block:: python

   import pytest
   from rexf import experiment, run

   @experiment
   def test_experiment(param1=42, param2="test"):
       """Test experiment for testing purposes."""
       return {"result": param1 * len(param2)}

   def test_single_experiment_execution(tmp_path):
       """Test that single experiments execute correctly."""
       # Use temporary database for testing
       runner = ExperimentRunner(storage_path=tmp_path / "test.db")
       
       # Run experiment
       run_id = runner.single(test_experiment, param1=10, param2="hello")
       
       # Verify results
       assert run_id is not None
       experiment = runner.get_by_id(run_id)
       assert experiment is not None
       assert experiment.status == "completed"
       assert experiment.metrics["result"] == 50  # 10 * len("hello")
       
       # Cleanup
       runner.close()

   def test_experiment_with_failure():
       """Test that failed experiments are handled properly."""
       @experiment
       def failing_experiment():
           raise ValueError("Test failure")
       
       # Should return run_id even for failed experiments
       run_id = run.single(failing_experiment)
       assert run_id is not None
       
       # Check that failure was recorded
       experiment = run.get_by_id(run_id)
       assert experiment.status == "failed"
       assert "error" in experiment.metadata

Test Guidelines:

1. **Use descriptive names**: ``test_experiment_with_custom_parameters``
2. **Test one thing per test**: Keep tests focused and simple
3. **Use fixtures**: Leverage ``conftest.py`` for shared setup
4. **Clean up resources**: Use temporary paths for databases
5. **Test edge cases**: Include error conditions and boundary values

Code Style and Quality
---------------------

Code Formatting
~~~~~~~~~~~~~~

We use Black for code formatting with these settings:

.. code-block:: bash

   # Format all code
   black .

   # Check formatting without making changes
   black --check .

Import Sorting
~~~~~~~~~~~~~

We use isort for import organization:

.. code-block:: bash

   # Sort imports
   isort .

   # Check import sorting
   isort --check-only .

Linting
~~~~~~

We use flake8 for linting:

.. code-block:: bash

   # Run linter
   flake8 rexf

   # Run with specific rules
   flake8 rexf --count --select=E9,F63,F7,F82 --show-source --statistics

Type Checking
~~~~~~~~~~~~

We use mypy for type checking:

.. code-block:: bash

   # Run type checker
   mypy rexf

   # Check specific file
   mypy rexf/core/simple_api.py

Code Style Guidelines:

1. **Follow PEP 8**: Use Black and flake8 to enforce this
2. **Use type hints**: Add type annotations to public APIs
3. **Write docstrings**: Use Google-style docstrings
4. **Keep functions small**: Aim for functions that do one thing well
5. **Use meaningful names**: Variable and function names should be descriptive

Documentation
------------

Building Documentation
~~~~~~~~~~~~~~~~~~~~~

Documentation is built with Sphinx:

.. code-block:: bash

   # Install documentation dependencies
   pip install -r docs/requirements.txt

   # Build documentation
   cd docs
   make html

   # View documentation
   open _build/html/index.rst

Documentation Guidelines
~~~~~~~~~~~~~~~~~~~~~~

1. **Write clear examples**: Include code examples that work
2. **Update API docs**: Keep docstrings in sync with code
3. **Test examples**: Ensure code examples actually run
4. **Use proper RST syntax**: Follow reStructuredText conventions

Writing Docstrings
~~~~~~~~~~~~~~~~~

Use Google-style docstrings:

.. code-block:: python

   def example_function(param1: str, param2: int = 42) -> dict:
       """
       Brief description of the function.

       Longer description if needed. Explain what the function does,
       any important behavior, and how to use it.

       Args:
           param1: Description of first parameter
           param2: Description of second parameter with default value

       Returns:
           Description of return value

       Raises:
           ValueError: When param1 is empty
           TypeError: When param2 is not an integer

       Example:
           >>> result = example_function("test", 100)
           >>> print(result["value"])
           200
       """
       if not param1:
           raise ValueError("param1 cannot be empty")
       
       return {"value": len(param1) * param2}

Contributing Guidelines
----------------------

What to Contribute
~~~~~~~~~~~~~~~~~

We welcome contributions in these areas:

**Core Features**:
- New intelligence algorithms
- Additional storage backends
- Enhanced query capabilities
- Performance improvements

**Integrations**:
- Support for new ML libraries
- Export formats
- Visualization enhancements
- CLI improvements

**Documentation**:
- Tutorial improvements
- API documentation
- Example experiments
- Use case guides

**Testing**:
- Test coverage improvements
- Performance benchmarks
- Integration tests
- Bug fixes

How to Contribute
~~~~~~~~~~~~~~~~

1. **Start with issues**: Look for issues labeled "good first issue" or "help wanted"
2. **Discuss first**: For large features, open an issue to discuss the approach
3. **Follow conventions**: Use our code style and testing practices
4. **Add tests**: All new features should include tests
5. **Update docs**: Update documentation for user-facing changes

Pull Request Process
~~~~~~~~~~~~~~~~~~~

1. **Create a feature branch**: ``git checkout -b feature/your-feature``
2. **Make focused changes**: Keep PRs small and focused on one thing
3. **Add tests**: Ensure your changes are tested
4. **Update documentation**: Update docs for user-facing changes
5. **Run quality checks**: Ensure all tests and style checks pass
6. **Write good commit messages**: Use clear, descriptive commit messages

Commit Message Format:

.. code-block:: text

   type(scope): brief description

   Longer description if needed. Explain what changed and why.

   Fixes #123

Types: ``feat``, ``fix``, ``docs``, ``style``, ``refactor``, ``test``, ``chore``

Example commit messages:

.. code-block:: text

   feat(intelligence): add adaptive parameter exploration

   Add new adaptive strategy that uses Bayesian optimization
   to suggest next experiments based on previous results.

   Fixes #45

   docs(tutorial): improve Monte Carlo example

   Add better explanations and more detailed code comments
   to make the tutorial easier to follow for beginners.

Code Review Process
------------------

What We Look For
~~~~~~~~~~~~~~~

During code review, we check for:

1. **Correctness**: Does the code work as intended?
2. **Testing**: Are there appropriate tests?
3. **Documentation**: Is the code well-documented?
4. **Style**: Does it follow our conventions?
5. **Performance**: Are there any performance concerns?
6. **Maintainability**: Is the code easy to understand and modify?

Responding to Feedback
~~~~~~~~~~~~~~~~~~~~~

- **Be responsive**: Try to address feedback promptly
- **Ask questions**: If feedback is unclear, ask for clarification
- **Explain decisions**: If you disagree with feedback, explain your reasoning
- **Be open to changes**: We're all learning and improving together

Community Guidelines
--------------------

Code of Conduct
~~~~~~~~~~~~~~

We are committed to providing a welcoming and inclusive environment for all contributors. Please:

- **Be respectful**: Treat others with respect and kindness
- **Be inclusive**: Welcome newcomers and help them learn
- **Be constructive**: Provide helpful feedback and suggestions
- **Be patient**: Remember that everyone has different experience levels

Getting Help
~~~~~~~~~~~

If you need help contributing:

1. **Check the documentation**: Start with this guide and the API docs
2. **Search existing issues**: Your question might already be answered
3. **Ask in discussions**: Use GitHub Discussions for questions
4. **Join our community**: We're here to help!

Recognition
----------

We recognize all contributors in our project:

- **Contributors list**: All contributors are listed in the repository
- **Release notes**: Significant contributions are mentioned in releases
- **Thank you**: We genuinely appreciate all contributions, big and small!

Development Roadmap
------------------

Current Priorities
~~~~~~~~~~~~~~~~~

1. **Performance optimization**: Improving speed for large experiment sets
2. **Advanced analytics**: More sophisticated analysis algorithms
3. **Integration improvements**: Better support for popular ML frameworks
4. **Documentation**: Comprehensive tutorials and examples

Future Directions
~~~~~~~~~~~~~~~~

- **Distributed experiments**: Support for distributed computing
- **Cloud integration**: Better cloud storage and compute support
- **Collaboration features**: Team-based experiment sharing
- **Advanced visualizations**: More sophisticated plotting and analysis tools

Getting Started
--------------

Ready to contribute? Here's how to get started:

1. **Explore the codebase**: Read through the code to understand the structure
2. **Run the examples**: Try the tutorials and examples
3. **Look for issues**: Find something that interests you in our issue tracker
4. **Start small**: Begin with documentation improvements or small bug fixes
5. **Ask questions**: Don't hesitate to reach out if you need help!

Thank you for considering contributing to RexF! Your contributions help make computational research more accessible and reproducible for everyone.
