# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2024-08-16

### Added
- Initial release of rexf (Reproducible Experiments Framework)
- Decorator-based API for marking experiment metadata
  - `@experiment()` for defining experiments
  - `@param()` for parameters
  - `@result()` for results
  - `@metric()` for metrics
  - `@artifact()` for artifacts
  - `@seed()` for random seeds
- Automatic reproducibility tracking
  - Random seed management (Python, NumPy, PyTorch)
  - Git commit hash and repository status
  - Environment information capture
  - Execution timestamps
- File-system based storage with SQLite backend
- Intelligent artifact management
  - Automatic type detection
  - File integrity checking with SHA256
  - Metadata storage for artifacts
- Built-in visualization tools
  - Metrics comparison plots
  - Parameter space exploration
  - Experiment timeline visualization
  - Metric correlation matrices
  - Comprehensive dashboards
- Export capabilities
  - JSON and YAML export formats
  - Single experiment or batch export
  - Comparison export functionality
- Extensible design with clear interfaces
- Comprehensive documentation and examples
- Monte Carlo π estimation demo
- Basic functionality test suite

### Technical Details
- Minimum Python version: 3.8
- Dependencies: numpy, matplotlib, pandas, pyyaml, gitpython
- Storage: SQLite for metadata, file system for artifacts
- License: MIT
- Development status: Alpha

[Unreleased]: https://github.com/dhruv1110/rexf/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/dhruv1110/rexf/releases/tag/v0.1.0
