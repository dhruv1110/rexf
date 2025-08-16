# GitHub Actions Setup Guide

This repository includes comprehensive GitHub Actions workflows for continuous integration, security scanning, and automated publishing.

## 🔄 Workflows Overview

### 1. **CI Workflow** (`.github/workflows/ci.yml`)
- **Triggers**: Push to main/dev, Pull Requests
- **Purpose**: Validate code quality, run tests, build package
- **Matrix Testing**: Python 3.8-3.12 on Ubuntu, Windows, macOS

### 2. **Publish Workflow** (`.github/workflows/publish.yml`)
- **Triggers**: Manual dispatch, GitHub releases
- **Purpose**: Build and publish package to PyPI/TestPyPI
- **Environments**: Supports both production and test publishing

### 3. **CodeQL Workflow** (`.github/workflows/codeql.yml`)
- **Triggers**: Push to main, PRs, weekly schedule
- **Purpose**: Security analysis and vulnerability scanning

## 🛠️ Required Setup

### 1. **Repository Secrets**
Navigate to your repository → Settings → Secrets and variables → Actions

Add these secrets:

#### For PyPI Publishing:
```
PYPI_API_TOKEN          # Production PyPI API token
TEST_PYPI_API_TOKEN     # Test PyPI API token (optional)
```

#### Getting PyPI API Tokens:
1. **PyPI Production**:
   - Go to https://pypi.org/manage/account/
   - Navigate to API tokens
   - Create new token with scope for this project
   - Copy token and add as `PYPI_API_TOKEN` secret

2. **Test PyPI** (optional but recommended):
   - Go to https://test.pypi.org/manage/account/
   - Create account if needed
   - Create API token
   - Copy token and add as `TEST_PYPI_API_TOKEN` secret

### 2. **Repository Environments** (Optional but Recommended)
Create environments for additional protection:

1. Go to Settings → Environments
2. Create `pypi` environment:
   - Add protection rules (require reviews)
   - Add `PYPI_API_TOKEN` as environment secret
3. Create `testpypi` environment:
   - Add `TEST_PYPI_API_TOKEN` as environment secret

### 3. **Branch Protection Rules** (Recommended)
1. Go to Settings → Branches
2. Add rule for `main` branch:
   - Require status checks to pass
   - Require branches to be up to date
   - Required status checks:
     - `test (ubuntu-latest, 3.11)`
     - `build`
     - `security`

## 📋 CI Workflow Details

### **Test Job**
- **Code Quality**: Black, isort, flake8, mypy
- **Testing**: pytest with coverage reporting
- **Matrix**: All Python versions (3.8-3.12) × All OS (Ubuntu, Windows, macOS)
- **Coverage**: Uploads to Codecov (requires `CODECOV_TOKEN` secret)

### **Build Job**
- **Package Building**: `python -m build`
- **Package Validation**: `twine check`
- **Artifacts**: Uploads built wheel and source distribution

### **Security Job**
- **Dependency Scanning**: Safety check for known vulnerabilities
- **Code Scanning**: Bandit for security issues
- **Reports**: Uploads security scan results as artifacts

## 🚀 Publishing Workflow

### **Automatic Publishing (Releases)**
1. Create a new release on GitHub
2. Workflow automatically publishes to PyPI
3. Attaches built packages to the release

### **Manual Publishing**
1. Go to Actions tab
2. Select "Publish to PyPI" workflow
3. Click "Run workflow"
4. Choose environment (testpypi/pypi)
5. Click "Run workflow"

### **Publishing Process**
1. **Build**: Creates wheel and source distribution
2. **Validate**: Runs `twine check` for compliance
3. **Test Publish** (if testpypi): Publishes to Test PyPI first
4. **Production Publish**: Publishes to PyPI
5. **Release Assets**: Attaches packages to GitHub release

## 🔒 Security Features

### **CodeQL Analysis**
- **Weekly Scans**: Automated security analysis
- **PR Scans**: Security check on every pull request
- **Vulnerability Database**: Up-to-date security intelligence

### **Dependency Scanning**
- **Safety**: Checks for known vulnerabilities in dependencies
- **Bandit**: Static analysis for common security issues
- **Reports**: Security findings uploaded as artifacts

### **Environment Protection**
- **Required Reviews**: Can require manual approval for production deployments
- **Environment Secrets**: Separate secrets for different environments
- **Branch Protection**: Ensures code quality before merging

## 📊 Workflow Status Badges

Add these badges to your README.md:

```markdown
[![CI](https://github.com/USERNAME/rexf/workflows/CI/badge.svg)](https://github.com/USERNAME/rexf/actions/workflows/ci.yml)
[![CodeQL](https://github.com/USERNAME/rexf/workflows/CodeQL/badge.svg)](https://github.com/USERNAME/rexf/actions/workflows/codeql.yml)
[![PyPI](https://img.shields.io/pypi/v/rexf)](https://pypi.org/project/rexf/)
[![Python Versions](https://img.shields.io/pypi/pyversions/rexf)](https://pypi.org/project/rexf/)
```

## 🐛 Troubleshooting

### **Common Issues**

1. **PyPI Token Issues**:
   - Ensure token has correct scope
   - Check token hasn't expired
   - Verify secret name matches workflow

2. **Test Failures**:
   - Check matrix for specific OS/Python version failures
   - Review detailed logs in Actions tab
   - Ensure all dependencies are properly specified

3. **Build Failures**:
   - Verify pyproject.toml syntax
   - Check MANIFEST.in includes all necessary files
   - Ensure version format is correct

4. **Security Scan Failures**:
   - Review Bandit and Safety reports
   - Update vulnerable dependencies
   - Add security exceptions if needed (carefully)

### **Local Testing**

Before pushing, test locally:

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run quality checks
black --check rexf/
isort --check-only rexf/
flake8 rexf/
mypy rexf/

# Run tests
pytest tests/ -v --cov=rexf

# Build package
python -m build
twine check dist/*
```

## 🎯 Best Practices

1. **Version Management**:
   - Update version in `pyproject.toml` before release
   - Use semantic versioning (MAJOR.MINOR.PATCH)
   - Update CHANGELOG.md for each release

2. **Testing Strategy**:
   - Write tests for new features
   - Maintain high test coverage (>90%)
   - Test on multiple Python versions locally

3. **Security**:
   - Regularly update dependencies
   - Review security scan results
   - Follow security best practices

4. **Release Process**:
   - Test on TestPyPI first
   - Create comprehensive release notes
   - Tag releases appropriately

This setup provides enterprise-grade CI/CD for your Python package! 🚀
