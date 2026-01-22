# Development Guide

This project uses **Hatch** for build/packaging/environments, **uv** for fast dependency management, and **ruff** for linting/formatting.

## Prerequisites

1. **Python 3.10+** (Python 3.12 recommended, specified in `.python-version`)
2. **uv** - Fast Python package installer
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
3. **Hatch** - Modern Python project manager
   ```bash
   uv tool install hatch
   # or: pipx install hatch
   ```

## Quick Start

### Install the package in development mode

```bash
# Using uv (recommended)
uv pip install --editable .

# Or using pip
pip install -e .
```

### Install with all optional dependencies

```bash
uv pip install --editable ".[dev]"
# or for all LLM/vector/framework instrumentations:
uv pip install --editable ".[all,dev]"
```

### Using uv's lock file

```bash
# Sync dependencies from uv.lock
uv sync

# Update lock file
uv lock

# Install with specific extras
uv sync --extra dev --extra openai
```

## Development Workflows

Hatch provides convenient commands for common development tasks:

### Running Tests

```bash
# Run all tests with coverage
hatch run test

# Run tests verbosely
hatch run test-verbose

# Run tests on specific Python version
hatch run +py=3.10 test
hatch run +py=3.11 test
hatch run +py=3.12 test
hatch run +py=3.13 test
hatch run +py=3.14 test

# Run tests across all Python versions (matrix)
hatch run test:run
```

### Code Quality

```bash
# Check code with ruff
hatch run lint

# Auto-fix linting issues
hatch run lint-fix

# Format code with ruff
hatch run fmt

# Check formatting without modifying
hatch run fmt-check

# Type check with mypy
hatch run typecheck

# Run all quality checks + tests
hatch run all
```

### Building

```bash
# Build wheel and source distribution
hatch build

# Build only wheel
hatch build --target wheel

# Build only source distribution
hatch build --target sdist

# Build to specific directory
hatch build --outdir dist/
```

### Version Management

Hatch can automatically manage versions in `basalt/_version.py`:

```bash
# Show current version
hatch version

# Bump patch version (1.1.0 -> 1.1.1)
hatch version patch

# Bump minor version (1.1.0 -> 1.2.0)
hatch version minor

# Bump major version (1.1.0 -> 2.0.0)
hatch version major

# Set specific version
hatch version 1.2.3
```

### Publishing to PyPI

```bash
# Build and publish to PyPI (requires credentials)
hatch publish

# Build and publish to TestPyPI
hatch publish -r test

# Dry run (build without publishing)
hatch build
```

## Hatch Environments

Hatch creates isolated virtual environments for different purposes:

- **default** - Development environment with pytest, ruff, mypy
- **test** - Matrix testing across Python 3.10-3.14
- **full** - Environment with all optional dependencies for comprehensive testing

You can access environments directly:

```bash
# Enter a shell in the default environment
hatch shell

# Run a command in a specific environment
hatch run full:test

# Show all environments
hatch env show
```

## Project Structure

```
basalt-python/
├── basalt/              # Main package source
│   ├── __init__.py
│   ├── _version.py      # Version file (managed by Hatch)
│   ├── client.py
│   └── ...
├── tests/               # Test suite
├── docs/                # Documentation
├── examples/            # Example scripts
├── pyproject.toml       # Project metadata and configuration
├── uv.lock              # Locked dependencies
├── .python-version      # Python version for uv
└── README.md
```

## CI/CD

GitHub Actions automatically:
- Runs tests on Python 3.10, 3.11, 3.12, 3.13, 3.14
- Uses uv for fast dependency installation
- Uses hatch for running test suite
- Generates coverage reports

## Optional Dependencies

The SDK provides many optional instrumentation packages:

### LLM Providers
```bash
pip install basalt-sdk[openai]
pip install basalt-sdk[anthropic]
pip install basalt-sdk[google-generativeai]
pip install basalt-sdk[bedrock]
pip install basalt-sdk[vertex-ai]
pip install basalt-sdk[mistralai]
# Or all at once:
pip install basalt-sdk[llm-all]
```

### Vector Databases
```bash
pip install basalt-sdk[chromadb]
pip install basalt-sdk[pinecone]
pip install basalt-sdk[qdrant]
# Or all at once:
pip install basalt-sdk[vector-all]
```

### Frameworks
```bash
pip install basalt-sdk[langchain]
pip install basalt-sdk[llamaindex]
# Or all at once:
pip install basalt-sdk[framework-all]
```

### Everything
```bash
pip install basalt-sdk[all]
```

## Troubleshooting

### Clean build artifacts

```bash
# Remove all build artifacts
rm -rf dist/ build/ *.egg-info .hatch/

# Rebuild from scratch
hatch build
```

### Reset environments

```bash
# Remove all Hatch environments
hatch env prune

# Recreate default environment
hatch env create
```

### Lock file issues

```bash
# Regenerate uv.lock
uv lock --upgrade

# Force reinstall all dependencies
uv sync --reinstall
```

## Additional Resources

- [Hatch Documentation](https://hatch.pypa.io/)
- [uv Documentation](https://docs.astral.sh/uv/)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
