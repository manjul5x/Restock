# Migration Guide: From pip/requirements.txt to uv

This guide helps you migrate from the old pip-based setup to the new uv-based setup.

## What Changed

### Old Setup (pip)
- `requirements.txt` - Dependencies list
- `setup.py` - Package configuration
- Manual virtual environment management
- `pip install -r requirements.txt`

### New Setup (uv)
- `pyproject.toml` - Modern Python packaging configuration
- `uv.lock` - Locked dependency versions
- Automatic virtual environment management
- `uv sync` - Install dependencies

## Migration Steps

### 1. Install uv (if not already installed)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Remove old virtual environment (if exists)
```bash
rm -rf venv/
rm -rf .venv/  # if using python -m venv
```

### 3. Install dependencies with uv
```bash
# Install production dependencies
uv sync

# Install development dependencies (optional)
uv sync --extra dev
```

### 4. Update your workflow

#### Old commands:
```bash
python run_customer_backtest.py
python run_safety_stock_calculation.py
python run_simulation.py
python webapp/app.py
```

#### New commands:
```bash
uv run python run_customer_backtest.py
uv run python run_safety_stock_calculation.py
uv run python run_simulation.py
uv run python webapp/app.py
```

#### Or use Makefile shortcuts:
```bash
make run-backtest
make run-safety-stocks
make run-simulation
make run-webapp
```

## Development Workflow

### Code Quality Tools
The new setup includes modern development tools:

```bash
# Format code
make format

# Run linting
make lint

# Run tests
make test

# Run all checks
make check
```

### Pre-commit Hooks
Set up automatic code quality checks:

```bash
make dev
```

This will install pre-commit hooks that run automatically on `git commit`.

## Benefits of the New Setup

1. **Faster**: uv is significantly faster than pip
2. **Reliable**: Locked dependencies ensure reproducible builds
3. **Modern**: Uses current Python packaging standards
4. **Developer-friendly**: Built-in code quality tools
5. **Simplified**: Automatic virtual environment management

## Troubleshooting

### If you get import errors
Make sure you're using `uv run` before your Python commands:
```bash
# ❌ Wrong
python script.py

# ✅ Correct
uv run python script.py
```

### If dependencies are missing
Sync dependencies again:
```bash
uv sync
```

### If you need to add a new dependency
Edit `pyproject.toml` and run:
```bash
uv sync
```

## File Changes Summary

### Removed Files
- `requirements.txt` - Replaced by `pyproject.toml`
- `setup.py` - Replaced by `pyproject.toml`

### New Files
- `pyproject.toml` - Modern Python project configuration
- `uv.lock` - Locked dependency versions
- `Makefile` - Development command shortcuts
- `.pre-commit-config.yaml` - Code quality hooks
- `MIGRATION_GUIDE.md` - This guide

### Updated Files
- `README.md` - Updated with uv instructions 