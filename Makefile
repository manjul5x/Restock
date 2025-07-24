.PHONY: help install install-dev sync sync-dev test lint format clean build run-backtest run-safety-stocks run-simulation run-webapp

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install the package in development mode
	uv sync

install-dev: ## Install the package with development dependencies
	uv sync --extra dev

sync: ## Sync dependencies
	uv sync

sync-dev: ## Sync dependencies with development extras
	uv sync --extra dev

test: ## Run tests
	uv run pytest

test-cov: ## Run tests with coverage
	uv run pytest --cov=forecaster --cov-report=html --cov-report=term-missing

lint: ## Run linting checks
	uv run black --check .
	uv run isort --check-only .
	uv run flake8 .
	uv run mypy forecaster/

format: ## Format code
	uv run black .
	uv run isort .

clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

build: ## Build the package
	uv build

run-backtest: ## Run backtesting
	uv run python run_customer_backtest.py

run-safety-stocks: ## Run safety stock calculation
	uv run python run_safety_stock_calculation.py

run-simulation: ## Run simulation
	uv run python run_simulation.py

run-webapp: ## Run the web application
	uv run python webapp/app.py

dev: install-dev ## Install development dependencies and set up pre-commit
	uv run pre-commit install

check: lint test ## Run all checks (linting and tests) 