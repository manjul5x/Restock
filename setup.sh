#!/bin/bash

# Inventory Forecasting & Analysis System Setup Script
# This script sets up the development environment for new users

set -e  # Exit on any error

echo "🚀 Setting up Inventory Forecasting & Analysis System..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "✅ uv installed successfully!"
else
    echo "✅ uv is already installed"
fi

# Install dependencies
echo "📚 Installing dependencies..."
uv sync

echo "🔧 Setting up development environment..."
uv sync --extra dev

echo "🎯 Setting up pre-commit hooks..."
uv run pre-commit install

echo "✅ Setup complete!"
echo ""
echo "🎉 You're ready to start! Try these commands:"
echo "  • make run-backtest     - Run backtesting"
echo "  • make run-safety-stocks - Calculate safety stocks"
echo "  • make run-simulation    - Run inventory simulation"
echo "  • make run-webapp        - Start the web interface"
echo "  • make check            - Run all quality checks"
echo ""
echo "📖 For more information, see QUICK_START_GUIDE.md" 