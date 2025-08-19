"""
Backtesting module for historical forecasting simulation.
"""

from .config import BacktestConfig

# Legacy imports - commented out to avoid import issues during testing
# from .unified_backtester import UnifiedBacktester, run_unified_backtest

# New pipeline
from .full_backtesting_pipeline import FullBacktestingPipeline, full_backtesting_pipeline

__all__ = [
    'BacktestConfig',
    'FullBacktestingPipeline',
    'full_backtesting_pipeline'
]
