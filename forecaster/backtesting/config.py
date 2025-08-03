"""
Configuration for the backtesting module.
"""

from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
from datetime import date, timedelta
import os


@dataclass
class BacktestConfig:
    """Configuration for backtesting runs."""

    # Output directory (will be set by DataLoader)

    # Backtesting parameters
    analysis_start_date: date = field(default_factory=lambda: date(2024, 4, 1))
    analysis_end_date: date = field(default_factory=lambda: date(2025, 4, 1))

    # Forecasting parameters
    demand_frequency: str = "d"  # 'd' for daily, 'w' for weekly, 'm' for monthly
    forecast_model: str = "moving_average"
    default_horizon: int = 1  # Default horizon for backtesting

    # Processing settings
    batch_size: int = 10
    max_workers: int = field(
        default_factory=lambda: os.cpu_count() or 4
    )  # Use max CPU cores
    validate_data: bool = True
    aggregation_enabled: bool = True

    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None

    def __post_init__(self):
        """Post-initialization setup."""
        # Initialize DataLoader
        from data.loader import DataLoader
        self.loader = DataLoader()

        # Set log file name if not provided
        if self.log_file is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file = str(self.loader.get_output_path("backtesting", f"backtest_run_{timestamp}.log"))

    def get_demand_file_path(self) -> Path:
        """Get full path to demand file."""
        return self.loader.get_output_path("customer_data", "customer_demand.csv")

    def get_product_master_file_path(self) -> Path:
        """Get full path to product master file."""
        return self.loader.get_output_path("customer_data", "customer_product_master.csv")

    def get_backtest_results_path(self) -> Path:
        """Get path for backtest results file."""
        return self.loader.get_output_path("backtesting", "backtest_results.csv")

    def get_accuracy_metrics_path(self) -> Path:
        """Get path for accuracy metrics file."""
        return self.loader.get_output_path("backtesting", "accuracy_metrics.csv")

    def get_forecast_comparison_path(self) -> Path:
        """Get path for forecast vs actual comparison file."""
        filename = self.loader.config['paths']['output_files']['forecast_comparison']
        return self.loader.get_output_path("backtesting", filename)

    def get_forecast_visualization_path(self) -> Path:
        """Get path for enhanced forecast visualization data file."""
        filename = self.loader.config['paths']['output_files']['forecast_visualization']
        return self.loader.get_output_path("backtesting", filename)

    def validate_dates(self) -> bool:
        """Validate that the dates are in correct order."""
        return (
            self.analysis_start_date
            <= self.analysis_end_date
        )

    def get_analysis_dates(self) -> list[date]:
        """Get list of dates to analyze based on demand frequency."""
        dates = []
        current_date = self.analysis_start_date

        # Calculate step size based on demand frequency
        if self.demand_frequency == "d":
            step_days = 1
        elif self.demand_frequency == "w":
            step_days = 7
        elif self.demand_frequency == "m":
            step_days = 30  # Approximate
        else:
            raise ValueError(f"Unsupported demand frequency: {self.demand_frequency}")

        # Calculate end date for analysis (risk_period + horizon steps before analysis_end_date)
        # We'll get the actual risk period from the product master data
        # For now, use a conservative estimate
        max_steps_back = 30  # Conservative estimate
        analysis_end = self.analysis_end_date - timedelta(days=max_steps_back)

        # Ensure we have at least some dates to analyze
        if analysis_end < current_date:
            # If the range is too short, just use the analysis_end_date
            analysis_end = self.analysis_end_date

        while current_date <= analysis_end:
            dates.append(current_date)
            current_date += timedelta(days=step_days)

        return dates
