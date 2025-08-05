#!/usr/bin/env python3
"""
Unified Backtesting Script

Script to run unified backtesting using the new UnifiedBacktester with CoreForecastingEngine.
This system uses method-specific parameter optimization (once per product) and supports
multiple forecasting methods with extensible framework.
"""

from forecaster.backtesting.config import BacktestConfig
from forecaster.backtesting.unified_backtester import (
    run_unified_backtest as run_unified_backtest_func,
)
from data.loader import DataLoader
from datetime import date
import sys
import os
from datetime import date, timedelta


def calculate_analysis_dates(loader, demand_frequency: str = "d") -> tuple[str, str]:
    """
    Calculate the analysis start and end dates based on the max ss_window_length and review dates.
    
    Args:
        loader: DataLoader instance
        demand_frequency: Demand frequency ('d', 'w', 'm')
    
    Returns:
        Tuple of (analysis_start_date, analysis_end_date) in YYYY-MM-DD format
        
    Raises:
        ValueError: If unable to calculate the dates (e.g., missing review dates or product master data)
    """
    try:
        # Load product master to get max ss_window_length
        product_master = loader.load_product_master()
        max_ss_window_length = product_master['ss_window_length'].max()
        
        # Get review dates from config
        review_dates = loader.config.get('safety_stock', {}).get('review_dates', [])
        if not review_dates:
            raise ValueError("No review dates found in config")
        
        first_review_date = date.fromisoformat(review_dates[0])
        last_review_date = date.fromisoformat(review_dates[-1])
        
        # Calculate days to subtract based on demand frequency and ss_window_length
        if demand_frequency == "d":
            days_to_subtract = int(max_ss_window_length)
        elif demand_frequency == "w":
            days_to_subtract = int(max_ss_window_length * 7)
        elif demand_frequency == "m":
            days_to_subtract = int(max_ss_window_length * 30)  # Approximate
        else:
            raise ValueError(f"Unsupported demand frequency: {demand_frequency}")
        
        # Calculate analysis start date
        analysis_start_date = first_review_date - timedelta(days=days_to_subtract)
        
        return analysis_start_date.isoformat(), last_review_date.isoformat()
        
    except Exception as e:
        raise ValueError(f"Failed to calculate analysis dates: {e}")


def run_unified_backtest_script(
    analysis_start_date: str = None,
    analysis_end_date: str = None,
    demand_frequency: str = "d",
    max_workers: int = 8,
    batch_size: int = 20,
    log_level: str = "INFO",
):
    """Run unified backtesting on customer data with method-specific parameter optimization."""

    # Initialize DataLoader to validate configuration
    try:
        loader = DataLoader()
        print("✅ DataLoader initialized successfully")
    except Exception as e:
        print(f"❌ DataLoader initialization failed: {e}")
        return False

    # Calculate analysis dates if not provided
    if analysis_start_date is None or analysis_end_date is None:
        print("📊 Calculating analysis dates based on max ss_window_length and review dates...")
        try:
            calculated_start, calculated_end = calculate_analysis_dates(loader, demand_frequency)
            if analysis_start_date is None:
                analysis_start_date = calculated_start
            if analysis_end_date is None:
                analysis_end_date = calculated_end
        except Exception as e:
            print(f"❌ Failed to calculate analysis dates: {e}")
            print("   Please provide analysis dates using --analysis-start-date and --analysis-end-date parameters")
            return False
    else:
        print(f"📅 Using provided analysis start date: {analysis_start_date}")
        print(f"📅 Using provided analysis end date: {analysis_end_date}")

    # Validate required parameters
    if not analysis_start_date:
        print("❌ Error: analysis_start_date is required")
        return False

    if not analysis_end_date:
        print("❌ Error: analysis_end_date is required")
        return False

    # Parse dates
    try:
        start_date = date.fromisoformat(analysis_start_date)
        end_date = date.fromisoformat(analysis_end_date)
    except ValueError as e:
        print(f"❌ Invalid date format: {e}")
        print("Use YYYY-MM-DD format (e.g., 2023-01-01)")
        return False

    # Initialize DataLoader to validate configuration
    try:
        loader = DataLoader()
        print("✅ DataLoader initialized successfully")
    except Exception as e:
        print(f"❌ DataLoader initialization failed: {e}")
        return False

    # Configuration
    config = BacktestConfig(
        # Backtesting parameters
        analysis_start_date=start_date,
        analysis_end_date=end_date,
        # Forecasting parameters
        demand_frequency=demand_frequency,
        forecast_model="unified",  # Use unified forecasting system
        default_horizon=1,
        # Processing settings
        batch_size=batch_size,
        max_workers=max_workers,
        validate_data=True,
        aggregation_enabled=True,
        # Logging
        log_level=log_level,
    )

    # Validate configuration
    if not config.validate_dates():
        print("❌ Invalid date configuration")
        return False
        return False

    print("=" * 70)
    print("🚀 Unified Backtesting")
    print("=" * 70)
    print(f"📅 Analysis Start Date: {start_date}")
    print(f"📅 Analysis End Date: {end_date}")
    print(f"🔄 Demand Frequency: {demand_frequency}")
    print(f"⚙️ Batch Size: {batch_size}")
    print(f"🚀 Max Workers: {max_workers}")
    print("=" * 70)
    print("⏱️  Starting backtesting process...")
    print("📊 Progress tracking and logging enabled")
    print("=" * 70)

    try:
        # Run unified backtesting
        result = run_unified_backtest_func(config)

        # The result is the summary itself, not a dict with success field
        if result:
            print("✅ Unified backtesting completed successfully!")
            return True
        else:
            print("❌ Unified backtesting failed")
            return False

    except Exception as e:
        print(f"❌ Unified backtesting failed with error: {e}")
        return False


def main():
    """Main function to run unified backtesting from command line."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run unified backtesting with method-specific parameter optimization"
    )

    # Analysis period arguments (optional - will auto-calculate if not provided)
    parser.add_argument(
        "--analysis-start-date",
        default=None,
        help="Analysis start date (YYYY-MM-DD format). If not provided, will be calculated based on max ss_window_length and first review date.",
    )
    parser.add_argument(
        "--analysis-end-date",
        default=None,
        help="Analysis end date (YYYY-MM-DD format). If not provided, will be set to the last review date.",
    )

    # Optional arguments
    parser.add_argument(
        "--demand-frequency",
        default="d",
        choices=["d", "w", "m"],
        help="Demand frequency (d=daily, w=weekly, m=monthly)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Maximum number of parallel workers",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Batch size for processing",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Run unified backtesting
    success = run_unified_backtest_script(
        analysis_start_date=args.analysis_start_date,
        analysis_end_date=args.analysis_end_date,
        demand_frequency=args.demand_frequency,
        max_workers=args.max_workers,
        batch_size=args.batch_size,
        log_level=args.log_level,
    )

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
