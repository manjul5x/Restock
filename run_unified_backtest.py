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


def run_unified_backtest_script(
    analysis_start_date: str = None,
    analysis_end_date: str = None,
    demand_frequency: str = "d",
    max_workers: int = 8,
    batch_size: int = 20,
    outlier_enabled: bool = True,
    log_level: str = "INFO",
):
    """Run unified backtesting on customer data with method-specific parameter optimization."""

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
        outlier_enabled=outlier_enabled,
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
    print(f"🔍 Outlier Handling: {'Enabled' if outlier_enabled else 'Disabled'}")
    print("=" * 70)
    print("⏱️  Starting backtesting process...")
    print("📊 Progress tracking and logging enabled")
    print("=" * 70)

    try:
        # Run unified backtesting
        result = run_unified_backtest_func(config)

        # The result is the summary itself, not a dict with success field
        print("\n✅ Unified backtesting completed successfully!")

        # Print summary
        if result:
            print(f"⏱️  Total time: {result.get('execution_time', 0):.2f} seconds")
            print(
                f"📊 Total forecasts: {result.get('results_summary', {}).get('total_forecasts', 0)}"
            )
            print(
                f"📈 Total comparisons: {result.get('results_summary', {}).get('total_comparisons', 0)}"
            )
            print(
                f"🔧 Products optimized: {result.get('results_summary', {}).get('products_optimized', 0)}"
            )

        return True

    except Exception as e:
        print(f"❌ Error during unified backtesting: {e}")
        return False

    except Exception as e:
        print(f"❌ Error during unified backtesting: {e}")
        return False


def main():
    """Main function for command line usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run unified backtesting with method-specific parameter optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with required parameters
  python run_unified_backtest.py --analysis-start-date 2024-01-01 --analysis-end-date 2024-12-01

  # With custom processing settings
  python run_unified_backtest.py --analysis-start-date 2024-01-01 --analysis-end-date 2024-12-01 --max-workers 8 --batch-size 20



  # Without outlier handling
  python run_unified_backtest.py --analysis-start-date 2024-01-01 --analysis-end-date 2024-12-01 --no-outliers

  # With custom log level for progress tracking
  python run_unified_backtest.py --analysis-start-date 2024-01-01 --analysis-end-date 2024-12-01 --log-level DEBUG

  # Note: For automatic date calculation, use run_complete_workflow.py instead
        """,
    )

    # Required arguments
    parser.add_argument(
        "--analysis-start-date",
        required=True,
        help="Analysis start date (YYYY-MM-DD format)",
    )
    parser.add_argument(
        "--analysis-end-date",
        required=True,
        help="Analysis end date (YYYY-MM-DD format)",
    )

    # Note: Output directory is now handled by DataLoader configuration
    parser.add_argument(
        "--demand-frequency",
        default="d",
        choices=["d", "w", "m"],
        help="Demand frequency: 'd' for daily, 'w' for weekly, 'm' for monthly (default: d)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Batch size for processing (default: 20)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Maximum number of parallel workers (default: 8)",
    )
    parser.add_argument(
        "--no-outliers", action="store_true", help="Disable outlier handling"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Run unified backtesting
    success = run_unified_backtest_script(
        analysis_start_date=args.analysis_start_date,
        analysis_end_date=args.analysis_end_date,
        demand_frequency=args.demand_frequency,
        max_workers=args.max_workers,
        batch_size=args.batch_size,
        outlier_enabled=not args.no_outliers,
        log_level=args.log_level,
    )

    if success:
        print("\n🎉 Unified backtesting completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Unified backtesting failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
