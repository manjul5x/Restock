#!/usr/bin/env python3
"""
Customer Backtesting Script

Simple script to run backtesting on customer data with minimal configuration.
"""

from forecaster.backtesting.config import BacktestConfig
from forecaster.backtesting.backtester import Backtester
from datetime import date
import sys
import os


def run_customer_backtest(
    data_dir: str,
    demand_file: str,
    product_master_file: str,
    output_dir: str = "output/customer_backtest",
    analysis_start_date: str = "2023-10-30",
    analysis_end_date: str = "2025-04-01",
    historic_start_date: str = None,
    demand_frequency: str = "d",
    max_workers: int = 4,
    batch_size: int = 20
):
    """Run backtesting on customer data."""
    
    # Parse dates
    try:
        start_date = date.fromisoformat(analysis_start_date)
        end_date = date.fromisoformat(analysis_end_date)
        
        # Parse historic start date
        if historic_start_date:
            historic_start = date.fromisoformat(historic_start_date)
        else:
            historic_start = start_date.replace(year=start_date.year - 1)  # Default: 1 year before analysis start
    except ValueError as e:
        print(f"❌ Invalid date format: {e}")
        print("Use YYYY-MM-DD format (e.g., 2023-01-01)")
        return False
    
    # Configuration
    config = BacktestConfig(
        # Data paths
        data_dir=data_dir,
        demand_file=demand_file,
        product_master_file=product_master_file,
        output_dir=output_dir,
        
        # Backtesting parameters
        historic_start_date=historic_start,
        analysis_start_date=start_date,
        analysis_end_date=end_date,
        
        # Forecasting parameters
        demand_frequency=demand_frequency,
        forecast_model="moving_average",
        default_horizon=1,
        
        # Processing settings
        batch_size=batch_size,
        max_workers=max_workers,
        validate_data=True,
        outlier_enabled=True,
        aggregation_enabled=True,
        
        # Logging
        log_level="INFO"
    )
    
    # Validate configuration
    if not config.validate_dates():
        print("❌ Invalid date configuration")
        return False
    
    # Check if data files exist
    demand_path = config.get_demand_file_path()
    product_master_path = config.get_product_master_file_path()
    
    if not demand_path.exists():
        print(f"❌ Demand file not found: {demand_path}")
        return False
    
    if not product_master_path.exists():
        print(f"❌ Product master file not found: {product_master_path}")
        return False
    
    print("=" * 60)
    print("CUSTOMER BACKTESTING")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Demand file: {demand_file}")
    print(f"Product master file: {product_master_file}")
    print(f"Output directory: {output_dir}")
    print(f"Historic start date: {historic_start_date or 'Auto-calculated'}")
    print(f"Analysis period: {analysis_start_date} to {analysis_end_date}")
    print(f"Demand frequency: {demand_frequency}")
    print(f"Max workers: {max_workers}")
    print(f"Batch size: {batch_size}")
    print("=" * 60)
    
    try:
        # Run backtesting
        print("Starting backtesting...")
        backtester = Backtester(config)
        results = backtester.run()
        
        print()
        print("=" * 60)
        print("✅ BACKTESTING COMPLETED SUCCESSFULLY")
        print("=" * 60)
        # Get total forecasts from the correct location in the summary
        total_forecasts = 0
        if 'accuracy_summary' in results and 'total_forecasts' in results['accuracy_summary']:
            total_forecasts = results['accuracy_summary']['total_forecasts']
        elif 'data_summary' in results and 'backtest_results' in results['data_summary']:
            total_forecasts = results['data_summary']['backtest_results']
        
        print(f"Generated {total_forecasts} forecasts")
        print(f"Results saved to: {output_dir}")
        
        # Show accuracy summary if available
        if 'accuracy_summary' in results:
            summary = results['accuracy_summary']
            print(f"Mean MAE: {summary.get('mean_mae', 'N/A'):.2f}")
            print(f"Mean MAPE: {summary.get('mean_mape', 'N/A'):.2f}%")
            print(f"Mean RMSE: {summary.get('mean_rmse', 'N/A'):.2f}")
            print(f"Mean Bias: {summary.get('mean_bias', 'N/A'):.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Backtesting failed: {e}")
        return False


def main():
    """Main function with command line interface."""
    # Default values for customer backtesting
    default_data_dir = "forecaster/data"
    default_demand_file = "customer_demand.csv"
    default_product_master_file = "customer_product_master.csv"
    
    # Parse arguments with defaults
    if len(sys.argv) < 2:
        # No arguments provided, use defaults
        data_dir = default_data_dir
        demand_file = default_demand_file
        product_master_file = default_product_master_file
        print("Using default arguments:")
        print(f"  Data directory: {data_dir}")
        print(f"  Demand file: {demand_file}")
        print(f"  Product master file: {product_master_file}")
        print()
    elif len(sys.argv) < 4:
        print("Usage: python run_customer_backtest.py [<data_dir> <demand_file> <product_master_file>] [options]")
        print()
        print("Arguments (all optional with defaults):")
        print("  data_dir              Directory containing your data files (default: forecaster/data)")
        print("  demand_file           Name of demand CSV file (default: customer_demand.csv)")
        print("  product_master_file   Name of product master CSV file (default: customer_product_master.csv)")
        print()
        print("Optional arguments:")
        print("  --output-dir DIR      Output directory (default: output/customer_backtest)")
        print("  --start-date DATE     Analysis start date YYYY-MM-DD (default: 2023-10-30)")
        print("  --end-date DATE       Analysis end date YYYY-MM-DD (default: 2025-04-01)")
        print("  --historic-start DATE Historic start date YYYY-MM-DD (default: 1 year before analysis start)")
        print("  --frequency FREQ      Demand frequency: d/w/m (default: d)")
        print("  --workers N           Number of parallel workers (default: 8)")
        print("  --batch-size N        Batch size for processing (default: 20)")
        print()
        print("Examples:")
        print("  python run_customer_backtest.py                                    # Use all defaults")
        print("  python run_customer_backtest.py customer_data demand.csv product_master.csv")
        print("  python run_customer_backtest.py --start-date 2023-06-01 --end-date 2023-08-31")
        sys.exit(1)
    else:
        # Arguments provided
        data_dir = sys.argv[1]
        demand_file = sys.argv[2]
        product_master_file = sys.argv[3]
    
    # Default values
    output_dir = "output/customer_backtest"
    analysis_start_date = "2023-10-30"
    analysis_end_date = "2025-04-01"
    historic_start_date = None
    demand_frequency = "d"
    max_workers = 8
    batch_size = 20
    
    # Parse optional arguments
    i = 4 if len(sys.argv) >= 4 else 1
    while i < len(sys.argv):
        if sys.argv[i] == "--output-dir" and i + 1 < len(sys.argv):
            output_dir = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--start-date" and i + 1 < len(sys.argv):
            analysis_start_date = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--end-date" and i + 1 < len(sys.argv):
            analysis_end_date = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--historic-start" and i + 1 < len(sys.argv):
            historic_start_date = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--frequency" and i + 1 < len(sys.argv):
            demand_frequency = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--workers" and i + 1 < len(sys.argv):
            max_workers = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--batch-size" and i + 1 < len(sys.argv):
            batch_size = int(sys.argv[i + 1])
            i += 2
        else:
            print(f"❌ Unknown argument: {sys.argv[i]}")
            sys.exit(1)
    
    # Validate demand frequency
    if demand_frequency not in ['d', 'w', 'm']:
        print("❌ Invalid demand frequency. Use 'd' (daily), 'w' (weekly), or 'm' (monthly)")
        sys.exit(1)
    
    # Run backtesting
    success = run_customer_backtest(
        data_dir=data_dir,
        demand_file=demand_file,
        product_master_file=product_master_file,
        output_dir=output_dir,
        analysis_start_date=analysis_start_date,
        analysis_end_date=analysis_end_date,
        historic_start_date=historic_start_date,
        demand_frequency=demand_frequency,
        max_workers=max_workers,
        batch_size=batch_size
    )
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main() 