#!/usr/bin/env python3
"""
Safety Stock Calculation Script

This script calculates safety stocks based on forecast comparison results.
"""

import pandas as pd
import sys
import argparse
from datetime import date, timedelta
from pathlib import Path
from forecaster.safety_stocks import SafetyStockCalculator
from forecaster.data import DemandDataLoader


def run_safety_stock_calculation(
    forecast_comparison_file: str,
    product_master_file: str,
    output_dir: str = "output/safety_stocks",
    review_dates: str = None,
    review_interval_days: int = 30
):
    """
    Run safety stock calculations.
    
    Args:
        forecast_comparison_file: Path to forecast comparison CSV
        product_master_file: Path to product master CSV
        output_dir: Output directory for results
        review_dates: Comma-separated list of review dates (YYYY-MM-DD)
        review_interval_days: Days between review dates if not specified
    """
    print("🔍 Safety Stock Calculation")
    print("=" * 50)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("📖 Loading data...")
    
    # Load forecast comparison data directly
    forecast_data = pd.read_csv(forecast_comparison_file)
    print(f"✅ Loaded {len(forecast_data)} forecast comparison records")
    
    # Convert datetime columns to date
    if 'analysis_date' in forecast_data.columns:
        forecast_data['analysis_date'] = pd.to_datetime(forecast_data['analysis_date']).dt.date
    if 'risk_period_end' in forecast_data.columns:
        forecast_data['risk_period_end'] = pd.to_datetime(forecast_data['risk_period_end']).dt.date
    
    # Load product master data directly
    product_master_data = pd.read_csv(product_master_file)
    print(f"✅ Loaded {len(product_master_data)} product master records")
    
    # Generate review dates
    if review_dates:
        # Parse provided review dates
        review_date_list = [date.fromisoformat(d.strip()) for d in review_dates.split(',')]
        print(f"📅 Using {len(review_date_list)} provided review dates")
    else:
        # Use default review dates: 1st, 8th, 15th, 22nd of each month from April 2024 to April 2025
        default_review_dates = [
            "2024-04-01", "2024-04-08", "2024-04-15", "2024-04-22",
            "2024-05-01", "2024-05-08", "2024-05-15", "2024-05-22",
            "2024-06-01", "2024-06-08", "2024-06-15", "2024-06-22",
            "2024-07-01", "2024-07-08", "2024-07-15", "2024-07-22",
            "2024-08-01", "2024-08-08", "2024-08-15", "2024-08-22",
            "2024-09-01", "2024-09-08", "2024-09-15", "2024-09-22",
            "2024-10-01", "2024-10-08", "2024-10-15", "2024-10-22",
            "2024-11-01", "2024-11-08", "2024-11-15", "2024-11-22",
            "2024-12-01", "2024-12-08", "2024-12-15", "2024-12-22",
            "2025-01-01", "2025-01-08", "2025-01-15", "2025-01-22",
            "2025-02-01", "2025-02-08", "2025-02-15", "2025-02-22",
            "2025-03-01", "2025-03-08", "2025-03-15", "2025-03-22",
            "2025-04-01"
        ]
        review_date_list = [date.fromisoformat(d) for d in default_review_dates]
        print(f"📅 Using {len(review_date_list)} default review dates (1st, 8th, 15th, 22nd of each month)")
    
    # Initialize safety stock calculator
    print("🔧 Initializing safety stock calculator...")
    calculator = SafetyStockCalculator(product_master_data)
    
    # Calculate safety stocks
    print("🧮 Calculating safety stocks...")
    safety_stock_results = calculator.calculate_safety_stocks(
        forecast_comparison_data=forecast_data,
        review_dates=review_date_list
    )
    
    # Save results
    output_file = Path(output_dir) / "safety_stock_results.csv"
    
    # Convert errors list to string for CSV storage
    results_for_csv = safety_stock_results.copy()
    results_for_csv['errors'] = results_for_csv['errors'].apply(lambda x: ','.join(map(str, x)) if x else '')
    
    results_for_csv.to_csv(output_file, index=False)
    print(f"💾 Saved results to: {output_file}")
    
    # Display summary
    print("\n📊 Safety Stock Calculation Summary:")
    print(f"  Total calculations: {len(safety_stock_results)}")
    print(f"  Products: {safety_stock_results['product_id'].nunique()}")
    print(f"  Locations: {safety_stock_results['location_id'].nunique()}")
    print(f"  Review dates: {safety_stock_results['review_date'].nunique()}")
    
    # Summary statistics
    print(f"\n📈 Safety Stock Statistics:")
    print(f"  Mean safety stock: {safety_stock_results['safety_stock'].mean():.2f}")
    print(f"  Median safety stock: {safety_stock_results['safety_stock'].median():.2f}")
    print(f"  Min safety stock: {safety_stock_results['safety_stock'].min():.2f}")
    print(f"  Max safety stock: {safety_stock_results['safety_stock'].max():.2f}")
    
    # Error count statistics
    print(f"\n📊 Error Data Statistics:")
    print(f"  Mean error count: {safety_stock_results['error_count'].mean():.1f}")
    print(f"  Median error count: {safety_stock_results['error_count'].median():.1f}")
    print(f"  Min error count: {safety_stock_results['error_count'].min()}")
    print(f"  Max error count: {safety_stock_results['error_count'].max()}")
    
    # Distribution types used
    print(f"\n🔧 Distribution Types:")
    dist_counts = safety_stock_results['distribution_type'].value_counts()
    for dist_type, count in dist_counts.items():
        print(f"  {dist_type}: {count}")
    
    return safety_stock_results


def main():
    """Main function."""
    # Default values for safety stock calculation
    default_forecast_comparison_file = "output/customer_backtest/forecast_comparison.csv"
    default_product_master_file = "forecaster/data/customer_product_master.csv"
    
    parser = argparse.ArgumentParser(description="Calculate safety stocks from forecast comparison results")
    parser.add_argument("forecast_comparison_file", nargs='?', default=default_forecast_comparison_file, 
                       help=f"Path to forecast comparison CSV file (default: {default_forecast_comparison_file})")
    parser.add_argument("product_master_file", nargs='?', default=default_product_master_file,
                       help=f"Path to product master CSV file (default: {default_product_master_file})")
    parser.add_argument("--output-dir", default="output/safety_stocks", help="Output directory (default: output/safety_stocks)")
    parser.add_argument("--review-dates", help="Comma-separated list of review dates (YYYY-MM-DD)")
    parser.add_argument("--review-interval", type=int, default=30, help="Days between review dates (default: 30)")
    
    args = parser.parse_args()
    
    # Show which files are being used
    print("Using default arguments:")
    print(f"  Forecast comparison file: {args.forecast_comparison_file}")
    print(f"  Product master file: {args.product_master_file}")
    print()
    
    # Validate input files
    if not Path(args.forecast_comparison_file).exists():
        print(f"❌ Error: Forecast comparison file not found: {args.forecast_comparison_file}")
        sys.exit(1)
    
    if not Path(args.product_master_file).exists():
        print(f"❌ Error: Product master file not found: {args.product_master_file}")
        sys.exit(1)
    
    try:
        results = run_safety_stock_calculation(
            forecast_comparison_file=args.forecast_comparison_file,
            product_master_file=args.product_master_file,
            output_dir=args.output_dir,
            review_dates=args.review_dates,
            review_interval_days=args.review_interval
        )
        print("\n✅ Safety stock calculation completed successfully!")
    except Exception as e:
        print(f"❌ Error during safety stock calculation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


# =============================================================================
# PERIOD REVIEW DATES - HISTORY
# =============================================================================
# Review dates have been restored to the original weekly schedule (1st, 8th, 15th, 22nd of each month)
# on 2025-07-23 after temporarily using monthly review dates (1st of each month)
#
# Current active review dates (weekly schedule):
# [
#     "2024-04-01", "2024-04-08", "2024-04-15", "2024-04-22",
#     "2024-05-01", "2024-05-08", "2024-05-15", "2024-05-22",
#     "2024-06-01", "2024-06-08", "2024-06-15", "2024-06-22",
#     "2024-07-01", "2024-07-08", "2024-07-15", "2024-07-22",
#     "2024-08-01", "2024-08-08", "2024-08-15", "2024-08-22",
#     "2024-09-01", "2024-09-08", "2024-09-15", "2024-09-22",
#     "2024-10-01", "2024-10-08", "2024-10-15", "2024-10-22",
#     "2024-11-01", "2024-11-08", "2024-11-15", "2024-11-22",
#     "2024-12-01", "2024-12-08", "2024-12-15", "2024-12-22",
#     "2025-01-01", "2025-01-08", "2025-01-15", "2025-01-22",
#     "2025-02-01", "2025-02-08", "2025-02-15", "2025-02-22",
#     "2025-03-01", "2025-03-08", "2025-03-15", "2025-03-22",
#     "2025-04-01"
# ]
#
# Previous monthly review dates (temporarily used):
# [
#     "2024-04-01", "2024-05-01", "2024-06-01", "2024-07-01", "2024-08-01", "2024-09-01",
#     "2024-10-01", "2024-11-01", "2024-12-01", "2025-01-01", "2025-02-01", "2025-03-01", "2025-04-01"
# ]
# ============================================================================= 