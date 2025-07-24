#!/usr/bin/env python3
"""
Safety Stock Calculation Script

This script calculates safety stock levels based on forecast comparison results.
It uses the forecast errors to determine appropriate safety stock levels for each product-location.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import argparse
from datetime import date
from forecaster.safety_stocks.safety_stock_calculator import SafetyStockCalculator


def run_safety_stock_calculation(
    forecast_comparison_file: str,
    product_master_file: str,
    output_dir: str = "output/safety_stocks",
    review_dates: str = None,
    review_interval_days: int = 30
):
    """
    Calculate safety stocks from forecast comparison results.
    
    Args:
        forecast_comparison_file: Path to forecast comparison CSV file
        product_master_file: Path to product master CSV file
        output_dir: Output directory for results
        review_dates: Comma-separated list of review dates (YYYY-MM-DD)
        review_interval_days: Days between review dates if not provided
    
    Returns:
        DataFrame with safety stock results
    """
    
    print("🔍 Safety Stock Calculation")
    print("=" * 50)
    print(f"Forecast comparison file: {forecast_comparison_file}")
    print(f"Product master file: {product_master_file}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n📊 Loading data...")
    try:
        forecast_data = pd.read_csv(forecast_comparison_file)
        product_master_data = pd.read_csv(product_master_file)
        
        print(f"✅ Loaded {len(forecast_data)} forecast comparison records")
        print(f"✅ Loaded {len(product_master_data)} product master records")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None
    
    # Handle column name mapping for different backtesting outputs
    print("\n🔧 Checking column mappings...")
    if 'error' in forecast_data.columns and 'forecast_error' not in forecast_data.columns:
        print("📝 Mapping 'error' column to 'forecast_error' for compatibility")
        forecast_data['forecast_error'] = forecast_data['error']
    elif 'forecast_error' not in forecast_data.columns:
        print("❌ No error column found. Available columns:")
        print(f"   {list(forecast_data.columns)}")
        return None
    
    # Handle product master column mappings
    if 'leadtime' in product_master_data.columns and 'lead_time' not in product_master_data.columns:
        print("📝 Mapping 'leadtime' column to 'lead_time' for compatibility")
        product_master_data['lead_time'] = product_master_data['leadtime']
    elif 'lead_time' not in product_master_data.columns:
        print("❌ No lead_time column found. Available columns:")
        print(f"   {list(product_master_data.columns)}")
        return None
    
    # Validate required columns
    required_forecast_columns = ['analysis_date', 'product_id', 'location_id', 'actual_demand', 'forecast_demand', 'forecast_error']
    missing_forecast_columns = [col for col in required_forecast_columns if col not in forecast_data.columns]
    
    if missing_forecast_columns:
        print(f"❌ Missing required columns in forecast comparison file: {missing_forecast_columns}")
        print(f"Available columns: {list(forecast_data.columns)}")
        return None
    
    required_product_columns = ['product_id', 'location_id', 'lead_time', 'risk_period']
    missing_product_columns = [col for col in required_product_columns if col not in product_master_data.columns]
    
    if missing_product_columns:
        print(f"❌ Missing required columns in product master file: {missing_product_columns}")
        return None
    
    # Process review dates
    print("\n📅 Processing review dates...")
    if review_dates:
        review_date_list = [date.fromisoformat(d.strip()) for d in review_dates.split(',')]
        print(f"📅 Using {len(review_date_list)} provided review dates")
    else:
        # Generate review dates based on interval
        print(f"📅 Generating review dates with {review_interval_days}-day intervals")
        
        # Get date range from forecast data
        forecast_data['analysis_date'] = pd.to_datetime(forecast_data['analysis_date'])
        min_date = forecast_data['analysis_date'].min().date()
        max_date = forecast_data['analysis_date'].max().date()
        
        # Generate review dates
        review_date_list = []
        current_date = min_date
        while current_date <= max_date:
            review_date_list.append(current_date)
            current_date = current_date.replace(day=current_date.day + review_interval_days)
        
        print(f"📅 Generated {len(review_date_list)} review dates from {min_date} to {max_date}")
    
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
    parser = argparse.ArgumentParser(
        description="Calculate safety stocks from forecast comparison results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with required parameters
  python run_safety_stock_calculation.py forecast_comparison.csv product_master.csv
  
  # Run with custom review dates
  python run_safety_stock_calculation.py forecast_comparison.csv product_master.csv --review-dates "2024-01-01,2024-02-01,2024-03-01"
  
  # Run with custom review interval
  python run_safety_stock_calculation.py forecast_comparison.csv product_master.csv --review-interval 14
  
  # Run with custom output directory
  python run_safety_stock_calculation.py forecast_comparison.csv product_master.csv --output-dir output/my_safety_stocks
        """
    )
    
    # Required arguments
    parser.add_argument("forecast_comparison_file", 
                       help="Path to forecast comparison CSV file")
    parser.add_argument("product_master_file",
                       help="Path to product master CSV file")
    
    # Optional arguments
    parser.add_argument("--output-dir", default="output/safety_stocks", 
                       help="Output directory (default: output/safety_stocks)")
    parser.add_argument("--review-dates", 
                       help="Comma-separated list of review dates (YYYY-MM-DD)")
    parser.add_argument("--review-interval", type=int, default=30, 
                       help="Days between review dates if not provided (default: 30)")
    
    args = parser.parse_args()
    
    # Validate input files
    if not Path(args.forecast_comparison_file).exists():
        print(f"❌ Error: Forecast comparison file not found: {args.forecast_comparison_file}")
        sys.exit(1)
    
    if not Path(args.product_master_file).exists():
        print(f"❌ Error: Product master file not found: {args.product_master_file}")
        sys.exit(1)
    
    # Run safety stock calculation
    result = run_safety_stock_calculation(
        forecast_comparison_file=args.forecast_comparison_file,
        product_master_file=args.product_master_file,
        output_dir=args.output_dir,
        review_dates=args.review_dates,
        review_interval_days=args.review_interval
    )
    
    if result is None:
        print("❌ Safety stock calculation failed")
        sys.exit(1)
    else:
        print("\n✅ Safety stock calculation completed successfully!")


if __name__ == "__main__":
    main() 