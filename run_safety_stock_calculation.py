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
from data.loader import DataLoader
from forecaster.utils.logger import get_logger, configure_workflow_logging


def run_safety_stock_calculation(
    forecast_comparison_file: str,
    review_dates: str = None,
    review_interval_days: int = 30,
    log_level: str = "INFO"
):
    """
    Calculate safety stocks from forecast comparison results.
    
    Args:
        forecast_comparison_file: Path to forecast comparison CSV file
        product_master_file: Path to product master CSV file (handled by DataLoader)
        output_dir: Output directory for results
        review_dates: Comma-separated list of review dates (YYYY-MM-DD). If not provided, will use dates from config file.
        review_interval_days: Days between review dates if not provided (deprecated - use config file instead)
        log_level: Logging level for the calculation
    
    Returns:
        DataFrame with safety stock results
    """
    
    # Setup logging FIRST before any other imports or logger creation
    from forecaster.utils.logger import setup_logging, configure_workflow_logging
    setup_logging(level=log_level, console_output=True, file_output=False)
    
    # Setup workflow logging
    logger = configure_workflow_logging(
        workflow_name="safety_stock_calculation",
        log_level=log_level,
        log_dir="output/logs"
    )
    
    logger.info("🔍 Safety Stock Calculation")
    logger.info(f"Forecast comparison file: {forecast_comparison_file}")
    logger.info("Product master file: (loaded via DataLoader)")
    
    # Initialize DataLoader
    loader = DataLoader()
    
    # Load data
    logger.info("📊 Loading data...")
    try:
        forecast_data = pd.read_csv(forecast_comparison_file)
        product_master_data = loader.load_product_master()
        
        logger.info(f"✅ Loaded {len(forecast_data)} forecast comparison records")
        logger.info(f"✅ Loaded {len(product_master_data)} product master records")
        
    except Exception as e:
        logger.log_error_with_context(e, "Error loading data")
        return None
    
    # Handle column name mapping for different backtesting outputs
    logger.info("🔧 Checking column mappings...")
    if 'error' in forecast_data.columns and 'forecast_error' not in forecast_data.columns:
        logger.info("📝 Mapping 'error' column to 'forecast_error' for compatibility")
        forecast_data['forecast_error'] = forecast_data['error']
    elif 'forecast_error' not in forecast_data.columns:
        logger.error("❌ No error column found. Available columns:")
        logger.error(f"   {list(forecast_data.columns)}")
        return None
    
    # Handle product master column mappings
    if 'leadtime' in product_master_data.columns and 'lead_time' not in product_master_data.columns:
        logger.info("📝 Mapping 'leadtime' column to 'lead_time' for compatibility")
        product_master_data['lead_time'] = product_master_data['leadtime']
    elif 'lead_time' not in product_master_data.columns:
        logger.error("❌ No lead_time column found. Available columns:")
        logger.error(f"   {list(product_master_data.columns)}")
        return None
    
    # Validate required columns
    required_forecast_columns = ['analysis_date', 'product_id', 'location_id', 'actual_demand', 'forecast_demand', 'forecast_error']
    missing_forecast_columns = [col for col in required_forecast_columns if col not in forecast_data.columns]
    
    if missing_forecast_columns:
        logger.error(f"❌ Missing required columns in forecast comparison file: {missing_forecast_columns}")
        logger.error(f"Available columns: {list(forecast_data.columns)}")
        return None
    
    required_product_columns = ['product_id', 'location_id', 'lead_time', 'risk_period']
    missing_product_columns = [col for col in required_product_columns if col not in product_master_data.columns]
    
    if missing_product_columns:
        logger.error(f"❌ Missing required columns in product master file: {missing_product_columns}")
        return None
    
    # Process review dates
    logger.info("📅 Processing review dates...")
    if review_dates:
        review_date_list = [date.fromisoformat(d.strip()) for d in review_dates.split(',')]
        logger.info(f"📅 Using {len(review_date_list)} provided review dates")
    else:
        # Try to get review dates from config
        try:
            config_review_dates = loader.config.get('safety_stock', {}).get('review_dates', [])
            if config_review_dates:
                review_date_list = [date.fromisoformat(d.strip()) for d in config_review_dates]
                logger.info(f"📅 Using {len(review_date_list)} review dates from config")
            else:
                raise ValueError("No review dates found in config")
        except Exception as e:
            logger.error("❌ Error: Review dates are required for safety stock calculation.")
            logger.error(f"   Please provide review dates via --review-dates parameter or add them to data/config/data_config.yaml")
            logger.log_error_with_context(e, "Review dates configuration")
            return None
    
    # Initialize safety stock calculator
    logger.info("🔧 Initializing safety stock calculator...")
    calculator = SafetyStockCalculator(product_master_data)
    
    # Calculate safety stocks
    logger.info("🧮 Calculating safety stocks...")
    safety_stock_results = calculator.calculate_safety_stocks(
        forecast_comparison_data=forecast_data,
        review_dates=review_date_list
    )
    
    # Save results using DataLoader
    loader.save_safety_stocks(safety_stock_results)
    filename = loader.config['paths']['output_files']['safety_stocks']
    output_file = loader.get_output_path("safety_stocks", filename)
    logger.info(f"💾 Saved results to: {output_file}")
    
    # Display summary
    logger.info("📊 Safety Stock Calculation Summary:")
    logger.info(f"  Total calculations: {len(safety_stock_results)}")
    
    if len(safety_stock_results) > 0:
        logger.info(f"  Products: {safety_stock_results['product_id'].nunique()}")
        logger.info(f"  Locations: {safety_stock_results['location_id'].nunique()}")
        logger.info(f"  Review dates: {safety_stock_results['review_date'].nunique()}")
        
        # Summary statistics
        logger.info("📈 Safety Stock Statistics:")
        logger.info(f"  Mean safety stock: {safety_stock_results['safety_stock'].mean():.2f}")
        logger.info(f"  Median safety stock: {safety_stock_results['safety_stock'].median():.2f}")
        logger.info(f"  Min safety stock: {safety_stock_results['safety_stock'].min():.2f}")
        logger.info(f"  Max safety stock: {safety_stock_results['safety_stock'].max():.2f}")
        
        # Error count statistics
        logger.info("📊 Error Data Statistics:")
        logger.info(f"  Mean error count: {safety_stock_results['error_count'].mean():.1f}")
        logger.info(f"  Median error count: {safety_stock_results['error_count'].median():.1f}")
        logger.info(f"  Min error count: {safety_stock_results['error_count'].min()}")
        logger.info(f"  Max error count: {safety_stock_results['error_count'].max()}")
        
        # Distribution types used
        logger.info("🔧 Distribution Types:")
        dist_counts = safety_stock_results['distribution_type'].value_counts()
        for dist_type, count in dist_counts.items():
            logger.info(f"  {dist_type}: {count}")
    else:
        logger.warning("  No safety stock calculations were performed")
        logger.warning("  This may be due to missing forecast comparison data for the specified product-location-method combinations")
    
    return safety_stock_results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Calculate safety stocks from forecast comparison results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with required parameters
  python run_safety_stock_calculation.py forecast_comparison.csv
  
  # Run with custom review dates
  python run_safety_stock_calculation.py forecast_comparison.csv --review-dates "2024-01-01,2024-02-01,2024-03-01"
  
  # Run with review dates from config file (recommended)
  python run_safety_stock_calculation.py forecast_comparison.csv
  
  # Run with custom review interval
  python run_safety_stock_calculation.py forecast_comparison.csv --review-interval 14
  
  # Run with custom output directory
  python run_safety_stock_calculation.py forecast_comparison.csv --output-dir output/my_safety_stocks
        """
    )
    
    # Required arguments
    parser.add_argument("forecast_comparison_file", 
                       help="Path to forecast comparison CSV file")
    
    # Note: Output directory is now handled by DataLoader configuration
    parser.add_argument("--review-dates", 
                       help="Comma-separated list of review dates (YYYY-MM-DD). If not provided, will use dates from config file.")
    parser.add_argument("--review-interval", type=int, default=30, 
                       help="Days between review dates if not provided (default: 30, deprecated - use config file instead)")
    parser.add_argument("--log-level", type=str, default="INFO",
                       help="Logging level for the calculation (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    
    args = parser.parse_args()
    
    # Setup logging FIRST before any other operations
    from forecaster.utils.logger import setup_logging, get_logger
    setup_logging(level=args.log_level, console_output=True, file_output=False)
    
    # Setup logging for main function
    logger = get_logger(__name__)
    
    # Validate input files
    if not Path(args.forecast_comparison_file).exists():
        logger.error(f"❌ Error: Forecast comparison file not found: {args.forecast_comparison_file}")
        sys.exit(1)
    
    # Run safety stock calculation
    result = run_safety_stock_calculation(
        forecast_comparison_file=args.forecast_comparison_file,
        review_dates=args.review_dates,
        review_interval_days=args.review_interval,
        log_level=args.log_level
    )
    
    if result is None:
        logger.error("❌ Safety stock calculation failed")
        sys.exit(1)
    else:
        logger.info("✅ Safety stock calculation completed successfully!")


if __name__ == "__main__":
    main() 