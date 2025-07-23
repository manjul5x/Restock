#!/usr/bin/env python3
"""
Create Customer Product Master Script

This script creates a product master file for the customer demand data
with appropriate forecasting parameters.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys


def create_product_master(demand_file: str, output_file: str):
    """Create product master file from demand data."""
    print(f"Loading demand data from: {demand_file}")
    
    # Load the demand data
    df = pd.read_csv(demand_file)
    print(f"Loaded {len(df)} demand records")
    
    # Get unique product-location combinations
    combinations = df.groupby(['product_id', 'location_id']).first().reset_index()
    combinations = combinations[['product_id', 'location_id', 'product_category']]
    
    print(f"Found {len(combinations)} unique product-location combinations")
    
    # Create product master with default parameters
    product_master = combinations.copy()
    
    # Add required columns with sensible defaults
    product_master['demand_frequency'] = 'd'  # Daily data
    product_master['risk_period'] = 14  # 14 days (1 week)
    product_master['forecast_window_length'] = 25  # 25 risk periods
    product_master['forecast_horizon'] = 2  # 2 risk periods
    product_master['outlier_method'] = 'iqr'  # IQR method for outlier detection
    product_master['outlier_threshold'] = 1.5  # Standard IQR threshold
    
    # Ensure correct column order
    required_columns = [
        'product_id', 'location_id', 'product_category', 'demand_frequency', 
        'risk_period', 'forecast_window_length', 'forecast_horizon', 
        'outlier_method', 'outlier_threshold'
    ]
    product_master = product_master[required_columns]
    
    # Convert data types
    product_master['product_id'] = product_master['product_id'].astype(str)
    product_master['location_id'] = product_master['location_id'].astype(str)
    product_master['product_category'] = product_master['product_category'].astype(str)
    product_master['demand_frequency'] = product_master['demand_frequency'].astype(str)
    product_master['risk_period'] = product_master['risk_period'].astype(int)
    product_master['forecast_window_length'] = product_master['forecast_window_length'].astype(int)
    product_master['forecast_horizon'] = product_master['forecast_horizon'].astype(int)
    product_master['outlier_method'] = product_master['outlier_method'].astype(str)
    product_master['outlier_threshold'] = product_master['outlier_threshold'].astype(float)
    
    # Sort by product_id, location_id for consistency
    product_master = product_master.sort_values(['product_id', 'location_id']).reset_index(drop=True)
    
    # Display the product master
    print(f"\nProduct Master Preview:")
    print(product_master.head(10).to_string(index=False))
    
    # Save the product master
    print(f"\nSaving product master to: {output_file}")
    product_master.to_csv(output_file, index=False)
    print("✓ Product master saved successfully!")
    
    # Display summary
    print(f"\nProduct Master Summary:")
    print(f"Total products: {len(product_master)}")
    print(f"Unique locations: {product_master['location_id'].nunique()}")
    print(f"Demand frequency: {product_master['demand_frequency'].iloc[0]}")
    print(f"Risk period: {product_master['risk_period'].iloc[0]} days")
    print(f"Forecast window length: {product_master['forecast_window_length'].iloc[0]} risk periods")
    print(f"Forecast horizon: {product_master['forecast_horizon'].iloc[0]} risk periods")
    print(f"Outlier method: {product_master['outlier_method'].iloc[0]}")
    print(f"Outlier threshold: {product_master['outlier_threshold'].iloc[0]}")
    
    return product_master


def main():
    """Main function."""
    # Define paths
    demand_file = "forecaster/data/customer_demand.csv"
    output_file = "forecaster/data/customer_product_master.csv"
    
    # Check if demand file exists
    if not Path(demand_file).exists():
        print(f"❌ Demand file not found: {demand_file}")
        print("Please run format_customer_data.py first to create the demand file.")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create the product master
    try:
        product_master = create_product_master(demand_file, output_file)
        
        print("\n" + "="*60)
        print("✅ PRODUCT MASTER CREATED SUCCESSFULLY")
        print("="*60)
        print(f"Product master saved to: {output_file}")
        print(f"Total products: {len(product_master)}")
        print()
        print("Next steps:")
        print("1. Review and adjust the product master parameters if needed")
        print("2. Fill in the product_category column with appropriate categories")
        print("3. Run validation: python validate_customer_data.py customer_demand.csv customer_product_master.csv")
        print("4. Run backtesting: python run_customer_backtest.py forecaster/data customer_demand.csv customer_product_master.csv")
        
    except Exception as e:
        print(f"❌ Error creating product master: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 