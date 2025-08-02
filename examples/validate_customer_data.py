#!/usr/bin/env python3
"""
Customer Data Validation Script

This script validates customer data files before running backtesting to ensure
they meet the required schema and data quality standards.
"""

import pandas as pd
import sys
from pathlib import Path
from datetime import datetime
from forecaster.validation.schema import DemandSchema
from forecaster.validation.product_master_schema import ProductMasterSchema


def validate_demand_data(file_path: str) -> dict:
    """Validate demand data file."""
    print(f"Validating demand data: {file_path}")
    
    try:
        # Load data
        df = pd.read_csv(file_path)
        print(f"  ✓ Loaded {len(df)} rows")
        
        # Check required columns
        required_cols = ['product_id', 'product_category', 'location_id', 'date', 'demand', 'stock_level']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            return {"valid": False, "error": f"Missing required columns: {missing_cols}"}
        print(f"  ✓ All required columns present")
        
        # Check for missing values
        missing_counts = df[required_cols].isnull().sum()
        if missing_counts.sum() > 0:
            print(f"  ⚠ Missing values found:")
            for col, count in missing_counts.items():
                if count > 0:
                    print(f"    - {col}: {count} missing values")
        else:
            print(f"  ✓ No missing values in required columns")
        
        # Check data types
        try:
            df['date'] = pd.to_datetime(df['date'])
            print(f"  ✓ Date column converted to datetime")
        except Exception as e:
            return {"valid": False, "error": f"Date column conversion failed: {e}"}
        
        # Check numeric columns
        try:
            df['demand'] = pd.to_numeric(df['demand'], errors='coerce')
            df['stock_level'] = pd.to_numeric(df['stock_level'], errors='coerce')
            print(f"  ✓ Numeric columns converted")
        except Exception as e:
            return {"valid": False, "error": f"Numeric conversion failed: {e}"}
        
        # Check for negative values
        negative_demand = (df['demand'] < 0).sum()
        negative_stock = (df['stock_level'] < 0).sum()
        
        if negative_demand > 0:
            print(f"  ⚠ Found {negative_demand} negative demand values")
        else:
            print(f"  ✓ No negative demand values")
            
        if negative_stock > 0:
            print(f"  ⚠ Found {negative_stock} negative stock values")
        else:
            print(f"  ✓ No negative stock values")
        
        # Check date range
        date_range = f"{df['date'].min().date()} to {df['date'].max().date()}"
        print(f"  ✓ Date range: {date_range}")
        
        # Check unique combinations
        unique_products = df['product_id'].nunique()
        unique_locations = df['location_id'].nunique()
        unique_categories = df['product_category'].nunique()
        unique_combinations = df.groupby(['product_id', 'location_id']).size().shape[0]
        
        print(f"  ✓ Unique products: {unique_products}")
        print(f"  ✓ Unique locations: {unique_locations}")
        print(f"  ✓ Unique categories: {unique_categories}")
        print(f"  ✓ Unique product-location combinations: {unique_combinations}")
        
        return {
            "valid": True,
            "data": df,
            "stats": {
                "rows": len(df),
                "unique_products": unique_products,
                "unique_locations": unique_locations,
                "unique_categories": unique_categories,
                "unique_combinations": unique_combinations,
                "date_range": date_range,
                "negative_demand": negative_demand,
                "negative_stock": negative_stock
            }
        }
        
    except Exception as e:
        return {"valid": False, "error": f"Failed to load or validate demand data: {e}"}


def validate_product_master(file_path: str) -> dict:
    """Validate product master file."""
    print(f"Validating product master: {file_path}")
    
    try:
        # Load data
        df = pd.read_csv(file_path)
        print(f"  ✓ Loaded {len(df)} rows")
        
        # Check required columns
        required_cols = ['product_id', 'location_id', 'product_category', 'demand_frequency', 'risk_period']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            return {"valid": False, "error": f"Missing required columns: {missing_cols}"}
        print(f"  ✓ All required columns present")
        
        # Check for missing values
        missing_counts = df[required_cols].isnull().sum()
        if missing_counts.sum() > 0:
            print(f"  ⚠ Missing values found:")
            for col, count in missing_counts.items():
                if count > 0:
                    print(f"    - {col}: {count} missing values")
        else:
            print(f"  ✓ No missing values in required columns")
        
        # Check demand frequencies
        valid_frequencies = ['d', 'w', 'm']
        invalid_freq = set(df['demand_frequency'].unique()) - set(valid_frequencies)
        if invalid_freq:
            return {"valid": False, "error": f"Invalid demand frequencies: {invalid_freq}"}
        print(f"  ✓ All demand frequencies valid")
        
        # Check risk periods
        try:
            df['risk_period'] = pd.to_numeric(df['risk_period'], errors='coerce')
            negative_risk = (df['risk_period'] <= 0).sum()
            if negative_risk > 0:
                return {"valid": False, "error": f"Found {negative_risk} non-positive risk periods"}
            print(f"  ✓ All risk periods positive")
        except Exception as e:
            return {"valid": False, "error": f"Risk period conversion failed: {e}"}
        
        # Check for required forecasting columns
        forecasting_cols = ['forecast_window_length', 'forecast_horizon']
        missing_forecast_cols = set(forecasting_cols) - set(df.columns)
        if missing_forecast_cols:
            print(f"  ⚠ Missing forecasting columns: {missing_forecast_cols}")
            print(f"    These will be set to defaults (window_length=100, horizon=10)")
        else:
            print(f"  ✓ All forecasting columns present")
        
        # Check unique combinations
        unique_combinations = df.groupby(['product_id', 'location_id']).size().shape[0]
        print(f"  ✓ Unique product-location combinations: {unique_combinations}")
        
        # Check frequency distribution
        freq_dist = df['demand_frequency'].value_counts()
        print(f"  ✓ Demand frequency distribution:")
        for freq, count in freq_dist.items():
            print(f"    - {freq}: {count} products")
        
        return {
            "valid": True,
            "data": df,
            "stats": {
                "rows": len(df),
                "unique_combinations": unique_combinations,
                "frequency_distribution": freq_dist.to_dict()
            }
        }
        
    except Exception as e:
        return {"valid": False, "error": f"Failed to load or validate product master: {e}"}


def check_data_coverage(demand_data: pd.DataFrame, product_master_data: pd.DataFrame) -> dict:
    """Check that all demand combinations exist in product master."""
    print("Checking data coverage...")
    
    # Get combinations from both datasets
    demand_combinations = set(zip(demand_data['product_id'], demand_data['location_id']))
    master_combinations = set(zip(product_master_data['product_id'], product_master_data['location_id']))
    
    # Find missing combinations
    missing_in_master = demand_combinations - master_combinations
    missing_in_demand = master_combinations - demand_combinations
    
    if missing_in_master:
        print(f"  ❌ Found {len(missing_in_master)} product-location combinations in demand data that are missing from product master:")
        for prod, loc in list(missing_in_master)[:10]:  # Show first 10
            print(f"    - {prod} at {loc}")
        if len(missing_in_master) > 10:
            print(f"    ... and {len(missing_in_master) - 10} more")
        return {"valid": False, "error": f"{len(missing_in_master)} combinations missing from product master"}
    
    if missing_in_demand:
        print(f"  ⚠ Found {len(missing_in_demand)} product-location combinations in product master that are not in demand data")
        print(f"    This is okay if these are new products/locations")
    
    print(f"  ✓ All demand combinations covered in product master")
    return {"valid": True}


def main():
    """Main validation function."""
    if len(sys.argv) != 3:
        print("Usage: python validate_customer_data.py <demand_file> <product_master_file>")
        print("Example: python validate_customer_data.py customer_demand.csv customer_product_master.csv")
        sys.exit(1)
    
    demand_file = sys.argv[1]
    product_master_file = sys.argv[2]
    
    print("=" * 60)
    print("CUSTOMER DATA VALIDATION")
    print("=" * 60)
    
    # Validate demand data
    demand_result = validate_demand_data(demand_file)
    if not demand_result["valid"]:
        print(f"❌ Demand data validation failed: {demand_result['error']}")
        sys.exit(1)
    
    print()
    
    # Validate product master
    product_master_result = validate_product_master(product_master_file)
    if not product_master_result["valid"]:
        print(f"❌ Product master validation failed: {product_master_result['error']}")
        sys.exit(1)
    
    print()
    
    # Check data coverage
    coverage_result = check_data_coverage(demand_result["data"], product_master_result["data"])
    if not coverage_result["valid"]:
        print(f"❌ Data coverage check failed: {coverage_result['error']}")
        sys.exit(1)
    
    print()
    print("=" * 60)
    print("✅ VALIDATION COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print("Your data is ready for backtesting!")
    print()
    print("Next steps:")
    print("1. Create a configuration script using the data paths above")
    print("2. Run a small test with limited date range")
    print("3. Run the full backtesting once the test succeeds")
    print()
    print("See CUSTOMER_DATA_GUIDE.md for detailed instructions.")


if __name__ == "__main__":
    main() 