#!/usr/bin/env python3
"""
Format Customer Data Script

This script formats the customer demand data to match the required schema
and moves it to the correct location for backtesting.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys


def format_demand_data(input_file: str, output_file: str):
    """Format demand data to match required schema."""
    print(f"Loading data from: {input_file}")
    
    # Load the data
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} rows")
    
    # Display original structure
    print(f"Original columns: {list(df.columns)}")
    print(f"Data types:")
    for col in df.columns:
        print(f"  {col}: {df[col].dtype}")
    
    # Rename demand_qty to demand
    if 'demand_qty' in df.columns:
        df = df.rename(columns={'demand_qty': 'demand'})
        print("✓ Renamed 'demand_qty' to 'demand'")
    
    # Rename closing_stock to stock_level
    if 'closing_stock' in df.columns:
        df = df.rename(columns={'closing_stock': 'stock_level'})
        print("✓ Renamed 'closing_stock' to 'stock_level'")
    
    # Add product_category column with NaN values
    df['product_category'] = np.nan
    print("✓ Added 'product_category' column with NaN values")
    
    # Ensure correct column order
    required_columns = ['product_id', 'product_category', 'location_id', 'date', 'demand', 'stock_level']
    df = df[required_columns]
    print("✓ Reordered columns to match required schema")
    
    # Convert data types
    print("Converting data types...")
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    print("  ✓ Date column converted to datetime")
    
    # Convert demand and stock_level to float
    df['demand'] = pd.to_numeric(df['demand'], errors='coerce').astype(float)
    df['stock_level'] = pd.to_numeric(df['stock_level'], errors='coerce').astype(float)
    print("  ✓ Demand and stock_level converted to float")
    
    # Convert string columns
    df['product_id'] = df['product_id'].astype(str)
    df['location_id'] = df['location_id'].astype(str)
    df['product_category'] = df['product_category'].astype(str)
    print("  ✓ String columns converted")
    
    # Sort by date, product, location for consistency
    df = df.sort_values(['date', 'product_id', 'location_id']).reset_index(drop=True)
    print("✓ Data sorted by date, product_id, location_id")
    
    # Display final structure
    print(f"\nFinal columns: {list(df.columns)}")
    print(f"Final data types:")
    for col in df.columns:
        print(f"  {col}: {df[col].dtype}")
    
    # Check for any issues
    print("\nData quality checks:")
    
    # Check for missing values
    missing_counts = df.isnull().sum()
    if missing_counts.sum() > 0:
        print("⚠ Missing values found:")
        for col, count in missing_counts.items():
            if count > 0:
                print(f"  - {col}: {count} missing values")
    else:
        print("✓ No missing values in required columns")
    
    # Check for negative values
    negative_demand = (df['demand'] < 0).sum()
    negative_stock = (df['stock_level'] < 0).sum()
    
    if negative_demand > 0:
        print(f"⚠ Found {negative_demand} negative demand values")
    else:
        print("✓ No negative demand values")
        
    if negative_stock > 0:
        print(f"⚠ Found {negative_stock} negative stock values")
    else:
        print("✓ No negative stock values")
    
    # Check date range
    date_range = f"{df['date'].min().date()} to {df['date'].max().date()}"
    print(f"✓ Date range: {date_range}")
    
    # Check unique combinations
    unique_products = df['product_id'].nunique()
    unique_locations = df['location_id'].nunique()
    unique_combinations = df.groupby(['product_id', 'location_id']).size().shape[0]
    
    print(f"✓ Unique products: {unique_products}")
    print(f"✓ Unique locations: {unique_locations}")
    print(f"✓ Unique product-location combinations: {unique_combinations}")
    
    # Save the formatted data
    print(f"\nSaving formatted data to: {output_file}")
    df.to_csv(output_file, index=False)
    print("✓ Data saved successfully!")
    
    return df


def main():
    """Main function."""
    # Define paths
    input_file = "forecaster/data/PR_consolidated_df.csv"
    output_file = "forecaster/data/customer_demand.csv"
    
    # Check if input file exists
    if not Path(input_file).exists():
        print(f"❌ Input file not found: {input_file}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Format the data
    try:
        formatted_df = format_demand_data(input_file, output_file)
        
        print("\n" + "="*60)
        print("✅ DATA FORMATTING COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Formatted data saved to: {output_file}")
        print(f"Total rows: {len(formatted_df)}")
        print(f"Date range: {formatted_df['date'].min().date()} to {formatted_df['date'].max().date()}")
        print(f"Products: {formatted_df['product_id'].nunique()}")
        print(f"Locations: {formatted_df['location_id'].nunique()}")
        print()
        print("Next steps:")
        print("1. Create a product master file with the same product-location combinations")
        print("2. Fill in the product_category column with appropriate categories")
        print("3. Run validation: python validate_customer_data.py customer_demand.csv product_master.csv")
        print("4. Run backtesting: python run_customer_backtest.py forecaster/data customer_demand.csv product_master.csv")
        
    except Exception as e:
        print(f"❌ Error formatting data: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 