#!/usr/bin/env python3
"""
Update Customer Product Master with Safety Stock Parameters

This script adds safety stock parameters to the customer product master file.
"""

import pandas as pd
import sys
from pathlib import Path

def update_product_master(input_file: str, output_file: str = None):
    """
    Update product master with safety stock parameters.
    
    Args:
        input_file: Path to input product master CSV
        output_file: Path to output product master CSV (optional, defaults to input_file)
    """
    print(f"📖 Loading product master from: {input_file}")
    
    # Load the product master
    df = pd.read_csv(input_file)
    print(f"✅ Loaded {len(df)} product-location combinations")
    
    # Add safety stock parameters with default values
    print("🔧 Adding safety stock parameters...")
    
    # Distribution type (default: kde)
    if 'distribution' not in df.columns:
        df['distribution'] = 'kde'
        print("  ➕ Added 'distribution' column with default value 'kde'")
    
    # Service level (default: 0.95)
    if 'service_level' not in df.columns:
        df['service_level'] = 0.95
        print("  ➕ Added 'service_level' column with default value 0.95")
    
    # Review window length (default: 180 days)
    if 'ss_window_length' not in df.columns:
        df['ss_window_length'] = 180
        print("  ➕ Added 'ss_window_length' column with default value 180")
    
    # Save the updated product master
    output_path = output_file or input_file
    df.to_csv(output_path, index=False)
    print(f"💾 Saved updated product master to: {output_path}")
    
    # Display summary
    print("\n📊 Updated Product Master Summary:")
    print(f"  Total records: {len(df)}")
    print(f"  Distribution types: {df['distribution'].value_counts().to_dict()}")
    print(f"  Service level range: {df['service_level'].min():.2f} - {df['service_level'].max():.2f}")
    print(f"  Safety stock window range: {df['ss_window_length'].min()} - {df['ss_window_length'].max()}")
    
    return df

def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python update_customer_product_master.py <input_file> [output_file]")
        print("Example: python update_customer_product_master.py forecaster/data/customer_product_master.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not Path(input_file).exists():
        print(f"❌ Error: Input file not found: {input_file}")
        sys.exit(1)
    
    try:
        update_product_master(input_file, output_file)
        print("\n✅ Product master updated successfully!")
    except Exception as e:
        print(f"❌ Error updating product master: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 