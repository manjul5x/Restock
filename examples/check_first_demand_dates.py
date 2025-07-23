#!/usr/bin/env python3
"""
Script to find the first demand date for each material in customer product master.
"""

import pandas as pd
import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

from forecaster.data.loader import DemandDataLoader

def check_first_demand_dates():
    """Check the first demand date for each material in customer product master."""
    
    print("=" * 60)
    print("FIRST DEMAND DATES BY MATERIAL")
    print("=" * 60)
    
    # Load data
    loader = DemandDataLoader('forecaster/data')
    
    print("\n1. Loading customer demand data...")
    demand_data = loader.load_csv('customer_demand.csv')
    
    print("\n2. Loading customer product master...")
    product_master = loader.load_csv('customer_product_master.csv')
    
    print(f"\n3. Found {len(product_master)} products in product master:")
    for _, row in product_master.iterrows():
        product_id = row['product_id']
        location_id = row['location_id']
        product_key = f"{product_id}-{location_id}"
        
        # Filter demand data for this product-location
        product_demand = demand_data[
            (demand_data['product_id'] == product_id) & 
            (demand_data['location_id'] == location_id)
        ]
        
        if len(product_demand) > 0:
            first_date = product_demand['date'].min()
            last_date = product_demand['date'].max()
            total_records = len(product_demand)
            print(f"   {product_key}:")
            print(f"     First date: {first_date}")
            print(f"     Last date:  {last_date}")
            print(f"     Total records: {total_records}")
        else:
            print(f"   {product_key}: NO DEMAND DATA FOUND")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    # Summary statistics
    all_first_dates = []
    for _, row in product_master.iterrows():
        product_id = row['product_id']
        location_id = row['location_id']
        
        product_demand = demand_data[
            (demand_data['product_id'] == product_id) & 
            (demand_data['location_id'] == location_id)
        ]
        
        if len(product_demand) > 0:
            all_first_dates.append(product_demand['date'].min())
    
    if all_first_dates:
        print(f"Earliest demand date across all products: {min(all_first_dates)}")
        print(f"Latest first demand date across all products: {max(all_first_dates)}")
        print(f"Products with demand data: {len(all_first_dates)}")
        print(f"Products in master: {len(product_master)}")
    else:
        print("No demand data found for any products!")

if __name__ == "__main__":
    check_first_demand_dates() 