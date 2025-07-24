#!/usr/bin/env python3
"""
Create filtered data files for Prophet combinations.
"""

import pandas as pd
from pathlib import Path
import sys

def create_prophet_filtered_data():
    """Filter data for Prophet combinations and save to files."""
    
    print("="*60)
    print("PROPHET FILTERED DATA CREATION")
    print("="*60)
    
    # Load data
    print("Loading data...")
    demand_data = pd.read_csv("forecaster/data/customer_demand.csv")
    product_master = pd.read_csv("forecaster/data/customer_product_master.csv")
    
    print(f"Original demand data: {len(demand_data)} records")
    print(f"Original product master: {len(product_master)} combinations")
    
    # Filter for Prophet combinations
    prophet_combinations = product_master[product_master['forecast_method'] == 'prophet']
    
    print(f"\nFound {len(prophet_combinations)} Prophet combinations:")
    for _, row in prophet_combinations.iterrows():
        print(f"  - {row['product_id']} at {row['location_id']}")
    
    if len(prophet_combinations) == 0:
        print("No Prophet combinations found!")
        return
    
    # Create filtered product master
    filtered_product_master = prophet_combinations.copy()
    filtered_product_master_path = Path("forecaster/data/temp_prophet_product_master.csv")
    filtered_product_master.to_csv(filtered_product_master_path, index=False)
    
    # Filter demand data for Prophet combinations
    prophet_product_locations = prophet_combinations[['product_id', 'location_id']].values.tolist()
    filtered_demand = demand_data[
        demand_data[['product_id', 'location_id']].apply(
            lambda x: tuple(x) in prophet_product_locations, axis=1
        )
    ].copy()
    
    # Alternative filtering approach
    mask = demand_data.set_index(['product_id', 'location_id']).index.isin(
        prophet_combinations.set_index(['product_id', 'location_id']).index
    )
    filtered_demand = demand_data[mask].copy()
    
    filtered_demand_path = Path("forecaster/data/temp_prophet_demand.csv")
    filtered_demand.to_csv(filtered_demand_path, index=False)
    
    print(f"\nFiltered data created:")
    print(f"  Demand records: {len(filtered_demand)} (from {len(demand_data)})")
    print(f"  Product combinations: {len(filtered_product_master)} (from {len(product_master)})")
    
    print(f"\nFiles created:")
    print(f"  - {filtered_demand_path}")
    print(f"  - {filtered_product_master_path}")
    
    print(f"\nTo run the backtest with these files, use:")
    print(f"python run_unified_backtest.py --data-dir forecaster/data --demand-file temp_prophet_demand.csv --product-master-file temp_prophet_product_master.csv")

if __name__ == "__main__":
    create_prophet_filtered_data() 