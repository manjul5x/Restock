#!/usr/bin/env python3
"""
Create filtered data files for Prophet combinations.
"""

import pandas as pd
from pathlib import Path
import sys
from data.loader import DataLoader

def create_prophet_filtered_data():
    """Filter data for Prophet combinations and save to files."""
    
    print("="*60)
    print("PROPHET FILTERED DATA CREATION")
    print("="*60)
    
    # Initialize DataLoader
    loader = DataLoader()
    
    # Load data
    print("Loading data...")
    product_master = loader.load_product_master()
    demand_data = loader.load_outflow(product_master=product_master)
    
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
    filtered_product_master_path = loader.get_output_path("temp", "prophet_product_master.csv")
    loader.save_results(filtered_product_master, "temp", "prophet_product_master.csv")
    
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
    
    filtered_demand_path = loader.get_output_path("temp", "prophet_demand.csv")
    loader.save_results(filtered_demand, "temp", "prophet_demand.csv")
    
    print(f"\nFiltered data created:")
    print(f"  Demand records: {len(filtered_demand)} (from {len(demand_data)})")
    print(f"  Product combinations: {len(filtered_product_master)} (from {len(product_master)})")
    
    print(f"\nFiles created:")
    print(f"  - {filtered_demand_path}")
    print(f"  - {filtered_product_master_path}")
    
    print(f"\nTo run the backtest with these files, use:")
    print(f"python run_unified_backtest.py")

if __name__ == "__main__":
    create_prophet_filtered_data() 