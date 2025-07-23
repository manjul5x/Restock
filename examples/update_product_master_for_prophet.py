#!/usr/bin/env python3
"""
Script to update product master to use Prophet forecasting for some combinations.
"""

import pandas as pd
from pathlib import Path

def update_product_master_for_prophet():
    """Update some product-location combinations to use Prophet forecasting."""
    
    # Load the product master
    file_path = Path("forecaster/data/customer_product_master.csv")
    df = pd.read_csv(file_path)
    
    print(f"Original product master: {len(df)} combinations")
    print(f"Current forecast methods: {df['forecast_method'].value_counts().to_dict()}")
    
    # Update first 5 combinations to use Prophet
    prophet_combinations = df.head(5)
    prophet_combinations['forecast_method'] = 'prophet'
    
    # Update the dataframe
    df.loc[prophet_combinations.index, 'forecast_method'] = 'prophet'
    
    print(f"\nUpdated {len(prophet_combinations)} combinations to use Prophet:")
    for _, row in prophet_combinations.iterrows():
        print(f"  - {row['product_id']} at {row['location_id']}")
    
    # Save the updated file
    df.to_csv(file_path, index=False)
    
    print(f"\nUpdated forecast methods: {df['forecast_method'].value_counts().to_dict()}")
    print(f"File saved to: {file_path}")
    
    return df

if __name__ == "__main__":
    update_product_master_for_prophet() 