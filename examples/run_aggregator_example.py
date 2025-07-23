#!/usr/bin/env python3
"""
Run aggregator for specific product and location with date 2023-08-23.
"""

from forecaster.data import DemandDataLoader, DemandAggregator, create_risk_period_buckets
from datetime import datetime
import pandas as pd

def main():
    """Run aggregator for specific example"""
    print("Running Aggregator Example")
    print("=" * 50)
    
    # Set the cutoff date
    cutoff_date = datetime(2023, 8, 24)
    print(f"Cutoff date: {cutoff_date.date()}")
    
    # Load data to see available products and locations
    loader = DemandDataLoader()
    demand_df = loader.load_dummy_data(frequency="daily")
    
    # Choose a specific product and location
    chosen_product = "PROD_001"
    chosen_location = "LOC_001"
    
    print(f"Selected product: {chosen_product}")
    print(f"Selected location: {chosen_location}")
    
    # Create buckets for all products/locations
    buckets_df = create_risk_period_buckets(cutoff_date, "daily")
    
    print(f"\nTotal buckets created: {len(buckets_df):,}")
    print(f"Products: {buckets_df['product_id'].nunique()}")
    print(f"Locations: {buckets_df['location_id'].nunique()}")
    
    # Filter for our chosen product and location
    product_location_buckets = buckets_df[
        (buckets_df['product_id'] == chosen_product) &
        (buckets_df['location_id'] == chosen_location)
    ].copy()
    
    print(f"\nBuckets for {chosen_product}-{chosen_location}: {len(product_location_buckets)}")
    
    if len(product_location_buckets) > 0:
        print(f"Date range: {product_location_buckets['bucket_start_date'].min().date()} to {product_location_buckets['bucket_end_date'].max().date()}")
        print(f"Total demand: {product_location_buckets['total_demand'].sum():.2f}")
        
        # Show the latest few buckets
        latest_buckets = product_location_buckets.sort_values('bucket_start_date').tail(5)
        print(f"\nLatest 5 buckets:")
        for _, bucket in latest_buckets.iterrows():
            start_date = bucket['bucket_start_date'].date()
            end_date = bucket['bucket_end_date'].date()
            total_demand = bucket['total_demand']
            avg_demand = bucket['avg_demand']
            print(f"  {start_date} to {end_date}: {total_demand:.2f} total ({avg_demand:.2f} avg/day)")
    
    # Save the full output to CSV
    output_filename = f"aggregation_output_{cutoff_date.date()}.csv"
    buckets_df.to_csv(output_filename, index=False)
    print(f"\nFull output saved to: {output_filename}")
    
    # Save just the product-location buckets to a separate file
    product_output_filename = f"aggregation_{chosen_product}_{chosen_location}_{cutoff_date.date()}.csv"
    product_location_buckets.to_csv(product_output_filename, index=False)
    print(f"Product-location output saved to: {product_output_filename}")
    
    # Show summary statistics
    print(f"\nSummary Statistics:")
    print(f"  Total buckets: {len(buckets_df):,}")
    print(f"  Total demand: {buckets_df['total_demand'].sum():,.2f}")
    print(f"  Average demand per bucket: {buckets_df['total_demand'].mean():.2f}")
    print(f"  Min demand per bucket: {buckets_df['total_demand'].min():.2f}")
    print(f"  Max demand per bucket: {buckets_df['total_demand'].max():.2f}")
    
    print(f"\nAggregation complete!")

if __name__ == "__main__":
    main() 