#!/usr/bin/env python3
"""
Script to generate product master data with separate daily and weekly tables.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def create_daily_product_master():
    """
    Create product master data for daily demand.
    Daily risk period: 7 days
    """

    # Define products and categories (matching our existing data)
    products = [f"PROD_{i:03d}" for i in range(1, 11)]  # 10 products
    locations = [f"LOC_{i:03d}" for i in range(1, 6)]  # 5 locations

    # Product categories (matching our existing data)
    product_categories = {
        "PROD_001": "ELECTRONICS",
        "PROD_002": "CLOTHING",
        "PROD_003": "FOOD",
        "PROD_004": "FOOD",
        "PROD_005": "FOOD",
        "PROD_006": "CLOTHING",
        "PROD_007": "SPORTS",
        "PROD_008": "SPORTS",
        "PROD_009": "SPORTS",
        "PROD_010": "HOME",
    }

    records = []

    # Create records for each product-location combination (daily only)
    for product_id in products:
        category = product_categories[product_id]

        for location_id in locations:
            # Add outlier parameters with some variation
            outlier_methods = ["iqr", "zscore", "mad", "rolling", "no"]
            outlier_method = outlier_methods[
                hash(product_id + location_id) % len(outlier_methods)
            ]

            # Vary thresholds based on category
            base_threshold = 1.5
            if category == "ELECTRONICS":
                outlier_threshold = (
                    base_threshold + 0.5
                )  # Higher threshold for electronics
            elif category == "FOOD":
                outlier_threshold = base_threshold - 0.3  # Lower threshold for food
            else:
                outlier_threshold = base_threshold

            # Add forecast parameters with some variation
            base_window = 100
            base_horizon = 10

            # Vary window length based on category
            if category == "ELECTRONICS":
                forecast_window_length = (
                    base_window + 20
                )  # Longer window for electronics
            elif category == "FOOD":
                forecast_window_length = base_window - 30  # Shorter window for food
            else:
                forecast_window_length = base_window

            # Vary horizon based on category
            if category == "SPORTS":
                forecast_horizon = base_horizon + 5  # Longer horizon for sports
            else:
                forecast_horizon = base_horizon

            records.append(
                {
                    "product_id": product_id,
                    "location_id": location_id,
                    "product_category": category,
                    "demand_frequency": "d",
                    "risk_period": 7,  # 7 days
                    "outlier_method": outlier_method,
                    "outlier_threshold": outlier_threshold,
                    "forecast_window_length": forecast_window_length,
                    "forecast_horizon": forecast_horizon,
                }
            )

    return pd.DataFrame(records)


def create_weekly_product_master():
    """
    Create product master data for weekly demand.
    Weekly risk period: 2 weeks
    """

    # Define products and categories (matching our existing data)
    products = [f"PROD_{i:03d}" for i in range(1, 11)]  # 10 products
    locations = [f"LOC_{i:03d}" for i in range(1, 6)]  # 5 locations

    # Product categories (matching our existing data)
    product_categories = {
        "PROD_001": "ELECTRONICS",
        "PROD_002": "CLOTHING",
        "PROD_003": "FOOD",
        "PROD_004": "FOOD",
        "PROD_005": "FOOD",
        "PROD_006": "CLOTHING",
        "PROD_007": "SPORTS",
        "PROD_008": "SPORTS",
        "PROD_009": "SPORTS",
        "PROD_010": "HOME",
    }

    records = []

    # Create records for each product-location combination (weekly only)
    for product_id in products:
        category = product_categories[product_id]

        for location_id in locations:
            # Add outlier parameters with some variation
            outlier_methods = ["iqr", "zscore", "mad", "rolling", "no"]
            outlier_method = outlier_methods[
                hash(product_id + location_id) % len(outlier_methods)
            ]

            # Vary thresholds based on category
            base_threshold = 1.5
            if category == "ELECTRONICS":
                outlier_threshold = (
                    base_threshold + 0.5
                )  # Higher threshold for electronics
            elif category == "FOOD":
                outlier_threshold = base_threshold - 0.3  # Lower threshold for food
            else:
                outlier_threshold = base_threshold

            # Add forecast parameters with some variation
            base_window = 100
            base_horizon = 10

            # Vary window length based on category
            if category == "ELECTRONICS":
                forecast_window_length = (
                    base_window + 20
                )  # Longer window for electronics
            elif category == "FOOD":
                forecast_window_length = base_window - 30  # Shorter window for food
            else:
                forecast_window_length = base_window

            # Vary horizon based on category
            if category == "SPORTS":
                forecast_horizon = base_horizon + 5  # Longer horizon for sports
            else:
                forecast_horizon = base_horizon

            records.append(
                {
                    "product_id": product_id,
                    "location_id": location_id,
                    "product_category": category,
                    "demand_frequency": "w",
                    "risk_period": 2,  # 2 weeks
                    "outlier_method": outlier_method,
                    "outlier_threshold": outlier_threshold,
                    "forecast_window_length": forecast_window_length,
                    "forecast_horizon": forecast_horizon,
                }
            )

    return pd.DataFrame(records)


def main():
    """Generate and save product master data"""
    print("Generating product master data...")

    # Get current directory (should be the dummy folder)
    data_dir = Path(__file__).parent

    # Generate daily product master data
    print("\n1. Generating daily product master...")
    daily_product_master_df = create_daily_product_master()
    daily_product_master_df.to_csv(data_dir / "product_master_daily.csv", index=False)

    print(f"   Daily product master created: {len(daily_product_master_df)} records")
    print(f"   Products: {daily_product_master_df['product_id'].nunique()}")
    print(f"   Locations: {daily_product_master_df['location_id'].nunique()}")
    print(f"   Categories: {daily_product_master_df['product_category'].nunique()}")
    print(f"   Risk period: {daily_product_master_df['risk_period'].iloc[0]} days")

    # Generate weekly product master data
    print("\n2. Generating weekly product master...")
    weekly_product_master_df = create_weekly_product_master()
    weekly_product_master_df.to_csv(data_dir / "product_master_weekly.csv", index=False)

    print(f"   Weekly product master created: {len(weekly_product_master_df)} records")
    print(f"   Products: {weekly_product_master_df['product_id'].nunique()}")
    print(f"   Locations: {weekly_product_master_df['location_id'].nunique()}")
    print(f"   Categories: {weekly_product_master_df['product_category'].nunique()}")
    print(f"   Risk period: {weekly_product_master_df['risk_period'].iloc[0]} weeks")

    # Show sample data
    print("\n3. Sample daily product master data:")
    print(daily_product_master_df.head())

    print("\n4. Sample weekly product master data:")
    print(weekly_product_master_df.head())

    # Show file paths
    print("\nFiles saved to:")
    print(f"  Daily: {data_dir / 'product_master_daily.csv'}")
    print(f"  Weekly: {data_dir / 'product_master_weekly.csv'}")


if __name__ == "__main__":
    main()
