#!/usr/bin/env python3
"""
Script to generate dummy demand data for testing the forecasting system.
Creates realistic data with seasonal patterns, trends, and noise.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

def generate_demand_series(
    start_date: datetime,
    end_date: datetime,
    base_demand: float,
    trend: float = 0.0,
    seasonality: float = 0.2,
    noise: float = 0.1,
    weekly_pattern: bool = True
) -> pd.Series:
    """
    Generate a realistic demand series
    
    Args:
        start_date: Start date for the series
        end_date: End date for the series
        base_demand: Base demand level
        trend: Linear trend per day (positive = increasing)
        seasonality: Seasonal amplitude (0-1)
        noise: Random noise level (0-1)
        weekly_pattern: Whether to include weekly patterns
        
    Returns:
        Series with demand values
    """
    # Create date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    n_days = len(date_range)
    
    # Base demand
    demand = np.full(n_days, base_demand)
    
    # Add trend
    if trend != 0:
        trend_component = np.arange(n_days) * trend
        demand += trend_component
    
    # Add seasonality (annual)
    if seasonality > 0:
        # Annual seasonality (365 days)
        annual_cycle = np.sin(2 * np.pi * np.arange(n_days) / 365.25)
        demand += demand * seasonality * annual_cycle
    
    # Add weekly pattern
    if weekly_pattern:
        # Weekly pattern (lower demand on weekends)
        day_of_week = np.array([d.weekday() for d in date_range])
        weekly_pattern = np.where(day_of_week >= 5, 0.7, 1.0)  # 30% reduction on weekends
        demand *= weekly_pattern
    
    # Add noise
    if noise > 0:
        noise_std = max(noise * demand.mean(), 0.1)  # Ensure positive standard deviation
        noise_component = np.random.normal(0, noise_std, n_days)
        demand += noise_component
    
    # Ensure non-negative
    demand = np.maximum(demand, 0)
    
    return pd.Series(demand, index=date_range)

def generate_stock_levels(demand_series: pd.Series, safety_stock: float = 0.5) -> pd.Series:
    """
    Generate realistic stock levels based on demand
    
    Args:
        demand_series: Series of demand values
        safety_stock: Safety stock as fraction of average demand
        
    Returns:
        Series with stock levels
    """
    avg_demand = demand_series.mean()
    base_stock = avg_demand * (1 + safety_stock)
    
    # Add some variation to stock levels
    stock_variation = np.random.normal(0, 0.2 * base_stock, len(demand_series))
    stock_levels = base_stock + stock_variation
    
    # Ensure non-negative
    stock_levels = np.maximum(stock_levels, 0)
    
    return pd.Series(stock_levels, index=demand_series.index)

def assign_product_categories(n_products: int) -> dict:
    """
    Assign product categories with uneven distribution
    
    Args:
        n_products: Number of products to assign categories to
        
    Returns:
        Dictionary mapping product_id to category
    """
    # Define 5 categories with different characteristics
    categories = {
        'ELECTRONICS': {'weight': 0.15, 'base_demand_range': (20, 50), 'trend_range': (0.1, 0.3)},
        'CLOTHING': {'weight': 0.25, 'base_demand_range': (30, 80), 'trend_range': (-0.1, 0.2)},
        'FOOD': {'weight': 0.35, 'base_demand_range': (50, 120), 'trend_range': (0.0, 0.1)},
        'HOME': {'weight': 0.15, 'base_demand_range': (15, 40), 'trend_range': (0.05, 0.15)},
        'SPORTS': {'weight': 0.10, 'base_demand_range': (10, 30), 'trend_range': (0.2, 0.4)}
    }
    
    # Calculate number of products per category based on weights
    category_counts = {}
    remaining_products = n_products
    
    for category, props in categories.items():
        if category == list(categories.keys())[-1]:  # Last category gets remaining products
            count = remaining_products
        else:
            count = max(1, int(n_products * props['weight']))
            remaining_products -= count
        category_counts[category] = count
    
    # Assign categories to products
    product_categories = {}
    product_counter = 1
    
    for category, count in category_counts.items():
        for _ in range(count):
            product_id = f"PROD_{product_counter:03d}"
            product_categories[product_id] = category
            product_counter += 1
    
    return product_categories

def create_dummy_dataset(
    start_date: datetime = datetime(2022, 1, 1),
    end_date: datetime = datetime(2023, 12, 31),
    n_products: int = 10,
    n_locations: int = 5
) -> pd.DataFrame:
    """
    Create a complete dummy dataset
    
    Args:
        start_date: Start date for the dataset
        end_date: End date for the dataset
        n_products: Number of products to generate
        n_locations: Number of locations to generate
        
    Returns:
        DataFrame with demand data
    """
    records = []
    
    # Generate product and location IDs
    products = [f"PROD_{i:03d}" for i in range(1, n_products + 1)]
    locations = [f"LOC_{i:03d}" for i in range(1, n_locations + 1)]
    
    # Assign product categories
    product_categories = assign_product_categories(n_products)
    
    # Define category characteristics
    category_props = {
        'ELECTRONICS': {'base_demand_range': (20, 50), 'trend_range': (0.1, 0.3), 'seasonality': (0.2, 0.4)},
        'CLOTHING': {'base_demand_range': (30, 80), 'trend_range': (-0.1, 0.2), 'seasonality': (0.3, 0.5)},
        'FOOD': {'base_demand_range': (50, 120), 'trend_range': (0.0, 0.1), 'seasonality': (0.1, 0.2)},
        'HOME': {'base_demand_range': (15, 40), 'trend_range': (0.05, 0.15), 'seasonality': (0.15, 0.25)},
        'SPORTS': {'base_demand_range': (10, 30), 'trend_range': (0.2, 0.4), 'seasonality': (0.25, 0.35)}
    }
    
    for product in products:
        category = product_categories[product]
        props = category_props[category]
        
        for location in locations:
            # Generate different demand patterns for each product-location combination
            base_demand = np.random.uniform(*props['base_demand_range'])
            trend = np.random.uniform(*props['trend_range'])
            seasonality = np.random.uniform(*props['seasonality'])
            noise = np.random.uniform(0.05, 0.15)
            
            # Generate demand series
            demand_series = generate_demand_series(
                start_date=start_date,
                end_date=end_date,
                base_demand=base_demand,
                trend=trend,
                seasonality=seasonality,
                noise=noise
            )
            
            # Generate stock levels
            stock_series = generate_stock_levels(demand_series)
            
            # Create records
            for date, demand, stock in zip(demand_series.index, demand_series.values, stock_series.values):
                records.append({
                    'product_id': product,
                    'product_category': category,
                    'location_id': location,
                    'date': date,
                    'demand': round(demand, 2),
                    'stock_level': round(stock, 2)
                })
    
    return pd.DataFrame(records)

def create_weekly_dataset(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create weekly aggregated dataset from daily data
    
    Args:
        daily_df: Daily demand DataFrame
        
    Returns:
        Weekly aggregated DataFrame
    """
    # Group by week, product, and location
    weekly_df = daily_df.copy()
    weekly_df['date'] = pd.to_datetime(weekly_df['date'])
    weekly_df['week_start'] = weekly_df['date'].dt.to_period('W').dt.start_time
    
    # Aggregate by week
    weekly_agg = weekly_df.groupby(['week_start', 'product_id', 'product_category', 'location_id']).agg({
        'demand': 'sum',
        'stock_level': 'mean'  # Average stock level for the week
    }).reset_index()
    
    # Rename columns
    weekly_agg = weekly_agg.rename(columns={'week_start': 'date'})
    
    return weekly_agg

def main():
    """Generate and save dummy datasets"""
    print("Generating dummy demand data...")
    
    # Get current directory (should be the dummy folder)
    data_dir = Path(__file__).parent
    
    # Generate daily data
    print("Creating daily dataset...")
    daily_df = create_dummy_dataset()
    daily_df.to_csv(data_dir / "sku_demand_daily.csv", index=False)
    print(f"Daily dataset created: {len(daily_df)} records")
    
    # Generate weekly data
    print("Creating weekly dataset...")
    weekly_df = create_weekly_dataset(daily_df)
    weekly_df.to_csv(data_dir / "sku_demand_weekly.csv", index=False)
    print(f"Weekly dataset created: {len(weekly_df)} records")
    
    # Print summary statistics
    print("\nDataset Summary:")
    print(f"Date range: {daily_df['date'].min()} to {daily_df['date'].max()}")
    print(f"Products: {daily_df['product_id'].nunique()}")
    print(f"Locations: {daily_df['location_id'].nunique()}")
    print(f"Total daily records: {len(daily_df)}")
    print(f"Total weekly records: {len(weekly_df)}")
    print(f"Average daily demand: {daily_df['demand'].mean():.2f}")
    print(f"Average stock level: {daily_df['stock_level'].mean():.2f}")
    
    # Print category distribution
    print("\nProduct Category Distribution:")
    category_counts = daily_df.groupby('product_category')['product_id'].nunique()
    for category, count in category_counts.items():
        percentage = (count / daily_df['product_id'].nunique()) * 100
        print(f"  {category}: {count} products ({percentage:.1f}%)")
    
    print(f"\nFiles saved to: {data_dir}")

if __name__ == "__main__":
    main() 