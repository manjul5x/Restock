#!/usr/bin/env python3
"""
Simple example demonstrating how to use the visualization module.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add the forecaster package to the path
sys.path.append(str(Path(__file__).parent))

from forecaster.data import DemandDataLoader
from forecaster.utils import DemandVisualizer

def main():
    """Demonstrate visualization capabilities"""
    
    # Load data
    print("Loading demand data...")
    loader = DemandDataLoader()
    data = loader.load_dummy_data(frequency="daily")
    
    # Initialize visualizer
    viz = DemandVisualizer(data)
    
    # Example 1: Simple demand trend (all data aggregated)
    print("\n1. Creating total demand trend...")
    fig1 = viz.plot_demand_trend(title="Total Demand Over Time")
    viz.save_plot(fig1, "example_total_demand.png")
    
    # Example 2: Filter by specific locations and categories
    print("\n2. Creating filtered demand plot...")
    fig2 = viz.plot_demand_trend(
        locations=['LOC_001', 'LOC_002'],
        categories=['FOOD', 'CLOTHING'],
        group_by=['date', 'product_category'],
        title="Food & Clothing Demand (Locations 001 & 002)"
    )
    viz.save_plot(fig2, "example_filtered_demand.png")
    
    # Example 3: SKU-location level granularity
    print("\n3. Creating SKU-location specific plot...")
    sku_locations = [
        ('PROD_001', 'LOC_001'),
        ('PROD_002', 'LOC_001'),
        ('PROD_003', 'LOC_002')
    ]
    fig3 = viz.plot_sku_location_demand(
        sku_location_pairs=sku_locations,
        title="Specific SKU-Location Demand"
    )
    viz.save_plot(fig3, "example_sku_location.png")
    
    # Example 4: Time-filtered plot
    print("\n4. Creating time-filtered plot...")
    start_date = datetime(2022, 6, 1)
    end_date = datetime(2022, 8, 31)
    fig4 = viz.plot_demand_trend(
        categories=['ELECTRONICS'],
        start_date=start_date,
        end_date=end_date,
        group_by=['date', 'location_id'],
        title=f"Electronics Demand by Location (Summer 2022)"
    )
    viz.save_plot(fig4, "example_time_filtered.png")
    
    # Example 5: Category comparison
    print("\n5. Creating category comparison...")
    fig5 = viz.plot_category_comparison(
        locations=['LOC_001'],
        title="Demand by Category (Location 001)"
    )
    viz.save_plot(fig5, "example_category_comparison.png")
    
    print("\n✅ All example plots created!")
    print("Check the generated PNG files to see the visualizations.")

if __name__ == "__main__":
    main() 