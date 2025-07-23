#!/usr/bin/env python3
"""
Create a simple PNG simulation plot using matplotlib
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

def create_simulation_plot():
    # Load simulation data
    simulation_file = Path("output/simulation/detailed_results/RSWQ_WB_simulation.csv")
    
    if not simulation_file.exists():
        print(f"Error: Simulation file not found: {simulation_file}")
        return
    
    # Load data
    simulation_data = pd.read_csv(simulation_file)
    simulation_data['date'] = pd.to_datetime(simulation_data['date'])
    
    # Ensure numeric columns are properly formatted
    numeric_columns = ['actual_inventory', 'inventory_on_hand', 'actual_demand', 'safety_stock', 'order_placed', 'min_level', 'max_level']
    for col in numeric_columns:
        if col in simulation_data.columns:
            simulation_data[col] = pd.to_numeric(simulation_data[col], errors='coerce').fillna(0)
    
    print(f"Simulation data shape: {simulation_data.shape}")
    print(f"Date range: {simulation_data['date'].min()} to {simulation_data['date'].max()}")
    
    # Create the plot with single y-axis
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Plot 1: Inventory Levels
    # Actual inventory as shaded bars
    ax.bar(simulation_data['date'], simulation_data['actual_inventory'], 
           alpha=0.4, color='lightblue', label='Actual Inventory', width=1)
    
    # Simulated stock on hand as line
    ax.plot(simulation_data['date'], simulation_data['inventory_on_hand'], 
            color='blue', linewidth=2, label='Simulated Stock on Hand')
    
    # Actual demand as line
    ax.plot(simulation_data['date'], simulation_data['actual_demand'], 
            color='orange', linewidth=2, label='Actual Demand')
    
    # Safety stock as dashed line
    ax.plot(simulation_data['date'], simulation_data['safety_stock'], 
            color='red', linewidth=2, linestyle='--', label='Safety Stock')
    
    # Min level as dotted line
    ax.plot(simulation_data['date'], simulation_data['min_level'], 
            color='green', linewidth=2, linestyle=':', label='Min Level')
    
    # Max level as dotted line
    ax.plot(simulation_data['date'], simulation_data['max_level'], 
            color='purple', linewidth=2, linestyle=':', label='Max Level')
    
    # Plot 2: Orders Placed (same y-axis)
    ax.bar(simulation_data['date'], simulation_data['order_placed'], 
           color='cyan', alpha=0.7, label='Orders Placed', width=1)
    
    # Set labels and titles
    ax.set_xlabel('Date')
    ax.set_ylabel('Quantity')
    ax.set_title('Inventory Simulation: RSWQ at WB')
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add legend
    ax.legend(loc='upper left')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    output_file = "simulation_plot.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Simulation plot saved to {output_file}")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    create_simulation_plot() 