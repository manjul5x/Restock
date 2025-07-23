"""
Visualization module for demand forecasting data.
Provides interactive line graphs with filtering and aggregation capabilities.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Dict, Tuple
from datetime import date, timedelta
import warnings

# Suppress matplotlib warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

class DemandVisualizer:
    """
    Visualization class for demand data with filtering and aggregation capabilities.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the visualizer with demand data.
        
        Args:
            data: DataFrame with demand data (must include columns: 
                  product_id, product_category, location_id, date, demand)
        """
        self.data = data.copy()
        self.data['date'] = pd.to_datetime(self.data['date']).dt.date
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def get_available_filters(self) -> Dict[str, List]:
        """
        Get available filter options from the data.
        
        Returns:
            Dictionary with available locations, categories, and products
        """
        return {
            'locations': sorted(self.data['location_id'].unique().tolist()),
            'categories': sorted(self.data['product_category'].unique().tolist()),
            'products': sorted(self.data['product_id'].unique().tolist())
        }
    
    def filter_data(self, 
                   locations: Optional[List[str]] = None,
                   categories: Optional[List[str]] = None,
                   products: Optional[List[str]] = None,
                   start_date: Optional[date] = None,
                   end_date: Optional[date] = None) -> pd.DataFrame:
        """
        Filter data based on specified criteria.
        
        Args:
            locations: List of location IDs to include
            categories: List of product categories to include
            products: List of product IDs to include
            start_date: Start date for filtering
            end_date: End date for filtering
            
        Returns:
            Filtered DataFrame
        """
        filtered_data = self.data.copy()
        
        if locations:
            filtered_data = filtered_data[filtered_data['location_id'].isin(locations)]
        
        if categories:
            filtered_data = filtered_data[filtered_data['product_category'].isin(categories)]
        
        if products:
            filtered_data = filtered_data[filtered_data['product_id'].isin(products)]
        
        if start_date:
            filtered_data = filtered_data[filtered_data['date'] >= start_date]
        
        if end_date:
            filtered_data = filtered_data[filtered_data['date'] <= end_date]
        
        return filtered_data
    
    def aggregate_data(self, 
                      data: pd.DataFrame,
                      group_by: List[str],
                      time_aggregation: Optional[str] = None) -> pd.DataFrame:
        """
        Aggregate data by specified grouping columns and time period.
        
        Args:
            data: DataFrame to aggregate
            group_by: List of columns to group by (e.g., ['date', 'location_id'])
            time_aggregation: Time aggregation level ('daily', 'weekly', 'monthly', None)
            
        Returns:
            Aggregated DataFrame
        """
        # Ensure date is in the group_by for time series
        if 'date' not in group_by:
            group_by = ['date'] + group_by
        
        # Apply time aggregation if specified
        if time_aggregation:
            data = data.copy()
            if time_aggregation == 'weekly':
                data['date'] = data['date'].dt.to_period('W').dt.start_time
            elif time_aggregation == 'monthly':
                data['date'] = data['date'].dt.to_period('M').dt.start_time
            # For 'daily', no change needed
        
        # Aggregate demand by summing
        aggregated = data.groupby(group_by)['demand'].sum().reset_index()
        
        return aggregated.sort_values('date')
    
    def plot_demand_trend(self,
                         locations: Optional[List[str]] = None,
                         categories: Optional[List[str]] = None,
                         products: Optional[List[str]] = None,
                         start_date: Optional[date] = None,
                         end_date: Optional[date] = None,
                         group_by: Optional[List[str]] = None,
                         time_aggregation: Optional[str] = None,
                         figsize: Tuple[int, int] = (12, 6),
                         title: Optional[str] = None,
                         show_legend: bool = True) -> plt.Figure:
        """
        Create a line plot of demand trends with filtering and aggregation.
        
        Args:
            locations: List of location IDs to include
            categories: List of product categories to include
            products: List of product IDs to include
            start_date: Start date for filtering
            end_date: End date for filtering
            group_by: Columns to group by for aggregation (e.g., ['location_id', 'product_category'])
            figsize: Figure size (width, height)
            title: Plot title
            show_legend: Whether to show legend
            
        Returns:
            Matplotlib figure object
        """
        # Filter data
        filtered_data = self.filter_data(
            locations=locations,
            categories=categories,
            products=products,
            start_date=start_date,
            end_date=end_date
        )
        
        if filtered_data.empty:
            raise ValueError("No data matches the specified filters")
        
        # Determine grouping
        if group_by is None:
            # Default: group by date only (total demand over time)
            group_by = ['date']
        
        # Aggregate data
        aggregated_data = self.aggregate_data(filtered_data, group_by, time_aggregation)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot based on grouping
        if len(group_by) == 1 and group_by[0] == 'date':
            # Simple time series
            ax.plot(aggregated_data['date'], aggregated_data['demand'], 
                   linewidth=2, marker='o', markersize=4)
            ax.set_ylabel('Total Demand')
            
        elif 'date' in group_by and len(group_by) == 2:
            # Group by date and one other dimension
            other_dim = [col for col in group_by if col != 'date'][0]
            
            for value in aggregated_data[other_dim].unique():
                subset = aggregated_data[aggregated_data[other_dim] == value]
                ax.plot(subset['date'], subset['demand'], 
                       linewidth=2, marker='o', markersize=4, label=str(value))
            
            ax.set_ylabel('Demand')
            if show_legend:
                ax.legend(title=other_dim.replace('_', ' ').title())
                
        elif 'date' in group_by and len(group_by) == 3:
            # Group by date and two other dimensions
            other_dims = [col for col in group_by if col != 'date']
            
            # Create a unique identifier for each combination
            aggregated_data['combination'] = aggregated_data[other_dims[0]] + ' - ' + aggregated_data[other_dims[1]]
            
            for combo in aggregated_data['combination'].unique():
                subset = aggregated_data[aggregated_data['combination'] == combo]
                ax.plot(subset['date'], subset['demand'], 
                       linewidth=2, marker='o', markersize=4, label=combo)
            
            ax.set_ylabel('Demand')
            if show_legend:
                ax.legend(title=f"{other_dims[0].replace('_', ' ').title()} - {other_dims[1].replace('_', ' ').title()}")
        
        # Customize plot
        ax.set_xlabel('Date')
        ax.grid(True, alpha=0.3)
        
        if title:
            ax.set_title(title)
        else:
            # Generate title based on filters
            title_parts = []
            if locations:
                title_parts.append(f"Locations: {', '.join(locations)}")
            if categories:
                title_parts.append(f"Categories: {', '.join(categories)}")
            if products:
                title_parts.append(f"Products: {', '.join(products)}")
            
            if time_aggregation:
                title_parts.append(f"Aggregation: {time_aggregation.title()}")
            
            if title_parts:
                ax.set_title(" | ".join(title_parts))
            else:
                ax.set_title("Demand Trend")
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
    
    def plot_sku_location_demand(self,
                                sku_location_pairs: List[Tuple[str, str]],
                                start_date: Optional[date] = None,
                                end_date: Optional[date] = None,
                                figsize: Tuple[int, int] = (12, 6),
                                title: Optional[str] = None) -> plt.Figure:
        """
        Plot demand for specific SKU-location combinations.
        
        Args:
            sku_location_pairs: List of (product_id, location_id) tuples
            start_date: Start date for filtering
            end_date: End date for filtering
            figsize: Figure size
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        for product_id, location_id in sku_location_pairs:
            # Filter for specific SKU-location combination
            subset = self.filter_data(
                products=[product_id],
                locations=[location_id],
                start_date=start_date,
                end_date=end_date
            )
            
            if not subset.empty:
                # Sort by date
                subset = subset.sort_values('date')
                ax.plot(subset['date'], subset['demand'], 
                       linewidth=2, marker='o', markersize=4, 
                       label=f"{product_id} - {location_id}")
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Demand')
        if title:
            ax.set_title(title)
        else:
            ax.set_title('SKU-Location Demand Trends')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
    
    def plot_category_comparison(self,
                                locations: Optional[List[str]] = None,
                                start_date: Optional[date] = None,
                                end_date: Optional[date] = None,
                                time_aggregation: Optional[str] = None,
                                figsize: Tuple[int, int] = (12, 6),
                                title: Optional[str] = None) -> plt.Figure:
        """
        Compare demand trends across different product categories.
        
        Args:
            locations: List of location IDs to include
            start_date: Start date for filtering
            end_date: End date for filtering
            figsize: Figure size
            
        Returns:
            Matplotlib figure object
        """
        # Filter data
        filtered_data = self.filter_data(
            locations=locations,
            start_date=start_date,
            end_date=end_date
        )
        
        # Aggregate by date and category
        aggregated_data = self.aggregate_data(filtered_data, ['date', 'product_category'], time_aggregation)
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        for category in aggregated_data['product_category'].unique():
            subset = aggregated_data[aggregated_data['product_category'] == category]
            ax.plot(subset['date'], subset['demand'], 
                   linewidth=2, marker='o', markersize=4, label=category)
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Total Demand')
        if title:
            ax.set_title(title)
        else:
            ax.set_title('Demand by Product Category')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
    
    def plot_location_comparison(self,
                                categories: Optional[List[str]] = None,
                                start_date: Optional[date] = None,
                                end_date: Optional[date] = None,
                                time_aggregation: Optional[str] = None,
                                figsize: Tuple[int, int] = (12, 6),
                                title: Optional[str] = None) -> plt.Figure:
        """
        Compare demand trends across different locations.
        
        Args:
            categories: List of product categories to include
            start_date: Start date for filtering
            end_date: End date for filtering
            figsize: Figure size
            
        Returns:
            Matplotlib figure object
        """
        # Filter data
        filtered_data = self.filter_data(
            categories=categories,
            start_date=start_date,
            end_date=end_date
        )
        
        # Aggregate by date and location
        aggregated_data = self.aggregate_data(filtered_data, ['date', 'location_id'], time_aggregation)
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        for location in aggregated_data['location_id'].unique():
            subset = aggregated_data[aggregated_data['location_id'] == location]
            ax.plot(subset['date'], subset['demand'], 
                   linewidth=2, marker='o', markersize=4, label=location)
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Total Demand')
        if title:
            ax.set_title(title)
        else:
            ax.set_title('Demand by Location')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
    
    def save_plot(self, fig: plt.Figure, filename: str, dpi: int = 300):
        """
        Save a plot to file.
        
        Args:
            fig: Matplotlib figure object
            filename: Output filename
            dpi: Resolution for saving
        """
        fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        print(f"Plot saved as {filename}")
    
    def show_plot(self, fig: plt.Figure):
        """
        Display a plot.
        
        Args:
            fig: Matplotlib figure object
        """
        plt.show() 