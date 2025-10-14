"""
Demand classification functions for insights.

This module contains functionality for classifying products based on their demand patterns,
following the methodology of Syntetos et al. (2005).
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Parameterising the demand interval
INTERDEMAND_INTERVAL = 1.32

# Parameterising the CV2 interval
CV2_INTERVAL = 0.49


def demand_classification(table, group_variables, unit_to_round_over='day', revenue_column=None, start_date=None, end_date=None):
    """
    Returns demand classifications by product from a table containing daily demand and inventory data
    
    Parameters:
    -----------
    table : pd.DataFrame
        A dataframe containing daily inventory and demand data for various products
    group_variables : list
        List of column names to group by (e.g., ['product_id'] or ['product_id', 'location_id'])
    unit_to_round_over : str, optional
        Unit to round over. Can be 'day', 'week', 'month', or 'year'. Default is 'day'.
    revenue_column : str, optional
        Name of the column containing revenue data. If provided, revenue statistics will be included.
    start_date : str or datetime, optional
        Start date for filtering data (inclusive). Format: 'YYYY-MM-DD'
    end_date : str or datetime, optional
        End date for filtering data (inclusive). Format: 'YYYY-MM-DD'
    
    Returns:
    --------
    pd.DataFrame
        A dataframe with a row per product and associated demand classification
    """
    
    # Set frequency based on rounding unit
    freq_map = {
        'year': 365,
        'month': 30,
        'week': 7,
        'day': 1
    }
    
    if unit_to_round_over in freq_map:
        freq = freq_map[unit_to_round_over]
    else:
        freq = 1
    
    # Ensure date column is datetime
    table = table.copy()
    table['date'] = pd.to_datetime(table['date'])
    
    # Filter by date range if provided
    if start_date is not None:
        start_date = pd.to_datetime(start_date)
        table = table[table['date'] >= start_date]
    
    if end_date is not None:
        end_date = pd.to_datetime(end_date)
        table = table[table['date'] <= end_date]
    
    # Get demand mean_interdemand_interval & CV2
    # Filter for non-zero demand
    demand_data = table[table['demand'] != 0].copy()
    
    # Round dates based on unit
    if unit_to_round_over == 'day':
        demand_data['date'] = demand_data['date'].dt.date
    elif unit_to_round_over == 'week':
        demand_data['date'] = demand_data['date'].dt.to_period('W').dt.start_time.dt.date
    elif unit_to_round_over == 'month':
        demand_data['date'] = demand_data['date'].dt.to_period('M').dt.start_time.dt.date
    elif unit_to_round_over == 'year':
        demand_data['date'] = demand_data['date'].dt.to_period('Y').dt.start_time.dt.date
    
    # Group by specified variables and date, sum demand
    agg_dict = {'demand': 'sum'}
    if revenue_column:
        agg_dict[revenue_column] = 'sum'
    
    grouped_demand = (demand_data
                     .groupby(group_variables + ['date'])
                     .agg(agg_dict)
                     .reset_index())
    
    # Calculate interdemand intervals and statistics for each group
    def calculate_stats(group):
        group = group.sort_values('date')
        
        # Calculate interdemand intervals
        if len(group) > 1:
            # Convert dates to datetime for diff calculation
            group['date'] = pd.to_datetime(group['date'])
            date_diffs = group['date'].diff().dt.days / freq
            interdemand_intervals = date_diffs.dropna().round(0)
            mean_interdemand_interval = interdemand_intervals.mean()
        else:
            mean_interdemand_interval = np.nan
        
        # Calculate CV2 (coefficient of variation squared)
        demand_values = group['demand']
        if len(demand_values) > 1 and demand_values.mean() > 0:
            cv2 = (demand_values.std() / demand_values.mean()) ** 2
        else:
            cv2 = np.nan
        
        stats = {
            'nyearlyorders': len(group),
            'mean_interdemand_interval': mean_interdemand_interval,
            'CV2': cv2
        }
        
        return pd.Series(stats)
    
    # Use a safer approach for groupby/apply
    result_list = []
    for name, group in grouped_demand.groupby(group_variables):
        # Handle both single and multiple group variables
        if isinstance(name, tuple):
            group_dict = {var: val for var, val in zip(group_variables, name)}
        else:
            group_dict = {group_variables[0]: name}
        
        # Calculate stats for this group
        stats = calculate_stats(group)
        
        # Combine group info with stats
        result = {**group_dict, **stats.to_dict()}
        result_list.append(result)
    
    # Convert results to DataFrame
    x_classified = pd.DataFrame(result_list)
    
    # Classify based on purchase patterns
    only_one = x_classified[x_classified['nyearlyorders'] == 1].copy()
    only_one['type'] = "One Purchase"
    
    more_than_one = x_classified[x_classified['nyearlyorders'] > 1].copy()
    
    # Lumpy: high interdemand interval AND high CV2
    lumpy = more_than_one[
        (more_than_one['mean_interdemand_interval'] > INTERDEMAND_INTERVAL) & 
        (more_than_one['CV2'] >= CV2_INTERVAL)
    ].copy()
    lumpy = lumpy.sort_values(['mean_interdemand_interval', 'CV2'], ascending=[False, False])
    lumpy['type'] = "Lumpy"
    
    # Intermittent: high interdemand interval BUT low CV2
    intermittent = more_than_one[
        (more_than_one['mean_interdemand_interval'] > INTERDEMAND_INTERVAL) & 
        (more_than_one['CV2'] < CV2_INTERVAL)
    ].copy()
    intermittent = intermittent.sort_values(['mean_interdemand_interval', 'CV2'], ascending=[True, False])
    intermittent['type'] = "Intermittent"
    
    # Erratic: low interdemand interval BUT high CV2
    erratic = more_than_one[
        (more_than_one['mean_interdemand_interval'] <= INTERDEMAND_INTERVAL) & 
        (more_than_one['CV2'] > CV2_INTERVAL)
    ].copy()
    erratic = erratic.sort_values(['mean_interdemand_interval', 'CV2'], ascending=[True, False])
    erratic['type'] = "Erratic"
    
    # Smooth: low interdemand interval AND low CV2
    smooth = more_than_one[
        (more_than_one['mean_interdemand_interval'] <= INTERDEMAND_INTERVAL) & 
        (more_than_one['CV2'] <= CV2_INTERVAL)
    ].copy()
    smooth = smooth.sort_values(['mean_interdemand_interval', 'CV2'], ascending=[True, True])
    smooth['type'] = "Smooth"
    
    # Combine all classifications
    classification = pd.concat([only_one, lumpy, intermittent, erratic, smooth], ignore_index=True)
    
    # Calculate summary statistics for the original table
    agg_dict = {
        'demand': ['mean', 'sum']
    }
    if revenue_column:
        agg_dict[revenue_column] = ['mean', 'sum']
    
    table_summary = (table
                    .groupby(group_variables, observed=True)
                    .agg(agg_dict)
                    .round(2))
    
    # Flatten column names
    table_summary.columns = [f"{col[0]}_{col[1]}" for col in table_summary.columns]
    table_summary = table_summary.reset_index()
    
    # Join with classifications
    table_classified = table_summary.merge(
        classification[group_variables + ['type', 'mean_interdemand_interval', 'CV2', 'nyearlyorders']], 
        on=group_variables, 
        how='left'
    )
    
    # Fill missing types with "Zero Demand"
    table_classified['type'] = table_classified['type'].fillna("Zero Demand")
    
    # Set factor levels (category order)
    type_order = ["Smooth", "Intermittent", "Erratic", "Lumpy", "One Purchase", "Zero Demand"]
    table_classified['type'] = pd.Categorical(
        table_classified['type'], 
        categories=type_order, 
        ordered=True
    )
    
    # Fill missing values with zeros
    for col in ['mean_interdemand_interval', 'CV2', 'nyearlyorders']:
        if col in table_classified.columns:
            table_classified[col] = table_classified[col].fillna(0)
    
    return table_classified