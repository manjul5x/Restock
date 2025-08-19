"""
Recency Regressor for the new backtesting pipeline.

This module provides the compute_recency_weight function that computes
weights based on distance from the last review date.
"""

import pandas as pd
import yaml
from pathlib import Path
from typing import Optional
from typeguard import typechecked


@typechecked
def compute_recency_weight(
    df: pd.DataFrame,
    product_master_df: pd.DataFrame,
    **kwargs
) -> pd.Series:
    """
    Compute weight based on distance from last review date.
    
    Weight tiers:
    - 1.0: Within 1 year of last review date
    - 0.8: 1-2 years from last review date  
    - 0.6: More than 2 years from last review date
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with date column
    product_master_df : pd.DataFrame
        Product master dataframe (not used but required for interface consistency)
    **kwargs : Additional parameters (not used but required for interface consistency)
        
    Returns
    -------
    pd.Series
        Weight values: 1.0, 0.8, or 0.6
    """
    # Validate inputs
    if 'date' not in df.columns:
        raise ValueError("DataFrame must contain 'date' column")
    
    # Read review dates from data_config
    config_path = Path("data/config/data_config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Data config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    review_dates = data_config.get('safety_stock', {}).get('review_dates', [])
    if not review_dates:
        raise ValueError("No review dates found in data_config.yaml under safety_stock.review_dates")
    
    # Convert review dates to datetime for proper max() comparison
    review_dates_dt = [pd.to_datetime(date_str) for date_str in review_dates]
    last_review_dt = max(review_dates_dt)
    
    # Calculate days since last review
    days_since_review = (df["date"] - last_review_dt).dt.days
    
    # Apply tiered weight system
    # 1.0: ≤ 365 days (within 1 year)
    # 0.8: 366-730 days (1-2 years)  
    # 0.6: > 730 days (more than 2 years)
    
    weights = pd.Series(0.6, index=df.index)  # Default weight
    weights[days_since_review <= 365] = 1.0    # Within 1 year
    weights[(days_since_review > 365) & (days_since_review <= 730)] = 0.8  # 1-2 years
    
    return weights
