"""
In-Season Regressor for the new backtesting pipeline.

This module provides the compute_in_season function that computes
a binary flag indicating if the risk period midpoint falls within a season window.
"""

import pandas as pd
import numpy as np
from forecaster.validation.product_master_schema import ProductMasterSchema
from typing import Optional, List
from typeguard import typechecked


@typechecked
def compute_in_season(
    df: pd.DataFrame,
    product_master_df: pd.DataFrame,
    season_start: str,
    season_end: str,
) -> pd.Series:
    """
    Compute binary flag indicating if risk period midpoint falls within season window.
    
    Uses month/day comparison (ignoring year) to handle multi-year data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with date, product_id, location_id columns
    product_master_df : pd.DataFrame
        Product master dataframe with risk_period and demand_frequency columns
    season_start : str, default "09-01"
        Season start date in MM-DD format
    season_end : str, default "12-31"
        Season end date in MM-DD format
        
    Returns
    -------
    pd.Series
        Binary flag: 1 if in season, 0 if not
        
    Raises
    ------
    ValueError
        If risk period information is missing for any product/location
    """
    # Validate inputs
    if not all(col in df.columns for col in ['date', 'product_id', 'location_id']):
        raise ValueError("DataFrame must contain 'date', 'product_id', 'location_id' columns")
    
    if not all(col in product_master_df.columns for col in ['product_id', 'location_id', 'risk_period', 'demand_frequency']):
        raise ValueError("Product master DataFrame must contain 'product_id', 'location_id', 'risk_period', 'demand_frequency' columns")
    
    # Create a copy to avoid modifying original
    result_df = df.copy()
    
    # Merge risk period information
    merge_cols = ["product_id", "location_id"]
    result_df = result_df.merge(
        product_master_df[merge_cols + ["risk_period", "demand_frequency"]],
        on=merge_cols,
        how="left",
        suffixes=("", "_pm")
    )
    
    # Check for missing risk period information
    missing_risk_period = result_df[['risk_period', 'demand_frequency']].isnull().any(axis=1)
    if missing_risk_period.any():
        missing_products = result_df[missing_risk_period][merge_cols].drop_duplicates()
        raise ValueError(f"Missing risk period information for products: {missing_products.to_dict('records')}")
    
    # Convert risk_period to risk_period_days
    result_df["risk_period_days"] = result_df.apply(
        lambda row: ProductMasterSchema.get_risk_period_days(row["demand_frequency"], row["risk_period"]),
        axis=1
    )
    
    # Calculate risk period midpoint
    result_df["risk_period_midpoint"] = result_df["date"] + pd.to_timedelta(result_df["risk_period_days"] / 2, unit="D")
    
    # Normalize all dates to year 2000 for consistent season comparison
    # Extract month and day as integers and create normalized dates
    midpoint_month = result_df["risk_period_midpoint"].dt.month
    midpoint_day = result_df["risk_period_midpoint"].dt.day
    
    # Create normalized dates in year 2000
    normalized_midpoints_2000 = pd.to_datetime({
        'year': 2000,
        'month': midpoint_month,
        'day': midpoint_day
    })
    
    # Create season start and end dates in year 2000
    season_start_2000 = f"2000-{season_start}"
    season_end_2000 = f"2000-{season_end}"
    
    # Convert to datetime for comparison
    season_start_dt = pd.to_datetime(season_start_2000)
    season_end_dt = pd.to_datetime(season_end_2000)
    
    # Check if season spans year boundary (e.g., 11-01 to 02-01)
    if season_start_dt > season_end_dt:
        # Season spans year boundary - flip the logic
        # In season if: (midpoint >= start) OR (midpoint <= end)
        in_season = (
            (normalized_midpoints_2000 <= season_start_dt) | 
            (normalized_midpoints_2000 >= season_end_dt)
        )
    else:
        # Season within same year (e.g., 03-01 to 08-31)
        # In season if: (midpoint >= start) AND (midpoint <= end)
        in_season = (
            (normalized_midpoints_2000 >= season_start_dt) & 
            (normalized_midpoints_2000 <= season_end_dt)
        )
    
    return in_season.astype(int)


