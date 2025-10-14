"""
Lead/Lag Aggregation Regressor for the new backtesting pipeline.

This module provides the compute_lead_lag_aggregation function that computes
rolling sums over time windows with support for both forward-looking (lead)
and backward-looking (lag) windows.
"""

import pandas as pd
import numpy as np
try:
    from forecaster.validation.product_master_schema import ProductMasterSchema
except ImportError:
    # Schema module requires pydantic which may not be installed
    ProductMasterSchema = None
from typing import Union, List, Optional, Literal
from typeguard import typechecked

@typechecked
def compute_lead_lag_aggregation(
    df: pd.DataFrame,
    product_master_df: Optional[pd.DataFrame], #used for risk period lookup, pass None if not needed
    lead_or_lag: Literal["lead", "lag"], # "lead" for forward, "lag" for backward
    value_col: str = "demand",
    date_col: str = "date",
    group_cols: Optional[List[str]] = None,
    window_days: Union[int, Literal["rp"]] = "rp", # "rp" for risk period, int for days
    rp_scaler: Optional[float] = None,
    incomplete: Literal["partial", "nan", "scale"] = "nan" # "partial", "nan", or "scale"
) -> pd.Series:
    """
    Compute rolling sum over a time window in days per group using efficient cumulative sums.
    
    This function handles forward-looking (lead) and backward-looking (lag) windows differently:
    - Lead: Includes current date + next (window_days-1) days
    - Lag: Includes previous window_days days (excludes current date)
    
    Uses merge_asof for efficient date boundary lookups on large datasets.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with datetime column and value column.
    product_master_df : pd.DataFrame
        Product master dataframe with product configurations
    value_col : str, default "demand"
        Name of the column containing values to sum.
    date_col : str, default "date"
        Name of the datetime column for window boundaries.
    group_cols : list, default None
        List of columns to group by (e.g., ['product_id', 'location_id']).
        If None, defaults to ['product_id', 'location_id'].
    lead_or_lag : str, default "lead"
        "lead": Forward-looking window (current date + future days)
        "lag": Backward-looking window (past days, excluding current)
    window_days : int | str, default "rp"
        Size of the rolling window in days.
        "rp": Risk period fetched from product master
        int: Fixed number of days
    rp_scaler : float, default None
        Scaler if basing window on risk period
        0.5: half risk period
        1: full risk period
        2: 2x risk period
    incomplete : str, default "nan"
        How to handle incomplete windows at data boundaries:
        "partial": Allow incomplete windows
        "nan": Set incomplete windows to NaN (default)
        "scale": Scale up incomplete windows proportionally to how many days are missing
    
    Returns
    -------
    pd.Series
        Rolling sum aligned to the original df's index.
    
    Examples
    --------
    # Forward-looking 3-day window (current + next 2 days)
    df['future_3d'] = compute_lead_lag_aggregation(
        df, product_master_df, 'demand', 'date', ['product_id', 'location_id'], 3, 'lead'
    )
    
    # Backward-looking 3-day window (previous 3 days, excluding current)
    df['past_3d'] = compute_lead_lag_aggregation(
        df, product_master_df, 'demand', 'date', ['product_id', 'location_id'], 3, 'lag'
    )
    
    # Risk period based window with half risk period
    df['half_rp_lag'] = compute_lead_lag_aggregation(
        df, product_master_df, 'demand', 'date', ['product_id', 'location_id'], 'rp', 0.5, 'lag'
    )
    
    Notes
    -----
    - For lead=3: sums current_date + next 2 days (3 total days)
    - For lag=3: sums 3 days ago + 2 days ago + 1 day ago (3 total days, excluding current)
    - Uses cumulative sums for O(n) performance on large datasets
    - Handles missing dates gracefully through merge_asof
    
    Performance Characteristics:
    - Time complexity: O(n log n) due to sorting and merge_asof
    - Space complexity: O(n) for temporary columns
    - Optimized for large datasets with many groups
    - Cumulative sum approach avoids repeated date range calculations
    - merge_asof handles missing dates efficiently
    """

    # --- Domain checks ---

    # column existence
    for col in [value_col, date_col] + (group_cols or []):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataframe")

    # group_cols must not be empty list
    if group_cols is not None and not group_cols:
        raise ValueError("group_cols cannot be an empty list if provided")

    # window_days must be positive if integer
    if isinstance(window_days, int) and window_days <= 0:
        raise ValueError("window_days must be a positive integer")

    # rp_scaler must be positive if given
    if rp_scaler is not None and rp_scaler <= 0:
        raise ValueError("rp_scaler must be > 0 if provided")

    #  -----  Main function logic -----
    
    # Sort and copy to avoid modifying original 
    orig_index = df.index
    sort_cols = group_cols + [date_col]
    key_index = df.set_index(sort_cols).index
    df_sorted = df.sort_values(sort_cols).copy()
    
    # Determine direction and sign
    diff_sign = -1 if lead_or_lag == 'lag' else 1

    # Cumulative sum per group
    df_sorted["_cumsum"] = df_sorted.groupby(group_cols, sort=False)[value_col].cumsum()
    
    # Add a window_days_col to df_sorted, one per row, based on window_days argument
    if isinstance(window_days, int):
        # Use the same window_days for all rows
        df_sorted["_window_days"] = window_days
    elif window_days == "rp":
        # Need to get risk period per product_id/location_id, and apply window scaler if provided
        # Merge risk_period_days from product_master_df
        # Default: product_id, location_id
        merge_cols = ["product_id", "location_id"]
        df_sorted = df_sorted.merge(
            product_master_df[merge_cols + ["risk_period", "demand_frequency"]],
            on=merge_cols,
            how="left",
            suffixes=("", "_pm")
        )

        # Convert risk_period to risk_period_days using get_risk_period_days from ProductMasterSchema
        df_sorted["risk_period_days"] = df_sorted.apply(
            lambda row: ProductMasterSchema.get_risk_period_days(row["demand_frequency"], row["risk_period"]),
            axis=1
        )
        
        # Apply window scaler if provided
        if rp_scaler is not None:
            df_sorted["_window_days"] = (df_sorted["risk_period_days"] * rp_scaler).round().astype(int)
        else:
            df_sorted["_window_days"] = df_sorted["risk_period_days"]
    else:
        raise ValueError("window_days must be int or 'rp' (optionally with scaler)")

    # Set anchor dates for window boundaries
    if lead_or_lag == 'lead':
        # Lead: anchor at (current_date + window_days) for backward merge
        # This ensures we include current_date when looking backward from future point
        # end of the window is excluded in the merge direction later
        # No, we do not have to use pd.Timedelta here; we can use pd.to_timedelta, which is vectorized and more efficient for Series.
        df_sorted["_anchor_date"] = df_sorted[date_col] + pd.to_timedelta(df_sorted["_window_days"], unit="D")
        allow_exact_matches = False
    else:
        # Lag: anchor at (current_date - window_days - 1) for backward merge  
        # This ensures we exclude current_date when looking backward from past point
        # cumulative sum at the start of the window-1 is the baseline for the window
        df_sorted["_anchor_date"] = df_sorted[date_col] - pd.to_timedelta(df_sorted["_window_days"] + 1, unit="D")
        allow_exact_matches = True

    # Prepare lookup for window end cumsum
    window_boundaries = df_sorted[[date_col, "_cumsum"] + group_cols].rename(
        columns={date_col: "_anchor_date", "_cumsum": "window_boundary_cumsum"}
    )

    # Use merge_asof to efficiently find cumulative sums at window boundaries
    # This avoids expensive date range filtering on large datasets
    merged = pd.merge_asof(
        df_sorted.sort_values(["_anchor_date"] + group_cols),
        window_boundaries.sort_values(["_anchor_date"] + group_cols),
        on="_anchor_date",
        by=group_cols,
        direction="backward",
        allow_exact_matches=allow_exact_matches
    ).fillna(0)  # fillna(0) for the very first entry in the lag scenario

    # Compute rolling sum using cumulative sum differences
    merged["_roll_sum"] = abs(diff_sign * (merged["window_boundary_cumsum"] - merged["_cumsum"] + merged[value_col]))
    
    # Handle incomplete windows
    if incomplete != 'partial':
        cutoff_date = merged.groupby(group_cols, sort=False)[date_col].transform(
            'min' if lead_or_lag == 'lag' else 'max')
        merged['days_missing'] = (diff_sign * ((merged['_anchor_date'] - cutoff_date).dt.days) - 1).clip(lower=0)
        if incomplete == 'scale':
            denom = merged['_window_days'] - merged['days_missing']
            merged["scaler"] = merged['_window_days'] / denom.replace(0, np.nan)
            merged["roll_dem"] = merged["_roll_sum"] * merged["scaler"]
            merged["roll_dem"] = merged["roll_dem"].fillna(merged["roll_dem"].median())
        elif incomplete == 'nan':
            merged["roll_dem"] = np.where(merged['days_missing'] > 0, np.nan, merged["_roll_sum"])
    else:
        merged["roll_dem"] = merged["_roll_sum"]

    # reset the index so that rows are aligned and series matches correctly
    result = merged.set_index(sort_cols)["roll_dem"]
    result = result.reindex(key_index)
    result.index = orig_index

    return result
