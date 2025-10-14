"""
Week of Month Regressor for the new backtesting pipeline.

This module provides the compute_week_of_month function that computes
one-hot encoding for week of month (4 weeks).
"""

import pandas as pd
from typing import List, Optional
from typeguard import typechecked


@typechecked
def compute_week_of_month(
    df: pd.DataFrame,
    product_master_df: Optional[pd.DataFrame],
    **kwargs
) -> List[pd.Series]:
    """
    Compute one-hot encoding for week of month (4 weeks).
    
    Returns a list of 4 pandas Series, where each Series is a binary flag
    indicating if the date falls in that specific week of the month.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with date column
    product_master_df : pd.DataFrame
        Product master dataframe (not used but required for interface consistency)
        
    Returns
    -------
    List[pd.Series]
        List of 4 binary Series: [week_1, week_2, week_3, week_4]
        Each Series contains 1 if date is in that week, 0 otherwise
        
    Examples
    --------
    # Returns 4 Series: week_1, week_2, week_3, week_4
    week_series = compute_week_of_month(df, product_master_df)
    
    # Each Series is binary (0 or 1)
    # week_1: 1 for days 1-7, 0 otherwise
    # week_2: 1 for days 8-14, 0 otherwise  
    # week_3: 1 for days 15-21, 0 otherwise
    # week_4: 1 for days 22-31, 0 otherwise
    """
    # Validate inputs
    if 'date' not in df.columns:
        raise ValueError("DataFrame must contain 'date' column")
    
    # Calculate week of month (1-4)
    # Week 1: days 1-7, Week 2: days 8-14, Week 3: days 15-21, Week 4: days 22-31
    week_of_month = ((df["date"].dt.day - 1) // 7 + 1).clip(upper=4)
    
    # Create one-hot encoding as list of Series
    week_series = []
    for week_num in range(1, 5):
        week_flag = (week_of_month == week_num).astype(int)
        week_series.append(week_flag)
    
    return week_series
