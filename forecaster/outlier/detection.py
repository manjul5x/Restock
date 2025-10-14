"""
Outlier detection utilities for demand data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import date
from pathlib import Path
from data.loader import DataLoader
try:
    from ..validation import ProductMasterSchema
except ImportError:
    # Schema module requires pydantic which may not be installed
    ProductMasterSchema = None

class OutlierDetector:
    """Utilities for detecting outliers in demand data"""
    
    def __init__(self, data_loader: DataLoader = None):
        self.data_loader = data_loader or DataLoader()
    
    def detect_outliers_iqr(self, 
                           demand_series: pd.Series,
                           multiplier: float = 1.5) -> pd.Series:
        """
        Detect outliers using Interquartile Range (IQR) method
        
        Args:
            demand_series: Series of demand values
            multiplier: IQR multiplier (default 1.5)
            
        Returns:
            Boolean series indicating outliers
        """
        # Exclude zeros for threshold calculation
        non_zero_demand = demand_series[demand_series > 0]
        
        # If all values are zero or only one non-zero value, no outliers
        if len(non_zero_demand) <= 1:
            return pd.Series([False] * len(demand_series), index=demand_series.index)
        
        Q1 = non_zero_demand.quantile(0.25)
        Q3 = non_zero_demand.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        outliers = (demand_series < lower_bound) | (demand_series > upper_bound)
        return outliers
    
    def detect_outliers_zscore(self, 
                              demand_series: pd.Series,
                              threshold: float = 3.0) -> pd.Series:
        """
        Detect outliers using Z-score method
        
        Args:
            demand_series: Series of demand values
            threshold: Z-score threshold (default 3.0)
            
        Returns:
            Boolean series indicating outliers
        """
        # Exclude zeros for threshold calculation
        non_zero_demand = demand_series[demand_series > 0]
        
        # If all values are zero or only one non-zero value, no outliers
        if len(non_zero_demand) <= 1:
            return pd.Series([False] * len(demand_series), index=demand_series.index)
        
        mean_val = non_zero_demand.mean()
        std_val = non_zero_demand.std()
        
        # Avoid division by zero
        if std_val == 0:
            return pd.Series([False] * len(demand_series), index=demand_series.index)
        
        z_scores = np.abs((demand_series - mean_val) / std_val)
        outliers = z_scores > threshold
        return outliers
    
    def detect_outliers_mad(self, 
                           demand_series: pd.Series,
                           threshold: float = 3.5) -> pd.Series:
        """
        Detect outliers using Median Absolute Deviation (MAD) method
        
        Args:
            demand_series: Series of demand values
            threshold: MAD threshold (default 3.5)
            
        Returns:
            Boolean series indicating outliers
        """
        # Exclude zeros for threshold calculation
        non_zero_demand = demand_series[demand_series > 0]
        
        # If all values are zero or only one non-zero value, no outliers
        if len(non_zero_demand) <= 1:
            return pd.Series([False] * len(demand_series), index=demand_series.index)
        
        median = non_zero_demand.median()
        mad = np.median(np.abs(non_zero_demand - median))
        
        # Convert MAD to standard deviation approximation
        mad_std = mad * 1.4826
        
        # Avoid division by zero
        if mad_std == 0:
            return pd.Series([False] * len(demand_series), index=demand_series.index)
        
        z_scores = np.abs((demand_series - median) / mad_std)
        outliers = z_scores > threshold
        return outliers
    
    def detect_outliers_rolling(self, 
                               demand_series: pd.Series,
                               window: int = 30,
                               multiplier: float = 2.0) -> pd.Series:
        """
        Detect outliers using rolling statistics
        
        Args:
            demand_series: Series of demand values
            window: Rolling window size (default 30)
            multiplier: Standard deviation multiplier (default 2.0)
            
        Returns:
            Boolean series indicating outliers
        """
        # For rolling method, we need to handle zeros differently
        # We'll use a minimum window size and handle edge cases
        min_window = min(window, len(demand_series) // 2)
        if min_window < 3:
            min_window = 3
        
        rolling_mean = demand_series.rolling(window=min_window, center=True, min_periods=1).mean()
        rolling_std = demand_series.rolling(window=min_window, center=True, min_periods=1).std()
        
        # Fill NaN values with overall statistics (excluding zeros)
        non_zero_demand = demand_series[demand_series > 0]
        if len(non_zero_demand) > 0:
            overall_mean = non_zero_demand.mean()
            overall_std = non_zero_demand.std()
        else:
            overall_mean = 0
            overall_std = 0
        
        rolling_mean = rolling_mean.fillna(overall_mean)
        rolling_std = rolling_std.fillna(overall_std)
        
        upper_bound = rolling_mean + multiplier * rolling_std
        lower_bound = rolling_mean - multiplier * rolling_std
        
        outliers = (demand_series > upper_bound) | (demand_series < lower_bound)
        return outliers
    
    def get_outlier_detection_method(self, method_name: str):
        """Get outlier detection method by name"""
        methods = {
            'iqr': self.detect_outliers_iqr,
            'zscore': self.detect_outliers_zscore,
            'mad': self.detect_outliers_mad,
            'rolling': self.detect_outliers_rolling
        }
        return methods.get(method_name, self.detect_outliers_iqr)

def detect_outliers_iqr(demand_series: pd.Series, multiplier: float = 1.5) -> pd.Series:
    """Quick function to detect outliers using IQR method"""
    detector = OutlierDetector()
    return detector.detect_outliers_iqr(demand_series, multiplier)

def detect_outliers_zscore(demand_series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Quick function to detect outliers using Z-score method"""
    detector = OutlierDetector()
    return detector.detect_outliers_zscore(demand_series, threshold)
