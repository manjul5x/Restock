"""
Outlier detection utilities for demand data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import date
from pathlib import Path
from data.loader import DataLoader
from ..validation import ProductMasterSchema

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
        Q1 = demand_series.quantile(0.25)
        Q3 = demand_series.quantile(0.75)
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
        z_scores = np.abs((demand_series - demand_series.mean()) / demand_series.std())
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
        median = demand_series.median()
        mad = np.median(np.abs(demand_series - median))
        
        # Convert MAD to standard deviation approximation
        mad_std = mad * 1.4826
        
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
        rolling_mean = demand_series.rolling(window=window, center=True).mean()
        rolling_std = demand_series.rolling(window=window, center=True).std()
        
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
