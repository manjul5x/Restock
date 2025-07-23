"""
Final forecaster implementation ready for integration.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime
from data_prep import prepare_data, get_training_data, get_target
from weighted_ma_forecaster import WeightedMAForecaster
from seasonal_forecaster import SeasonalForecaster

class FinalForecaster:
    """
    Final forecaster implementation that combines multiple methods
    for optimal performance. Uses:
    1. Linear weighted MA (34%)
    2. Exponential weighted MA (34%)
    3. Seasonal patterns (31%)
    """
    
    def __init__(self, window_length: int = 25):
        """
        Initialize forecaster
        
        Args:
            window_length: Number of historical periods to use
        """
        self.window_length = window_length
        self.is_fitted = False
        
        # Initialize component forecasters with optimal configuration
        self.forecasters = {
            'linear_ma': WeightedMAForecaster(
                window_length=window_length,
                weight_type='linear'
            ),
            'exp_ma': WeightedMAForecaster(
                window_length=window_length,
                weight_type='exponential',
                alpha=0.2
            ),
            'seasonal': SeasonalForecaster(
                window_length=window_length,
                seasonal_period=7
            )
        }
        
        # Set optimal weights based on evaluation
        self.weights = {
            'linear_ma': 0.3429,
            'exp_ma': 0.3438,
            'seasonal': 0.3133
        }
        
    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit the forecaster to training data
        
        Args:
            data: Training data DataFrame with required columns:
                - historical_bucket_start_dates: List of historical dates
                - historical_demands: List of historical demand values
                - forecast_horizon_start_dates: Target forecast date
        """
        if len(data) == 0:
            raise ValueError("Empty training data")
            
        # Store the last row for forecasting
        self.last_row = data.iloc[-1]
        
        # Get training data
        dates, demands = get_training_data(self.last_row)
        
        # Use only the specified window length if provided
        if self.window_length is not None:
            dates = dates[-self.window_length:]
            demands = demands[-self.window_length:]
            
        # Fit all component forecasters
        for forecaster in self.forecasters.values():
            forecaster._fit(dates, demands)
            
        self.is_fitted = True
        
    def forecast(self, steps: int = 1) -> List[float]:
        """
        Generate forecasts
        
        Args:
            steps: Number of steps ahead to forecast
            
        Returns:
            List of forecasted values
        """
        if not self.is_fitted:
            raise ValueError("Forecaster must be fitted before forecasting")
            
        # Get target date
        target_date = self.last_row['forecast_horizon_start_dates']
        
        # Generate forecasts
        forecasts = []
        for i in range(steps):
            # Get forecasts from each component
            step_forecasts = {}
            for name, forecaster in self.forecasters.items():
                forecast = forecaster._forecast(target_date)
                step_forecasts[name] = forecast
                
            # Compute weighted average
            weighted_forecast = sum(forecast * self.weights[name]
                                  for name, forecast in step_forecasts.items())
            forecasts.append(float(weighted_forecast))
            
            # Update target date for next step
            target_date = target_date + pd.Timedelta(days=14)  # Assuming 14-day periods
            
        return forecasts
        
    def get_component_weights(self) -> Dict[str, float]:
        """
        Get the weights of each component forecaster
        
        Returns:
            Dictionary mapping forecaster name to weight
        """
        return self.weights.copy()
        
    @classmethod
    def create(cls, **kwargs) -> 'FinalForecaster':
        """
        Factory method to create forecaster instance
        
        Args:
            **kwargs: Keyword arguments for initialization
            
        Returns:
            Configured forecaster instance
        """
        return cls(
            window_length=kwargs.get('window_length', 25)
        ) 