"""
Moving average forecasting implementation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import date, timedelta
from .base import BaseForecaster, ForecastingError, validate_forecast_parameters

class MovingAverageForecaster(BaseForecaster):
    """Moving average forecasting model"""
    
    def __init__(self, window_length: int = None, horizon: int = 1, risk_period_days: int = None):
        """
        Initialize moving average forecaster
        
        Args:
            window_length: Maximum number of periods to include (rolling window limit). 
                          If None, uses all available data.
            horizon: Number of steps to forecast ahead
            risk_period_days: Risk period in days (if None, will be determined from data)
        """
        super().__init__(name="moving_average")
        self.window_length = window_length
        self.horizon = horizon
        self.risk_period_days = risk_period_days
        self.data = None
        self.last_values = None
        
    def _determine_risk_period(self, data: pd.DataFrame, **kwargs) -> int:
        """
        Determine the risk period in days from data or parameters.
        
        Args:
            data: Input data
            **kwargs: Additional arguments that may contain risk_period
            
        Returns:
            Risk period in days
        """
        # First check if risk_period is provided in kwargs
        if 'risk_period' in kwargs:
            risk_period = kwargs['risk_period']
            if isinstance(risk_period, int):
                return risk_period
            elif isinstance(risk_period, str):
                # Try to parse as days
                try:
                    return int(risk_period)
                except ValueError:
                    pass
        
        # Check if risk_period_days is provided in kwargs
        if 'risk_period_days' in kwargs:
            return int(kwargs['risk_period_days'])
        
        # Try to determine from data frequency
        if len(data) >= 2:
            dates = pd.to_datetime(data['date'])
            date_diffs = dates.diff().dropna()
            if len(date_diffs) > 0:
                # Get the most common interval
                most_common_interval = date_diffs.mode().iloc[0]
                if pd.notna(most_common_interval):
                    days = most_common_interval.days
                    if days > 0:
                        return days
        
        # Default to 14 days if we can't determine
        return 14
        
    def fit(self, data: pd.DataFrame, **kwargs) -> 'MovingAverageForecaster':
        """
        Fit the moving average model to the data
        
        Args:
            data: DataFrame with columns ['product_id', 'location_id', 'date', 'demand']
            
        Returns:
            Self for method chaining
        """
        # Validate and prepare data
        self.validate_data(data)
        
        # Prepare data (this data is already aggregated into buckets)
        self.data = self.prepare_data(data)
        
        # Apply rolling window if specified (after bucketing)
        if self.window_length is not None and len(self.data) > self.window_length:
            # Sort by date and take the most recent window_length data points
            self.data = self.data.sort_values('date').tail(self.window_length).reset_index(drop=True)
        
        # Determine risk period if not already set
        if self.risk_period_days is None:
            self.risk_period_days = self._determine_risk_period(data, **kwargs)
        
        # Store last values for forecasting (use all available data)
        if len(self.data) > 0:
            self.last_values = self.data['demand'].values
        
        self.is_fitted = True
        return self
    
    def forecast(self, steps: Optional[int] = None, **kwargs) -> pd.Series:
        """
        Generate forecast for specified number of steps
        
        Args:
            steps: Number of steps to forecast (defaults to self.horizon)
            
        Returns:
            Series with forecast values
        """
        if not self.is_fitted:
            raise ForecastingError("Model must be fitted before forecasting")
        
        if steps is None:
            steps = self.horizon
        
        if steps <= 0:
            raise ForecastingError("Steps must be positive")
        
        if self.last_values is None or len(self.last_values) == 0:
            raise ForecastingError("No data available for forecasting")
        
        # Calculate moving average
        if self.window_length is not None and len(self.last_values) >= self.window_length:
            # Use full window
            window_data = self.last_values[-self.window_length:]
        else:
            # Use available data if window is larger than available data or window_length is None
            window_data = self.last_values
        
        # Calculate moving average
        forecast_value = np.mean(window_data)
        
        # Generate forecast series
        forecast_series = pd.Series([forecast_value] * steps)
        
        return forecast_series
    
    def update(self, new_data: pd.DataFrame) -> 'MovingAverageForecaster':
        """
        Update the model with new data
        
        Args:
            new_data: New data to add to the model
            
        Returns:
            Self for method chaining
        """
        if self.data is None:
            return self.fit(new_data)
        
        # Validate and prepare new data
        self.validate_data(new_data)
        new_data_prepared = self.prepare_data(new_data)
        
        # Combine with existing data
        combined_data = pd.concat([self.data, new_data_prepared], ignore_index=True)
        combined_data = combined_data.drop_duplicates(subset=['product_id', 'location_id', 'date'])
        combined_data = combined_data.sort_values('date').reset_index(drop=True)
        
        # Update stored data and last values
        self.data = combined_data
        if len(self.data) > 0:
            self.last_values = self.data['demand'].tail(self.window_length).values
        
        return self
    
    def get_parameters(self) -> Dict:
        """Get model parameters"""
        return {
            'window_length': self.window_length,
            'horizon': self.horizon,
            'risk_period_days': self.risk_period_days,
            'name': self.name
        }
    
    def set_parameters(self, parameters: Dict) -> 'MovingAverageForecaster':
        """
        Set model parameters
        
        Args:
            parameters: Dictionary with 'window_length' and 'horizon'
            
        Returns:
            Self for method chaining
        """
        validate_forecast_parameters(parameters)
        
        self.window_length = int(parameters['window_length'])
        self.horizon = int(parameters['horizon'])
        if 'risk_period_days' in parameters:
            self.risk_period_days = parameters['risk_period_days']
        
        return self

def create_moving_average_forecaster(parameters: Dict) -> MovingAverageForecaster:
    """
    Factory function to create moving average forecaster
    
    Args:
        parameters: Dictionary with 'window_length' and 'horizon'
        
    Returns:
        Configured MovingAverageForecaster instance
    """
    validate_forecast_parameters(parameters)
    
    forecaster = MovingAverageForecaster(
        window_length=int(parameters['window_length']),
        horizon=int(parameters['horizon'])
    )
    
    # Set risk period if provided
    if 'risk_period_days' in parameters:
        forecaster.risk_period_days = parameters['risk_period_days']
    
    return forecaster

def forecast_product_location(data: pd.DataFrame, 
                             product_id: str, 
                             location_id: str,
                             parameters: Dict,
                             forecast_date: Optional[date] = None) -> Dict:
    """
    Forecast for a specific product-location combination
    
    Args:
        data: Aggregated demand data
        product_id: Product identifier
        location_id: Location identifier
        parameters: Forecasting parameters
        forecast_date: Date to forecast from (defaults to latest date in data)
        
    Returns:
        Dictionary with forecast results
    """
    # Filter data for product-location
    product_data = data[
        (data['product_id'] == product_id) & 
        (data['location_id'] == location_id)
    ].copy()
    
    if len(product_data) == 0:
        raise ForecastingError(f"No data found for {product_id}-{location_id}")
    
    # Sort by date
    product_data = product_data.sort_values('date').reset_index(drop=True)
    
    # Determine forecast date
    if forecast_date is None:
        forecast_date = product_data['date'].max()
    
    # Filter data up to forecast date
    historical_data = product_data[product_data['date'] <= forecast_date].copy()
    
    if len(historical_data) == 0:
        raise ForecastingError(f"No historical data found for {product_id}-{location_id} up to {forecast_date}")
    
    # Create and fit forecaster
    forecaster = create_moving_average_forecaster(parameters)
    
    # Pass risk period information if available
    fit_kwargs = {}
    if 'risk_period' in parameters:
        fit_kwargs['risk_period'] = parameters['risk_period']
    elif 'risk_period_days' in parameters:
        fit_kwargs['risk_period_days'] = parameters['risk_period_days']
    
    forecaster.fit(historical_data, **fit_kwargs)
    
    # Generate forecast
    forecast_values = forecaster.forecast(steps=parameters['horizon'])
    
    # Create forecast dates
    last_date = historical_data['date'].max()
    forecast_dates = []
    
    # Assuming the data is aggregated by risk period, calculate next dates
    if len(historical_data) >= 2:
        # Calculate average period length from historical data
        date_diffs = historical_data['date'].diff().dropna()
        avg_period_days = date_diffs.mean().days
        
        for i in range(parameters['horizon']):
            next_date = last_date + timedelta(days=avg_period_days * (i + 1))
            forecast_dates.append(next_date)
    else:
        # Fallback: assume weekly periods
        for i in range(parameters['horizon']):
            next_date = last_date + timedelta(weeks=i + 1)
            forecast_dates.append(next_date)
    
    # Create result
    result = {
        'product_id': product_id,
        'location_id': location_id,
        'forecast_date': forecast_date,
        'forecast_values': forecast_values.tolist(),
        'forecast_dates': forecast_dates,
        'parameters': parameters,
        'historical_data_points': len(historical_data),
        'last_actual_date': last_date,
        'last_actual_demand': historical_data['demand'].iloc[-1]
    }
    
    return result 