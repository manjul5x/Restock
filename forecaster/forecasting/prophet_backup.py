"""
Enhanced Prophet forecasting implementation.

This enhanced version handles insufficient data periods by:
1. Using available data even when below minimum window_length
2. Automatically adjusting seasonality settings based on data availability
3. Providing fallback forecasting methods for very limited data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from datetime import date, timedelta
from prophet import Prophet
import warnings
from .base import BaseForecaster, ForecastingError, validate_forecast_parameters

class ProphetForecaster(BaseForecaster):
    """
    Enhanced Prophet forecaster that handles insufficient data gracefully.
    """
    
    def __init__(self, window_length: int = 10, horizon: int = 1, 
                 yearly_seasonality: bool = True, weekly_seasonality: bool = True, 
                 daily_seasonality: bool = False, seasonality_mode: str = 'additive',
                 min_data_points: int = 5, auto_adjust_seasonality: bool = True):
        """
        Initialize Enhanced Prophet forecaster
        
        Args:
            window_length: Preferred number of periods for training
            horizon: Number of steps to forecast ahead
            yearly_seasonality: Whether to include yearly seasonality
            weekly_seasonality: Whether to include weekly seasonality
            daily_seasonality: Whether to include daily seasonality
            seasonality_mode: 'additive' or 'multiplicative'
            min_data_points: Minimum data points required (default: 5)
            auto_adjust_seasonality: Whether to automatically adjust seasonality based on data
        """
        super().__init__(name="prophet")
        self.window_length = window_length
        self.horizon = horizon
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.seasonality_mode = seasonality_mode
        self.min_data_points = min_data_points
        self.auto_adjust_seasonality = auto_adjust_seasonality
        
        self.data = None
        self.model = None
        self.last_date = None
        self.is_fitted = False
        self.actual_window_length = None
        self.seasonality_adjusted = False
        
    def _adjust_seasonality_for_data(self, data_length: int, date_range: Tuple[date, date]) -> Dict[str, bool]:
        """
        Automatically adjust seasonality settings based on available data.
        
        Args:
            data_length: Number of data points available
            date_range: Tuple of (start_date, end_date)
            
        Returns:
            Dict with adjusted seasonality settings
        """
        start_date, end_date = date_range
        date_span = (end_date - start_date).days
        
        # Default settings
        seasonality_settings = {
            'yearly_seasonality': self.yearly_seasonality,
            'weekly_seasonality': self.weekly_seasonality,
            'daily_seasonality': self.daily_seasonality
        }
        
        if not self.auto_adjust_seasonality:
            return seasonality_settings
        
        # Adjust based on data availability
        if date_span < 365:  # Less than 1 year
            seasonality_settings['yearly_seasonality'] = False
            warnings.warn(f"Insufficient data for yearly seasonality (need ~1 year, have {date_span} days)")
        
        if date_span < 28:  # Less than 4 weeks
            seasonality_settings['weekly_seasonality'] = False
            warnings.warn(f"Insufficient data for weekly seasonality (need ~4 weeks, have {date_span} days)")
        
        if data_length < 7:  # Less than 7 data points
            seasonality_settings['weekly_seasonality'] = False
            seasonality_settings['daily_seasonality'] = False
            warnings.warn(f"Very limited data ({data_length} points), disabling complex seasonality")
        
        return seasonality_settings
    
    def _simple_fallback_forecast(self, data: pd.DataFrame, steps: int) -> pd.Series:
        """
        Simple fallback forecasting method for very limited data.
        
        Args:
            data: DataFrame with demand data
            steps: Number of steps to forecast
            
        Returns:
            Series with forecast values
        """
        if len(data) == 0:
            return pd.Series([0] * steps)
        
        # Use simple moving average or last value
        if len(data) >= 3:
            # 3-period moving average
            forecast_value = data['demand'].tail(3).mean()
        else:
            # Use last available value
            forecast_value = data['demand'].iloc[-1]
        
        return pd.Series([forecast_value] * steps)
        
    def fit(self, data: pd.DataFrame, **kwargs) -> 'ProphetForecaster':
        """
        Fit the Enhanced Prophet model to the data
        
        Args:
            data: DataFrame with columns ['product_id', 'location_id', 'date', 'demand']
            
        Returns:
            Self for method chaining
        """
        # Validate and prepare data
        self.validate_data(data)
        self.data = self.prepare_data(data)
        
        # Check data availability
        data_length = len(self.data)
        self.actual_window_length = data_length
        
        if data_length < self.min_data_points:
            raise ForecastingError(f"Insufficient data: need at least {self.min_data_points} periods, got {data_length}")
        
        # Prepare data for Prophet (requires 'ds' and 'y' columns)
        prophet_data = self.data[['date', 'demand']].copy()
        prophet_data.columns = ['ds', 'y']
        
        # Remove any NaN values
        prophet_data = prophet_data.dropna()
        
        if len(prophet_data) == 0:
            raise ForecastingError("No valid data after removing NaN values")
        
        # Auto-adjust seasonality if enabled
        if self.auto_adjust_seasonality:
            start_date = prophet_data['ds'].min()
            end_date = prophet_data['ds'].max()
            adjusted_seasonality = self._adjust_seasonality_for_data(len(prophet_data), (start_date, end_date))
            
            if adjusted_seasonality != {
                'yearly_seasonality': self.yearly_seasonality,
                'weekly_seasonality': self.weekly_seasonality,
                'daily_seasonality': self.daily_seasonality
            }:
                self.seasonality_adjusted = True
                print(f"Auto-adjusted seasonality for {len(prophet_data)} data points: {adjusted_seasonality}")
        else:
            adjusted_seasonality = {
                'yearly_seasonality': self.yearly_seasonality,
                'weekly_seasonality': self.weekly_seasonality,
                'daily_seasonality': self.daily_seasonality
            }
        
        # Initialize and fit Prophet model
        self.model = Prophet(
            yearly_seasonality=adjusted_seasonality['yearly_seasonality'],
            weekly_seasonality=adjusted_seasonality['weekly_seasonality'],
            daily_seasonality=adjusted_seasonality['daily_seasonality'],
            seasonality_mode=self.seasonality_mode
        )
        
        try:
            self.model.fit(prophet_data)
            self.last_date = prophet_data['ds'].max()
            self.is_fitted = True
            
            # Log information about the fit
            if data_length < self.window_length:
                print(f"⚠️  Using {data_length} data points (preferred: {self.window_length})")
            else:
                print(f"✅ Using {data_length} data points for forecasting")
                
        except Exception as e:
            raise ForecastingError(f"Failed to fit Prophet model: {str(e)}")
        
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
        
        if self.model is None or self.last_date is None:
            raise ForecastingError("No fitted model available")
        
        # Use fallback method for very limited data
        if self.actual_window_length < 5:
            print(f"⚠️  Using fallback forecasting method (only {self.actual_window_length} data points)")
            return self._simple_fallback_forecast(self.data, steps)
        
        try:
            # Create future dates for forecasting
            future_dates = []
            current_date = self.last_date
            
            for i in range(1, steps + 1):
                if isinstance(current_date, pd.Timestamp):
                    next_date = current_date + pd.Timedelta(days=1)
                else:
                    next_date = current_date + timedelta(days=1)
                future_dates.append(next_date)
                current_date = next_date
            
            future_df = pd.DataFrame({'ds': future_dates})
            
            # Generate forecast
            forecast_result = self.model.predict(future_df)
            
            # Extract forecast values (yhat column)
            forecast_values = forecast_result['yhat'].values
            
            # Ensure non-negative values (demand can't be negative)
            forecast_values = np.maximum(forecast_values, 0)
            
            return pd.Series(forecast_values)
            
        except Exception as e:
            # Fallback to simple method if Prophet fails
            print(f"⚠️  Prophet forecast failed, using fallback method: {str(e)}")
            return self._simple_fallback_forecast(self.data, steps)
    
    def update(self, new_data: pd.DataFrame) -> 'ProphetForecaster':
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
        
        # Combine existing and new data
        combined_data = pd.concat([self.data, new_data_prepared], ignore_index=True)
        combined_data = combined_data.drop_duplicates(subset=['date']).sort_values('date').reset_index(drop=True)
        
        # Refit the model with combined data
        return self.fit(combined_data)
    
    def get_parameters(self) -> Dict:
        """Get current parameters"""
        return {
            'window_length': self.window_length,
            'horizon': self.horizon,
            'yearly_seasonality': self.yearly_seasonality,
            'weekly_seasonality': self.weekly_seasonality,
            'daily_seasonality': self.daily_seasonality,
            'seasonality_mode': self.seasonality_mode,
            'min_data_points': self.min_data_points,
            'auto_adjust_seasonality': self.auto_adjust_seasonality,
            'actual_window_length': self.actual_window_length,
            'seasonality_adjusted': self.seasonality_adjusted
        }
    
    def set_parameters(self, parameters: Dict) -> 'ProphetForecaster':
        """Set parameters"""
        if 'window_length' in parameters:
            self.window_length = parameters['window_length']
        if 'horizon' in parameters:
            self.horizon = parameters['horizon']
        if 'yearly_seasonality' in parameters:
            self.yearly_seasonality = parameters['yearly_seasonality']
        if 'weekly_seasonality' in parameters:
            self.weekly_seasonality = parameters['weekly_seasonality']
        if 'daily_seasonality' in parameters:
            self.daily_seasonality = parameters['daily_seasonality']
        if 'seasonality_mode' in parameters:
            self.seasonality_mode = parameters['seasonality_mode']
        if 'min_data_points' in parameters:
            self.min_data_points = parameters['min_data_points']
        if 'auto_adjust_seasonality' in parameters:
            self.auto_adjust_seasonality = parameters['auto_adjust_seasonality']
        
        return self

def create_prophet_forecaster(parameters: Dict) -> ProphetForecaster:
    """
    Create an Enhanced Prophet forecaster with given parameters
    
    Args:
        parameters: Dictionary of parameters
        
    Returns:
        ProphetForecaster instance
    """
    validate_forecast_parameters(parameters)
    
    return ProphetForecaster(
        window_length=parameters.get('window_length', 10),
        horizon=parameters.get('horizon', 1),
        yearly_seasonality=parameters.get('yearly_seasonality', True),
        weekly_seasonality=parameters.get('weekly_seasonality', True),
        daily_seasonality=parameters.get('daily_seasonality', False),
        seasonality_mode=parameters.get('seasonality_mode', 'additive'),
        min_data_points=parameters.get('min_data_points', 5),
        auto_adjust_seasonality=parameters.get('auto_adjust_seasonality', True)
    )

def forecast_product_location(data: pd.DataFrame, 
                             product_id: str, 
                             location_id: str,
                             parameters: Dict,
                             forecast_date: Optional[date] = None) -> Dict:
    """
    Forecast demand for a specific product-location combination using Prophet
    
    Args:
        data: Full demand dataset
        product_id: Product ID to forecast
        location_id: Location ID to forecast
        parameters: Forecasting parameters
        forecast_date: Date to forecast from (defaults to latest date in data)
        
    Returns:
        Dictionary with forecast results
    """
    # Filter data for specific product-location
    product_data = data[
        (data['product_id'] == product_id) & 
        (data['location_id'] == location_id)
    ].copy()
    
    if len(product_data) == 0:
        raise ForecastingError(f"No data found for product {product_id} at location {location_id}")
    
    # Determine forecast date
    if forecast_date is None:
        forecast_date = product_data['date'].max()
    
    # Filter data up to forecast date
    historical_data = product_data[product_data['date'] <= forecast_date].copy()
    
    if len(historical_data) < parameters.get('window_length', 10):
        raise ForecastingError(f"Insufficient historical data for forecasting")
    
    # Create and fit forecaster
    forecaster = create_prophet_forecaster(parameters)
    forecaster.fit(historical_data)
    
    # Generate forecast
    forecast_values = forecaster.forecast()
    
    # Create forecast dates
    last_date = historical_data['date'].max()
    forecast_dates = []
    for i in range(len(forecast_values)):
        if isinstance(last_date, pd.Timestamp):
            forecast_date = last_date + pd.Timedelta(days=i)
        else:
            forecast_date = last_date + timedelta(days=i)
        forecast_dates.append(forecast_date)
    
    return {
        'product_id': product_id,
        'location_id': location_id,
        'forecast_date': forecast_date,
        'forecast_values': forecast_values.tolist(),
        'forecast_dates': [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d) for d in forecast_dates],
        'model': 'prophet',
        'parameters': forecaster.get_parameters()
    }


