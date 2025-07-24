"""
ARIMA forecasting implementation.

This module provides ARIMA-based forecasting with automatic parameter selection
and integration with the existing forecasting framework.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from datetime import date, timedelta
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
from .base import BaseForecaster, ForecastingError, validate_forecast_parameters

# Suppress specific warnings that are handled gracefully
warnings.filterwarnings('ignore', category=UserWarning, module='statsmodels.tsa.statespace.sarimax')

class ARIMAForecaster(BaseForecaster):
    """
    ARIMA forecaster with automatic parameter selection.
    """
    
    def __init__(self, window_length: int = None, horizon: int = 1, 
                 auto_arima: bool = True, max_p: int = 3, max_d: int = 2, max_q: int = 3,
                 seasonal: bool = False, m: int = 1, min_data_points: int = 5):
        """
        Initialize ARIMA forecaster
        
        Args:
            window_length: Maximum number of periods to include (rolling window limit).
                          If None, uses all available data.
            horizon: Number of steps to forecast ahead
            auto_arima: Whether to automatically select ARIMA parameters
            max_p: Maximum AR order
            max_d: Maximum differencing order
            max_q: Maximum MA order
            seasonal: Whether to use seasonal ARIMA
            m: Seasonal period (if seasonal=True)
            min_data_points: Minimum data points required
        """
        super().__init__(name="arima")
        self.window_length = window_length
        self.horizon = horizon
        self.auto_arima = auto_arima
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.seasonal = seasonal
        self.m = m
        self.min_data_points = min_data_points
        
        self.data = None
        self.model = None
        self.order = None
        self.seasonal_order = None
        self.is_fitted = False
        self.actual_window_length = None
        
    def _check_stationarity(self, series: pd.Series) -> Tuple[bool, int]:
        """
        Check if series is stationary and determine differencing order.
        
        Args:
            series: Time series to check
            
        Returns:
            Tuple of (is_stationary, differencing_order)
        """
        # Remove any remaining NaN values
        series = series.dropna()
        
        if len(series) < 4:  # Need at least 4 points for ADF test
            return False, 0
        
        # Check for constant series
        if series.std() == 0:
            return True, 0
        
        # Check for stationarity
        try:
            adf_result = adfuller(series)
            is_stationary = adf_result[1] < 0.05  # p-value < 0.05
        except (ValueError, np.linalg.LinAlgError):
            # If ADF test fails, assume non-stationary
            is_stationary = False
        
        if is_stationary:
            return True, 0
        
        # Try differencing
        d = 0
        current_series = series.copy()
        
        for i in range(1, self.max_d + 1):
            current_series = current_series.diff().dropna()
            if len(current_series) < 4:  # Need at least 4 points for ADF test
                break
                
            # Check if differenced series is constant
            if current_series.std() == 0:
                return True, i
                
            try:
                adf_result = adfuller(current_series)
                if adf_result[1] < 0.05:
                    return True, i
            except (ValueError, np.linalg.LinAlgError):
                pass
            d = i
        
        return False, d
    
    def _validate_arima_parameters(self, p: int, d: int, q: int, series_length: int) -> bool:
        """
        Validate ARIMA parameters to avoid estimation issues.
        
        Args:
            p: AR order
            d: Differencing order
            q: MA order
            series_length: Length of the time series
            
        Returns:
            True if parameters are valid, False otherwise
        """
        # Check if we have enough data points after differencing
        effective_length = series_length - d
        if effective_length < max(p, q) + 2:
            return False
        
        # Check for reasonable parameter combinations
        if p == 0 and q == 0:
            return False  # No AR or MA terms
        
        # Avoid overfitting
        total_params = p + q
        if total_params > effective_length // 3:
            return False
        
        return True
    
    def _fit_arima_model(self, series: pd.Series, order: Tuple[int, int, int], 
                        seasonal_order: Optional[Tuple[int, int, int, int]] = None) -> Optional[object]:
        """
        Fit ARIMA model with error handling and parameter validation.
        
        Args:
            series: Time series data
            order: ARIMA order (p, d, q)
            seasonal_order: Seasonal ARIMA order (P, D, Q, m)
            
        Returns:
            Fitted model or None if fitting fails
        """
        p, d, q = order
        
        # Validate parameters
        if not self._validate_arima_parameters(p, d, q, len(series)):
            return None
        
        try:
            # Create model
            if seasonal_order:
                model = ARIMA(series, order=order, seasonal_order=seasonal_order)
            else:
                model = ARIMA(series, order=order)
            
            # Fit with method='lbfgs' for better convergence
            fitted_model = model.fit(method='lbfgs')
            
            # Check if model converged
            if not fitted_model.mle_retvals['converged']:
                return None
            
            # Check for reasonable AIC (not too high)
            if fitted_model.aic > 1e6:
                return None
            
            return fitted_model
            
        except (ValueError, np.linalg.LinAlgError, RuntimeWarning, RuntimeError):
            return None
    
    def _auto_select_arima_params(self, series: pd.Series) -> Tuple[Tuple[int, int, int], Optional[Tuple[int, int, int, int]]]:
        """
        Automatically select ARIMA parameters using grid search with improved validation.
        
        Args:
            series: Time series data
            
        Returns:
            Tuple of (order, seasonal_order)
        """
        # Check stationarity and determine d
        is_stationary, d = self._check_stationarity(series)
        
        if not is_stationary and d >= self.max_d:
            # Force stationarity with max differencing
            d = self.max_d
        
        best_aic = np.inf
        best_order = (1, d, 1)  # Default order
        best_seasonal_order = None
        best_model = None
        
        # Define parameter search space
        p_range = range(0, min(self.max_p + 1, len(series) // 4))
        q_range = range(0, min(self.max_q + 1, len(series) // 4))
        
        # Try simpler models first (faster and more stable)
        simple_orders = [(1, d, 0), (0, d, 1), (1, d, 1), (2, d, 0), (0, d, 2)]
        
        # Test simple orders first
        for order in simple_orders:
            p, d_test, q = order
            if d_test != d:
                continue  # Use the determined d value
                
            model = self._fit_arima_model(series, (p, d, q))
            if model is not None and model.aic < best_aic:
                best_aic = model.aic
                best_order = (p, d, q)
                best_model = model
        
        # If simple models work well, use the best one
        if best_model is not None and best_aic < np.inf:
            return best_order, best_seasonal_order
        
        # Otherwise, try grid search for more complex models
        for p in p_range:
            for q in q_range:
                # Skip if we already found a good simple model
                if (p, d, q) in simple_orders:
                    continue
                    
                model = self._fit_arima_model(series, (p, d, q))
                if model is not None and model.aic < best_aic:
                    best_aic = model.aic
                    best_order = (p, d, q)
                    best_model = model
        
        # If no model worked, use a simple fallback
        if best_model is None:
            # Try (1, d, 1) as fallback
            fallback_model = self._fit_arima_model(series, (1, d, 1))
            if fallback_model is not None:
                best_order = (1, d, 1)
                best_model = fallback_model
            else:
                # Last resort: use (0, d, 1) or (1, d, 0)
                for fallback_order in [(0, d, 1), (1, d, 0)]:
                    fallback_model = self._fit_arima_model(series, fallback_order)
                    if fallback_model is not None:
                        best_order = fallback_order
                        best_model = fallback_model
                        break
        
        return best_order, best_seasonal_order
    
    def fit(self, data: pd.DataFrame, **kwargs) -> 'ARIMAForecaster':
        """
        Fit the ARIMA model to the data
        
        Args:
            data: DataFrame with columns ['product_id', 'location_id', 'date', 'demand']
            
        Returns:
            Self for method chaining
        """
        # Validate and prepare data
        self.validate_data(data)
        
        # Prepare data (this data is already aggregated into buckets)
        self.data = self.prepare_data(data)
        
        # Store the first date before applying window (for tracking purposes)
        self.original_first_date = self.data['date'].min() if len(self.data) > 0 else None
        
        # Apply rolling window if specified (after bucketing)
        if self.window_length is not None and len(self.data) > self.window_length:
            # Sort by date and take the most recent window_length data points
            self.data = self.data.sort_values('date').tail(self.window_length).reset_index(drop=True)
        
        # Store the first date of data actually used for forecasting
        self.first_date_used = self.data['date'].min() if len(self.data) > 0 else None
        
        # Check data availability
        data_length = len(self.data)
        self.actual_window_length = data_length
        
        if data_length < self.min_data_points:
            raise ForecastingError(f"Insufficient data: need at least {self.min_data_points} periods, got {data_length}")
        
        # Prepare series for ARIMA
        series = self.data['demand'].copy()
        
        # Remove any NaN values
        series = series.dropna()
        
        if len(series) == 0:
            raise ForecastingError("No valid data after removing NaN values")
        
        # Check for constant series
        if series.std() == 0:
            # For constant series, use simple model
            self.order = (0, 0, 0)
            self.seasonal_order = None
        else:
            # Auto-select parameters if enabled
            if self.auto_arima:
                self.order, self.seasonal_order = self._auto_select_arima_params(series)
                print(f"Auto-selected ARIMA order: {self.order}")
                if self.seasonal_order:
                    print(f"Seasonal order: {self.seasonal_order}")
            else:
                # Use default parameters
                self.order = (1, 1, 1)
                self.seasonal_order = None
        
        # Fit ARIMA model with error handling
        try:
            if self.seasonal_order:
                self.model = ARIMA(series, order=self.order, seasonal_order=self.seasonal_order)
            else:
                self.model = ARIMA(series, order=self.order)
            
            # Use LBFGS method for better convergence
            fitted_model = self.model.fit(method='lbfgs')
            
            # Check convergence
            if not fitted_model.mle_retvals['converged']:
                print(f"⚠️  ARIMA model did not converge, but proceeding with results")
            
            self.model = fitted_model
            self.is_fitted = True
            
            print(f"✅ ARIMA model fitted successfully with AIC: {fitted_model.aic:.2f}")
            
        except Exception as e:
            # Try fallback to simpler model
            print(f"⚠️  Failed to fit ARIMA model with order {self.order}: {str(e)}")
            print("Trying fallback model...")
            
            try:
                # Try simple fallback models
                fallback_orders = [(1, 1, 0), (0, 1, 1), (1, 0, 1)]
                for fallback_order in fallback_orders:
                    try:
                        fallback_model = ARIMA(series, order=fallback_order)
                        fitted_fallback = fallback_model.fit()
                        self.model = fitted_fallback
                        self.order = fallback_order
                        self.is_fitted = True
                        print(f"✅ Fallback ARIMA model fitted with order {fallback_order}, AIC: {fitted_fallback.aic:.2f}")
                        break
                    except:
                        continue
                
                if not self.is_fitted:
                    raise ForecastingError(f"All ARIMA models failed to fit")
                    
            except Exception as fallback_error:
                raise ForecastingError(f"Failed to fit ARIMA model and fallbacks: {str(fallback_error)}")
        
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
        
        if self.model is None:
            raise ForecastingError("No fitted model available")
        
        try:
            # Generate forecast
            forecast_result = self.model.forecast(steps=steps)
            
            # Ensure forecast values are non-negative
            forecast_values = np.maximum(forecast_result, 0)
            
            return pd.Series(forecast_values, index=range(len(forecast_values)))
            
        except Exception as e:
            raise ForecastingError(f"Failed to generate forecast: {str(e)}")
    
    def update(self, new_data: pd.DataFrame) -> 'ARIMAForecaster':
        """
        Update the model with new data
        
        Args:
            new_data: New data to add to the model
            
        Returns:
            Self for method chaining
        """
        if not self.is_fitted:
            raise ForecastingError("Model must be fitted before updating")
        
        # Combine existing and new data
        combined_data = pd.concat([self.data, new_data], ignore_index=True)
        
        # Refit the model with combined data
        return self.fit(combined_data)
    
    def get_parameters(self) -> Dict:
        """Get current parameters."""
        return {
            'window_length': self.window_length,
            'horizon': self.horizon,
            'auto_arima': self.auto_arima,
            'max_p': self.max_p,
            'max_d': self.max_d,
            'max_q': self.max_q,
            'seasonal': self.seasonal,
            'm': self.m,
            'min_data_points': self.min_data_points,
            'order': self.order,
            'seasonal_order': self.seasonal_order,
            'actual_window_length': self.actual_window_length,
            'is_fitted': self.is_fitted,
            'method_name': self.name
        }
    
    def get_first_date_used(self) -> Optional[date]:
        """
        Get the first date of data that was actually used for forecasting
        (after applying the window).
        
        Returns:
            First date used, or None if no data was used
        """
        return self.first_date_used

    def set_parameters(self, parameters: Dict) -> 'ARIMAForecaster':
        """
        Set model parameters
        
        Args:
            parameters: Dictionary of parameters to set
            
        Returns:
            Self for method chaining
        """
        if 'window_length' in parameters:
            self.window_length = parameters['window_length']
        if 'horizon' in parameters:
            self.horizon = parameters['horizon']
        if 'auto_arima' in parameters:
            self.auto_arima = parameters['auto_arima']
        if 'max_p' in parameters:
            self.max_p = parameters['max_p']
        if 'max_d' in parameters:
            self.max_d = parameters['max_d']
        if 'max_q' in parameters:
            self.max_q = parameters['max_q']
        if 'seasonal' in parameters:
            self.seasonal = parameters['seasonal']
        if 'm' in parameters:
            self.m = parameters['m']
        if 'min_data_points' in parameters:
            self.min_data_points = parameters['min_data_points']
        
        return self

def create_arima_forecaster(parameters: Dict) -> ARIMAForecaster:
    """
    Create an ARIMA forecaster with given parameters
    
    Args:
        parameters: Dictionary of parameters
        
    Returns:
        ARIMAForecaster instance
    """
    validate_forecast_parameters(parameters)
    
    return ARIMAForecaster(
        window_length=parameters.get('window_length', 10),
        horizon=parameters.get('horizon', 1),
        auto_arima=parameters.get('auto_arima', True),
        max_p=parameters.get('max_p', 3),
        max_d=parameters.get('max_d', 2),
        max_q=parameters.get('max_q', 3),
        seasonal=parameters.get('seasonal', False),
        m=parameters.get('m', 1),
        min_data_points=parameters.get('min_data_points', 5)
    )

def forecast_product_location(data: pd.DataFrame, 
                             product_id: str, 
                             location_id: str,
                             parameters: Dict,
                             forecast_date: Optional[date] = None) -> Dict:
    """
    Forecast demand for a specific product-location combination using ARIMA
    
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
    forecaster = create_arima_forecaster(parameters)
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
        'model': 'arima',
        'parameters': forecaster.get_parameters()
    }

