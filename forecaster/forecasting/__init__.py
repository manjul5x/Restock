"""
Forecasting module for demand prediction.
"""

from .base import BaseForecaster, ForecastingError, calculate_forecast_metrics, validate_forecast_parameters
from .moving_average import MovingAverageForecaster, create_moving_average_forecaster, forecast_product_location
from .prophet import ProphetForecaster, create_prophet_forecaster, forecast_product_location as prophet_forecast_product_location
from .arima import ARIMAForecaster, create_arima_forecaster, forecast_product_location as arima_forecast_product_location

__all__ = [
    'BaseForecaster',
    'ForecastingError',
    'calculate_forecast_metrics',
    'validate_forecast_parameters',
    'MovingAverageForecaster',
    'create_moving_average_forecaster',
    'forecast_product_location',
    'ProphetForecaster',
    'create_prophet_forecaster',
    'prophet_forecast_product_location',
    'ARIMAForecaster',
    'create_arima_forecaster',
    'arima_forecast_product_location'
]
