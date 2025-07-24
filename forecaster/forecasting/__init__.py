"""
Forecasting module for demand prediction.
"""

from .base import BaseForecaster, calculate_forecast_metrics
from .moving_average import MovingAverageForecaster, create_moving_average_forecaster, forecast_product_location as moving_average_forecast_product_location
from .prophet import ProphetForecaster, create_prophet_forecaster, forecast_product_location as prophet_forecast_product_location
from .arima import ARIMAForecaster, create_arima_forecaster, forecast_product_location as arima_forecast_product_location
from .parameter_optimizer import (
    ParameterOptimizer,
    ParameterOptimizerFactory,
    MovingAverageParameterOptimizer,
    ProphetParameterOptimizer,
    ARIMAParameterOptimizer
)
from .core_engine import CoreForecastingEngine

__all__ = [
    'BaseForecaster',
    'calculate_forecast_metrics',
    'MovingAverageForecaster',
    'create_moving_average_forecaster',
    'moving_average_forecast_product_location',
    'ProphetForecaster',
    'create_prophet_forecaster',
    'prophet_forecast_product_location',
    'ARIMAForecaster',
    'create_arima_forecaster',
    'arima_forecast_product_location',
    'ParameterOptimizer',
    'ParameterOptimizerFactory',
    'MovingAverageParameterOptimizer',
    'ProphetParameterOptimizer',
    'ARIMAParameterOptimizer',
    'CoreForecastingEngine'
]
