"""
Forecasting module for demand prediction.
"""

# Import new forecasting models
from .base import BaseForecastingModel
from .moving_average import MovingAverageModel
from .prophet import ProphetModel
from .engine import ForecastingEngine

__all__ = [
    'BaseForecastingModel',
    'MovingAverageModel', 
    'ProphetModel',
    'ForecastingEngine'
]
