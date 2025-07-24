"""
Base forecasting module with abstract base class and common utilities.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from datetime import date, timedelta
from pathlib import Path


class BaseForecaster(ABC):
    """Abstract base class for all forecasters"""

    def __init__(self, name: str = "base"):
        self.name = name
        self.is_fitted = False

    @abstractmethod
    def fit(self, data: pd.DataFrame, **kwargs) -> "BaseForecaster":
        """Fit the forecasting model to the data"""
        pass

    @abstractmethod
    def forecast(self, steps: int, **kwargs) -> pd.Series:
        """Generate forecast for specified number of steps"""
        pass

    @abstractmethod
    def get_parameters(self) -> Dict:
        """Get model parameters"""
        pass

    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data format"""
        required_columns = ["product_id", "location_id", "date", "demand"]
        missing_cols = set(required_columns) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        return True

    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for forecasting (sort by date)"""
        if len(data) == 0:
            return data

        prepared_data = data.copy()
        prepared_data["date"] = pd.to_datetime(prepared_data["date"]).dt.date
        prepared_data = prepared_data.sort_values("date").reset_index(drop=True)
        return prepared_data


class ForecastingError(Exception):
    """Custom exception for forecasting errors"""

    pass


def calculate_forecast_metrics(actual, forecast) -> Dict[str, float]:
    """
    Calculate forecast accuracy metrics

    Args:
        actual: Actual values (pandas Series or list)
        forecast: Forecasted values (pandas Series or list)

    Returns:
        Dictionary of metrics
    """
    # Convert to pandas Series if they're lists
    if isinstance(actual, list):
        actual = pd.Series(actual)
    if isinstance(forecast, list):
        forecast = pd.Series(forecast)

    if len(actual) != len(forecast):
        raise ValueError("Actual and forecast series must have same length")

    # Remove any NaN values
    mask = ~(actual.isna() | forecast.isna())
    actual_clean = actual[mask]
    forecast_clean = forecast[mask]

    if len(actual_clean) == 0:
        return {"mae": np.nan, "mape": np.nan, "rmse": np.nan, "bias": np.nan}

    # Calculate metrics
    errors = actual_clean - forecast_clean

    mae = np.mean(np.abs(errors))
    mape = np.mean(np.abs(errors / actual_clean)) * 100
    rmse = np.sqrt(np.mean(errors**2))
    bias = np.mean(errors)

    return {"mae": mae, "mape": mape, "rmse": rmse, "bias": bias}


def validate_forecast_parameters(parameters: Dict) -> bool:
    """
    Validate forecast parameters

    Args:
        parameters: Dictionary of parameters

    Returns:
        True if valid, raises ValueError if not
    """
    required_params = ["window_length", "horizon"]

    for param in required_params:
        if param not in parameters:
            raise ValueError(f"Missing required parameter: {param}")

        # Allow None for window_length (use all available data)
        if param == "window_length" and parameters[param] is None:
            continue

        if not isinstance(parameters[param], (int, float)):
            raise ValueError(f"Parameter {param} must be numeric")

        if parameters[param] <= 0:
            raise ValueError(f"Parameter {param} must be positive")

    return True
