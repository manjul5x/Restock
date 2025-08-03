"""
Core Forecasting Engine

This module provides the core forecasting logic that can be used by backtesting.
No parameter optimization is done here - parameters are passed in from outside.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import date

from .parameter_optimizer import ParameterOptimizerFactory
from .base import calculate_forecast_metrics
from .prophet import create_prophet_forecaster
from .arima import ARIMAForecaster
from .moving_average import MovingAverageForecaster
from ..utils.logger import get_logger


class CoreForecastingEngine:
    """
    Core forecasting engine that handles pure forecasting logic.
    No parameter optimization - parameters are provided externally.
    """
    
    def __init__(self):
        self.parameter_optimizer_factory = ParameterOptimizerFactory()
    
    def generate_forecast(
        self,
        product_data: pd.DataFrame,
        product_record: pd.Series,
        forecast_date: date,
        horizon: int,
        optimized_parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a single forecast using provided optimized parameters.
        
        Args:
            product_data: Historical demand data for this product-location
            product_record: Product master record
            forecast_date: Date to forecast from
            horizon: Number of periods to forecast
            optimized_parameters: Pre-optimized parameters for this product-location
            
        Returns:
            Dictionary containing forecast results
        """
        logger = get_logger(__name__)
        try:
            # Get forecasting method
            forecast_method = product_record.get('forecast_method', 'moving_average')
            
            # Create forecaster directly from optimized parameters (no parameter optimization)
            forecaster = self._create_forecaster_directly(forecast_method, optimized_parameters)
            
            # Fit the model to the provided data
            forecaster.fit(product_data)
            
            # Generate forecast
            # All forecasters should generate daily forecasts that we then aggregate to risk periods
            # This ensures consistency across all methods
            risk_period = product_record.get('risk_period', 1)
            daily_steps_needed = horizon * risk_period
            forecast_series = forecaster.forecast(steps=daily_steps_needed)
            
            # Get the first date of data used for forecasting
            first_date_used = forecaster.get_first_date_used()
            
            # Aggregate daily forecasts to risk periods for all methods
            # This ensures consistent aggregation regardless of the forecaster type
            daily_forecast_values = forecast_series.tolist()
            aggregated_forecast_values = self.aggregate_daily_forecast_to_risk_period(
                daily_forecast_values, product_record
            )
            
            # Return forecast result
            return {
                'forecast_values': aggregated_forecast_values,
                'forecast_method': forecast_method,
                'forecast_date': forecast_date,
                'parameters_used': optimized_parameters,
                'data_points_used': len(product_data),
                'horizon': horizon,
                'first_date_used': first_date_used
            }
            
        except Exception as e:
            logger.error(f"Error generating forecast for {product_record.get('product_id', 'unknown')}: {e}")
            return None
    
    def aggregate_daily_forecast_to_risk_period(
        self, 
        daily_forecast: List[float], 
        product_record: pd.Series
    ) -> List[float]:
        """
        Aggregate daily forecast values to risk period values.
        
        Args:
            daily_forecast: List of daily forecast values
            product_record: Product master record containing risk_period
            
        Returns:
            List of aggregated forecast values for risk periods
        """
        logger = get_logger(__name__)
        try:
            risk_period = product_record.get('risk_period', 1)
            
            if risk_period == 1:
                # No aggregation needed
                return daily_forecast
            
            # Aggregate daily values to risk period values
            aggregated_forecast = []
            for i in range(0, len(daily_forecast), risk_period):
                period_values = daily_forecast[i:i + risk_period]
                aggregated_value = sum(period_values)
                aggregated_forecast.append(aggregated_value)
            
            return aggregated_forecast
            
        except Exception as e:
            logger.error(f"Error aggregating forecast to risk period: {e}")
            return daily_forecast
    
    def _create_forecaster_directly(self, forecast_method: str, parameters: Dict[str, Any]):
        """
        Create forecaster directly from parameters without using parameter optimizer.
        
        Args:
            forecast_method: The forecasting method to use
            parameters: Optimized parameters for the forecaster
            
        Returns:
            Forecaster instance
        """
        logger = get_logger(__name__)
        
        try:
            if forecast_method == "prophet":
                # Ensure seasonality analysis is disabled for regular forecasting
                forecasting_parameters = parameters.copy()
                forecasting_parameters["run_seasonality_analysis"] = False
                return create_prophet_forecaster(forecasting_parameters)
                
            elif forecast_method == "arima":
                return ARIMAForecaster(
                    window_length=parameters.get('window_length'),
                    horizon=parameters.get('horizon'),
                    auto_arima=parameters.get('auto_arima', True),
                    max_p=parameters.get('max_p', 3),
                    max_d=parameters.get('max_d', 2),
                    max_q=parameters.get('max_q', 3),
                    seasonal=parameters.get('seasonal', False),
                    m=parameters.get('m', 1),
                    min_data_points=parameters.get('min_data_points', 5),
                    order=parameters.get('order', (1, 1, 1)),
                    seasonal_order=parameters.get('seasonal_order')
                )
                
            elif forecast_method == "moving_average":
                return MovingAverageForecaster(
                    window_length=parameters.get('window_length'),
                    horizon=parameters.get('horizon'),
                    risk_period_days=parameters.get('risk_period_days')
                )
                
            else:
                raise ValueError(f"Unsupported forecasting method: {forecast_method}")
                
        except Exception as e:
            logger.error(f"Error creating forecaster for method {forecast_method}: {e}")
            raise 