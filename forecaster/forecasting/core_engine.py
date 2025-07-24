"""
Core Forecasting Engine

This module provides the core forecasting logic that can be used by backtesting.
No parameter optimization is done here - parameters are passed in from outside.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import date
import logging

from .parameter_optimizer import ParameterOptimizerFactory
from .base import calculate_forecast_metrics

logger = logging.getLogger(__name__)


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
        try:
            # Get forecasting method
            forecast_method = product_record.get('forecast_method', 'moving_average')
            
            # Get parameter optimizer for creating forecaster
            optimizer = self.parameter_optimizer_factory.get_optimizer(forecast_method)
            
            # Create forecaster with the provided optimized parameters
            forecaster = optimizer.create_forecaster(optimized_parameters)
            
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
        Aggregate daily forecast values to risk period level.
        
        Args:
            daily_forecast: List of daily forecast values
            product_record: Product master record containing risk_period
            
        Returns:
            List of aggregated forecast values at risk period level
        """
        try:
            risk_period = product_record.get('risk_period', 1)
            
            if risk_period == 1:
                # If risk period is 1 day, return daily forecasts as-is
                return daily_forecast
            
            # Aggregate daily forecasts into risk period buckets
            aggregated_forecasts = []
            for i in range(0, len(daily_forecast), risk_period):
                period_forecasts = daily_forecast[i:i + risk_period]
                aggregated_value = sum(period_forecasts)
                aggregated_forecasts.append(aggregated_value)
            
            return aggregated_forecasts
            
        except Exception as e:
            logger.error(f"Error aggregating forecast to risk period: {e}")
            return daily_forecast  # Return original if aggregation fails 