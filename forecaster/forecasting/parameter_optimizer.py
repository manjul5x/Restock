"""
Parameter Optimization Framework for Forecasting Methods

This module provides a unified framework for optimizing parameters for different
forecasting methods. Each method has its own optimization strategy.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from abc import ABC, abstractmethod
import logging
from datetime import date

from .seasonality_analyzer import SeasonalityAnalyzer
from .prophet import ProphetForecaster, create_prophet_forecaster
from .moving_average import MovingAverageForecaster
from .arima import ARIMAForecaster

logger = logging.getLogger(__name__)


class ParameterOptimizer(ABC):
    """
    Abstract base class for parameter optimization strategies.
    Each forecasting method should implement its own optimizer.
    """
    
    def __init__(self, method_name: str):
        """
        Initialize parameter optimizer.
        
        Args:
            method_name: Name of the forecasting method
        """
        self.method_name = method_name
    
    @abstractmethod
    def optimize_parameters(
        self, 
        data: pd.DataFrame, 
        product_record: pd.Series,
        base_parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Optimize parameters for the specific forecasting method.
        
        Args:
            data: Historical demand data
            product_record: Product master record
            base_parameters: Base parameters from product master
            
        Returns:
            Optimized parameters dictionary
        """
        pass
    
    @abstractmethod
    def create_forecaster(self, parameters: Dict[str, Any]) -> Any:
        """
        Create a forecaster instance with the given parameters.
        
        Args:
            parameters: Optimized parameters
            
        Returns:
            Forecaster instance
        """
        pass


class MovingAverageParameterOptimizer(ParameterOptimizer):
    """
    Parameter optimizer for Moving Average forecasting.
    No optimization needed - uses product master parameters directly.
    """
    
    def __init__(self):
        super().__init__("moving_average")
    
    def optimize_parameters(
        self, 
        data: pd.DataFrame, 
        product_record: pd.Series,
        base_parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        For Moving Average, no optimization is needed.
        Returns the base parameters from product master.
        """
        logger.debug(f"No parameter optimization needed for Moving Average")
        
        # Get risk period to convert window_length from risk periods to data points
        risk_period = product_record.get('risk_period', 1)
        forecast_window_length_risk_periods = product_record.get('forecast_window_length', base_parameters.get('window_length', 25))
        
        # Convert from risk periods to data points
        window_length_data_points = forecast_window_length_risk_periods * risk_period
        
        # Use base parameters from product master
        optimized_parameters = {
            'window_length': window_length_data_points,
            'horizon': product_record.get('forecast_horizon', base_parameters.get('horizon', 1)),
            'risk_period_days': base_parameters.get('risk_period_days'),
            'method_name': self.method_name
        }
        
        return optimized_parameters
    
    def create_forecaster(self, parameters: Dict[str, Any]) -> MovingAverageForecaster:
        """Create Moving Average forecaster with parameters."""
        return MovingAverageForecaster(
            window_length=parameters['window_length'],
            horizon=parameters['horizon'],
            risk_period_days=parameters.get('risk_period_days')
        )


class ProphetParameterOptimizer(ParameterOptimizer):
    """
    Parameter optimizer for Prophet forecasting.
    Uses seasonality analysis to optimize parameters.
    """
    
    def __init__(self):
        super().__init__("prophet")
        self.seasonality_analyzer = SeasonalityAnalyzer()
    
    def optimize_parameters(
        self, 
        data: pd.DataFrame, 
        product_record: pd.Series,
        base_parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Optimize Prophet parameters using seasonality analysis.
        """
        logger.info(f"Optimizing Prophet parameters for {product_record['product_id']}")
        
        try:
            # Create a temporary Prophet forecaster for seasonality analysis
            temp_forecaster = ProphetForecaster(
                window_length=None  # Use entire available data for analysis
            )
            
            # Fit the model for analysis
            temp_forecaster.fit(data)
            
            # Get seasonality analysis results
            seasonality_analysis = temp_forecaster.get_seasonality_analysis()
            
            if seasonality_analysis and "recommendations" in seasonality_analysis:
                # Use best model parameters from seasonality analysis
                best_parameters = seasonality_analysis["recommendations"].get(
                    "best_model_parameters", {}
                )
                
                # Merge with default parameters
                default_parameters = {
                    "changepoint_range": 0.8,
                    "n_changepoints": 25,
                    "changepoint_prior_scale": 0.05,
                    "seasonality_prior_scale": 10.0,
                    "holidays_prior_scale": 10.0,
                    "seasonality_mode": "multiplicative",
                    "weekly_seasonality": True,
                    "daily_seasonality": False,
                    "include_indian_holidays": True,
                    "include_regional_holidays": False,
                    "include_quarterly_effects": True,
                    "include_monthly_effects": True,
                    "include_festival_seasons": True,
                    "include_monsoon_effect": False,
                    "min_data_points": 60,
                    "window_length": product_record.get('forecast_window_length') * product_record.get('risk_period', 1),
                    "horizon": product_record.get('forecast_horizon', 1),
                    "method_name": self.method_name
                }
                
                # Update with best parameters from seasonality analysis
                optimized_parameters = {**default_parameters, **best_parameters}
                
                logger.debug(f"Optimized Prophet parameters: {optimized_parameters}")
                return optimized_parameters
            else:
                # Fallback to default parameters
                logger.warning(f"No seasonality analysis available, using default parameters")
                return self._get_default_prophet_parameters(product_record, base_parameters)
                
        except Exception as e:
            logger.error(f"Error optimizing Prophet parameters: {e}")
            return self._get_default_prophet_parameters(product_record, base_parameters)
    
    def _get_default_prophet_parameters(
        self, 
        product_record: pd.Series, 
        base_parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get default Prophet parameters as fallback."""
        return {
            "changepoint_range": 0.8,
            "n_changepoints": 25,
            "changepoint_prior_scale": 0.05,
            "seasonality_prior_scale": 10.0,
            "holidays_prior_scale": 10.0,
            "seasonality_mode": "multiplicative",
            "weekly_seasonality": True,
            "daily_seasonality": False,
            "include_indian_holidays": True,
            "include_regional_holidays": False,
            "include_quarterly_effects": True,
            "include_monthly_effects": True,
            "include_festival_seasons": True,
            "include_monsoon_effect": False,
            "min_data_points": 60,
            "window_length": product_record.get('forecast_window_length') * product_record.get('risk_period', 1),
            "horizon": product_record.get('forecast_horizon', 1),
            "method_name": self.method_name
        }
    
    def create_forecaster(self, parameters: Dict[str, Any]) -> ProphetForecaster:
        """Create Prophet forecaster with optimized parameters."""
        return create_prophet_forecaster(parameters)


class ARIMAParameterOptimizer(ParameterOptimizer):
    """
    Parameter optimizer for ARIMA forecasting.
    Uses auto-ARIMA for parameter selection.
    """
    
    def __init__(self):
        super().__init__("arima")
    
    def optimize_parameters(
        self, 
        data: pd.DataFrame, 
        product_record: pd.Series,
        base_parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Optimize ARIMA parameters using auto-ARIMA.
        """
        logger.info(f"Optimizing ARIMA parameters for {product_record['product_id']}")
        
        try:
            # Create a temporary ARIMA forecaster for parameter optimization
            temp_forecaster = ARIMAForecaster(
                window_length=product_record.get('forecast_window_length') * product_record.get('risk_period', 1),
                horizon=product_record.get('forecast_horizon', 1),
                auto_arima=True,
                max_p=3,
                max_d=2,
                max_q=3,
                seasonal=False,
                min_data_points=5
            )
            
            # Fit the model to get optimized parameters
            temp_forecaster.fit(data)
            
            # Get the optimized parameters
            optimized_parameters = {
                'window_length': product_record.get('forecast_window_length') * product_record.get('risk_period', 1),
                'horizon': product_record.get('forecast_horizon', 1),
                'auto_arima': True,
                'max_p': 3,
                'max_d': 2,
                'max_q': 3,
                'seasonal': False,
                'm': 1,
                'min_data_points': 5,
                'order': temp_forecaster.order,
                'seasonal_order': temp_forecaster.seasonal_order,
                'method_name': self.method_name
            }
            
            logger.debug(f"Optimized ARIMA parameters: {optimized_parameters}")
            return optimized_parameters
            
        except Exception as e:
            logger.error(f"Error optimizing ARIMA parameters: {e}")
            return self._get_default_arima_parameters(product_record, base_parameters)
    
    def _get_default_arima_parameters(
        self, 
        product_record: pd.Series, 
        base_parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get default ARIMA parameters as fallback."""
        return {
            'window_length': product_record.get('forecast_window_length') * product_record.get('risk_period', 1),
            'horizon': product_record.get('forecast_horizon', 1),
            'auto_arima': True,
            'max_p': 3,
            'max_d': 2,
            'max_q': 3,
            'seasonal': False,
            'm': 1,
            'min_data_points': 5,
            'order': (1, 1, 1),
            'seasonal_order': None,
            'method_name': self.method_name
        }
    
    def create_forecaster(self, parameters: Dict[str, Any]) -> ARIMAForecaster:
        """Create ARIMA forecaster with optimized parameters."""
        return ARIMAForecaster(
            window_length=parameters['window_length'],
            horizon=parameters['horizon'],
            auto_arima=parameters['auto_arima'],
            max_p=parameters['max_p'],
            max_d=parameters['max_d'],
            max_q=parameters['max_q'],
            seasonal=parameters['seasonal'],
            m=parameters['m'],
            min_data_points=parameters['min_data_points']
        )


class ParameterOptimizerFactory:
    """
    Factory class for creating parameter optimizers based on forecasting method.
    """
    
    _optimizers = {
        'moving_average': MovingAverageParameterOptimizer,
        'prophet': ProphetParameterOptimizer,
        'arima': ARIMAParameterOptimizer
    }
    
    @classmethod
    def get_optimizer(cls, method_name: str) -> ParameterOptimizer:
        """
        Get parameter optimizer for the specified method.
        
        Args:
            method_name: Name of the forecasting method
            
        Returns:
            Parameter optimizer instance
            
        Raises:
            ValueError: If method is not supported
        """
        if method_name not in cls._optimizers:
            raise ValueError(f"Unsupported forecasting method: {method_name}")
        
        return cls._optimizers[method_name]()
    
    @classmethod
    def register_optimizer(cls, method_name: str, optimizer_class: type):
        """
        Register a new parameter optimizer for a forecasting method.
        
        Args:
            method_name: Name of the forecasting method
            optimizer_class: Optimizer class to register
        """
        cls._optimizers[method_name] = optimizer_class
    
    @classmethod
    def get_supported_methods(cls) -> List[str]:
        """Get list of supported forecasting methods."""
        return list(cls._optimizers.keys()) 