"""
Forecasting Engine - Model-Agnostic Orchestrator

This engine provides a unified interface for different forecasting models,
handling data preparation, model selection, training, prediction, and output
composition. It's designed to work with the new backtesting pipeline.

Key Features:
- Model-agnostic design with dynamic model selection
- Automatic regressor inference and handling
- Standardized input/output schemas
- Support for Prophet, MovingAverage, and future models
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import date, datetime
import warnings

from pandas.core.config_init import parquet_engine_doc

# Import model classes (these will be implemented separately)
try:
    from .prophet import ProphetModel
    from .moving_average import MovingAverageModel
    from forecaster.validation.product_master_schema import ProductMasterSchema
except ImportError:
    # Fallback for development/testing
    ProphetModel = None
    MovingAverageModel = None


class ForecastingEngine:
    """
    Model-agnostic forecasting engine that orchestrates the complete
    forecasting workflow from data preparation to output composition.
    
    This engine replaces the legacy forecasting system and provides a
    unified interface for different forecasting models.
    """
    
    def __init__(self):
        """Initialize the forecasting engine."""
        self.available_models = {
            'prophet': ProphetModel,
            'moving_average': MovingAverageModel,
        }
        
        # Identifier columns to exclude from regressors
        self.identifier_columns = {'product_id', 'location_id'}
        
        # Metadata columns to exclude from regressors
        self.metadata_columns = {'date', 'outflow'}
    
    def generate_forecast(self,
                         forecast_method: str,
                         training_data: pd.DataFrame,
                         product_record: Dict[str, Any],
                         future_data_frame: pd.DataFrame,
                         override_config: Optional[Dict[str, Any]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate forecast using the specified method.
        
        Args:
            forecast_method: Name of the forecasting method to use
            training_data: Historical data for training (includes regressors)
            product_record: Product configuration including risk_period, demand_frequency
            future_data_frame: Future data frame for prediction (includes regressors)
            override_config: Optional configuration override for the model
            
        Returns:
            Tuple of (forecast_comparison_df, components_df)
            
        Raises:
            ValueError: If forecast_method is not supported
            KeyError: If required columns are missing
        """
        
        self.product_id = product_record['product_id']
        self.location_id = product_record['location_id']

        # Select and initialize model with optional override configuration
        model = self._select_and_initialize_model(forecast_method, product_record, override_config)
        
        # Prepare training data
        train_df = self._prepare_training_data(training_data, model)
        
        # Prepare future data
        future_df = self._prepare_future_data(future_data_frame, model)
        
        # Train model and generate predictions
        model.fit(train_df)
        predict_df = model.predict(future_df)
        
        # Compose outputs
        forecast_comparison_df = self._compose_forecast_comparison(
            train_df, future_df, predict_df, product_record, forecast_method
        )
        
        return forecast_comparison_df, predict_df
    
    def _select_and_initialize_model(self, 
                                   forecast_method: str,
                                   product_record: Dict[str, Any],
                                   override_config: Optional[Dict[str, Any]] = None) -> Any:
        """
        Select and initialize the appropriate forecasting model.
        
        Args:
            forecast_method: Name of the forecasting method
            product_record: Product configuration
            override_config: Optional configuration override for the model
            
        Returns:
            Initialized model instance
            
        Raises:
            ValueError: If forecast_method is not supported
        """
        if forecast_method not in self.available_models:
            available = list(self.available_models.keys())
            raise ValueError(f"Unsupported forecast_method: {forecast_method}. "
                           f"Available methods: {available}")
        
        model_class = self.available_models[forecast_method]
        if model_class is None:
            raise ValueError(f"Model class for {forecast_method} is not available. "
                           "Please ensure the model is properly imported.")
        
        # Initialize model with product configuration and optional override
        if forecast_method == 'prophet':
            model = model_class(
                product_id=product_record['product_id'],
                location_id=product_record['location_id'],
                override_config=override_config
            )
        else:
            # For other models, initialize without override (maintain backward compatibility)
            model = model_class(
                product_id=product_record['product_id'],
                location_id=product_record['location_id']
            )
        
        # Store required regressors or empty list
        self.required_regressors = getattr(model, 'required_regressors', [])
        
        return model
    
    def _prepare_training_data(self, 
                              training_data: pd.DataFrame,
                              model: Any) -> pd.DataFrame:
        """
        Prepare training data for the model.
        
        Args:
            training_data: Raw training data
            model: Model instance to get required regressors
            
        Returns:
            Prepared training DataFrame
        """
        # Create copy to avoid modifying original
        train_df = training_data.copy()
        
        # Rename core columns
        train_df = train_df.rename(columns={'date': 'ds', 'outflow': 'y'})

        # Get required regressors from model
        required_regressors = self.required_regressors
        
        # Select only required columns
        required_columns = ['ds', 'y'] + required_regressors
        available_columns = [col for col in required_columns if col in train_df.columns]
        
        # Warn if required regressors are missing
        missing_regressors = set(required_regressors) - set(train_df.columns)
        if missing_regressors:
            raise ValueError(f"{self.product_id} at {self.location_id} missing required regressors: {missing_regressors}")
        
        # Select final columns
        train_df = train_df[available_columns].copy()
        
        # Ensure ds column is datetime (Prophet requires datetime)
        if 'ds' in train_df.columns:
            train_df['ds'] = pd.to_datetime(train_df['ds'])
        
        # Sort by date
        if 'ds' in train_df.columns:
            train_df = train_df.sort_values('ds').reset_index(drop=True)
        
        return train_df
    
    def _prepare_future_data(self, 
                            future_data_frame: pd.DataFrame,
                            model: Any) -> pd.DataFrame:
        """
        Prepare future data for prediction.
        
        Args:
            future_data_frame: Raw future data
            model: Model instance to get required regressors
            
        Returns:
            Prepared future DataFrame
        """
        # Create copy to avoid modifying original
        future_df = future_data_frame.copy()
        
        # Rename core columns
        future_df = future_df.rename(columns={'date': 'ds', 'outflow': 'y'})
        
        # Get required regressors from model
        required_regressors = self.required_regressors
        
        # Select only required columns
        required_columns = ['ds', 'y'] + required_regressors
        available_columns = [col for col in required_columns if col in future_df.columns]

        # Raise if required regressors are missing
        missing_regressors = set(required_regressors) - set(future_df.columns)
        if missing_regressors:
            raise ValueError(f"{self.product_id} at {self.location_id} missing required regressors: {missing_regressors}")
        
        # Select final columns
        future_df = future_df[available_columns].copy()
        
        # Ensure ds column is datetime (Prophet requires datetime)
        if 'ds' in future_df.columns:
            future_df['ds'] = pd.to_datetime(future_df['ds'])
        
        # Sort by date
        if 'ds' in future_df.columns:
            future_df = future_df.sort_values('ds').reset_index(drop=True)
        
        return future_df
    
    def _compose_forecast_comparison(self,
                                   train_df: pd.DataFrame,
                                   future_df: pd.DataFrame,
                                   predict_df: pd.DataFrame,
                                   product_record: Dict[str, Any],
                                   forecast_method: str) -> pd.DataFrame:
        """
        Compose the forecast comparison DataFrame using vectorized operations.
        
        Args:
            future_df: Prepared future data
            predict_df: Model predictions
            product_record: Product configuration
            forecast_method: Name of the forecasting method
            
        Returns:
            Forecast comparison DataFrame
        """
        # Calculate risk period days using function from product_master_schema
        risk_period_days = ProductMasterSchema.get_risk_period_days(
            product_record['demand_frequency'],
            product_record['risk_period']
        )
        
        # Create base DataFrame with vectorized operations
        comparison_df = pd.DataFrame({
            'analysis_date': predict_df['ds'].dt.date,
            'product_id': product_record.get('product_id'),
            'location_id': product_record.get('location_id'),
            'forecast_method': forecast_method,
            'risk_period': risk_period_days,
            'demand_frequency': product_record['demand_frequency'],
            'first_date_used': train_df['ds'].min().date(),
            'step': range(1, len(predict_df) + 1),
            'risk_period_start': predict_df['ds'],
            'risk_period_end': predict_df['ds'] + pd.Timedelta(days=risk_period_days),
            'actual_demand': predict_df['y'],
            'forecast_demand': predict_df['yhat']
        })
        
        # Vectorized error calculations
        mask = comparison_df['actual_demand'].notna() & comparison_df['forecast_demand'].notna()
        
        # Initialize error columns with NaN
        comparison_df['forecast_error'] = np.nan
        comparison_df['absolute_error'] = np.nan
        comparison_df['percentage_error'] = np.nan
        
        # Calculate errors where both values are available
        comparison_df.loc[mask, 'forecast_error'] = (
            comparison_df.loc[mask, 'actual_demand'] - comparison_df.loc[mask, 'forecast_demand']
        )
        
        comparison_df.loc[mask, 'absolute_error'] = comparison_df.loc[mask, 'forecast_error'].abs()
        
        # Calculate percentage error (avoid division by zero)
        zero_mask = comparison_df.loc[mask, 'actual_demand'] != 0
        comparison_df.loc[mask & zero_mask, 'percentage_error'] = (
            comparison_df.loc[mask & zero_mask, 'absolute_error'] / 
            comparison_df.loc[mask & zero_mask, 'actual_demand'] * 100
        )
        
        return comparison_df
    
    def get_available_methods(self) -> List[str]:
        """
        Get list of available forecasting methods.
        
        Returns:
            List of available method names
        """
        return list(self.available_models.keys())

