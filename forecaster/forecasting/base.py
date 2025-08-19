"""
Base Forecasting Model - Abstract Base Class

This module defines the base class that all forecasting models must inherit from.
It provides a consistent interface for the ForecastingEngine to work with different
forecasting algorithms.

All forecasting models must implement:
- Initialization with product_id and location_id
- fit() method for training
- predict() method for forecasting
- required_regressors attribute
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any


class BaseForecastingModel(ABC):
    """
    Abstract base class for all forecasting models.
    
    This class defines the contract that all forecasting models must implement
    to work with the ForecastingEngine. Models should inherit from this class
    and implement the abstract methods.
    
    Attributes:
        product_id: Product identifier
        location_id: Location identifier
        required_regressors: List of regressor column names required by this model
    """
    
    def __init__(self, product_id: str, location_id: str):
        """
        Initialize the forecasting model.
        
        Args:
            product_id: Product identifier
            location_id: Location identifier
        """
        self.product_id = product_id
        self.location_id = location_id
        self.required_regressors: List[str] = []
        self._is_fitted = False
        
        # Call the model-specific initialization
        self._initialize_model()
    
    @abstractmethod
    def _initialize_model(self) -> None:
        """
        Initialize the specific forecasting model.
        
        This method should be implemented by subclasses to:
        - Set up model-specific parameters
        - Initialize the underlying forecasting algorithm
        - Set self.required_regressors to the list of regressor columns needed
        
        The method should not take any arguments beyond self. 
        Product and location id can be accessed from self.product_id and self.location_id
        """
        pass
    
    def fit(self, train_df: pd.DataFrame) -> None:
        """
        Fit the forecasting model to training data.
        
        Args:
            train_df: Training DataFrame with columns 'ds' (date), 'y' (target),
                     and potentially regressor columns. Must not be mutated.
        
        Raises:
            ValueError: If required columns are missing or data is invalid
            RuntimeError: If model fails to fit
        """
        # Validate input data
        self._validate_training_data(train_df)
        
        # Create a copy to avoid mutating the original
        train_copy = train_df.copy()
        
        try:
            # Call the model-specific fitting logic
            self._fit_model(train_copy)
            self._is_fitted = True
            
        except Exception as e:
            raise RuntimeError(f"Model fitting failed for {self.product_id} at {self.location_id}: {e}")
    
    def predict(self, future_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for future data.
        
        Args:
            future_df: Future DataFrame with columns 'ds' (date), 'y' (actual if available else NaN),
                      and potentially regressor columns.
        
        Returns:
            DataFrame with predictions. Must contain at least 'ds', 'y', and 'yhat' columns.
            Length must equal len(future_df) and align on 'ds'. May include additional
            columns like upper/lower bounds, component decomposition, etc.
        
        Raises:
            RuntimeError: If model is not fitted
            ValueError: If required columns are missing or data is invalid
        """
        if not self._is_fitted:
            raise RuntimeError(f"Model must be fitted before prediction for {self.product_id} at {self.location_id}")
        
        # Validate input data
        self._validate_prediction_data(future_df)
        
        # Create a copy to avoid mutating the original
        future_copy = future_df.copy()
        future_copy = future_copy.sort_values(by='ds', ascending=True)
        future_copy['y'] = np.nan
        
        try:
            # Call the model-specific prediction logic
            predictions_df = self._predict_model(future_copy)

            # add a function to add y back into the predictions df from the future df based on ds
            predictions_df = self._add_y_to_predictions(predictions_df, future_df)
            
            # Validate output   
            self._validate_predictions_output(predictions_df, future_df)
            
            return predictions_df
            
        except Exception as e:
            raise RuntimeError(f"Model prediction failed for {self.product_id} at {self.location_id}: {e}")
    
    def _validate_training_data(self, train_df: pd.DataFrame) -> None:
        """
        Validate training data format and content.
        
        Args:
            train_df: Training DataFrame to validate
            
        Raises:
            ValueError: If validation fails
        """
        # Check required columns
        required_columns = {'ds', 'y'}
        missing_columns = required_columns - set(train_df.columns)
        if missing_columns:
            raise ValueError(f"Training data missing required columns: {missing_columns}")
        
        # Check for empty data
        if train_df.empty:
            raise ValueError("Training data cannot be empty")
        
        # Check for required regressors if specified
        if self.required_regressors:
            missing_regressors = set(self.required_regressors) - set(train_df.columns)
            if missing_regressors:
                raise ValueError(f"Missing required regressors: {missing_regressors}")
        
        # Validate data types
        if not pd.api.types.is_datetime64_any_dtype(train_df['ds']):
            raise ValueError("Column 'ds' must be datetime type")
        
        if not pd.api.types.is_numeric_dtype(train_df['y']):
            raise ValueError("Column 'y' must be numeric type")
        
        # Check for NaN values in required columns
        if train_df['ds'].isna().any():
            raise ValueError("Column 'ds' cannot contain NaN values")
        
        if train_df['y'].isna().any():
            import warnings
            warnings.warn(f"{self.product_id} at {self.location_id} has NaN values in training data", RuntimeWarning)
    
    def _validate_prediction_data(self, future_df: pd.DataFrame) -> None:
        """
        Validate future data format and content.
        
        Args:
            future_df: Future DataFrame to validate
            
        Raises:
            ValueError: If validation fails
        """
        # Check required columns
        required_columns = {'ds', 'y'}
        missing_columns = required_columns - set(future_df.columns)
        if missing_columns:
            raise ValueError(f"Future data missing required columns: {missing_columns}")
        
        # Check for empty data
        if future_df.empty:
            raise ValueError("Future data cannot be empty")
        
        # Check for required regressors if specified
        if self.required_regressors:
            missing_regressors = set(self.required_regressors) - set(future_df.columns)
            if missing_regressors:
                raise ValueError(f"{self.product_id} at {self.location_id} missing required regressors: {missing_regressors}")
        
        # Validate data types
        if not pd.api.types.is_datetime64_any_dtype(future_df['ds']):
            raise ValueError("Column 'ds' must be datetime type")
        
        # Allow 'y' to be all NaN, all numeric, or a mix of numeric and NaN
        # Check that all non-NaN values are numeric
        non_nan_y = future_df['y'].dropna()
        if not non_nan_y.empty and not pd.api.types.is_numeric_dtype(non_nan_y):
            raise ValueError("Column 'y' must be numeric type (or NaN)")
        
        # Check for NaN values in required columns
        if future_df['ds'].isna().any():
            raise ValueError("Column 'ds' cannot contain NaN values")
    
    def _validate_predictions_output(self, predictions_df: pd.DataFrame, future_df: pd.DataFrame) -> None:
        """
        Validate that predictions output meets requirements.
        
        Args:
            predictions_df: Predictions DataFrame to validate
            future_df: Original future DataFrame for comparison
            
        Raises:
            ValueError: If validation fails
        """
        # Check required output columns
        required_columns = {'ds', 'y', 'yhat'}
        missing_columns = required_columns - set(predictions_df.columns)
        if missing_columns:
            raise ValueError(f"Predictions missing required columns: {missing_columns}")
        
        # Check length matches
        if len(predictions_df) != len(future_df):
            raise ValueError(f"Predictions length ({len(predictions_df)}) must equal future data length ({len(future_df)})")
        
        # Check ds column alignment
        if not predictions_df['ds'].equals(future_df['ds']):
            raise ValueError("Predictions 'ds' column must align with future data 'ds' column")
        
        # Validate data types
        if not pd.api.types.is_datetime64_any_dtype(predictions_df['ds']):
            raise ValueError("Predictions column 'ds' must be datetime type")
        
        if not pd.api.types.is_numeric_dtype(predictions_df['yhat']):
            raise ValueError("Predictions column 'yhat' must be numeric type")
    
    @abstractmethod
    def _fit_model(self, train_df: pd.DataFrame) -> None:
        """
        Fit the specific forecasting model to training data.
        
        Args:
            train_df: Copy of training DataFrame with columns 'ds', 'y', and regressors
        
        This method should be implemented by subclasses to contain the actual
        model fitting logic. The input DataFrame is a copy and can be modified
        if needed for the specific model implementation.
        """
        pass
    
    @abstractmethod
    def _predict_model(self, future_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions using the fitted model.
        
        Args:
            future_df: Copy of future DataFrame with columns 'ds', 'y', and regressors
        
        Returns:
            DataFrame with predictions including 'ds', 'y', 'yhat', and potentially
            additional columns like bounds, components, etc.
        
        This method should be implemented by subclasses to contain the actual
        prediction logic. The input DataFrame is a copy and can be modified
        if needed for the specific model implementation.
        """
        pass
    
    @property
    def is_fitted(self) -> bool:
        """
        Check if the model has been fitted.
        
        Returns:
            True if model is fitted, False otherwise
        """
        return self._is_fitted
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            'model_type': self.__class__.__name__,
            'product_id': self.product_id,
            'location_id': self.location_id,
            'required_regressors': self.required_regressors,
            'is_fitted': self._is_fitted
        }
    
    def _add_y_to_predictions(self, predictions_df: pd.DataFrame, future_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add the 'y' column from future_df to predictions_df based on 'ds' alignment.
        
        Args:
            predictions_df: DataFrame with predictions (must have 'ds' column)
            future_df: DataFrame with future data including 'y' column
            
        Returns:
            predictions_df with 'y' column added/updated
        """
        # Create a copy to avoid modifying the original
        result_df = predictions_df.copy()
        
        # Use merge to efficiently align 'y' values based on 'ds'
        # This is more efficient than iterating through rows
        # Merge 'y' from future_df into result_df on 'ds' without changing order or index
        result_df = result_df.merge(future_df, on='ds', how='left', suffixes=('', '_raw')).sort_values(by='ds', ascending=True)
        result_df.rename(columns={'y_raw': 'y'}, inplace=True)
        return result_df
