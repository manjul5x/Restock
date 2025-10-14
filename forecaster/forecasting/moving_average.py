"""
Moving Average Forecasting Model

A simple baseline forecasting model that predicts a constant value equal to
the mean of the training data. This model serves as a baseline comparison
for more sophisticated forecasting methods.

Inherits from BaseForecastingModel and implements the required interface.
"""

import pandas as pd
import numpy as np
from .base import BaseForecastingModel


class MovingAverageModel(BaseForecastingModel):
    """
    Simple moving average forecasting model.
    
    This model computes the mean of training data and uses it as a constant
    forecast for all future periods. It's useful as a baseline comparison
    and for cases where simple, interpretable forecasts are needed.
    
    Attributes:
        average: The computed mean value from training data
        required_regressors: Empty list (no regressors needed)
    """
    
    def _initialize_model(self) -> None:
        """
        Initialize the moving average model.
        
        Sets required_regressors to empty list since this model
        doesn't use any regressors.
        """
        self.required_regressors = []
        self.average = None
    
    def _fit_model(self, train_df: pd.DataFrame) -> None:
        """
        Fit the moving average model to training data.
        
        Computes the mean of the 'y' column and stores it for prediction.
        
        Args:
            train_df: Training DataFrame with 'ds' and 'y' columns
        """
        # Compute the mean of training data
        self.average = train_df['y'].mean()
        
    
    def _predict_model(self, future_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions using the fitted moving average model.
        
        Returns a constant forecast equal to the computed average
        for all future periods.
        
        Args:
            future_df: Future DataFrame with 'ds' column
            
        Returns:
            DataFrame with 'ds' and 'yhat' columns where yhat = self.average
        """
        # Create predictions DataFrame
        predictions_df = pd.DataFrame({
            'ds': future_df['ds'],
            'yhat': self.average
        })
        
        return predictions_df
