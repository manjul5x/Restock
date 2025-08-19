"""
Input Data Prepper for the new backtesting pipeline.

This class orchestrates the computation of all regressor features and returns
a final DataFrame ready for forecasting.
"""

import pandas as pd
import numpy as np
from datetime import date, datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import yaml

try:
    from .regressors import AVAILABLE_REGRESSORS
except ImportError:
    # Fallback for when running as script
    from regressors import AVAILABLE_REGRESSORS


class InputDataPrepper:
    """
    Main class for preparing input data with all regressor features.
    
    This class orchestrates the computation of:
    - Forward-looking outflow (aggregated demand)
    - Lag features (rp_lag, half_rp_lag)
    - Seasonality features (season, week_of_month)
    - Recency weights
    
    The architecture allows users to easily add custom regressors without
    modifying the core pipeline code.
    """
    
    def __init__(self, config_path: str = "data/config/regressor_config.yaml"):
        """
        Initialize the InputDataPrepper.
        
        Args:
            config_path: Path to regressor configuration file
        """
        self.config_path = config_path
        self.logger = self._get_logger()
        self.config = self._load_config()
        
        self.logger.info("InputDataPrepper initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load regressor configuration from file."""
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                self.logger.info(f"Loaded regressor config from {self.config_path}")
            else:
                # No default config - must be provided via config file
                raise FileNotFoundError(f"Regressor config file not found: {self.config_path}")
        except Exception as e:
            self.logger.error(f"Failed to load config from {self.config_path}: {e}")
            raise
        
        return config
    
    # No default config - must be provided via config file
    
    def _get_logger(self):
        """Get logger instance."""
        try:
            from forecaster.utils.logger import get_logger
            return get_logger(__name__)
        except ImportError:
            # Fallback to basic logging if forecaster logger not available
            import logging
            return logging.getLogger(__name__)
    
    def prepare_data(self, 
                    df: pd.DataFrame,
                    product_master_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare input data with all regressor features.
        
        Args:
            df: Input DataFrame with demand, date, product_id, location_id columns
            product_master_df: Product master DataFrame with all product configurations
            
        Returns:
            DataFrame with all computed regressor features
        """
        self.logger.info("Preparing input data with regressors")
        
        # Step 1: Validate input DataFrame
        self._validate_input_dataframe(df)
        
        # Step 2: Validate regressor configuration
        self._validate_regressor_config()
        
        # Step 3: Get enabled regressors
        enabled_regressors = self._get_enabled_regressors()
        
        # Create a copy to avoid modifying original
        result_df = df.copy()
        
        # Step 4: Loop through enabled regressors to add columns
        for regr_name, regr_cfg in enabled_regressors.items():
            self.logger.debug(f"Computing {regr_name} feature")
            
            # Get function name and parameters
            function_name = regr_cfg["function_name"]
            params = regr_cfg.get("parameters", {})
            
            # Get the actual function from the regressors module
            compute_func = self._get_regressor_function(function_name)
            
            # Compute the feature(s) and add to DataFrame
            column_name = regr_cfg["column_name"]
            feature_result = compute_func(result_df, product_master_df, **params)
            if isinstance(column_name, list):
                if not isinstance(feature_result, list) or len(feature_result) != len(column_name):
                    raise ValueError(
                        f"Regressor function '{function_name}' must return a list of Series with the same length as column_name list."
                    )
                for col, series in zip(column_name, feature_result):
                    result_df[col] = series
            else:
                result_df[column_name] = feature_result
        
        # Log NaN values
        result_df = self._log_nan_values(result_df)
        
        self.logger.info(f"Data preparation completed. Final shape: {result_df.shape}")
        
        return result_df
    
    def _validate_input_dataframe(self, df: pd.DataFrame) -> None:
        """Validate that input DataFrame has required columns."""
        required_columns = ['demand', 'date', 'product_id', 'location_id']
        missing_columns = set(required_columns) - set(df.columns)
        
        if missing_columns:
            raise ValueError(
                f"Input DataFrame missing required columns: {missing_columns}. "
                f"Available columns: {list(df.columns)}"
            )
        
        # Ensure date column is datetime type
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            raise ValueError("'date' column must be datetime type")
    
    
    def _log_nan_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Log information about NaN values in the result DataFrame."""
        # Log info about NaN values
        nan_counts = df.isnull().sum()
        if nan_counts.sum() > 0:
            self.logger.info(f"NaN value counts: {nan_counts[nan_counts > 0].to_dict()}")
        
        # Keep all rows for now - NaN handling should be done upstream
        # when constructing training data
        return df
    
    def get_enabled_regressors(self) -> List[str]:
        """Get list of enabled regressor names."""
        enabled = []
        for regressor_name, config in self.config.items():
            if config.get('enabled', True):
                enabled.append(regressor_name)
        return enabled
    
    def _validate_regressor_config(self) -> None:
        """Validate regressor configuration."""
        if not isinstance(self.config, dict):
            raise ValueError("Regressor config must be a dictionary")
        
        for regressor_name, regressor_config in self.config.items():
            if not isinstance(regressor_config, dict):
                raise ValueError(f"Config for regressor '{regressor_name}' must be a dictionary")
            
            # Check required fields
            required_fields = ['column_name', 'description', 'enabled', 'function_name']
            missing_fields = set(required_fields) - set(regressor_config.keys())
            if missing_fields:
                raise ValueError(f"Regressor '{regressor_name}' missing required fields: {missing_fields}")
            
            # Check if regressor is enabled
            if not regressor_config.get('enabled', True):
                continue
            
            # Validate function name exists
            function_name = regressor_config['function_name']
            if function_name not in AVAILABLE_REGRESSORS:
                raise ValueError(f"Unknown function '{function_name}' for regressor '{regressor_name}'. Available: {list(AVAILABLE_REGRESSORS)}")
    
    def _get_enabled_regressors(self) -> Dict[str, Any]:
        """Get dictionary of enabled regressors."""
        enabled = {}
        
        for regressor_name, regressor_config in self.config.items():
            if regressor_config.get('enabled', True):
                enabled[regressor_name] = regressor_config
        
        return enabled
    
    def _get_regressor_function(self, function_name: str):
        """Get the actual function dynamically from the regressors module."""
        # Import the function dynamically from the regressors module
        try:
            from .regressors import AVAILABLE_REGRESSORS
            if function_name not in AVAILABLE_REGRESSORS:
                raise ValueError(f"Unknown function '{function_name}'. Available: {list(AVAILABLE_REGRESSORS)}")
            module = __import__('.regressors', fromlist=[function_name])
        except ImportError:
            # Fallback for when running as script
            try:
                from regressors import AVAILABLE_REGRESSORS
                if function_name not in AVAILABLE_REGRESSORS:
                    raise ValueError(f"Unknown function '{function_name}'. Available: {list(AVAILABLE_REGRESSORS)}")
                module = __import__('regressors', fromlist=[function_name])
            except ImportError:
                # Final fallback - try absolute import
                import sys
                sys.path.insert(0, str(Path(__file__).parent))
                from regressors import AVAILABLE_REGRESSORS
                if function_name not in AVAILABLE_REGRESSORS:
                    raise ValueError(f"Unknown function '{function_name}'. Available: {list(AVAILABLE_REGRESSORS)}")
                module = __import__('regressors', fromlist=[function_name])
        return getattr(module, function_name)

