"""
Prophet Forecasting Model

A configuration-driven Prophet forecasting model that inherits from BaseForecastingModel.
This model reads product-specific Prophet parameters from a JSON configuration file and
automatically sets up the model with the exact parameters needed for that product-location
combination.

Features:
- Configuration-driven initialization from JSON
- Dynamic regressor management
- Comprehensive seasonality handling (built-in + custom)
- Full Prophet component decomposition output
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
import json
import warnings
import logging
from pathlib import Path
from data.config.paths import get_input_file_path
from data.loader import DataLoader

logger = logging.getLogger(__name__)
try:
    from prophet import Prophet
except ImportError:
    Prophet = None
    warnings.warn("Prophet library not available. Please install prophet package.")

from .base import BaseForecastingModel



class ProphetModel(BaseForecastingModel):
    """
    Configuration-driven Prophet forecasting model.
    
    This model reads Prophet parameters from a JSON configuration file and automatically
    sets up the model with the exact parameters needed for that product-location
    combination. It handles dynamic regressors, seasonalities, and holidays based
    on the configuration.
    
    Attributes:
        config: Product-specific Prophet configuration
        model: Prophet model instance
        required_regressors: List of regressor column names required by this model
    """
    
    def __init__(self, product_id: str, location_id: str, override_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Prophet model with product-specific configuration.
        
        Args:
            product_id: Product identifier
            location_id: Location identifier
            override_config: Optional configuration override (takes precedence over JSON config)
        """
        # Initialize required_regressors before calling parent
        self.required_regressors = []
        self.config = None
        self.model = None
        self.override_config = override_config
        
        # Call parent constructor
        super().__init__(product_id, location_id)
    
    def _initialize_model(self) -> None:
        """
        Initialize the Prophet model from configuration.
        
        This method:
        1. Reads configuration from JSON for the specific product-location
        2. Creates Prophet model with configured parameters
        3. Sets up seasonalities, regressors, and holidays
        4. Sets self.required_regressors for data subsetting
        """
        # Load configuration
        self.config = self._load_product_config()

        # Create Prophet model
        self.model = self._create_prophet_model()
        
        # Add seasonalities
        self._add_seasonalities()
        
        # Add regressors
        self._add_regressors()
        
    
    def _load_product_config(self) -> Dict[str, Any]:
        """
        Load product-specific configuration from JSON or override.
        
        Returns:
            Dictionary containing Prophet configuration parameters
            
        Raises:
            FileNotFoundError: If configuration file doesn't exist
            KeyError: If product-location combination not found
        """
        # Check for override configuration first (takes precedence)
        if self.override_config is not None:
            logger.debug(f"Using override configuration for {self.product_id} at {self.location_id}")
            
            # Validate override configuration structure
            if self.override_config:
                self._validate_config_structure(self.override_config)
            
            return self.override_config
        
        # Fall back to JSON configuration
        try:
            # Load data config and get path to prophet parameters
            config_path = get_input_file_path('prophet_parameters')
            
            # Check if file exists
            if not Path(config_path).exists():
                raise FileNotFoundError(f"Prophet parameters file not found: {config_path}")
            
            # Read configuration JSON
            try:
                with open(config_path, 'r') as f:
                    all_configs = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in prophet parameters file: {e}")
            except Exception as e:
                raise RuntimeError(f"Failed to read prophet parameters file: {e}")
            
            # Validate JSON structure
            if not isinstance(all_configs, dict):
                raise ValueError("Prophet parameters file must contain a JSON object")
            
            # Look up by product_id and location_id tuple
            key = str((self.product_id, self.location_id))
            if key not in all_configs:
                raise KeyError(f"Prophet configuration not found for {self.product_id} at {self.location_id}")
            
            # Get configuration for this product-location
            config = all_configs[key]
            
            # Validate configuration structure
            if config is not None:
                self._validate_config_structure(config)
            
            # If config is empty ({}), return empty dict - Prophet will use defaults
            if not config:
                logger.debug(f"Using default Prophet parameters for {self.product_id} at {self.location_id}")
                return {}
            
            return config
            
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration for {self.product_id} at {self.location_id}: {e}")
    
    def _validate_config_structure(self, config: Dict[str, Any]) -> None:
        """
        Validate the structure of a configuration dictionary.
        
        Args:
            config: Configuration dictionary to validate
            
        Raises:
            ValueError: If configuration structure is invalid
        """
        if not isinstance(config, dict):
            raise ValueError(f"Configuration must be a dictionary, got {type(config)}")
        
        # Check for required fields if config is not empty
        if config:
            # Validate regressors if present
            if 'regressors' in config:
                if not isinstance(config['regressors'], dict):
                    raise ValueError("Regressors must be a dictionary")
                
                for regressor_name, regressor_config in config['regressors'].items():
                    if not isinstance(regressor_config, dict):
                        raise ValueError(f"Regressor {regressor_name} configuration must be a dictionary")
                    
                    # Validate prior_scale range
                    if 'prior_scale' in regressor_config:
                        prior_scale = regressor_config['prior_scale']
                        if not isinstance(prior_scale, (int, float)) or prior_scale < 0.01 or prior_scale > 20.0:
                            raise ValueError(f"Regressor {regressor_name} prior_scale must be between 0.01 and 20.0")
                    
                    # Validate mode
                    if 'mode' in regressor_config:
                        mode = regressor_config['mode']
                        if mode not in ['additive', 'multiplicative']:
                            raise ValueError(f"Regressor {regressor_name} mode must be 'additive' or 'multiplicative'")
            
            # Validate seasonality settings
            if 'yearly_seasonality' in config and not isinstance(config['yearly_seasonality'], bool):
                raise ValueError("yearly_seasonality must be a boolean")
            
            if 'weekly_seasonality' in config and not isinstance(config['weekly_seasonality'], bool):
                raise ValueError("weekly_seasonality must be a boolean")
            
            if 'daily_seasonality' in config and not isinstance(config['daily_seasonality'], bool):
                raise ValueError("daily_seasonality must be a boolean")
            
            if 'seasonality_mode' in config and config['seasonality_mode'] not in ['additive', 'multiplicative']:
                raise ValueError("seasonality_mode must be 'additive' or 'multiplicative'")
            
            # Validate custom seasonalities
            if config.get('add_custom') and 'custom_seasonalities' in config:
                custom_seasonalities = config['custom_seasonalities']
                if not isinstance(custom_seasonalities, dict):
                    raise ValueError("custom_seasonalities must be a dictionary")
                
                for name, seasonality_config in custom_seasonalities.items():
                    if not isinstance(seasonality_config, dict):
                        raise ValueError(f"Custom seasonality {name} must be a dictionary")
                    
                    # Validate period range
                    if 'period' in seasonality_config:
                        period = seasonality_config['period']
                        if not isinstance(period, (int, float)) or period < 2 or period > 365:
                            raise ValueError(f"Custom seasonality {name} period must be between 2 and 365 days")
                    
                    # Validate fourier order
                    if 'fourier' in seasonality_config:
                        fourier = seasonality_config['fourier']
                        if not isinstance(fourier, int) or fourier < 1 or fourier > 15:
                            raise ValueError(f"Custom seasonality {name} fourier order must be between 1 and 15")
                    
                    # Validate prior scale
                    if 'prior' in seasonality_config:
                        prior = seasonality_config['prior']
                        if not isinstance(prior, (int, float)) or prior < 0.01 or prior > 20.0:
                            raise ValueError(f"Custom seasonality {name} prior must be between 0.01 and 20.0")
            
            # Validate growth settings
            if 'growth' in config and config['growth'] not in ['linear', 'logistic', 'flat']:
                raise ValueError("growth must be 'linear', 'logistic', or 'flat'")
            
            # Validate changepoint settings
            if 'n_changepoints' in config:
                n_changepoints = config['n_changepoints']
                if not isinstance(n_changepoints, int) or n_changepoints < 5 or n_changepoints > 50:
                    raise ValueError("n_changepoints must be between 5 and 50")
            
            if 'changepoint_range' in config:
                changepoint_range = config['changepoint_range']
                if not isinstance(changepoint_range, (int, float)) or changepoint_range < 0.1 or changepoint_range > 1.0:
                    raise ValueError("changepoint_range must be between 0.1 and 1.0")
            
            if 'changepoint_prior_scale' in config:
                changepoint_prior_scale = config['changepoint_prior_scale']
                if not isinstance(changepoint_prior_scale, (int, float)) or changepoint_prior_scale < 0.001 or changepoint_prior_scale > 0.5:
                    raise ValueError("changepoint_prior_scale must be between 0.001 and 0.5")
    
    def _create_prophet_model(self) -> Prophet:
        """
        Create Prophet model with basic configuration parameters.
        
        Returns:
            Configured Prophet model instance
        """
        if Prophet is None:
            raise RuntimeError("Prophet library not available")

        holidays = self._determine_holidays()
        
        # Extract basic Prophet parameters
        growth = self.config.get('growth', 'linear')
        growth_floor = self.config.get('growth_floor', 0)
        growth_ceiling = self.config.get('growth_ceiling')
        
        # Create Prophet model
        model = Prophet(
            growth=growth,
            changepoint_prior_scale=self.config.get('changepoint_prior_scale', 0.05),
            n_changepoints=self.config.get('n_changepoints', 5),
            changepoint_range=self.config.get('changepoint_range', 0.8),
            holidays=holidays,
            holidays_prior_scale=self.config.get('holidays_prior_scale', 5.0),
            yearly_seasonality=self.config.get('yearly_seasonality', True),
            weekly_seasonality=self.config.get('weekly_seasonality', True),
            daily_seasonality=self.config.get('daily_seasonality', False),
            seasonality_mode=self.config.get('seasonality_mode', 'additive'),
            seasonality_prior_scale=self.config.get('seasonality_prior_scale', 10.0)
        )
        
        # Set growth floor/ceiling for logistic growth
        if growth == 'logistic':
            if growth_floor is not None:
                model.growth_floor = growth_floor
            if growth_ceiling is not None:
                model.growth_ceiling = growth_ceiling
            else:
                raise ValueError(f"{self.product_id} at {self.location_id} growth_ceiling must be set for logistic growth")
        
        return model
    
    def _add_seasonalities(self) -> None:
        """Add built-in and custom seasonalities to the Prophet model."""
        
        # Add custom seasonalities
        if 'custom_seasonalities' in self.config:
            custom_seasonalities = self.config['custom_seasonalities']
            if isinstance(custom_seasonalities, dict):
                for seasonality_name, seasonality_config in custom_seasonalities.items():
                    if isinstance(seasonality_config, dict):
                        self.model.add_seasonality(
                            name=seasonality_config.get('name', seasonality_name),
                            period=seasonality_config.get('period'),
                            fourier_order=seasonality_config.get('fourier', 5),
                            mode=seasonality_config.get('mode', 'additive'),
                            prior_scale=seasonality_config.get('prior', 1.0)
                        )
            else:
                raise ValueError(f"{self.product_id} at {self.location_id} custom_seasonalities in config must be a dictionary")
    
    def _add_regressors(self) -> None:
        """Add regressors to the Prophet model and set required_regressors."""
        regressors_config = self.config.get('regressors', {})
        
        if isinstance(regressors_config, dict):
            for regressor_name, regressor_config in regressors_config.items():
                if isinstance(regressor_config, dict):
                    # Add regressor to Prophet model
                    self.model.add_regressor(
                        name=regressor_name,
                        prior_scale=regressor_config.get('prior_scale', 10.0),
                        standardize=regressor_config.get('standardize', 'auto'),
                        mode=regressor_config.get('mode', 'additive')
                    )
                    
                    # Add to required regressors list
                    self.required_regressors.append(regressor_name)
        else:
            raise ValueError(f"{self.product_id} at {self.location_id} regressors in configmust be a dictionary")

    
    def _determine_holidays(self) -> None:
        """Add holiday effects to the Prophet model."""
        try:
            # Get holidays value from config
            holidays_value = self.config.get('selected_holidays', None)

            # Normalize holidays_value to a list or None
            def is_empty(val):
                # Helper to robustly check for "empty" values
                if val is None:
                    return True
                if isinstance(val, float) and pd.isna(val):
                    return True
                if isinstance(val, str) and val.strip() == '':
                    return True
                if isinstance(val, (list, tuple)) and len(val) == 0:
                    return True
                if val is False or val == 0:
                    return True
                return False

            if is_empty(holidays_value):
                return None

            # If we get here, holidays are enabled, so try to load them
            try:
                data_loader = DataLoader()
                holidays_df = data_loader.load_holidays(location=self.location_id)

                if holidays_df is not None and not holidays_df.empty:
                    # Normalize holidays_value to a list of names if provided
                    holiday_names = None
                    if isinstance(holidays_value, str):
                        holiday_names = [holidays_value]
                    elif isinstance(holidays_value, (list, tuple, np.ndarray)):
                        # Convert to list, flatten if necessary
                        holiday_names = list(holidays_value)
                        print(holiday_names)    
                    # If holiday_names is set and not empty, filter
                    if holiday_names is not None and len(holiday_names) > 0:
                        selected_holidays = holidays_df[holidays_df["holiday"].isin(holiday_names)]
                        print(selected_holidays.head(10))

                    if selected_holidays.empty:
                        logger.warning(f"No holidays found for {self.product_id} at {self.location_id}")
                        return None

                    return selected_holidays
                else:
                    logger.warning(f"No holiday data available for location {self.location_id}")
                    return None

            except Exception as e:
                logger.warning(f"Failed to load holidays for {self.product_id} at {self.location_id}: {e}")
                return None

        except Exception as e:
            logger.warning(f"Failed to add holidays for {self.product_id} at {self.location_id}: {e}")
            return None
    
    def _fit_model(self, train_df: pd.DataFrame) -> None:
        """
        Fit the Prophet model to training data.
        
        Args:
            train_df: Training DataFrame with 'ds', 'y', and regressor columns
        """

        # silence verbose cdmstan info output
        # TODO: see if this works
        logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

        # Prophet expects specific column names, so ensure we have them
        if 'ds' not in train_df.columns or 'y' not in train_df.columns:
            raise ValueError("Training data must contain 'ds' and 'y' columns")
        
        #fall back to moving average if we don't have enough data
        self.using_fallback = False
        if len(train_df) < 25:
            logger.debug(f"Insufficient training data for Prophet ({len(train_df)} periods) for {self.product_id} at {self.location_id}. Falling back to Moving Average.")
            self.fallback_average = train_df['y'].mean()
            self.using_fallback = True
            return

        # Fit the model
        self.model.fit(train_df)
    
    def _predict_model(self, future_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions using the fitted Prophet model.
        
        Args:
            future_df: Future DataFrame with 'ds' and regressor columns
            
        Returns:
            DataFrame with full Prophet output including components
        """
        # Ensure we have the required columns
        if 'ds' not in future_df.columns:
            raise ValueError("Future data must contain 'ds' column")

        #if we're using the fallback, return a constant forecast equal to the computed average
        if self.using_fallback:
            logger.debug(f"Using Moving Average fallback for {self.product_id} at {self.location_id}")
            return pd.DataFrame({'ds': future_df['ds'], 'yhat': self.fallback_average, 'yhat_lower': self.fallback_average, 'yhat_upper': self.fallback_average})
        
        # Generate predictions with full components
        predictions_df = self.model.predict(future_df)
        
        return predictions_df
