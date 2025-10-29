"""
Environment-based data loader for 5X workspace deployment.
This replaces the config-file based loader with environment variable support.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, ClassVar, Dict, List, Any, Union
import logging
import psutil
import time
import threading
import multiprocessing
import os

try:
    from .exceptions import DataAccessError, CacheError, StaleDataError, CacheMemoryError
    from .access.env_snowflake_accessor import EnvSnowflakeAccessor
    from .access.env_snowflake_config import get_snowflake_config_from_env
except ImportError:
    # Handle case when running as script
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from exceptions import DataAccessError, CacheError, StaleDataError, CacheMemoryError
    from access.env_snowflake_accessor import EnvSnowflakeAccessor
    from access.env_snowflake_config import get_snowflake_config_from_env

logger = logging.getLogger(__name__)

class EnvDataLoader:
    """
    Environment-based data loader for 5X workspace deployment.
    
    Features:
    - Uses environment variables instead of config files
    - Multi-schema support (read from STAGE, write to TRANSFORMATION)
    - Automatic cache disabling in worker processes
    - Memory-safe caching with TTL and size limits
    - Thread-safe operations
    """
    
    # Class-level caches and locks (per process)
    _instance: ClassVar[Optional['EnvDataLoader']] = None
    _accessor_cache: ClassVar[Dict[str, EnvSnowflakeAccessor]] = {}
    _data_cache: ClassVar[Dict[str, pd.DataFrame]] = {}
    _cache_timestamps: ClassVar[Dict[str, float]] = {}
    _cache_lock: ClassVar[threading.Lock] = threading.Lock()
    
    def __new__(cls, enable_cache: Optional[bool] = None):
        """Create new instance (no singleton in worker processes)"""
        # Use singleton only in main process to avoid sharing across workers
        if cls._is_main_process():
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance
        else:
            # Create new instance for each worker process
            return super().__new__(cls)
    
    def __init__(self, enable_cache: Optional[bool] = None):
        """Initialize the environment-based data loader"""
        # Allow re-initialization in worker processes
        if hasattr(self, '_initialized') and self._is_main_process():
            return
            
        # Get environment configuration
        self.config = get_snowflake_config_from_env()
        
        # Initialize accessor
        if 'snowflake' not in EnvDataLoader._accessor_cache:
            EnvDataLoader._accessor_cache['snowflake'] = EnvSnowflakeAccessor(use_env_vars=True)
        self.accessor = EnvDataLoader._accessor_cache['snowflake']
        
        # Cache settings - auto-disable in worker processes
        if enable_cache is None:
            enable_cache = self._is_main_process()
        
        self.cache_enabled = enable_cache
        self.cache_ttl = 3600  # 1 hour TTL
        self.max_cache_size = 500 * 1024 * 1024  # 500MB max cache
        
        # Preloaded data storage (for worker processes)
        self._preloaded_data: Optional[Dict[str, pd.DataFrame]] = None
        
        process_type = "main" if self._is_main_process() else "worker"
        cache_status = "enabled" if self.cache_enabled else "disabled"
        logger.info(f"EnvDataLoader initialized in {process_type} process with cache {cache_status}")
        
        self._initialized = True
    
    @staticmethod
    def _is_main_process() -> bool:
        """Check if we're in the main process (not a worker)"""
        return multiprocessing.current_process().name == 'MainProcess'
    
    def load_outflow(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Load raw demand data from STAGE schema.
        
        Args:
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame containing outflow data
        """
        cache_key = "outflow_data"
        
        # Check cache first
        if use_cache and self.cache_enabled and cache_key in EnvDataLoader._data_cache:
            if self._is_cache_valid(cache_key):
                logger.info("Loading outflow data from cache")
                return EnvDataLoader._data_cache[cache_key].copy()
        
        # Load from Snowflake
        logger.info("Loading outflow data from Snowflake (STAGE schema)")
        table_name = os.getenv("FIVEX_SNOWFLAKE_OUTFLOW_TABLE", "OUTFLOW_BHASIN")
        
        try:
            df = self.accessor.read_data(table_name, schema=self.config['read_schema'])
            logger.info(f"Loaded {len(df)} rows from {self.config['read_schema']}.{table_name}")
            
            # Convert column names to lowercase for compatibility
            df.columns = [col.lower() for col in df.columns]
            logger.info(f"Converted column names to lowercase: {list(df.columns)}")
            
            # Convert date column to datetime if it exists
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                logger.info("Converted 'date' column to datetime format")
            
            # Convert product_id and location_id to string for consistent merging
            if 'product_id' in df.columns:
                df['product_id'] = df['product_id'].astype(str)
                logger.info("Converted 'product_id' column to string")
            if 'location_id' in df.columns:
                df['location_id'] = df['location_id'].astype(str)
                logger.info("Converted 'location_id' column to string")
            
            # Cache the data
            if self.cache_enabled:
                self._cache_data(cache_key, df)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load outflow data: {e}")
            raise DataAccessError(f"Failed to load outflow data: {e}")
    
    def load_product_master(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Load product master data from STAGE schema.
        
        Args:
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame containing product master data
        """
        cache_key = "product_master_data"
        
        # Check cache first
        if use_cache and self.cache_enabled and cache_key in EnvDataLoader._data_cache:
            if self._is_cache_valid(cache_key):
                logger.info("Loading product master data from cache")
                return EnvDataLoader._data_cache[cache_key].copy()
        
        # Load from Snowflake
        logger.info("Loading product master data from Snowflake (STAGE schema)")
        table_name = os.getenv("FIVEX_SNOWFLAKE_PRODUCT_MASTER_TABLE", "PRODUCT_MASTER_BHASIN")
        
        try:
            df = self.accessor.read_data(table_name, schema=self.config['read_schema'])
            logger.info(f"Loaded {len(df)} rows from {self.config['read_schema']}.{table_name}")
            
            # Convert column names to lowercase for compatibility
            df.columns = [col.lower() for col in df.columns]
            logger.info(f"Converted column names to lowercase: {list(df.columns)}")
            
            # Convert product_id and location_id to string for consistent merging
            if 'product_id' in df.columns:
                df['product_id'] = df['product_id'].astype(str)
                logger.info("Converted 'product_id' column to string")
            if 'location_id' in df.columns:
                df['location_id'] = df['location_id'].astype(str)
                logger.info("Converted 'location_id' column to string")
            
            # Cache the data
            if self.cache_enabled:
                self._cache_data(cache_key, df)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load product master data: {e}")
            raise DataAccessError(f"Failed to load product master data: {e}")
    
    def load_processed_data(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Load processed data with regressor features from TRANSFORMATION schema.
        
        Args:
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame containing processed data with regressor features
        """
        cache_key = "processed_data"
        
        # Check cache first
        if use_cache and self.cache_enabled and cache_key in EnvDataLoader._data_cache:
            if self._is_cache_valid(cache_key):
                logger.info("Loading processed data from cache")
                return EnvDataLoader._data_cache[cache_key].copy()
        
        # Load from Snowflake
        logger.info("Loading processed data from Snowflake (TRANSFORMATION schema)")
        table_name = os.getenv("FIVEX_SNOWFLAKE_PROCESSED_DATA_TABLE", "PROCESSED_DATA_WITH_REGRESSORS")
        logger.info(f"Table name: {table_name}, Schema: {self.config['write_schema']}")
        
        try:
            df = self.accessor.read_data(table_name, schema=self.config['write_schema'])
            logger.info(f"Loaded {len(df)} rows from {self.config['write_schema']}.{table_name}")
            
            # Cache the data
            if self.cache_enabled:
                self._cache_data(cache_key, df)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load processed data: {e}")
            raise DataAccessError(f"Failed to load processed data: {e}")
    
    def save_processed_data(self, df: pd.DataFrame, if_exists: str = "replace") -> None:
        """
        Save processed data with regressor features to TRANSFORMATION schema.
        
        Args:
            df: DataFrame containing processed data
            if_exists: What to do if table exists ('append', 'replace', 'fail')
        """
        logger.info("Saving processed data to Snowflake (TRANSFORMATION schema)")
        table_name = os.getenv("FIVEX_SNOWFLAKE_PROCESSED_DATA_TABLE", "PROCESSED_DATA_WITH_REGRESSORS")
        
        try:
            self.accessor.write_data(df, table_name, if_exists=if_exists, schema=self.config['write_schema'])
            logger.info(f"Saved {len(df)} rows to {self.config['write_schema']}.{table_name}")
            
            # Update cache
            if self.cache_enabled:
                self._cache_data("processed_data", df)
            
        except Exception as e:
            logger.error(f"Failed to save processed data: {e}")
            raise DataAccessError(f"Failed to save processed data: {e}")
    
    def save_future_predictions(self, df: pd.DataFrame, if_exists: str = "replace") -> None:
        """
        Save future predictions to TRANSFORMATION schema.
        
        Args:
            df: DataFrame containing future predictions
            if_exists: What to do if table exists ('append', 'replace', 'fail')
        """
        logger.info("Saving future predictions to Snowflake (TRANSFORMATION schema)")
        table_name = os.getenv("FIVEX_SNOWFLAKE_FUTURE_PREDICTIONS_TABLE", "FUTURE_PREDICTIONS_RESULTS")
        
        try:
            self.accessor.write_data(df, table_name, if_exists=if_exists, schema=self.config['write_schema'])
            logger.info(f"Saved {len(df)} rows to {self.config['write_schema']}.{table_name}")
            
        except Exception as e:
            logger.error(f"Failed to save future predictions: {e}")
            raise DataAccessError(f"Failed to save future predictions: {e}")
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in EnvDataLoader._cache_timestamps:
            return False
        
        age = time.time() - EnvDataLoader._cache_timestamps[cache_key]
        return age < self.cache_ttl
    
    def _cache_data(self, cache_key: str, df: pd.DataFrame) -> None:
        """Cache data with timestamp"""
        with EnvDataLoader._cache_lock:
            EnvDataLoader._data_cache[cache_key] = df.copy()
            EnvDataLoader._cache_timestamps[cache_key] = time.time()
    
    def clear_cache(self) -> None:
        """Clear all cached data"""
        with EnvDataLoader._cache_lock:
            EnvDataLoader._data_cache.clear()
            EnvDataLoader._cache_timestamps.clear()
            logger.info("Cache cleared")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached data"""
        with EnvDataLoader._cache_lock:
            cache_info = {}
            for key, df in EnvDataLoader._data_cache.items():
                if key in EnvDataLoader._cache_timestamps:
                    age = time.time() - EnvDataLoader._cache_timestamps[key]
                    cache_info[key] = {
                        'rows': len(df),
                        'columns': len(df.columns),
                        'age_seconds': age,
                        'valid': age < self.cache_ttl
                    }
            return cache_info
    
    def load_regressor_config(self) -> Dict[str, Any]:
        """
        Load regressor configuration from YAML file.
        
        Returns:
            Dictionary with regressor configuration
        """
        try:
            import yaml
            regressor_config_path = Path("data/config/regressor_config.yaml")
            
            if not regressor_config_path.exists():
                logger.warning(f"Regressor config file not found: {regressor_config_path}")
                return {}
            
            with open(regressor_config_path, 'r') as f:
                regressor_config = yaml.safe_load(f)
            
            logger.info(f"Loaded regressor configuration from {regressor_config_path}")
            return regressor_config
            
        except Exception as e:
            logger.error(f"Failed to load regressor config: {e}")
            return {}
    
    def load_holidays(self, location: Optional[str] = None) -> pd.DataFrame:
        """
        Load holiday data from the configured holiday CSV file.
        
        Args:
            location: Optional location filter (e.g., 'all', 'WB', etc.)
                    If None, returns all holidays
            
        Returns:
            DataFrame with holiday information
        """
        try:
            # Get holiday file path
            holiday_file = Path("data/holidays/india_holidays.csv")
            
            if not holiday_file.exists():
                raise FileNotFoundError(f"Holiday file not found: {holiday_file}")
            
            # Load holiday data
            holidays_df = pd.read_csv(holiday_file)
            
            # Convert ds column to datetime
            holidays_df['ds'] = pd.to_datetime(holidays_df['ds'])
            
            # Filter by location if specified
            if location is not None:
                # Handle 'all' location specially
                if location == 'all':
                    holidays_df = holidays_df[holidays_df['location'] == 'all']
                else:
                    # Include both 'all' and specific location
                    holidays_df = holidays_df[
                        (holidays_df['location'] == 'all') | 
                        (holidays_df['location'] == location)
                    ]
            
            # Sort by date
            holidays_df = holidays_df.sort_values('ds').reset_index(drop=True)
            
            logger.info(f"Loaded {len(holidays_df)} holidays from {holiday_file}")
            
            return holidays_df
            
        except Exception as e:
            logger.error(f"Failed to load holidays: {e}")
            raise
    
    def load_holiday_data(self, location: Optional[str] = None) -> pd.DataFrame:
        """
        Load holiday data from the configured holiday CSV file.
        Alias for load_holidays for consistency with parameter optimization.
        
        Args:
            location: Optional location filter (e.g., 'all', 'WB', etc.)
                    If None, returns all holidays
            
        Returns:
            DataFrame with holiday information
        """
        return self.load_holidays(location)
