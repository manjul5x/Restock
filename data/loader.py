"""
Unified data loader with caching and parallel processing support.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, ClassVar, Dict, List, Any, Union
import yaml
import logging
import psutil
import time
import threading
import multiprocessing
try:
    from .exceptions import DataAccessError, CacheError, StaleDataError, CacheMemoryError
    from .access.base import DataAccessor
    from .access.access_and_storage import get_accessor
except ImportError:
    # Handle case when running as script
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from exceptions import DataAccessError, CacheError, StaleDataError, CacheMemoryError
    from access.base import DataAccessor
    from access.access_and_storage import get_accessor

logger = logging.getLogger(__name__)

class DataLoader:
    """
    Unified data loader with caching and parallel processing support.
    
    Features:
    - Automatic cache disabling in worker processes
    - Preloaded data support for parallel processing
    - Memory-safe caching with TTL and size limits
    - Thread-safe operations
    """
    
    # Class-level caches and locks (per process)
    _instance: ClassVar[Optional['DataLoader']] = None
    _config_cache: ClassVar[Optional[Dict]] = None
    _accessor_cache: ClassVar[Dict[str, DataAccessor]] = {}
    _data_cache: ClassVar[Dict[str, pd.DataFrame]] = {}
    _cache_timestamps: ClassVar[Dict[str, float]] = {}
    _cache_lock: ClassVar[threading.Lock] = threading.Lock()
    
    def __new__(cls, config_path: str = "data/config/data_config.yaml", 
                enable_cache: Optional[bool] = None):
        """Create new instance (no singleton in worker processes)"""
        # Use singleton only in main process to avoid sharing across workers
        if cls._is_main_process():
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance
        else:
            # Create new instance for each worker process
            return super().__new__(cls)
    
    def __init__(self, config_path: str = "data/config/data_config.yaml", 
                 enable_cache: Optional[bool] = None):
        """Initialize the data loader"""
        # Allow re-initialization in worker processes
        if hasattr(self, '_initialized') and self._is_main_process():
            return
            
        # Load config
        if DataLoader._config_cache is None:
            with open(config_path, 'r') as f:
                DataLoader._config_cache = yaml.safe_load(f)
        self.config = DataLoader._config_cache
        
        # Get storage type
        self.storage_type = self.config['storage']['type']
        
        # Initialize accessor
        if self.storage_type not in DataLoader._accessor_cache:
            DataLoader._accessor_cache[self.storage_type] = self._get_accessor()
        self.accessor = DataLoader._accessor_cache[self.storage_type]
        
        # Cache settings - auto-disable in worker processes
        if enable_cache is None:
            enable_cache = self._is_main_process()
        
        self.cache_enabled = enable_cache and self.config['cache']['enabled']
        self.cache_ttl = self.config['cache']['ttl_seconds']
        self.max_cache_size = self.config['cache']['max_size_mb'] * 1024 * 1024  # Convert to bytes
        
        # Preloaded data storage (for worker processes)
        self._preloaded_data: Optional[Dict[str, pd.DataFrame]] = None
        
        process_type = "main" if self._is_main_process() else "worker"
        cache_status = "enabled" if self.cache_enabled else "disabled"
        logger.info(f"DataLoader initialized in {process_type} process with cache {cache_status}")
        
        self._initialized = True
    
    @staticmethod
    def _is_main_process() -> bool:
        """Check if running in the main process"""
        return multiprocessing.current_process().name == 'MainProcess'
    
    def _get_accessor(self) -> DataAccessor:
        """Get appropriate accessor based on storage type"""
        return get_accessor(self.storage_type, self.config)
    
    def _get_cache_key(self, data_type: str, **kwargs) -> str:
        """Generate cache key"""
        key_parts = [data_type]
        for k, v in sorted(kwargs.items()):
            if isinstance(v, pd.DataFrame):
                # Use shape and memory usage for DataFrame keys
                key_parts.append(f"{k}_{v.shape}_{v.memory_usage().sum()}")
            else:
                key_parts.append(f"{k}_{v}")
        return "_".join(key_parts)
    
    def _check_cache_memory(self):
        """Check cache memory usage"""
        cache_size = sum(df.memory_usage().sum() for df in self._data_cache.values())
        if cache_size > self.max_cache_size:
            raise CacheMemoryError(f"Cache size ({cache_size} bytes) exceeds limit ({self.max_cache_size} bytes)")
    
    def _is_cache_valid(self, cache_key: str, path: Union[str, Path]) -> bool:
        """Check if cached data is still valid"""
        if not self.cache_enabled:
            return False
            
        if cache_key not in self._cache_timestamps:
            return False
            
        # Check TTL
        if time.time() - self._cache_timestamps[cache_key] > self.cache_ttl:
            return False
            
        # Check file changes
        if self.config['cache']['check_file_changes']:
            if self.accessor.has_file_changed(path):
                return False
        
        return True
    
    def _cache_data(self, cache_key: str, df: pd.DataFrame):
        """Cache data with thread safety"""
        with self._cache_lock:
            self._check_cache_memory()
            self._data_cache[cache_key] = df
            self._cache_timestamps[cache_key] = time.time()
    
    def load_product_master(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Load product master data"""
        # Check for preloaded data first (worker process optimization)
        if self._preloaded_data and 'product_master' in self._preloaded_data:
            logger.debug("Using preloaded product master data")
            df = self._preloaded_data['product_master'].copy()
            
            # Apply column selection if needed
            if columns:
                df = df[columns]
            
            return df
        
        # Normal loading path
        path = Path(self.config['paths']['base_dir']) / self.config['paths']['product_master']
        cache_key = self._get_cache_key('product_master', columns=columns)
        
        # Check cache
        if self._is_cache_valid(cache_key, path):
            logger.debug(f"Cache hit for product_master")
            return self._data_cache[cache_key].copy()
        
        # Load data
        logger.info(f"Loading product master from {path}")
        df = self.accessor.read_data(path, columns=columns)
        
        # Cache if enabled
        if self.cache_enabled:
            self._cache_data(cache_key, df)
        
        return df.copy()
    
    def load_outflow(self, 
                    product_master: Optional[pd.DataFrame] = None,
                    columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Load outflow data with optional filtering"""
        # Check for preloaded data first (worker process optimization)
        if self._preloaded_data and 'outflow' in self._preloaded_data:
            logger.debug("Using preloaded outflow data")
            df = self._preloaded_data['outflow'].copy()
            
            # Apply column selection if needed
            if columns:
                df = df[columns]
            
            # Apply filtering if needed
            if product_master is not None:
                logger.debug("Filtering preloaded outflow by product master")
                product_locations = set(
                    product_master[["product_id", "location_id"]].apply(tuple, axis=1)
                )
                df = df[
                    df[["product_id", "location_id"]].apply(tuple, axis=1).isin(product_locations)
                ]
                logger.info(f"Filtered to {len(df)} records for {len(product_locations)} product-location combinations")
            
            return df
        
        # Normal loading path
        path = Path(self.config['paths']['base_dir']) / self.config['paths']['outflow']
        cache_key = self._get_cache_key('outflow', 
                                       product_master=product_master, 
                                       columns=columns)
        
        # Check cache
        if self._is_cache_valid(cache_key, path):
            logger.debug(f"Cache hit for outflow")
            return self._data_cache[cache_key].copy()
        
        # Load data
        logger.info(f"Loading outflow data from {path}")
        df = self.accessor.read_data(path, columns=columns)
        
        # Filter by product master
        if product_master is not None:
            logger.debug("Filtering outflow by product master")
            product_locations = set(
                product_master[["product_id", "location_id"]].apply(tuple, axis=1)
            )
            df = df[
                df[["product_id", "location_id"]].apply(tuple, axis=1).isin(product_locations)
            ]
            logger.info(f"Filtered to {len(df)} records for {len(product_locations)} product-location combinations")
        
        # Cache if enabled
        if self.cache_enabled:
            self._cache_data(cache_key, df)
        
        return df.copy()
    
    def get_output_path(self, category: str, filename: str, date: Optional[str] = None) -> Path:
        """Get path for output files"""
        base_path = Path(self.config['paths']['output_dir']) / category
        if date:
            return base_path / f"{date}_{filename}"
        return base_path / filename
    
    def save_results(self, 
                    df: pd.DataFrame, 
                    category: str, 
                    filename: str,
                    date: Optional[str] = None) -> None:
        """Save results with safety checks"""
        path = self.get_output_path(category, filename, date)
        logger.info(f"Saving results to {path}")
        self.accessor.write_data(df, path)
    
    def save_safety_stocks(self, df: pd.DataFrame) -> None:
        """Save safety stock results"""
        filename = self.config['paths']['output_files']['safety_stocks']
        self.save_results(df, "safety_stocks", filename)
    
    def save_forecast_comparison(self, df: pd.DataFrame) -> None:
        """Save forecast comparison results"""
        filename = self.config['paths']['output_files']['forecast_comparison']
        self.save_results(df, "backtesting", filename)
    
    def save_simulation_results(self, df: pd.DataFrame) -> None:
        """Save simulation results"""
        filename = self.config['paths']['output_files']['simulation_results']
        self.save_results(df, "simulation", filename)
    
    def clear_cache(self):
        """Clear all cached data"""
        with self._cache_lock:
            self._data_cache.clear()
            self._cache_timestamps.clear()
        logger.info("Cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        cache_size = sum(df.memory_usage().sum() for df in self._data_cache.values())
        return {
            'cached_datasets': len(self._data_cache),
            'cache_keys': list(self._data_cache.keys()),
            'total_size_mb': cache_size / (1024 * 1024),
            'max_size_mb': self.max_cache_size / (1024 * 1024),
            'usage_percent': (cache_size / self.max_cache_size) * 100,
            'oldest_cache_age': min(
                [time.time() - ts for ts in self._cache_timestamps.values()], 
                default=0
            ) if self._cache_timestamps else 0,
            'process_type': 'main' if self._is_main_process() else 'worker',
            'cache_enabled': self.cache_enabled,
            'has_preloaded_data': self._preloaded_data is not None
        }
    
    # ============================================================================
    # PARALLEL PROCESSING SUPPORT
    # ============================================================================
    
    @classmethod
    def preload_for_parallel_processing(cls, 
                                       config_path: str = "data/config/data_config.yaml",
                                       columns: Optional[Dict[str, List[str]]] = None) -> Dict[str, pd.DataFrame]:
        """
        Preload all data in the main process for sharing with workers.
        
        Args:
            config_path: Path to configuration file
            columns: Optional column selection for each dataset
                    e.g., {'product_master': ['product_id', 'location_id'], 'outflow': ['product_id', 'demand']}
        
        Returns:
            Dictionary containing all preloaded data
        """
        logger.info("Preloading data for parallel processing...")
        
        # Create a loader instance with caching enabled
        loader = cls(config_path, enable_cache=True)
        
        # Load product master
        pm_columns = columns.get('product_master') if columns else None
        product_master = loader.load_product_master(columns=pm_columns)
        logger.info(f"Preloaded product master: {len(product_master)} records")
        
        # Load outflow data (filtered by product master for efficiency)
        outflow_columns = columns.get('outflow') if columns else None
        outflow = loader.load_outflow(product_master=product_master, columns=outflow_columns)
        logger.info(f"Preloaded outflow data: {len(outflow)} records")
        
        preloaded_data = {
            'product_master': product_master,
            'outflow': outflow
        }
        
        # Calculate total memory usage
        total_memory = sum(df.memory_usage(deep=True).sum() for df in preloaded_data.values())
        logger.info(f"Total preloaded data size: {total_memory / (1024*1024):.2f} MB")
        
        return preloaded_data
    
    @classmethod
    def create_for_worker(cls, 
                         preloaded_data: Dict[str, pd.DataFrame],
                         config_path: str = "data/config/data_config.yaml") -> 'DataLoader':
        """
        Create a DataLoader instance for worker processes with preloaded data.
        
        Args:
            preloaded_data: Data preloaded by preload_for_parallel_processing()
            config_path: Path to configuration file
            
        Returns:
            DataLoader instance optimized for worker processes
        """
        # Create loader with cache disabled (worker process)
        loader = cls(config_path, enable_cache=False)
        
        # Set preloaded data
        loader._preloaded_data = preloaded_data
        
        logger.info(f"Created worker DataLoader with {len(preloaded_data)} preloaded datasets")
        return loader
    
    def has_preloaded_data(self) -> bool:
        """Check if this loader has preloaded data"""
        return self._preloaded_data is not None
    
    def get_preloaded_datasets(self) -> List[str]:
        """Get list of preloaded dataset names"""
        return list(self._preloaded_data.keys()) if self._preloaded_data else []
    
