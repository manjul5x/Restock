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
            
            # Convert date column to datetime type if it exists (needed for regressor operations)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
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
        
        # Convert date column to datetime type if it exists (needed for regressor operations)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
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
    
    def save_input_data_with_regressors(self, df: pd.DataFrame) -> None:
        """Save input data with regressors"""
        filename = self.config['paths']['output_files']['input_data_with_regressors']
        self.save_results(df, "backtesting", filename)
    
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
    
    # ============================================================================
    # CHUNKED PERSISTENCE FOR BACKTESTING
    # ============================================================================
    
    def save_results_chunk(self, 
                          df: pd.DataFrame, 
                          category: str, 
                          base_filename: str, 
                          run_id: str, 
                          chunk_idx: int,
                          file_format: str = "parquet") -> str:
        """
        Save a chunk of results to intermediate storage.
        
        Args:
            df: DataFrame to save
            category: Category directory (e.g., 'backtesting')
            base_filename: Base filename without extension
            run_id: Unique run identifier
            chunk_idx: Sequential chunk index
            file_format: File format ('parquet' or 'csv')
            
        Returns:
            Path to saved chunk file
        """
        # Create chunk directory structure
        chunk_dir = Path(self.config['paths']['output_dir']) / category / 'chunks' / base_filename / run_id
        chunk_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate chunk filename
        chunk_filename = f"chunk_{chunk_idx:06d}.{file_format}"
        chunk_path = chunk_dir / chunk_filename
        
        # Save chunk
        try:
            if file_format == "parquet":
                df.to_parquet(chunk_path, index=False)
            else:  # csv
                df.to_csv(chunk_path, index=False)
            
            logger.debug(f"Saved chunk {chunk_idx} to {chunk_path} ({len(df)} rows)")
            return str(chunk_path)
            
        except Exception as e:
            logger.error(f"Failed to save chunk {chunk_idx}: {e}")
            raise
    
    def finalize_results_from_chunks(self, 
                                   category: str, 
                                   base_filename: str, 
                                   run_id: str,
                                   file_format: str = "parquet") -> str:
        """
        Concatenate all chunks and create final consolidated output file.
        
        Args:
            category: Category directory (e.g., 'backtesting')
            base_filename: Base filename without extension
            run_id: Unique run identifier
            file_format: File format of chunks ('parquet' or 'csv')
            
        Returns:
            Path to final consolidated file
        """
        # Find chunk directory
        chunk_dir = Path(self.config['paths']['output_dir']) / category / 'chunks' / base_filename / run_id
        
        if not chunk_dir.exists():
            raise FileNotFoundError(f"Chunk directory not found: {chunk_dir}")
        
        # Find all chunk files
        chunk_pattern = f"chunk_*.{file_format}"
        chunk_files = sorted(chunk_dir.glob(chunk_pattern))
        
        if not chunk_files:
            raise FileNotFoundError(f"No chunk files found in {chunk_dir}")
        
        logger.info(f"Found {len(chunk_files)} chunk files to consolidate")
        
        # Read and concatenate chunks
        chunks = []
        total_rows = 0
        
        for chunk_file in chunk_files:
            try:
                if file_format == "parquet":
                    chunk_df = pd.read_parquet(chunk_file)
                else:  # csv
                    chunk_df = pd.read_csv(chunk_file)
                
                chunks.append(chunk_df)
                total_rows += len(chunk_df)
                logger.debug(f"Loaded chunk {chunk_file.name}: {len(chunk_df)} rows")
                
            except Exception as e:
                logger.error(f"Failed to load chunk {chunk_file.name}: {e}")
                raise
        
        # Concatenate all chunks
        if chunks:
            consolidated_df = pd.concat(chunks, ignore_index=True)
            logger.info(f"Consolidated {len(chunks)} chunks into {len(consolidated_df)} total rows")
        else:
            consolidated_df = pd.DataFrame()
            logger.warning("No chunks to consolidate")
        
        # Save final consolidated file
        final_filename = f"{base_filename}.csv"  # Always output as CSV for downstream compatibility
        final_path = Path(self.config['paths']['output_dir']) / category / final_filename
        
        try:
            consolidated_df.to_csv(final_path, index=False)
            logger.info(f"Saved consolidated results to {final_path}")
            
            # Optionally clean up chunk files
            if self.config.get('cleanup_chunks', False):
                for chunk_file in chunk_files:
                    chunk_file.unlink()
                chunk_dir.rmdir()  # Remove empty chunk directory
                logger.info("Cleaned up chunk files")
            
            return str(final_path)
            
        except Exception as e:
            logger.error(f"Failed to save consolidated results: {e}")
            raise
    
    def list_available_runs(self, category: str, base_filename: str) -> List[str]:
        """
        List available run IDs for a given category and base filename.
        
        Args:
            category: Category directory (e.g., 'backtesting')
            base_filename: Base filename without extension
            
        Returns:
            List of available run IDs
        """
        chunk_base_dir = Path(self.config['paths']['output_dir']) / category / 'chunks' / base_filename
        
        if not chunk_base_dir.exists():
            return []
        
        run_ids = [d.name for d in chunk_base_dir.iterdir() if d.is_dir()]
        return sorted(run_ids)
    
    def get_chunk_info(self, category: str, base_filename: str, run_id: str) -> Dict[str, Any]:
        """
        Get information about chunks for a specific run.
        
        Args:
            category: Category directory (e.g., 'backtesting')
            base_filename: Base filename without extension
            run_id: Unique run identifier
            
        Returns:
            Dictionary with chunk information
        """
        chunk_dir = Path(self.config['paths']['output_dir']) / category / 'chunks' / base_filename / run_id
        
        if not chunk_dir.exists():
            return {'run_id': run_id, 'chunks': [], 'total_rows': 0, 'status': 'not_found'}
        
        chunk_files = sorted(chunk_dir.glob("chunk_*.parquet"))
        chunk_info = []
        total_rows = 0
        
        for chunk_file in chunk_files:
            try:
                chunk_df = pd.read_parquet(chunk_file)
                chunk_info.append({
                    'filename': chunk_file.name,
                    'rows': len(chunk_df),
                    'size_mb': chunk_file.stat().st_size / (1024 * 1024)
                })
                total_rows += len(chunk_df)
            except Exception as e:
                logger.warning(f"Could not read chunk {chunk_file.name}: {e}")
        
        return {
            'run_id': run_id,
            'chunks': chunk_info,
            'total_rows': total_rows,
            'status': 'available' if chunk_info else 'empty'
        }
    
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
    
    def load_regressor_config(self) -> Dict[str, Any]:
        """
        Load regressor configuration from YAML file.
        
        Returns:
            Dictionary with regressor configuration
        """
        try:
            regressor_config_path = Path(self.config['paths']['input_files']['regressor_config'])
            
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
    
    def load_input_data_with_regressors(self) -> pd.DataFrame:
        """
        Load input data with regressors from CSV file.
        
        Returns:
            DataFrame with input data including regressors
        """
        try:
            # Get path from config - this should point to the input_data_with_regressors.csv
            input_data_path = Path(self.config['paths']['input_files']['input_data_with_regressors'])
            
            if not input_data_path.exists():
                raise FileNotFoundError(f"Input data file not found: {input_data_path}")
            
            # Load data
            input_data = pd.read_csv(input_data_path)
            
            # Convert date column to datetime
            if 'date' in input_data.columns:
                input_data['date'] = pd.to_datetime(input_data['date'])
            
            logger.info(f"Loaded input data with regressors from {input_data_path}")
            return input_data
            
        except Exception as e:
            logger.error(f"Failed to load input data with regressors: {e}")
            raise

    def load_simulation_detailed_data(self, filters: Dict[str, List[str]] = None) -> pd.DataFrame:
        """
        Load detailed simulation data from detailed_results directory
        
        Args:
            filters: Optional dictionary of filters with keys:
                    - products: List of product IDs to include
                    - locations: List of location IDs to include
                    - forecast_methods: List of forecast methods to include
                    
        Returns:
            DataFrame containing filtered simulation data
            
        Raises:
            DataAccessError: If no simulation data is available
        """
        detailed_dir = self.get_output_path("simulation", "detailed_results")
        detailed_data = []
        
        if not detailed_dir.exists():
            raise DataAccessError("No detailed simulation data available")
        
        for file_path in detailed_dir.glob("*_simulation.csv"):
            # Parse filename to extract product_id, location_id, forecast_method
            filename = file_path.stem
            parts = filename.split("_")
            
            if len(parts) >= 4:
                product_id = parts[0]
                location_id = parts[1]
                
                # Extract forecast method
                simulation_index = -1
                for i, part in enumerate(parts):
                    if part == "simulation":
                        simulation_index = i
                        break
                
                if simulation_index > 2:
                    forecast_method = "_".join(parts[2:simulation_index])
                else:
                    forecast_method = parts[2] if len(parts) > 2 else "unknown"
                
                # Apply filters if provided
                if filters:
                    if 'products' in filters and product_id not in filters['products']:
                        continue
                    if 'locations' in filters and location_id not in filters['locations']:
                        continue
                    if 'forecast_methods' in filters and forecast_method not in filters['forecast_methods']:
                        continue
                
                try:
                    data = pd.read_csv(file_path)
                    data["date"] = pd.to_datetime(data["date"])
                    detailed_data.append(data)
                except Exception as e:
                    logger.warning(f"Error loading {file_path}: {e}")
                    continue
        
        if not detailed_data:
            raise DataAccessError("No detailed simulation data available")
        
        return pd.concat(detailed_data, ignore_index=True)
        
    def load_holidays(self, location: Optional[str] = None) -> pd.DataFrame:
        """
        Load holiday data from the configured holiday CSV file.
        
        Args:
            location: Optional location filter (e.g., 'all', 'WB', etc.)
                    If None, returns all holidays
            
        Returns:
            DataFrame with holiday information including:
            - holiday: Holiday name
            - ds: Date (datetime)
            - lower_window: Lower window for holiday effect
            - upper_window: Upper window for holiday effect
            - location: Location identifier
            - source: Source of holiday data
            
        Raises:
            FileNotFoundError: If holiday file doesn't exist
            Exception: If file cannot be read or parsed
        """
        try:
            # Get holiday file path from config
            holiday_file = Path(self.config['paths']['input_files']['india_holidays'])
            
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
    
