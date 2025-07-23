"""
Configuration settings for the forecasting pipeline runner.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path
import os
from datetime import datetime, date


@dataclass
class RunnerConfig:
    """Configuration for the forecasting pipeline runner."""
    
    # Data paths
    data_dir: Path = field(default_factory=lambda: Path("forecaster/data/dummy"))
    demand_file: str = "sku_demand_daily.csv"
    product_master_daily_file: str = "product_master_daily.csv"
    product_master_weekly_file: str = "product_master_weekly.csv"
    
    # Output paths
    output_dir: Path = field(default_factory=lambda: Path("output"))
    outlier_output_file: str = "demand_outliers_removed.csv"
    aggregated_output_file: str = "demand_aggregated.csv"
    forecast_output_file: str = "forecasts.csv"
    
    # Pipeline settings
    run_date: Optional[date] = None  # Will be set to latest date in data if None
    demand_frequency: str = "d"  # 'd', 'w', 'm'
    
    # Batching settings
    batch_size: int = 10  # Number of product-location combinations per batch
    max_workers: int = 4  # Maximum number of parallel workers
    
    # Validation settings
    validate_data: bool = True
    validate_coverage: bool = True
    validate_completeness: bool = True
    
    # Outlier settings
    outlier_enabled: bool = True
    outlier_output_insights: bool = True
    
    # Aggregation settings
    aggregation_enabled: bool = True
    
    # Forecasting settings
    forecasting_enabled: bool = True
    forecast_model: str = "moving_average"  # 'moving_average', 'arima', 'prophet'
    
    # Logging settings
    log_level: str = "INFO"
    log_file: Optional[Path] = None
    
    # Performance settings
    chunk_size: int = 1000  # For processing large datasets
    memory_limit_gb: float = 4.0  # Memory limit for processing
    
    def __post_init__(self):
        """Post-initialization setup."""
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set log file if not specified
        if self.log_file is None:
            self.log_file = self.output_dir / f"forecast_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    def get_demand_file_path(self) -> Path:
        """Get the full path to the demand file."""
        return self.data_dir / self.demand_file
    
    def get_product_master_file_path(self) -> Path:
        """Get the full path to the appropriate product master file."""
        if self.demand_frequency == 'd':
            return self.data_dir / self.product_master_daily_file
        else:
            return self.data_dir / self.product_master_weekly_file
    
    def get_outlier_output_path(self) -> Path:
        """Get the full path for outlier output."""
        return self.output_dir / self.outlier_output_file
    
    def get_aggregated_output_path(self) -> Path:
        """Get the full path for aggregated output."""
        return self.output_dir / self.aggregated_output_file
    
    def get_forecast_output_path(self) -> Path:
        """Get the full path for forecast output."""
        return self.output_dir / self.forecast_output_file
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for logging."""
        return {
            'data_dir': str(self.data_dir),
            'demand_file': self.demand_file,
            'product_master_file': self.get_product_master_file_path().name,
            'output_dir': str(self.output_dir),
            'run_date': self.run_date.isoformat() if self.run_date else None,
            'demand_frequency': self.demand_frequency,
            'batch_size': self.batch_size,
            'max_workers': self.max_workers,
            'validate_data': self.validate_data,
            'outlier_enabled': self.outlier_enabled,
            'aggregation_enabled': self.aggregation_enabled,
            'forecasting_enabled': self.forecasting_enabled,
            'forecast_model': self.forecast_model,
            'log_level': self.log_level
        }


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    
    batch_id: int
    product_locations: List[tuple]  # List of (product_id, location_id) tuples
    config: RunnerConfig
    
    def __post_init__(self):
        """Validate batch configuration."""
        if not self.product_locations:
            raise ValueError("Batch must contain at least one product-location combination")
        
        if len(self.product_locations) > self.config.batch_size:
            raise ValueError(f"Batch size {len(self.product_locations)} exceeds maximum {self.config.batch_size}")


def create_default_config() -> RunnerConfig:
    """Create a default configuration."""
    return RunnerConfig()


def create_config_from_env() -> RunnerConfig:
    """Create configuration from environment variables."""
    config = RunnerConfig()
    
    # Override with environment variables if present
    if os.getenv('FORECAST_DATA_DIR'):
        config.data_dir = Path(os.getenv('FORECAST_DATA_DIR'))
    
    if os.getenv('FORECAST_OUTPUT_DIR'):
        config.output_dir = Path(os.getenv('FORECAST_OUTPUT_DIR'))
    
    if os.getenv('FORECAST_BATCH_SIZE'):
        config.batch_size = int(os.getenv('FORECAST_BATCH_SIZE'))
    
    if os.getenv('FORECAST_MAX_WORKERS'):
        config.max_workers = int(os.getenv('FORECAST_MAX_WORKERS'))
    
    if os.getenv('FORECAST_DEMAND_FREQUENCY'):
        config.demand_frequency = os.getenv('FORECAST_DEMAND_FREQUENCY')
    
    if os.getenv('FORECAST_MODEL'):
        config.forecast_model = os.getenv('FORECAST_MODEL')
    
    if os.getenv('FORECAST_LOG_LEVEL'):
        config.log_level = os.getenv('FORECAST_LOG_LEVEL')
    
    return config
