"""
Utility modules for the forecaster package.
"""

try:
    from .visualization import DemandVisualizer
except ImportError:
    # Visualization module requires matplotlib which may not be installed
    DemandVisualizer = None
from .logger import (
    ForecasterLogger, 
    get_logger, 
    setup_logging,
    debug, info, warning, error, critical
)
from .standardize import (
    DataStandardizer,
    check_data_quality,
    standardize_dataframe,
    validate_and_clean
)

__all__ = [
    'ForecasterLogger',
    'get_logger',
    'setup_logging',
    'debug',
    'info', 
    'warning',
    'error',
    'critical',
    'DataStandardizer',
    'check_data_quality',
    'standardize_dataframe',
    'validate_and_clean'
]

# Add DemandVisualizer only if available
if DemandVisualizer is not None:
    __all__.append('DemandVisualizer')
