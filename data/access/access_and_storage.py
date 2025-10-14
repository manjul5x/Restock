"""
Factory function for creating data accessors.
"""

from typing import Dict, Any
from .base import DataAccessor
from .csv_accessor import CSVAccessor
from .snowflake_accessor import SnowflakeAccessor
from ..exceptions import DataAccessError


def get_accessor(storage_type: str, config: Dict[str, Any]) -> DataAccessor:
    """
    Factory function to get the appropriate data accessor.
    
    Args:
        storage_type: Type of storage ('csv' or 'snowflake')
        config: Configuration dictionary for the accessor
        
    Returns:
        DataAccessor instance
        
    Raises:
        DataAccessError: If storage type is not supported
    """
    if storage_type == 'csv':
        return CSVAccessor()
    elif storage_type == 'snowflake':
        return SnowflakeAccessor(config.get('snowflake', {}))
    else:
        raise DataAccessError(f"Unsupported storage type: {storage_type}") 