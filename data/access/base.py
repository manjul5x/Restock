"""
Base classes for data access implementations.
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional, Union, List, Dict, Any
from pathlib import Path

class DataAccessor(ABC):
    """Abstract base class for data access with safety checks"""
    
    @abstractmethod
    def read_data(self, 
                  path: Union[str, Path], 
                  columns: Optional[List[str]] = None,
                  **kwargs) -> pd.DataFrame:
        """
        Read data with safety checks
        
        Args:
            path: Path to data source
            columns: Optional list of columns to read
            **kwargs: Additional arguments for specific implementations
            
        Returns:
            DataFrame containing the requested data
        """
        pass
    
    @abstractmethod
    def write_data(self, 
                   df: pd.DataFrame, 
                   path: Union[str, Path],
                   **kwargs) -> None:
        """
        Write data with safety checks
        
        Args:
            df: DataFrame to write
            path: Path to write to
            **kwargs: Additional arguments for specific implementations
        """
        pass
    
    @abstractmethod
    def validate_connection(self) -> bool:
        """
        Validate connection/access is working
        
        Returns:
            True if connection is valid, False otherwise
        """
        pass
    
    @abstractmethod
    def get_data_info(self, path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get metadata about the data source
        
        Args:
            path: Path to data source
            
        Returns:
            Dictionary containing metadata (size, modification time, etc.)
        """
        pass
    
    @abstractmethod
    def has_file_changed(self, path: Union[str, Path]) -> bool:
        """
        Check if data source has changed since last access
        
        Args:
            path: Path to data source
            
        Returns:
            True if data has changed, False otherwise
        """
        pass