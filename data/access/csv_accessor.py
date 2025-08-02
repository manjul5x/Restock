"""
CSV file accessor implementation with safety checks.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
import os
import hashlib
from datetime import datetime
import logging
try:
    from ..exceptions import DataAccessError
    from .base import DataAccessor
except ImportError:
    from exceptions import DataAccessError
    from access.base import DataAccessor

logger = logging.getLogger(__name__)

class CSVAccessor(DataAccessor):
    """CSV file accessor with safety checks and change detection"""
    
    def __init__(self):
        self.file_hashes = {}  # Track file changes
        self.file_sizes = {}   # Track file sizes
        self.last_access = {}  # Track access times
    
    def _get_file_hash(self, path: Path) -> str:
        """Get file hash to detect changes"""
        try:
            return hashlib.md5(open(path, 'rb').read()).hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash for {path}: {e}")
            return ""
    
    def has_file_changed(self, path: Path) -> bool:
        """Check if file has changed since last read"""
        try:
            path = Path(path)
            current_hash = self._get_file_hash(path)
            current_size = os.path.getsize(path)
            current_mtime = os.path.getmtime(path)
            
            # Check if we have previous info
            if str(path) in self.file_hashes:
                # Quick check with size and mtime first
                if (current_size != self.file_sizes.get(str(path)) or
                    current_mtime > self.last_access.get(str(path), 0)):
                    # Only calculate hash if size/time changed
                    if current_hash != self.file_hashes[str(path)]:
                        logger.info(f"File {path} has changed")
                        return True
            
            # Update tracking info
            self.file_hashes[str(path)] = current_hash
            self.file_sizes[str(path)] = current_size
            self.last_access[str(path)] = current_mtime
            return False
            
        except Exception as e:
            logger.error(f"Error checking file changes for {path}: {e}")
            return True  # Assume changed on error
    
    def read_data(self, 
                  path: Union[str, Path], 
                  columns: Optional[List[str]] = None,
                  **kwargs) -> pd.DataFrame:
        """Read CSV with safety checks"""
        path = Path(path)
        
        if not path.exists():
            raise DataAccessError(f"File not found: {path}")
        
        try:
            read_kwargs = {
                'memory_map': True,  # Memory mapping for large files
                'low_memory': False  # Avoid mixed type inference warnings
            }
            if columns:
                read_kwargs['usecols'] = columns
            read_kwargs.update(kwargs)
            
            logger.debug(f"Reading CSV file: {path}")
            df = pd.read_csv(path, **read_kwargs)
            
            # Convert date column to datetime if it exists
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date']).dt.date
            
            # Update tracking info
            self.file_hashes[str(path)] = self._get_file_hash(path)
            self.file_sizes[str(path)] = os.path.getsize(path)
            self.last_access[str(path)] = os.path.getmtime(path)
            
            logger.info(f"Successfully read {len(df)} rows from {path}")
            return df
            
        except Exception as e:
            logger.error(f"Error reading {path}: {e}")
            raise DataAccessError(f"Error reading {path}: {str(e)}")
    
    def write_data(self, 
                   df: pd.DataFrame, 
                   path: Union[str, Path],
                   **kwargs) -> None:
        """Write CSV with safety checks"""
        path = Path(path)
        temp_path = None
        
        try:
            # Create directory if needed
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write with temp file for atomicity
            temp_path = path.with_suffix('.tmp')
            logger.debug(f"Writing to temporary file: {temp_path}")
            
            df.to_csv(temp_path, index=False, **kwargs)
            
            # Atomic rename
            temp_path.replace(path)
            
            # Update tracking info
            self.file_hashes[str(path)] = self._get_file_hash(path)
            self.file_sizes[str(path)] = os.path.getsize(path)
            self.last_access[str(path)] = os.path.getmtime(path)
            
            logger.info(f"Successfully wrote {len(df)} rows to {path}")
            
        except Exception as e:
            logger.error(f"Error writing {path}: {e}")
            if temp_path and temp_path.exists():
                temp_path.unlink()
            raise DataAccessError(f"Error writing {path}: {str(e)}")
    
    def validate_connection(self) -> bool:
        """Validate file system access"""
        try:
            test_file = Path('test_access.tmp')
            test_file.touch()
            test_file.unlink()
            return True
        except Exception as e:
            logger.error(f"File system access validation failed: {e}")
            return False
    
    def get_data_info(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Get file metadata"""
        path = Path(path)
        try:
            return {
                'size': os.path.getsize(path),
                'modified': datetime.fromtimestamp(os.path.getmtime(path)),
                'hash': self._get_file_hash(path),
                'last_access': self.last_access.get(str(path)),
                'tracked_size': self.file_sizes.get(str(path))
            }
        except Exception as e:
            logger.error(f"Error getting file info for {path}: {e}")
            return {}