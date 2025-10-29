"""Data loading package."""

# Import both the original DataLoader and the new EnvDataLoader
try:
    from .loader import DataLoader
except ImportError:
    DataLoader = None

try:
    from .loader_env import EnvDataLoader
except ImportError:
    EnvDataLoader = None

from .exceptions import DataAccessError, CacheError, StaleDataError, CacheMemoryError

__all__ = [
    'DataLoader',
    'EnvDataLoader', 
    'DataAccessError',
    'CacheError',
    'StaleDataError',
    'CacheMemoryError'
]