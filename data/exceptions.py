"""
Custom exceptions for data loading and caching.
"""

class DataAccessError(Exception):
    """Base exception for data access errors"""
    pass

class CacheError(Exception):
    """Base exception for caching errors"""
    pass

class StaleDataError(CacheError):
    """Raised when cached data might be stale"""
    pass

class CacheMemoryError(CacheError):
    """Raised when cache memory limits are exceeded"""
    pass

class DataValidationError(Exception):
    """Raised when data validation fails"""
    pass