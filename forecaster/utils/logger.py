"""
Logging utilities for the forecaster package.
Provides standardized logging across all modules.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import date

class ForecasterLogger:
    """Centralized logging for the forecaster package"""
    
    def __init__(self, name: str = "forecaster", level: str = "INFO", 
                 log_file: Optional[str] = None):
        """
        Initialize the logger
        
        Args:
            name: Logger name
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Optional file path for logging
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (if specified)
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)
    
    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)
    
    def critical(self, message: str):
        """Log critical message"""
        self.logger.critical(message)
    
    def log_data_loading(self, filename: str, record_count: int, duration: float):
        """Log data loading information"""
        self.info(f"Loaded {record_count:,} records from {filename} in {duration:.2f}s")
    
    def log_validation_result(self, filename: str, is_valid: bool, errors: list = None):
        """Log validation results"""
        if is_valid:
            self.info(f"Validation passed for {filename}")
        else:
            self.error(f"Validation failed for {filename}: {errors}")
    
    def log_processing_step(self, step_name: str, duration: float, details: str = ""):
        """Log processing step information"""
        message = f"Completed {step_name} in {duration:.2f}s"
        if details:
            message += f" - {details}"
        self.info(message)

# Global logger instance
_logger = None

def get_logger(name: str = "forecaster", level: str = "INFO", 
               log_file: Optional[str] = None) -> ForecasterLogger:
    """
    Get or create a logger instance
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional log file path
        
    Returns:
        ForecasterLogger instance
    """
    global _logger
    if _logger is None:
        _logger = ForecasterLogger(name, level, log_file)
    return _logger

def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """
    Setup logging for the forecaster package
    
    Args:
        level: Logging level
        log_file: Optional log file path
    """
    global _logger
    _logger = ForecasterLogger("forecaster", level, log_file)
    _logger.info("Forecaster logging initialized")

# Convenience functions
def debug(message: str):
    """Log debug message using global logger"""
    if _logger:
        _logger.debug(message)

def info(message: str):
    """Log info message using global logger"""
    if _logger:
        _logger.info(message)

def warning(message: str):
    """Log warning message using global logger"""
    if _logger:
        _logger.warning(message)

def error(message: str):
    """Log error message using global logger"""
    if _logger:
        _logger.error(message)

def critical(message: str):
    """Log critical message using global logger"""
    if _logger:
        _logger.critical(message)
