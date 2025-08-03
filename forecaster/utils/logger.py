"""
Standardized logging utilities for the forecaster package.
Provides consistent logging across all modules with proper configuration for multi-step workflows.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import os


class ForecasterLogger:
    """
    Centralized logging for the forecaster package with workflow-aware features.
    
    Features:
    - Hierarchical logging with module names
    - Workflow step tracking
    - Progress-aware logging levels
    - File and console output
    - Performance timing
    - Structured logging for complex data
    """
    
    # Global configuration
    _global_config: Dict[str, Any] = {
        'level': 'INFO',
        'log_file': None,
        'console_output': True,
        'file_output': True,
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'date_format': '%Y-%m-%d %H:%M:%S',
        'max_file_size': 10 * 1024 * 1024,  # 10MB
        'backup_count': 5
    }
    
    # Registry of all loggers
    _loggers: Dict[str, 'ForecasterLogger'] = {}
    
    def __init__(self, name: str, level: str = None, log_file: Optional[str] = None):
        """
        Initialize the logger
        
        Args:
            name: Logger name (usually __name__)
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Optional file path for logging
        """
        self.name = name
        self.logger = logging.getLogger(name)
        
        # Use global config if not specified
        level = level or self._global_config['level']
        log_file = log_file or self._global_config['log_file']
        
        # Set level
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Clear existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            self._global_config['format'],
            datefmt=self._global_config['date_format']
        )
        
        # Console handler
        if self._global_config['console_output']:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if self._global_config['file_output'] and log_file:
            self._setup_file_handler(log_file, formatter)
        
        # Prevent propagation to root logger to avoid duplicate messages
        self.logger.propagate = False
        
        # Register this logger
        ForecasterLogger._loggers[name] = self
    
    def _setup_file_handler(self, log_file: str, formatter: logging.Formatter):
        """Setup rotating file handler"""
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use rotating file handler for large log files
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=self._global_config['max_file_size'],
            backupCount=self._global_config['backup_count']
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    # Standard logging methods
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
    
    # Workflow-specific logging methods
    def log_workflow_step(self, step_name: str, step_number: int, total_steps: int, 
                         description: str = ""):
        """Log workflow step information"""
        message = f"Step {step_number}/{total_steps}: {step_name}"
        if description:
            message += f" - {description}"
        self.info(message)
    
    def log_step_completion(self, step_name: str, duration: float, 
                           details: Dict[str, Any] = None):
        """Log step completion with timing and details"""
        message = f"✅ {step_name} completed in {duration:.2f}s"
        if details:
            detail_str = ", ".join([f"{k}: {v}" for k, v in details.items()])
            message += f" - {detail_str}"
        self.info(message)
    
    def log_data_loading(self, filename: str, record_count: int, duration: float):
        """Log data loading information"""
        self.info(f"📂 Loaded {record_count:,} records from {filename} in {duration:.2f}s")
    
    def log_validation_result(self, filename: str, is_valid: bool, 
                             issue_count: int = 0, critical_count: int = 0):
        """Log validation results"""
        if is_valid:
            self.info(f"✅ Validation passed for {filename}")
        else:
            self.warning(f"⚠️ Validation failed for {filename}: {issue_count} issues ({critical_count} critical)")
    
    def log_processing_progress(self, current: int, total: int, 
                               operation: str, details: str = ""):
        """Log processing progress"""
        percentage = (current / total) * 100 if total > 0 else 0
        message = f"📊 {operation}: {current:,}/{total:,} ({percentage:.1f}%)"
        if details:
            message += f" - {details}"
        self.info(message)
    
    def log_performance_metrics(self, operation: str, metrics: Dict[str, Any]):
        """Log performance metrics"""
        metric_str = ", ".join([f"{k}: {v}" for k, v in metrics.items()])
        self.info(f"📈 {operation} performance: {metric_str}")
    
    def log_error_with_context(self, error: Exception, context: str = ""):
        """Log error with context information"""
        message = f"❌ Error in {context}: {str(error)}" if context else f"❌ Error: {str(error)}"
        self.error(message)
    
    # Configuration methods
    @classmethod
    def configure_global(cls, **kwargs):
        """Configure global logging settings"""
        cls._global_config.update(kwargs)
        
        # Update existing loggers
        for logger in cls._loggers.values():
            logger._update_config()
    
    @classmethod
    def set_console_output(cls, enabled: bool):
        """Enable/disable console output"""
        cls.configure_global(console_output=enabled)
    
    @classmethod
    def set_file_output(cls, enabled: bool, log_file: str = None):
        """Enable/disable file output"""
        cls.configure_global(file_output=enabled, log_file=log_file)
    
    @classmethod
    def set_level(cls, level: str):
        """Set global log level"""
        cls.configure_global(level=level)
    
    def _update_config(self):
        """Update logger configuration"""
        # Recreate handlers with new config
        self.logger.handlers.clear()
        
        formatter = logging.Formatter(
            self._global_config['format'],
            datefmt=self._global_config['date_format']
        )
        
        if self._global_config['console_output']:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        if self._global_config['file_output'] and self._global_config['log_file']:
            self._setup_file_handler(self._global_config['log_file'], formatter)


# Global logger instance for convenience
_global_logger = None

def get_logger(name: str = None, level: str = None, 
               log_file: Optional[str] = None) -> ForecasterLogger:
    """
    Get or create a logger instance
    
    Args:
        name: Logger name (defaults to 'forecaster')
        level: Logging level
        log_file: Optional log file path
        
    Returns:
        ForecasterLogger instance
    """
    global _global_logger
    
    if name is None:
        name = "forecaster"
    
    # Return existing logger if it exists
    if name in ForecasterLogger._loggers:
        return ForecasterLogger._loggers[name]
    
    # Create new logger
    return ForecasterLogger(name, level, log_file)


def setup_logging(level: str = "INFO", log_file: Optional[str] = None,
                  console_output: bool = True, file_output: bool = True):
    """
    Setup global logging configuration
    
    Args:
        level: Logging level
        log_file: Optional log file path
        console_output: Enable console output
        file_output: Enable file output
    """
    # Configure global settings
    ForecasterLogger.configure_global(
        level=level,
        log_file=log_file,
        console_output=console_output,
        file_output=file_output
    )
    
    # Also configure the root logger to ensure all modules inherit the level
    import logging
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers to avoid duplicates
    root_logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        ForecasterLogger._global_config['format'],
        datefmt=ForecasterLogger._global_config['date_format']
    )
    
    # Console handler
    if ForecasterLogger._global_config['console_output']:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if ForecasterLogger._global_config['file_output'] and ForecasterLogger._global_config['log_file']:
        log_path = Path(ForecasterLogger._global_config['log_file'])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            ForecasterLogger._global_config['log_file'],
            maxBytes=ForecasterLogger._global_config['max_file_size'],
            backupCount=ForecasterLogger._global_config['backup_count']
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def configure_workflow_logging(workflow_name: str, log_level: str = "INFO",
                              log_dir: str = "output/logs"):
    """
    Configure logging specifically for workflow runs
    
    Args:
        workflow_name: Name of the workflow
        log_level: Logging level
        log_dir: Directory for log files
    """
    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/{workflow_name}_{timestamp}.log"
    
    # Setup logging with root logger configuration
    setup_logging(
        level=log_level,
        log_file=log_file,
        console_output=True,
        file_output=True
    )
    
    # Get logger and log initialization
    logger = get_logger("workflow")
    logger.info(f"🚀 {workflow_name} workflow started")
    logger.info(f"📝 Log file: {log_file}")
    logger.info(f"🔧 Log level: {log_level}")
    
    return logger


# Convenience functions for backward compatibility
def debug(message: str):
    """Log debug message using global logger"""
    logger = get_logger()
    logger.debug(message)

def info(message: str):
    """Log info message using global logger"""
    logger = get_logger()
    logger.info(message)

def warning(message: str):
    """Log warning message using global logger"""
    logger = get_logger()
    logger.warning(message)

def error(message: str):
    """Log error message using global logger"""
    logger = get_logger()
    logger.error(message)

def critical(message: str):
    """Log critical message using global logger"""
    logger = get_logger()
    logger.critical(message)
