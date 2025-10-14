"""
Pipeline decorators for consistent logging and timing.

This module provides decorators that automatically handle:
- Workflow step logging
- Step timing
- Step completion logging
"""

import time
import functools
from typing import Callable, Any, Optional
try:
    from .logger import get_logger
except ImportError:
    # Fallback for when running from root directory
    from forecaster.utils.logger import get_logger


def pipeline_step(step_name: str, step_number: int, total_steps: int):
    """
    Decorator for pipeline steps that handles logging and timing.
    
    Args:
        step_name: Human-readable name of the step
        step_number: Current step number (1-based)
        total_steps: Total number of steps in the pipeline
        
    Usage:
        @pipeline_step("Loading data", 1, 6)
        def _load_and_validate_data(self):
            # Step implementation
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs) -> Any:
            # Get logger from the instance
            logger = getattr(self, 'logger', None)
            if logger is None:
                # Fallback to module logger if instance logger not available
                logger = get_logger(__name__)
            
            # Log workflow step start
            logger.log_workflow_step(step_name, step_number, total_steps)
            
            # Time the step execution
            step_start = time.time()
            try:
                result = func(self, *args, **kwargs)
                step_duration = time.time() - step_start
                
                # Log step completion with timing
                completion_message = f"{step_name} completed"
                logger.log_step_completion(completion_message, step_duration)
                
                return result
                
            except Exception as e:
                step_duration = time.time() - step_start
                logger.error(f"Step '{step_name}' failed after {step_duration:.2f}s: {e}")
                raise
        
        return wrapper
    return decorator


def pipeline_step_simple(step_name: str):
    """
    Simplified decorator for pipeline steps without step numbering.
    
    Args:
        step_name: Human-readable name of the step
        
    Usage:
        @pipeline_step_simple("Loading data")
        def _load_and_validate_data(self):
            # Step implementation
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs) -> Any:
            # Get logger from the instance
            logger = getattr(self, 'logger', None)
            if logger is None:
                # Fallback to module logger if instance logger not available
                logger = get_logger(__name__)
            
            # Log workflow step start
            logger.log_workflow_step(step_name)
            
            # Time the step execution
            step_start = time.time()
            try:
                result = func(self, *args, **kwargs)
                step_duration = time.time() - step_start
                
                # Log step completion with timing
                completion_message = f"{step_name} completed"
                logger.log_step_completion(completion_message, step_duration)
                
                return result
                
            except Exception as e:
                step_duration = time.time() - step_start
                logger.error(f"Step '{step_name}' failed after {step_duration:.2f}s: {e}")
                raise
        
        return wrapper
    return decorator
