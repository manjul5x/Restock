# Logging Standardization

## Overview

This document describes the standardized logging system implemented across the forecaster project. The system provides consistent, workflow-aware logging that supports both individual module execution and complete workflow runs.

## Key Features

### 1. **Centralized Configuration**
- Global logging configuration that applies to all modules
- Hierarchical logger management with module-specific loggers
- Configurable log levels, output destinations, and formatting

### 2. **Workflow-Aware Logging**
- Step-by-step progress tracking for multi-step workflows
- Performance timing and metrics logging
- Structured logging for complex operations

### 3. **Flexible Output Control**
- Console and file output support
- Rotating file handlers for large log files
- Configurable log levels per workflow run

### 4. **Professional Output**
- Clean, minimal output for production use
- Rich, detailed logging for debugging
- Consistent formatting across all modules

## Architecture

### Core Components

#### `ForecasterLogger` Class
```python
class ForecasterLogger:
    """Centralized logging for the forecaster package with workflow-aware features."""
    
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
```

#### Key Methods
- `log_workflow_step()` - Track workflow progress
- `log_step_completion()` - Log step completion with timing
- `log_data_loading()` - Standardized data loading logs
- `log_validation_result()` - Validation result logging
- `log_processing_progress()` - Progress tracking
- `log_performance_metrics()` - Performance monitoring
- `log_error_with_context()` - Error logging with context

### Configuration Functions

#### `configure_workflow_logging()`
```python
def configure_workflow_logging(workflow_name: str, log_level: str = "INFO",
                              log_dir: str = "output/logs"):
    """Configure logging specifically for workflow runs"""
```

#### `setup_logging()`
```python
def setup_logging(level: str = "INFO", log_file: Optional[str] = None,
                  console_output: bool = True, file_output: bool = True):
    """Setup global logging configuration"""
```

## Usage Patterns

### 1. **Module-Level Logging**
```python
from forecaster.utils.logger import get_logger

logger = get_logger(__name__)

def my_function():
    logger.info("Starting operation")
    # ... do work ...
    logger.info("Operation completed")
```

### 2. **Workflow-Level Logging**
```python
from forecaster.utils.logger import configure_workflow_logging

# Setup workflow logging
logger = configure_workflow_logging(
    workflow_name="unified_backtesting",
    log_level="INFO",
    log_dir="output/logs"
)

# Log workflow steps
logger.log_workflow_step("Loading data", 1, 7)
logger.log_step_completion("Data loading", 2.5)
```

### 3. **Progress Tracking**
```python
# Log processing progress
logger.log_processing_progress(100, 1000, "forecast generation")

# Log performance metrics
logger.log_performance_metrics("backtesting", {
    "total_tasks": 1000,
    "success_rate": 99.5,
    "avg_time": 0.15
})
```

## Log Levels

### **ERROR** - Only errors and critical issues
- Use for production runs where minimal output is desired
- Shows only errors and critical warnings
- Ideal for automated workflows

### **WARNING** - Warnings and errors
- Default for most production use
- Shows warnings, errors, and essential progress
- Clean output with important information

### **INFO** - Standard information
- Default level for development
- Shows all important operations and progress
- Good balance of detail and readability

### **DEBUG** - Detailed debugging information
- Use for troubleshooting and development
- Shows all operations, including internal details
- Most verbose output

## Implementation Examples

### Unified Backtester
```python
class UnifiedBacktester:
    def __init__(self, config: BacktestConfig):
        # Setup logging for this backtesting run
        self.logger = configure_workflow_logging(
            workflow_name="unified_backtesting",
            log_level=config.log_level,
            log_dir="output/logs"
        )
    
    def run(self) -> Dict:
        # Log workflow steps
        self.logger.log_workflow_step("Loading data", 1, 7)
        self._load_data()
        self.logger.log_step_completion("Data loading", time.time() - start_time)
```

### Complete Workflow
```python
def main():
    # Setup logging for the workflow
    logger = configure_workflow_logging(
        workflow_name="complete_workflow",
        log_level=args.log_level,
        log_dir="output/logs"
    )
    
    try:
        run_complete_workflow(..., logger=logger)
    except Exception as e:
        logger.log_error_with_context(e, "Complete workflow failed")
        sys.exit(1)
```

## Best Practices

### 1. **Use Module-Specific Loggers**
```python
# Good
logger = get_logger(__name__)

# Avoid
logger = get_logger("forecaster")
```

### 2. **Log at Appropriate Levels**
```python
# ERROR - Only for actual errors
logger.error("Failed to load data file")

# WARNING - For issues that don't stop execution
logger.warning("Missing optional configuration")

# INFO - For normal operations
logger.info("Processing 1000 records")

# DEBUG - For detailed troubleshooting
logger.debug("Internal state: {state}")
```

### 3. **Use Structured Logging Methods**
```python
# Use specialized methods when available
logger.log_data_loading("data.csv", 1000, 2.5)
logger.log_validation_result("config.yaml", True, 0, 0)
logger.log_step_completion("backtesting", 15.2, {"forecasts": 1000})
```

### 4. **Handle Exceptions Properly**
```python
try:
    # ... risky operation ...
except Exception as e:
    logger.log_error_with_context(e, "Operation failed")
    raise
```

## Configuration Options

### Environment Variables
```bash
export FORECASTER_LOG_LEVEL=WARNING
export FORECASTER_LOG_FILE=output/logs/app.log
export FORECASTER_CONSOLE_OUTPUT=false
```

### Command Line Arguments
```bash
# Run with minimal logging
python run_complete_workflow.py --log-level ERROR

# Run with detailed logging
python run_complete_workflow.py --log-level DEBUG
```

### Programmatic Configuration
```python
from forecaster.utils.logger import ForecasterLogger

# Configure global settings
ForecasterLogger.configure_global(
    level="WARNING",
    console_output=False,
    file_output=True,
    log_file="output/logs/custom.log"
)
```

## Output Examples

### Clean Production Output (WARNING level)
```
🚀 Unified Backtesting
Processing forecast tasks: 100%|████████████████████████████████████| 1014/1014 [00:15<00:00, 65.01task/s]
✅ Parallel backtesting completed!
📊 Successfully processed: 1,014/1,014 tasks
📈 Generated 1,014 forecast results
```

### Detailed Development Output (INFO level)
```
2025-08-02 19:29:07 - unified_backtester - INFO - Step 1: Loading and validating data
2025-08-02 19:29:07 - unified_backtester - INFO - Loaded 1338 demand records
2025-08-02 19:29:07 - unified_backtester - INFO - Step 2: Grouping products by forecasting method
2025-08-02 19:29:07 - unified_backtester - INFO - Method 'moving_average': 2 product-method combinations
```

## Migration Guide

### From Old Logging System
1. Replace `logging.getLogger(__name__)` with `get_logger(__name__)`
2. Replace print statements with appropriate logger calls
3. Use workflow-specific logging methods for multi-step processes
4. Configure logging at the workflow level

### Before (Old System)
```python
import logging
logger = logging.getLogger(__name__)

print("Starting backtesting...")
logger.info("Processing data")
print("✅ Completed!")
```

### After (New System)
```python
from forecaster.utils.logger import get_logger

logger = get_logger(__name__)

logger.log_workflow_step("Backtesting", 1, 5)
logger.info("Processing data")
logger.log_step_completion("Backtesting", 15.2)
```

## Benefits

1. **Consistency** - All modules use the same logging approach
2. **Flexibility** - Easy to adjust log levels and output
3. **Professional** - Clean, structured output
4. **Maintainable** - Centralized configuration and methods
5. **Workflow-Aware** - Built-in support for multi-step processes
6. **Performance** - Efficient logging with rotating files
7. **Debugging** - Rich debugging information when needed

## Future Enhancements

1. **Structured Logging** - JSON format for machine processing
2. **Log Aggregation** - Centralized log collection
3. **Metrics Integration** - Performance monitoring
4. **Alert System** - Automatic error notifications
5. **Log Analysis** - Built-in log analysis tools 