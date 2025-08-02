# Progress Tracking Guide for Backtesting

This guide explains the enhanced progress tracking and logging features implemented in the unified backtesting system.

## Overview

The backtesting system now provides comprehensive progress tracking that allows you to monitor the execution of backtest runs in real-time. This includes:

- **Step-by-step progress indicators** for the entire workflow
- **Parameter optimization progress bars** with success/failure tracking
- **Backtesting task progress** with real-time success rates
- **Detailed logging** at multiple levels
- **Completion summaries** with performance metrics

## Progress Tracking Features

### 1. Overall Workflow Progress

The backtesting workflow is divided into 7 main steps, each with clear progress indicators:

```
🚀 UNIFIED BACKTESTING WORKFLOW
================================================================================

📂 Step 1/7: Loading and validating data...
✅ Data loading completed

📋 Step 1.5/7: Expanding product master for multiple methods...
✅ Product master expansion completed

📊 Step 2/7: Grouping products by forecasting method...
✅ Product grouping completed

🔍 Step 3/7: Handling outliers...
✅ Outlier handling completed

🔧 Step 4/7: Optimizing parameters...
📊 Total products to optimize: 150
==================================================
📈 Method: prophet (75 products)
Optimizing prophet: 100%|██████████| 75/75 [00:45<00:00, 1.67product/s]
📈 Method: arima (50 products)
Optimizing arima: 100%|██████████| 50/50 [00:30<00:00, 1.67product/s]
📈 Method: moving_average (25 products)
Optimizing moving_average: 100%|██████████| 25/25 [00:15<00:00, 1.67product/s]

✅ Parameter optimization completed!
📊 Successfully optimized: 145/150 products
❌ Failed optimizations: 5 products

🚀 Step 5/7: Running backtesting...
🚀 Backtesting Progress
📊 Total analysis dates: 365
📦 Total product-method combinations: 150
🔢 Total forecast tasks: 54,750
⚙️ Processing mode: Parallel
🚀 Workers: 8
============================================================
📋 Created 54,750 forecast tasks
🚀 Starting parallel processing with 8 workers...
Processing forecast tasks: 100%|██████████| 54750/54750 [2:30:15<00:00, 6.08task/s]

📊 Progress: 100/54750 tasks completed (95.0% success rate)
📊 Progress: 200/54750 tasks completed (94.5% success rate)
...
📊 Progress: 54700/54750 tasks completed (94.8% success rate)

✅ Parallel backtesting completed!
📊 Successfully processed: 51,825/54,750 tasks
📈 Generated 51,825 forecast results
📋 Generated 155,475 forecast comparisons

📈 Step 6/7: Calculating accuracy metrics...
✅ Accuracy metrics calculated

💾 Step 7/7: Saving results...
✅ Results saved

🎉 UNIFIED BACKTESTING COMPLETED SUCCESSFULLY!
⏱️  Total execution time: 9120.45 seconds
================================================================================
```

### 2. Parameter Optimization Progress

The parameter optimization step shows detailed progress for each forecasting method:

- **Progress bars** for each method showing completion percentage
- **Success/failure counts** for optimization attempts
- **Real-time processing rates** (products per second)
- **Method-specific grouping** for better organization

### 3. Backtesting Task Progress

The main backtesting phase provides comprehensive progress tracking:

#### Parallel Processing
- **Task creation summary** showing total number of forecast tasks
- **Worker information** showing parallel processing configuration
- **Real-time progress bar** with completion percentage and ETA
- **Success rate tracking** with periodic updates (every 100 tasks)
- **Final summary** with success/failure statistics

#### Sequential Processing
- **Date-based progress** showing analysis date completion
- **Task-level tracking** with success rate monitoring
- **Periodic progress updates** (every 50 tasks)
- **Completion summary** with final statistics

### 4. Progress Bar Format

The progress bars use a detailed format showing:
- **Description**: What's being processed
- **Progress bar**: Visual completion indicator
- **Count**: Current/total items
- **Time**: Elapsed/remaining time
- **Rate**: Items processed per second

Example:
```
Processing forecast tasks: 45%|████▌     | 24637/54750 [1:12:30<1:28:45, 5.67task/s]
```

## Logging Levels

The system supports multiple logging levels for different detail requirements:

### INFO Level (Default)
- Shows step-by-step progress
- Displays completion summaries
- Reports success/failure counts
- Shows timing information

### DEBUG Level
- Includes detailed parameter optimization logs
- Shows individual task processing details
- Provides method-specific debugging information

### WARNING Level
- Shows only warnings and errors
- Minimal progress information
- Focus on issues and failures

## Configuration Options

### Progress Tracking Settings

```python
config = BacktestConfig(
    # Processing settings that affect progress tracking
    batch_size=20,           # Affects progress update frequency
    max_workers=8,           # Determines parallel vs sequential processing
    log_level="INFO",        # Controls logging detail level
    
    # Other settings...
)
```

### Progress Update Frequency

- **Parameter optimization**: Updates per product
- **Parallel backtesting**: Updates every 100 tasks
- **Sequential backtesting**: Updates every 50 tasks
- **Overall workflow**: Updates per step

## Monitoring Long-Running Backtests

For long-running backtests, the system provides several monitoring features:

### 1. Real-Time Progress
- Live progress bars with ETA
- Success rate monitoring
- Processing speed indicators

### 2. Periodic Summaries
- Regular progress updates
- Success/failure statistics
- Time-based performance metrics

### 3. Error Tracking
- Failed optimization attempts
- Failed forecast tasks
- Detailed error logging

### 4. Resource Monitoring
- Worker utilization
- Memory usage (via logging)
- Processing efficiency

## Example Usage

### Basic Backtest with Progress Tracking

```python
from forecaster.backtesting.config import BacktestConfig
from forecaster.backtesting.unified_backtester import run_unified_backtest

# Configure backtest with progress tracking
config = BacktestConfig(
    data_dir="forecaster/data",
    demand_file="customer_demand.csv",
    product_master_file="customer_product_master.csv",
    output_dir="output/backtest_with_progress",
    analysis_start_date=None,  # Will be calculated based on ss_window_length and first review date
analysis_end_date=None,     # Will be set to last review date
    max_workers=8,  # Enable parallel processing
    log_level="INFO",  # Enable progress tracking
)

# Run backtest with progress tracking
result = run_unified_backtest(config)
```

### Complete Workflow with Progress Tracking

The complete workflow (`run_complete_workflow.py`) now supports real-time progress tracking for all steps:

- **Data Validation**: Progress information and validation results
- **Backtesting**: Full progress tracking with parameter optimization and forecast progress
- **Safety Stock Calculation**: Progress bars and completion status
- **Inventory Simulation**: Real-time progress updates
- **Web Interface**: Startup progress and status

All progress tracking features from the individual backtesting are preserved when running through the complete workflow.

### Command Line Usage

#### Individual Backtesting
```bash
# Run with default progress tracking (analysis dates calculated automatically)
python run_unified_backtest.py \
    --data-dir forecaster/data \
    --demand-file customer_demand.csv \
    --product-master-file customer_product_master.csv \
    --max-workers 8

# Run with detailed logging (analysis dates calculated automatically)
python run_unified_backtest.py \
    --data-dir forecaster/data \
    --demand-file customer_demand.csv \
    --product-master-file customer_product_master.csv \
    --max-workers 8 \
    --log-level DEBUG
```

#### Complete Workflow with Progress Tracking
```bash
# Run complete workflow with progress tracking (analysis dates calculated automatically)
python run_complete_workflow.py \
    --data-dir forecaster/data \
    --demand-file customer_demand.csv \
    --product-master-file customer_product_master.csv \
    --max-workers 8 \
    --log-level INFO

# Run with web interface (analysis dates calculated automatically)
python run_complete_workflow.py \
    --data-dir forecaster/data \
    --demand-file customer_demand.csv \
    --product-master-file customer_product_master.csv \
    --max-workers 8 \
    --log-level INFO \
    --web-interface
```

## Testing Progress Tracking

Use the test scripts to see progress tracking in action:

### Individual Backtesting Test
```bash
python test_progress_tracking.py
```

This runs a small backtest (15 days) to demonstrate all progress tracking features.

### Complete Workflow Test
```bash
python test_complete_workflow_progress.py
```

This runs a small complete workflow to verify that progress tracking works when running through `run_complete_workflow.py`.

## Performance Considerations

### Progress Update Overhead
- Progress bars add minimal overhead (< 1% of total time)
- Logging can be controlled via log_level
- Periodic updates are configurable

### Memory Usage
- Progress tracking uses minimal additional memory
- Real-time statistics are lightweight
- No persistent storage of progress data

### Scalability
- Progress tracking scales with the number of tasks
- Parallel processing shows worker utilization
- Success rate tracking works for any task count

## Troubleshooting

### Progress Not Showing
- Check that `log_level` is set to "INFO" or "DEBUG"
- Ensure terminal supports progress bars
- Verify that `max_workers > 1` for parallel processing

### Slow Progress Updates
- Increase `batch_size` for fewer progress updates
- Reduce `max_workers` if system is overloaded
- Check system resources (CPU, memory)

### Missing Progress Information
- Verify that data files exist and are valid
- Check that product master contains valid forecasting methods
- Ensure analysis dates are within data range

## Future Enhancements

Planned improvements to progress tracking:

1. **Web-based progress dashboard** for remote monitoring
2. **Progress persistence** for resuming interrupted runs
3. **Custom progress callbacks** for integration with external systems
4. **Performance profiling** with detailed timing breakdowns
5. **Resource monitoring** with CPU/memory usage graphs

## Conclusion

The enhanced progress tracking system provides comprehensive visibility into backtesting execution, making it easier to:

- Monitor long-running backtests
- Identify performance bottlenecks
- Track success rates and failures
- Estimate completion times
- Debug issues in real-time

This makes the backtesting system more user-friendly and production-ready for large-scale forecasting operations. 