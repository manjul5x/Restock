# Inventory Forecasting & Analysis System

A comprehensive inventory forecasting and analysis system that provides end-to-end inventory optimization through forecasting, backtesting, safety stock calculation, and simulation.

## Brief Summary

This system analyzes your inventory data to:
- **Forecast demand** using multiple models (Prophet, Moving Average)
- **Test forecast accuracy** through historical backtesting
- **Calculate optimal safety stocks** based on forecast errors
- **Simulate inventory performance** with different ordering policies
- **Visualize results** through an interactive web interface

The complete workflow runs all these analyses in sequence, providing comprehensive insights into your inventory management.

## Creating a Fork and Environment

### Prerequisites
- Python 3.8 or higher
- UV package manager (recommended)
- Git

### Creating a Fork
1. **Fork the Repository**
   - Go to the original repository on GitHub
   - Click the "Fork" button in the top right
   - This creates your own copy of the repository

2. **Clone Your Fork**
   ```bash
   # Clone your forked repository
   git clone https://github.com/YOUR_USERNAME/Forecaster.git
   cd Forecaster
   
   # Add the original repository as upstream (optional, for updates)
   git remote add upstream https://github.com/ORIGINAL_OWNER/Forecaster.git
   ```

3. **Create a Development Branch**
   ```bash
   # Create and switch to a development branch
   git checkout -b dev-your-name
   
   # Or use the existing dev branch
   git checkout dev-samuel
   ```

### Quick Setup
```bash
# Run the setup script (installs uv, dependencies, and dev tools)
./setup.sh
```

### Manual Setup
```bash
# Install UV if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# For development (optional)
uv sync --extra dev
```

### Development Environment
```bash
# Install development tools
make dev

# Run tests to verify setup
make test

# Format code
make format
```

## Data Requirements

### Required Files
Replace the example data files with your own:
- **`data/customer_data/customer_demand.csv`** - Your demand data
- **`data/customer_data/customer_product_master.csv`** - Your product master data

### Demand Data Format (`customer_demand.csv`)
Required columns:
- `product_id` - Product identifier
- `product_category` - Product category  
- `location_id` - Location identifier
- `date` - Date in YYYY-MM-DD format
- `demand` - Demand quantity (must be ≥ 0)
- `stock_level` - Stock level at end of day (must be ≥ 0)
- `incoming_inventory` - Incoming inventory (can be 0)

### Product Master Format (`customer_product_master.csv`)
Required columns:
- `product_id` - Product identifier
- `location_id` - Location identifier
- `product_category` - Product category
- `demand_frequency` - 'd' (daily), 'w' (weekly), or 'm' (monthly)
- `risk_period` - Risk period as integer multiple of demand frequency
- `forecast_window_length` - Forecasting window length in risk periods
- `forecast_horizon` - Forecast horizon in risk periods
- `forecast_methods` - Comma-separated forecasting methods: 'prophet,moving_average'
- `service_level` - Service level percentage 0.0 to 1.0 (default: 0.95)
- `leadtime` - Lead time in demand frequency units
- `inventory_cost` - Unit cost of inventory (default: 0.0)
- `moq` - Minimum order quantity (default: 0)

## Data Validation

The system automatically validates your data before processing:

### Schema Validation
- Checks required columns are present
- Validates data types (dates, numbers, strings)
- Ensures no missing values in critical fields

### Completeness Validation
- Identifies missing dates in expected ranges
- Checks frequency consistency
- Detects large data gaps (>30 days)

### Quality Validation
- Finds negative values (demand, stock levels)
- Identifies statistical outliers
- Checks for data consistency issues

### Coverage Validation
- Ensures product-location combinations exist in both datasets
- Validates cross-references between demand and product master

Run validation independently:
```bash
python run_data_validation.py
```

## Using the Tool

### What You Need to Configure

Before running the workflow, you need to configure settings in two places:

#### 1. Product Master File (`data/customer_data/customer_product_master.csv`)
Configure these settings for each product:

**Forecasting Methods**
- `forecast_methods`: Choose from "prophet", "moving_average", or "prophet,moving_average"
- **Options**: Single method or comma-separated list for multiple methods

**Safety Stock Settings**
- `distribution`: Choose "kde" (default) or "normal"
- `service_level`: Target service level (0.0 to 1.0, default: 0.95)
- `ss_window_length`: Rolling window for error calculation (default: 180)

**Simulation Settings**
- `leadtime`: Lead time in demand frequency units
- `moq`: Minimum order quantity (default: 0)

**Outlier Detection**
- `outlier_method`: Choose "iqr" (default), "zscore", "mad", "rolling", or "no"
- `outlier_threshold`: Detection threshold (default: 1.5)

#### 2. Configuration File (`data/config/data_config.yaml`)
Configure these system-wide settings:

**Data Paths**
- `paths.outflow`: Path to your demand data file (default: "customer_demand.csv")
- `paths.product_master`: Path to your product master file (default: "customer_product_master.csv")

**Safety Stock Review Dates**
- `safety_stock.review_dates`: List of dates when safety stocks are calculated
- **What it does**: Safety stocks are calculated at each review date using forecast errors from the period leading up to that date
- **Default**: 1st, 8th, 15th, and 22nd of each month in 2024
- **Impact on Backtesting**: 
  - Analysis start date is automatically calculated based on the first review date and maximum safety stock window length
  - More review dates = more safety stock calculations but longer processing time
- **Impact on Simulation**: 
  - Simulation uses safety stocks calculated at each review date
  - More review dates = more frequent safety stock updates during simulation
  - Affects order timing and quantities in the simulation

**MOQ Constraints**
- `simulation.enable_moq`: Enable/disable Minimum Order Quantity constraints
- **What it does**: When enabled, orders must meet the minimum order quantity specified in the product master
- **Impact**: 
  - `true`: Orders are rounded up to meet MOQ (more realistic but may increase inventory)
  - `false`: Orders can be any positive quantity (more flexible but may not reflect real constraints)

### Running the Complete Workflow

```bash
# Run everything with default settings
python run_complete_workflow.py

# Custom configuration
python run_complete_workflow.py \
  --analysis-start-date 2023-01-01 \
  --analysis-end-date 2023-12-31 \
  --max-workers 4 \
  --log-level WARNING
```

This runs:
1. **Data Validation** - Ensures data quality
2. **Backtesting** - Tests forecasting models historically
3. **Safety Stock Calculation** - Optimizes inventory levels
4. **Inventory Simulation** - Simulates performance
5. **Results Generation** - Creates output files

### Go to the Web App

```bash
python webapp/run.py
```

Open `http://localhost:8080` to access the interactive interface.

#### What to Look At

**Forecast Visualization**
- Historical vs. forecasted demand comparison
- Multi-model performance analysis
- Interactive filtering by product, location, method

**Safety Stocks**
- Dynamic safety stock level analysis
- Distribution method comparison
- Trend analysis over time

**Simulation Visualization**
- Inventory simulation results
- Stock level tracking
- Performance metric visualization

**Inventory Comparison** ⭐ **Key Page**
- **Actual vs. Simulated Performance** with visual highlighting
- **Key Metrics**:
  - Service Level & Stockout Rate
  - Inventory Days (lower is better)
  - Total Inventory Holding (units & cost)
  - Missed Demand & Stockout Days
  - Overstocking/Understocking percentages
- **Performance Indicators**: Green checkmarks for improvements, red X for areas needing attention

#### What to Do

1. **Start with Inventory Comparison** - This shows the overall impact of your inventory strategy
2. **Drill down into specific products** - Use filters to focus on problematic items
3. **Compare forecasting methods** - See which models perform best for your data
4. **Analyze safety stock trends** - Understand how safety stocks change over time
5. **Review simulation results** - See how different ordering policies affect performance

## How It Actually Works
### How Forecasting and Backtesting Works

#### Forecasting Process
1. **Data Preparation**: Load and validate demand data
2. **Parameter Optimization**: Each forecasting method optimizes its parameters using historical data
3. **Model Fitting**: Fit the model to the training data
4. **Forecast Generation**: Generate predictions for the specified horizon
5. **Aggregation**: Convert daily forecasts to risk period aggregates

#### Backtesting Process
1. **Historical Simulation**: For each analysis date, use only data available up to that date
2. **Forecast Generation**: Create forecasts for future periods from each historical date
3. **Comparison**: Compare forecasts with actual demand
4. **Error Calculation**: Calculate accuracy metrics (MAE, MAPE, etc.)
5. **Performance Analysis**: Aggregate results across all products and dates

#### Forecasting Methods

**Prophet**
- Facebook's forecasting model
- Handles seasonality, trends, and holidays
- Uses seasonality analysis to optimize parameters
- Falls back to Newton optimization if LBFGS fails
- **Advanced Features**: 
  - Seasonality analysis with Fourier terms
  - Hyperparameter optimization with caching
  - Holiday effects and special events
  - Automatic fallback to Moving Average for insufficient data

**Moving Average**
- Simple average of recent demand
- Configurable window length
- Fast and interpretable
- No parameter optimization needed
- **Best for**: Stable demand patterns with minimal seasonality


### How Safety Stock Works

#### Error Analysis
1. **Historical Errors**: Calculate forecast errors from backtesting results
2. **Error Distribution**: Analyze the statistical distribution of errors
3. **Service Level**: Determine safety stock needed to meet target service level

#### Calculation Methods

**Kernel Density Estimation (KDE)**
- Non-parametric approach
- Adapts to any error distribution shape
- More accurate for non-normal distributions
- Default method
- **Best for**: Skewed or non-normal error distributions

**Normal Distribution**
- Assumes errors follow normal distribution
- Faster calculation
- Less accurate for skewed distributions
- **Best for**: Normally distributed forecast errors

#### Safety Stock Formula
```
Safety Stock = Error Percentile × Service Level Factor
```

The system calculates safety stocks for each product-location-method combination at each review date.

### How Simulation Works

#### Simulation Process
1. **Data Loading**: Load forecasts, safety stocks, and actual demand
2. **Initialization**: Set initial inventory levels
3. **Step-by-Step Simulation**: For each time period:
   - Calculate demand from actual data
   - Update inventory levels
   - Determine if order is needed
   - Place order if required
   - Update on-order inventory
4. **Performance Calculation**: Calculate final metrics

#### Order Policies

**Review Ordering Policy**
- Places orders only on review dates
- Order quantity = max(0, safety_stock + forecast - net_stock)
- Most common in practice

**Continuous Review Policy**
- Places orders whenever stock falls below reorder point
- Order quantity = max(0, order_up_to_level - net_stock)
- More responsive but higher ordering costs

**Min-Max Policy**
- Places orders when stock falls below minimum level
- Order quantity = max(0, maximum_level - net_stock)
- Simple but may not be optimal

#### MOQ Constraints
- **Enabled**: Orders must meet minimum order quantity
- **Disabled**: Orders can be any positive quantity
- Affects order timing and quantities

### What's in the Web App

#### Pages and Features

**Forecast Visualization**
- Interactive demand charts
- Multi-model comparison
- Filtering by product, location, date range
- Zoom and pan capabilities
- Historical vs. forecasted demand comparison

**Safety Stocks**
- Safety stock level charts
- Distribution method comparison
- Trend analysis over time
- Service level impact visualization
- Error distribution analysis

**Simulation Visualization**
- Inventory level tracking
- Order placement visualization
- Performance metric charts
- Policy comparison
- Stock level trends over time

**Inventory Comparison** ⭐ **Main Dashboard**
- Overall performance summary
- Actual vs. simulated metrics
- Performance indicators (green/red)
- Drill-down capabilities
- Key metrics comparison

**Seasonality Analysis** (Advanced)
- Fourier term analysis for Prophet models
- Seasonality strength assessment and visualization
- Component optimization recommendations
- Interactive seasonality charts with period detection
- Automatic seasonality parameter tuning

**Hyperparameter Analysis** (Advanced)
- Parameter optimization testing with performance tracking
- Performance comparison across multiple configurations
- Best parameter identification with statistical validation
- Optimization visualization with convergence analysis
- Cached optimization results for faster subsequent runs

#### Metrics Explained

**Service Level**
- Percentage of demand met from stock
- Higher is better (target: 95%+)

**Stockout Rate**
- Percentage of periods with zero stock
- Lower is better (target: <5%)

**Inventory Days**
- Average days of inventory on hand
- Lower is better (depends on business)

**Total Inventory Holding**
- Total units and cost of inventory
- Balance between service and cost

**Missed Demand**
- Total demand not met from stock
- Lower is better

**Stockout Days**
- Number of days with zero stock
- Lower is better

## Standardization

### How Logging Works

#### Logger Architecture
- **Unified Logger**: `ForecasterLogger` class provides consistent logging across all modules
- **Hierarchical Names**: Module-based logger names (e.g., `forecaster.backtesting`)
- **Workflow Awareness**: Tracks step completion and timing with detailed progress tracking
- **Multiple Outputs**: Console and file logging with automatic rotation
- **Progress Tracking**: Real-time progress bars with tqdm for long-running operations
- **Performance Monitoring**: Memory usage and timing information for optimization

#### Usage Pattern
```python
from forecaster.utils.logger import get_logger

logger = get_logger(__name__, level="INFO")
logger.info("Processing started")
logger.log_step_completion("Data Loading", 2.5)
```

#### Log Levels
- **DEBUG**: Detailed debugging information
- **INFO**: General information and progress
- **WARNING**: Potential issues that don't stop execution
- **ERROR**: Errors that affect functionality
- **CRITICAL**: Critical errors that may stop execution

### How Data Loading Works

#### DataLoader Architecture
- **Singleton Pattern**: Single instance in main process for consistency
- **Worker Process Support**: New instances for parallel processing with preloaded data
- **Caching System**: Memory-based caching with TTL and automatic size management
- **Multiple Storage Types**: CSV, Snowflake, and extensible for other data sources
- **Error Handling**: Graceful fallback mechanisms for data access issues

#### Caching Strategy
- **Main Process**: Full caching enabled with memory limits
- **Worker Processes**: Caching disabled, preloaded data to avoid I/O bottlenecks
- **Memory Management**: Automatic cache size limits and garbage collection
- **TTL**: Time-based cache invalidation to ensure data freshness
- **Performance Optimization**: Efficient data streaming for large datasets

#### Usage Pattern
```python
from data.loader import DataLoader

loader = DataLoader()
product_master = loader.load_product_master()
demand_data = loader.load_outflow(product_master=product_master)
```

### How Configuration Works

#### Configuration Files
- **`data/config/data_config.yaml`**: Main configuration
- **Paths**: File locations and output directories
- **Storage**: Data source configuration
- **Processing**: Parallel processing settings

#### Configuration Structure
```yaml
storage:
  type: csv  # or snowflake
  paths:
    demand_data: data/customer_data/customer_demand.csv
    product_master: data/customer_data/customer_product_master.csv

processing:
  max_workers: 8
  batch_size: 10

safety_stock:
  review_dates: ["2024-01-01", "2024-01-08", ...]
```

### How Parallel Processing Works

#### Processing Modes
- **Vectorized**: Process all dates for one product (for large datasets)
- **Parallel**: Process individual tasks in parallel (for smaller datasets)
- **Automatic Selection**: System chooses optimal mode based on data size and task count

#### Worker Management
- **ProcessPoolExecutor**: CPU-bound tasks
- **ThreadPoolExecutor**: I/O-bound tasks
- **Memory Safety**: Preloaded data for workers to avoid I/O bottlenecks
- **Error Handling**: Graceful failure handling with task isolation
- **Performance Optimization**: Automatic worker count adjustment based on system resources

### How Parameter Optimization Works

#### Method-Specific Optimization
- **Prophet**: Seasonality analysis + parameter tuning
  - Fourier term analysis for seasonality components
  - Hyperparameter optimization for prior scales
  - Holiday and special event detection
  - Automatic fallback mechanisms for optimization failures
- **Moving Average**: No optimization (uses window length)
- **Parameter Caching**: Optimized parameters stored for reuse across runs

#### Optimization Process
1. **Data Preparation**: Prepare training data with validation
2. **Parameter Search**: Try different parameter combinations with error handling
3. **Model Evaluation**: Score each combination using backtesting
4. **Best Selection**: Choose parameters with best performance
5. **Caching**: Store optimized parameters for reuse across multiple runs
6. **Validation**: Verify performance on holdout data
7. **Fallback**: Use default parameters if optimization fails

### How Error Handling Works

#### Error Types
- **DataAccessError**: File/system access issues
- **ValidationError**: Data quality problems
- **ProcessingError**: Computation failures
- **ConfigurationError**: Setup issues
- **ForecastingError**: Model fitting and prediction failures

#### Error Recovery
- **Graceful Degradation**: Continue with available data when possible
- **Fallback Mechanisms**: Use alternative methods (e.g., Prophet → Moving Average)
- **Detailed Logging**: Full error context with stack traces
- **User Feedback**: Clear error messages with actionable recommendations
- **Task Isolation**: Individual task failures don't stop entire workflow

### How Testing Works

#### Test Structure
- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end workflow testing
- **Performance Tests**: Speed and memory testing
- **Data Tests**: Validation and processing tests

#### Running Tests
```bash
# All tests
make test

# With coverage
make test-cov

# Specific categories
uv run python -m pytest forecaster/tests/ -v
```

### Output Files and Results

#### Generated Files
The complete workflow generates several output files in the `output/` directory:

**Backtesting Results**
- `forecast_comparison.csv`: Historical forecast accuracy metrics
- `forecast_visualization.csv`: Detailed forecast data for visualization
- `optimized_parameters.csv`: Best parameters found for each product

**Safety Stock Results**
- `safety_stocks.csv`: Calculated safety stock levels for each review date
- Error distribution analysis and service level calculations

**Simulation Results**
- `simulation_results.csv`: Performance metrics for each product-location
- Inventory level tracking and order placement data

#### Example Output
```
📊 Backtesting Results Summary:
   Total forecasts generated: 51,825
   Average MAPE: 12.3%
   Success rate: 94.8%

🛡️ Safety Stock Summary:
   Products processed: 42
   Review dates: 48
   Average safety stock: 156 units

🎮 Simulation Summary:
   Service level improvement: +8.2%
   Stockout rate reduction: -15.3%
   Inventory days reduction: -12.1%
```

This standardized approach ensures consistency, maintainability, and reliability across the entire system.

## Advanced Features

### Parameter Optimization
- **Caching**: Optimized parameters are cached for reuse across multiple runs
- **Method-specific**: Each forecasting method has specialized optimization strategies
- **Performance tracking**: Optimization results are logged and analyzed for improvement
- **Fallback mechanisms**: Automatic fallback to default parameters if optimization fails

### Parallel Processing Modes
- **Vectorized**: Process all dates for one product (optimal for large datasets)
- **Parallel**: Process individual tasks in parallel (optimal for smaller datasets)
- **Automatic selection**: System chooses optimal mode based on data size and available resources
- **Memory optimization**: Preloaded data for workers to avoid I/O bottlenecks

### Advanced Outlier Detection
- **Zero exclusion**: Automatically excludes zeros when calculating outlier thresholds
- **Multiple methods**: IQR, Z-score, MAD, Rolling statistics, or no detection
- **Configurable thresholds**: Per-product outlier sensitivity settings
- **Statistical validation**: Ensures outlier detection doesn't remove valid data patterns

### Performance & Scalability

#### Memory Management
- **Caching strategy**: Main process caches data, worker processes preload
- **Batch processing**: Configurable batch sizes for large datasets
- **Memory limits**: Automatic cache size management to prevent memory overflow
- **Data streaming**: Efficient data loading for very large datasets

#### Processing Optimization
- **Worker management**: ProcessPoolExecutor for CPU-bound tasks, ThreadPoolExecutor for I/O
- **Data preloading**: Workers receive preloaded data to avoid I/O bottlenecks
- **Error isolation**: Individual task failures don't stop entire workflow
- **Resource monitoring**: Automatic adjustment based on system resources

### Error Handling & Recovery

#### Graceful Degradation
- **Fallback mechanisms**: Prophet falls back to Moving Average if insufficient data
- **Parameter validation**: Automatic parameter adjustment for edge cases
- **Data recovery**: Continue processing with available data when possible
- **Partial results**: Return partial results even if some components fail

#### Comprehensive Logging
- **Progress tracking**: Real-time progress bars with tqdm for long-running operations
- **Error context**: Full error context with stack traces and debugging information
- **Performance metrics**: Timing and memory usage tracking for optimization
- **Workflow awareness**: Step completion tracking and timing analysis

## Configuration Reference

### Product Master Configuration

#### Required Columns
```csv
product_id,location_id,product_category,demand_frequency,risk_period,forecast_window_length,forecast_horizon,forecast_methods,outlier_method,outlier_threshold,distribution,service_level,ss_window_length,leadtime,inventory_cost,moq
```

#### Configuration Options

**Forecasting Methods (`forecast_methods`)**
- `"prophet"`: Facebook's forecasting model with seasonality detection
- `"moving_average"`: Simple average-based forecasting
- `"prophet,moving_average"`: Use both methods and compare results

**Outlier Detection (`outlier_method`)**
- `"iqr"`: Interquartile Range method (default)
- `"zscore"`: Z-score method
- `"mad"`: Median Absolute Deviation method
- `"rolling"`: Rolling statistics method
- `"no"`: No outlier detection (preserves all data)

**Safety Stock Distribution (`distribution`)**
- `"kde"`: Kernel Density Estimation (default, more accurate for non-normal distributions)
- `"normal"`: Normal distribution assumption (faster calculation)

**Service Level (`service_level`)**
- Range: 0.0 to 1.0
- Default: 0.95 (95% service level)
- Higher values = more safety stock

### System Configuration (`data/config/data_config.yaml`)

#### Data Paths
```yaml
paths:
  base_dir: "data/customer_data"
  outflow: "customer_demand.csv"           # Your demand data file
  product_master: "customer_product_master.csv"  # Your product master file
  output_dir: "output"
```

#### Safety Stock Review Dates
```yaml
safety_stock:
  review_dates:             # Dates when safety stocks are calculated
    - "2024-01-01"
    - "2024-01-08"
    - "2024-01-15"
    - "2024-01-22"
    # ... more dates throughout the year
```

**What Review Dates Do:**
- Safety stocks are calculated at each review date
- Uses forecast errors from the period leading up to that date
- **Backtesting Impact**: Analysis start date is automatically calculated based on the first review date and maximum safety stock window length from product master
- **Simulation Impact**: Simulation uses safety stocks calculated at each review date, affecting order timing and quantities
- More review dates = more frequent updates but longer processing
- Default: 1st, 8th, 15th, and 22nd of each month

#### MOQ Constraints
```yaml
simulation:
  enable_moq: true          # Enable/disable Minimum Order Quantity constraints
```

**What MOQ Does:**
- When `true`: Orders must meet the minimum order quantity from product master
- When `false`: Orders can be any positive quantity
- **Impact on Simulation:**
  - `true`: More realistic but may increase inventory levels
  - `false`: More flexible but may not reflect real business constraints

### Advanced Configuration Examples

#### Custom Review Dates Impact
```yaml
safety_stock:
  review_dates: ["2024-01-01", "2024-01-08", "2024-01-15"]
  # More dates = more frequent updates but longer processing
  # Affects both backtesting analysis period and simulation order timing
```

#### MOQ Constraints
```yaml
simulation:
  enable_moq: true  # Orders must meet minimum quantity
  # true: More realistic but may increase inventory
  # false: More flexible but may not reflect real constraints
```

#### Performance Tuning
```yaml
processing:
  max_workers: 8        # Adjust based on CPU cores
  batch_size: 10        # Larger = more memory, faster processing
  vectorized_mode: true # Enable for large datasets
```

### Command Line Options

#### Basic Usage
```bash
python run_complete_workflow.py
```

#### Advanced Options
```bash
python run_complete_workflow.py \
  --analysis-start-date 2023-01-01 \
  --analysis-end-date 2023-12-31 \
  --max-workers 4 \
  --log-level WARNING \
  --product-master-file custom_product_master.csv
```

#### Available Flags
- `--analysis-start-date`: Custom start date for backtesting
- `--analysis-end-date`: Custom end date for backtesting
- `--max-workers`: Number of parallel workers (default: 8)
- `--log-level`: Logging level (INFO, WARNING, ERROR)
- `--product-master-file`: Custom product master file path
- `--backtesting-enabled`: Enable/disable backtesting (default: True)
- `--safety-stock-enabled`: Enable/disable safety stock calculation (default: True)
- `--simulation-enabled`: Enable/disable simulation (default: True)

### Product Filtering

To run analysis on only specific products:

1. **Edit Product Master**: Remove unwanted products from the CSV file
2. **Create Subset**: Create a smaller product master file
   ```bash
   # Create subset for testing
   head -5 data/customer_data/customer_product_master.csv > subset.csv
   python run_complete_workflow.py --product-master-file subset.csv
   ```

**Benefits:**
- **Memory Reduction**: 76% reduction in memory usage for 5 vs 21 products
- **Faster Processing**: Reduced validation, backtesting, and simulation time
- **Focused Analysis**: Concentrate on specific product categories or locations

### Outlier Detection Details

#### Zero Exclusion
The system automatically excludes zeros when calculating outlier thresholds:
- **Problem**: Zeros skew statistics and cause false outliers
- **Solution**: Calculate thresholds using only non-zero demand values
- **Result**: More accurate outlier detection for demand data with many zeros

#### Example
```python
# Data with many zeros
demand_data = [0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 50, 60, 70, 80, 90, 100]

# Old method (including zeros): Values 50+ flagged as outliers
# New method (excluding zeros): Only truly extreme values flagged
```

## Troubleshooting

### Common Issues

#### Memory Errors
- **Symptom**: `MemoryError` or system becomes unresponsive
- **Solution**: Reduce `batch_size` and `max_workers` in configuration
- **Example**: `--batch-size 5 --max-workers 4`

#### Slow Processing
- **Symptom**: Workflow takes much longer than expected
- **Solution**: Enable vectorized mode for large datasets
- **Configuration**: Set `vectorized_mode: true` in processing settings

#### Data Quality Issues
- **Symptom**: Validation errors or poor forecast accuracy
- **Solution**: Run data validation first and fix reported issues
- **Command**: `python run_data_validation.py --log-level INFO`

#### Web Interface Not Loading
- **Symptom**: Cannot access http://localhost:8080
- **Solution**: Check if port 8080 is available, or change port in `webapp/run.py`
- **Alternative**: Use `--port 8081` when starting the web app

### Performance Tuning

#### For Large Datasets (>1000 products)
- **Use vectorized processing**: `--vectorized-mode true`
- **Increase batch size**: `--batch-size 20`
- **Optimize workers**: Set `max_workers` to 75% of CPU cores
- **Enable caching**: Ensure output directory is writable

#### For Limited Memory Systems
- **Reduce batch size**: `--batch-size 5`
- **Limit workers**: `--max-workers 2`
- **Disable caching**: Set `cache_enabled: false` in config
- **Process in chunks**: Use product filtering to process subsets

#### For Fast Processing
- **Increase workers**: Set `max_workers` to number of CPU cores
- **Larger batches**: `--batch-size 20` or higher
- **Disable logging**: `--log-level ERROR` for minimal output
- **Use SSD storage**: Faster I/O for data loading

### Debugging

#### Enable Detailed Logging
```bash
python run_complete_workflow.py --log-level DEBUG
```

#### Check Data Quality
```bash
python run_data_validation.py --log-level INFO
```

#### Test Individual Components
```bash
# Test backtesting only
python run_unified_backtest.py --analysis-start-date 2024-01-01 --analysis-end-date 2024-01-31

# Test safety stock calculation only
python run_safety_stock_calculation.py

# Test simulation only
python run_simulation.py
```

#### Monitor System Resources
- **Memory usage**: Monitor with `htop` or `top`
- **CPU usage**: Check for bottlenecks with `iostat`
- **Disk I/O**: Monitor with `iotop` for data loading issues

### Getting Help

#### Check Logs
- **Workflow logs**: `output/logs/workflow.log`
- **Validation logs**: `output/logs/validation.log`
- **Backtesting logs**: `output/logs/backtesting.log`

#### Common Error Messages
- **"No data found"**: Check file paths and data format
- **"Insufficient data"**: Ensure minimum data requirements are met
- **"Memory error"**: Reduce batch size and workers
- **"Validation failed"**: Fix data quality issues before proceeding

#### Performance Benchmarks
- **Small dataset** (<100 products): 5-15 minutes
- **Medium dataset** (100-500 products): 15-60 minutes
- **Large dataset** (>500 products): 1-4 hours
- **Very large dataset** (>1000 products): 4+ hours (consider subsetting)
