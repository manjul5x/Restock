# Restock - Inventory Forecasting & Analysis System

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
product_id,location_id,date,demand,unit_price,stock_level,incoming_inventory,product_category

- `product_id` - Product identifier
- `product_category` - Product category  
- `location_id` - Location identifier
- `date` - Date in YYYY-MM-DD format
- `demand` - Demand quantity (must be ≥ 0)
- `stock_level` - Stock level at end of day (must be ≥ 0)
- `incoming_inventory` - Incoming inventory (can be 0)
- `unit_price` - unit price at date

### Product Master Format (`customer_product_master.csv`)
Required columns:
- `product_id` - Product identifier
- `location_id` - Location identifier
- `product_category` - Product category
- `demand_frequency` - 'd' (daily), 'w' (weekly), or 'm' (monthly)
- `risk_period` - Risk period as integer multiple of demand frequency
- `forecast_window_length` - Forecasting window length in days
- `forecast_horizon` - Forecast horizon in days
- `forecast_methods` - Comma-separated forecasting methods: 'prophet,moving_average'
- `outlier_method` - Outlier detection method: 'iqr', 'zscore', 'mad', 'rolling', or 'no' (default: 'iqr')
- `outlier_threshold` - Outlier detection threshold (default: 1.5)
- `distribution` - Safety stock distribution: 'kde' or 'normal' (default: 'kde')
- `service_level` - Service level percentage 0.0 to 1.0 (default: 0.95)
- `ss_window_length` - Rolling window length for safety stock calculation in demand frequency units (default: 180)
- `leadtime` - Lead time in demand frequency units
- `inventory_cost` - Unit cost of inventory (default: 0.0)
- `moq` - Minimum order quantity (default: 0)
- `min_safety_stock` - Minimum safety stock level (cannot be negative, default: 0.0)
- 'max_safety_stock' - Maximum safety stock level (cannot be negative, default: 0.0)
- `sunset_date` - Date when product is sunset (YYYY-MM-DD format, empty string = not sunset)

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
```uv run
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
- `ss_window_length`: Rolling window for error calculation (days) (default: 180)
- `min_safety_stock`: Minimum safety stock level (cannot be negative, default: 0.0)
- `max_safety_stock`: Maximum safety stock level
- `sunset_date`: Date when product is sunset (YYYY-MM-DD format, empty string = not sunset)

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

**Overstock Management**
- `simulation.overstock_buffer_perc`: Buffer percentage for overstock detection (default: 20%)
- `simulation.overstock_boundary_window`: Time window in days for overstock boundary calculation (default: 30 days)
- `simulation.calc_metrics_after_lt`: Calculate metrics after lead time (default: true)

**Memory Monitoring**
I'm not sure if this is implemented correctly
- `monitoring.track_memory`: Enable memory usage tracking (default: true)
- `monitoring.alert_threshold_mb`: Memory alert threshold in MB (default: 400)
- `monitoring.log_access`: Enable access logging (default: false)

### Running the Complete Workflow

```uv run
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
3. **Safety Stock Calculation** - Calculates safety stock levels for each review date
4. **Inventory Simulation** - Simulates performance
5. **Results Generation** - Creates output files

### Running Individual Components

You can also run individual components separately:

```uv run
# Data validation only
python run_data_validation.py

# New backtesting pipeline with advanced features
python run_backtesting.py --analysis-start-date 2023-01-01 --analysis-end-date 2023-12-31

# Safety stock calculation
python run_safety_stock_calculation.py

# Inventory simulation
python run_simulation.py
```

#### Advanced Backtesting Features

The new backtesting pipeline (`run_backtesting.py`) includes:

**Chunked Persistence**
- Saves results in chunks to prevent data loss during long runs
- Automatic crash recovery and resume functionality
- Configurable chunk sizes for memory optimization

**Resume Capability**
```uv run
# Resume a previous run using the same run ID
python run_backtesting.py --run-id 20250101_120000 --resume
```

**Flexible Configuration**
```bash
# Custom run ID for organization
python run_backtesting.py --run-id my_analysis_2024 --analysis-start-date 2024-01-01

# Adjust worker count for performance
python run_backtesting.py --max-workers 16 --analysis-start-date 2024-01-01
```

### Go to the Web App

```uv run
python webapp/run.py
```

The web app will automatically find an available port and display the URL. Open the displayed URL to access the interactive interface.

#### What to Look At

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

1. **Data validation** The data validation model will check your demand and master table for valid input
2. **Data prep** In the backtesting module, there is a prep input data step. This will add to the demand table columns that will be used for forecasting.
These columnd are called regressors. The primary column is called outflow. Technically this isn't a regressor, but it is treated like one. This is the target
variable for our forecasting. One adds a regressor by adding to the regressor config file and using the functions created. The output of this step is input_data_with_regressors.csv
  2.1 **RP aggregation** our forecasting target for creating safety stocks is the sum of the demand over the upcoming risk period every day. As such, we model the input data to match
  that. So outflow is converted using a rolling lead aggregator. 
3. **Historic forecasts** The system then creates historic forecasts. This is currently called backtesting but should be renamed. The goal is to find out what forecast we would have 
created for all the dates in the past. 
  3.1 The first step is to determine which dates we need to create forecasts for. These are called analysis dates. they are programatically determined based on the review dates, safety stock window period, etc. 
  3.2 The historic forecasts scale compute very quickly so we parrallelise it. Each task is created as all of the required forecasts for a specific product location combination. 
  3.3 The forecasting engine is then called for each analysis date in that task. This engine can be used independently outside of the backtesting flow. 
  3.4 The results of the forecasts are stored in chucks with persistence based on the variables in data_config. At the end of it all, it's pulled to gether into forecast_comparison.csv
4. **Safety Stocks** safety stocks are recalculated for every review date for every product location combination.
  4.1 Safety stocks are there to cover us for the variability in the supply chain. When unexpected things occur, we need extra buffer stock to protect us against running out of stock. The safety stock calculation is there to determine how much extra stock we need. 
  4.2 If our forecast was 100% accurate, we would need to safety stock. Safety stock isn't determined on the volatility of our demand, but the volatility of the accuracy of our forecast. 
  4.3 So we create the relevant data by creating the historic forecasts, and then calculating an error of how accurate our forecast was. This error forms a distribution. 
  4.4 We use a KDE to model that distribution, and a CDF to determine the 95th percentile of that CDF. This is reflective of "95% of the time, our forecasts were at most X units under forecasting. Therefore if we hold X units extra stock, we should not run out of stock 95% of the time." 
  4.5 It's important the the period of the error aligns with the period of demand you need to cover. Pre-aggregating demand to risk period solves for this. 
  4.6 Because we are preaggregating demand, the days which are most recent cannot be used. the actual outflow point on those days is reflective of the sum of the demand from that day until one risk period after that day. This could potentially include data which is 'in the future' for that analysis date. SO the errors which are actually used for the distribution of the safety stock calculation are the safety_stock_window (set in the product master table) days ago (lets say 180) up until one risk period ago. If the risk period is 100 days, then it'll be the forecast errors for the analysis dates that are 180 days before the review date until 80 days before the review date that are used for the safety stock calculation. the last data point (80 days before) will be the forecast made on that day compared to the sum of the actual daily demand from that day up until one day before the review date. 
5. **Simulation** The simulation is done by creating arrays, and stepping through those arrays. 
  5.1 Find the arrays in line 237 of data loader. 
  5.2 Best to just look at the code for this one. 

### How Forecasting and Backtesting Works

#### Forecasting Process
1. **Data Preparation**: Load and validate demand data
2. **Model Fitting**: Fit the model to the training data
3. **Forecast Generation**: Generate predictions for the specified horizon

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
Safety Stock = min(max(Calculated Safety Stock, Minimum Safety Stock), Maximum Safety Stock)
```

Where:
- **Calculated Safety Stock** = 
- **Minimum Safety Stock** = User-defined minimum level (cannot be negative)
- **Maximum Safety Stock** = User-defined maximum level (empty = no maximum, cannot be negative)

The system calculates safety stocks for each product-location-method combination at each review date, ensuring they never fall below the specified minimum or exceed the specified maximum.

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


#### MOQ Constraints
- **Enabled**: Orders must meet minimum order quantity
- **Disabled**: Orders can be any positive quantity
- Affects order timing and quantities

### What's in the Web App

#### Pages and Features


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
This is actually still a bit of a mess and needs to be standardised properly

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

#### Worker Management
- **ProcessPoolExecutor**: CPU-bound tasks
- **ThreadPoolExecutor**: I/O-bound tasks
- **Memory Safety**: Preloaded data for workers to avoid I/O bottlenecks
- **Error Handling**: Graceful failure handling with task isolation
- **Performance Optimization**: Automatic worker count adjustment based on system resources

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

### Output Files and Results

#### Generated Files
The complete workflow generates several output files in the `output/` directory:

**Backtesting Results**
- `forecast_comparison.csv`: Historical forecast accuracy metrics

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
- Default: 1st and 15th of each month from July 2024 to June 2025

#### Simulation Configuration
```yaml
simulation:
  enable_moq: true                    # Enable/disable Minimum Order Quantity constraints
  overstock_buffer_perc: 20          # Buffer percentage for overstock detection
  overstock_boundary_window: 30      # Time window in days for overstock boundary calculation
  calc_metrics_after_lt: true        # Calculate metrics after lead time

monitoring:
  track_memory: true                 # Enable memory usage tracking
  alert_threshold_mb: 400           # Memory alert threshold in MB
  log_access: false                 # Enable access logging
```

**What These Settings Do:**
- **MOQ**: When `true`, orders must meet minimum order quantity from product master
- **Overstock Buffer**: Percentage threshold for detecting overstock situations
- **Overstock Boundary Window**: Days to look ahead for overstock boundary calculation
- **Metrics After Lead Time**: Whether to calculate performance metrics after lead time periods
- **Memory Monitoring**: Tracks memory usage and alerts when threshold is exceeded

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
  batch_size: 10        # Larger = more memory, faster processing I don't think this is used
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

#### New Backtesting Pipeline Flags (`run_backtesting.py`)
- `--run-id`: Custom run identifier for organization and resume functionality
- `--resume`: Resume a previous run using the same run ID
- `--demand-frequency`: Demand frequency (d/w/m, default: d)
- `--profile`: Enable performance profiling

### Product Filtering

To run analysis on only specific products:

1. **Edit Product Master**: Remove unwanted products from the CSV file
2. **Create Subset**: Create a smaller product master file
   ```bash
   # Create subset for testing
   head -5 data/customer_data/customer_product_master.csv > subset.csv
   python run_complete_workflow.py --product-master-file subset.csv
   ```

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


#### Data Quality Issues
- **Symptom**: Validation errors or poor forecast accuracy
- **Solution**: Run data validation first and fix reported issues
- **Command**: `python run_data_validation.py --log-level INFO`

#### Web Interface Not Loading
- **Symptom**: Cannot access http://localhost:8080
- **Solution**: Check if port 8080 is available, or change port in `webapp/run.py`
- **Alternative**: Use `--port 8081` when starting the web app
