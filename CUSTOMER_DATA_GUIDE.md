# Customer Data Integration Guide

## Overview
This guide explains how to run the backtesting module on real customer data, including data preparation, configuration, and execution.

## 1. Data Requirements

### Demand Data File (CSV)
Your demand data must have the following columns:
- `product_id` (string): Unique product identifier
- `product_category` (string): Product category/classification
- `location_id` (string): Unique location identifier
- `date` (datetime): Date of the demand record (YYYY-MM-DD format)
- `demand` (numeric): Demand quantity (must be ≥ 0)
- `stock_level` (numeric): Stock level at the end of the day (must be ≥ 0)

**Example:**
```csv
product_id,product_category,location_id,date,demand,stock_level
PROD_001,Electronics,LOC_001,2023-01-01,150.5,1200.0
PROD_001,Electronics,LOC_001,2023-01-02,180.2,1100.0
PROD_002,Clothing,LOC_002,2023-01-01,75.0,500.0
```

### Product Master File (CSV)
Your product master must have the following columns:
- `product_id` (string): Unique product identifier (must match demand data)
- `location_id` (string): Unique location identifier (must match demand data)
- `product_category` (string): Product category (must match demand data)
- `demand_frequency` (string): 'd' (daily), 'w' (weekly), or 'm' (monthly)
- `risk_period` (integer): Risk period as integer multiple of demand frequency
- `forecast_window_length` (integer): Forecasting window length in risk periods
- `forecast_horizon` (integer): Forecasting horizon in risk periods
- `outlier_method` (string, optional): 'iqr', 'zscore', 'mad', 'rolling', or 'no' (default: 'iqr')
- `outlier_threshold` (float, optional): Outlier detection threshold (default: 1.5)

**Example:**
```csv
product_id,location_id,product_category,demand_frequency,risk_period,forecast_window_length,forecast_horizon,outlier_method,outlier_threshold
PROD_001,LOC_001,Electronics,d,7,100,10,iqr,1.5
PROD_002,LOC_002,Clothing,d,7,80,8,zscore,2.0
PROD_003,LOC_003,Food,d,7,60,5,no,1.5
```

## 2. Data Validation Rules

### Demand Data Validation
- All required columns must be present
- Date column must be in datetime format
- Demand and stock_level must be numeric and non-negative
- No missing values in required columns
- Product-location combinations must exist in product master

### Product Master Validation
- All required columns must be present
- Demand frequency must be 'd', 'w', or 'm'
- Risk period must be positive and reasonable:
  - Daily: ≤ 365 days
  - Weekly: ≤ 52 weeks
  - Monthly: ≤ 12 months
- All product-location combinations in demand data must exist in product master

## 3. Configuration Setup

### Option 1: Direct Configuration (Recommended for testing)
Create a Python script to configure and run the backtesting:

```python
from forecaster.backtesting.config import BacktestConfig
from forecaster.backtesting.backtester import Backtester
from datetime import date

# Configure for your customer data
config = BacktestConfig(
    # Data paths
    data_dir="path/to/your/customer/data",
    demand_file="customer_demand.csv",
    product_master_file="customer_product_master.csv",
    output_dir="output/customer_backtest",
    
    # Backtesting parameters
    historic_start_date=date(2022, 1, 1),  # Start of your historical data
    analysis_start_date=date(2023, 1, 1),  # When to start backtesting
    analysis_end_date=date(2023, 12, 31),  # When to end backtesting
    
    # Forecasting parameters
    demand_frequency="d",  # 'd', 'w', or 'm' based on your data
    forecast_model="moving_average",
    default_horizon=2,  # Default horizon for backtesting
    
    # Processing settings
    batch_size=10,
    max_workers=4,  # Adjust based on your system
    validate_data=True,
    outlier_enabled=True,
    aggregation_enabled=True,
    
    # Logging
    log_level="INFO"
)

# Run backtesting
backtester = Backtester(config)
results = backtester.run()
```

### Option 2: Environment Variables
Set environment variables and use the default configuration:

```bash
export FORECASTER_DATA_DIR="path/to/your/customer/data"
export FORECASTER_DEMAND_FILE="customer_demand.csv"
export FORECASTER_PRODUCT_MASTER_FILE="customer_product_master.csv"
export FORECASTER_OUTPUT_DIR="output/customer_backtest"
export FORECASTER_ANALYSIS_START_DATE="2023-01-01"
export FORECASTER_ANALYSIS_END_DATE="2023-12-31"
export FORECASTER_DEMAND_FREQUENCY="d"
export FORECASTER_MAX_WORKERS="4"
```

## 4. Data Preparation Checklist

### Before Running
- [ ] Verify your demand data has all required columns
- [ ] Verify your product master has all required columns
- [ ] Check that all product-location combinations in demand data exist in product master
- [ ] Ensure date format is consistent (YYYY-MM-DD)
- [ ] Remove any rows with missing values in required columns
- [ ] Verify demand and stock_level are non-negative
- [ ] Check that risk_period values are reasonable for your business
- [ ] Ensure demand_frequency matches your actual data frequency

### Data Quality Checks
```python
import pandas as pd

# Load your data
demand_df = pd.read_csv("customer_demand.csv")
product_master_df = pd.read_csv("customer_product_master.csv")

# Check for missing values
print("Demand data missing values:")
print(demand_df.isnull().sum())

print("\nProduct master missing values:")
print(product_master_df.isnull().sum())

# Check date range
print(f"\nDemand data date range: {demand_df['date'].min()} to {demand_df['date'].max()}")

# Check product-location coverage
demand_combinations = set(zip(demand_df['product_id'], demand_df['location_id']))
master_combinations = set(zip(product_master_df['product_id'], product_master_df['location_id']))
missing_in_master = demand_combinations - master_combinations
print(f"\nProduct-location combinations missing in product master: {len(missing_in_master)}")
```

## 5. Running the Backtesting

### Step 1: Test with Small Dataset
Start with a subset of your data to verify everything works:

```python
# Create a test configuration with limited data
test_config = BacktestConfig(
    data_dir="path/to/your/customer/data",
    demand_file="customer_demand_sample.csv",  # Small sample
    product_master_file="customer_product_master.csv",
    output_dir="output/test_backtest",
    analysis_start_date=date(2023, 6, 1),
    analysis_end_date=date(2023, 6, 30),  # Short period
    max_workers=1  # Single worker for testing
)

# Run test
backtester = Backtester(test_config)
results = backtester.run()
```

### Step 2: Full Backtesting Run
Once the test works, run the full backtesting:

```python
# Full configuration
full_config = BacktestConfig(
    data_dir="path/to/your/customer/data",
    demand_file="customer_demand.csv",
    product_master_file="customer_product_master.csv",
    output_dir="output/full_backtest",
    analysis_start_date=date(2023, 1, 1),
    analysis_end_date=date(2023, 12, 31),
    max_workers=4  # Adjust based on your system
)

# Run full backtesting
backtester = Backtester(full_config)
results = backtester.run()
```

## 6. Output Files

The backtesting will generate three main output files:

### `backtest_results.csv`
- Summary of all forecasts generated
- Columns: analysis_date, cutoff_date, product_id, location_id, model, window_length, horizon, forecast_values, forecast_mean, risk_period, demand_frequency

### `forecast_comparison.csv`
- Detailed comparison of forecasts vs actuals
- Columns: analysis_date, risk_period_start, risk_period_end, product_id, location_id, step, actual_demand, forecast_demand, error, absolute_error, percentage_error, risk_period, demand_frequency

### `accuracy_metrics.csv`
- Aggregated accuracy metrics for each product-location combination
- Columns: analysis_date, product_id, location_id, horizon, mae, mape, rmse, bias, actual_demands, forecast_values, risk_period, demand_frequency

## 7. Common Issues and Solutions

### Issue: "Missing required columns"
**Solution**: Check that your CSV files have all required columns with exact names.

### Issue: "Invalid demand frequencies"
**Solution**: Ensure demand_frequency column only contains 'd', 'w', or 'm'.

### Issue: "Risk period values must be positive"
**Solution**: Check that all risk_period values in product master are positive integers.

### Issue: "Product-location combinations missing in product master"
**Solution**: Ensure all product-location combinations in demand data exist in product master.

### Issue: "Date column must be datetime type"
**Solution**: Ensure date column is in YYYY-MM-DD format or use pd.to_datetime() to convert.

### Issue: "Demand values cannot be negative"
**Solution**: Check for negative demand values and either correct them or set to 0.

## 8. Performance Optimization

### For Large Datasets
- Increase `max_workers` based on your CPU cores (typically 4-8)
- Adjust `batch_size` based on memory availability (10-50)
- Consider running on a subset of data first

### Memory Management
- Monitor memory usage during large runs
- Reduce `batch_size` if you encounter memory issues
- Consider processing data in chunks for very large datasets

## 9. Example Customer Data Script

```python
#!/usr/bin/env python3
"""
Example script for running backtesting on customer data.
"""

from forecaster.backtesting.config import BacktestConfig
from forecaster.backtesting.backtester import Backtester
from datetime import date
import os

def run_customer_backtest():
    """Run backtesting on customer data."""
    
    # Configuration for customer data
    config = BacktestConfig(
        # Update these paths to your actual data
        data_dir="customer_data",
        demand_file="demand_data.csv",
        product_master_file="product_master.csv",
        output_dir="output/customer_backtest",
        
        # Update these dates to match your data
        historic_start_date=date(2022, 1, 1),
        analysis_start_date=date(2023, 1, 1),
        analysis_end_date=date(2023, 12, 31),
        
        # Update frequency based on your data
        demand_frequency="d",  # 'd', 'w', or 'm'
        
        # Processing settings
        batch_size=20,
        max_workers=4,
        validate_data=True,
        outlier_enabled=True,
        aggregation_enabled=True,
        
        # Logging
        log_level="INFO"
    )
    
    # Validate configuration
    if not config.validate_dates():
        raise ValueError("Invalid date configuration")
    
    # Check if data files exist
    if not config.get_demand_file_path().exists():
        raise FileNotFoundError(f"Demand file not found: {config.get_demand_file_path()}")
    
    if not config.get_product_master_file_path().exists():
        raise FileNotFoundError(f"Product master file not found: {config.get_product_master_file_path()}")
    
    # Run backtesting
    print("Starting customer backtesting...")
    backtester = Backtester(config)
    results = backtester.run()
    
    print(f"Backtesting completed!")
    print(f"Generated {results.get('total_forecasts', 0)} forecasts")
    print(f"Results saved to: {config.output_dir}")
    
    return results

if __name__ == "__main__":
    run_customer_backtest()
```

## 10. Next Steps

1. **Prepare your data** according to the schema requirements
2. **Test with a small subset** to verify everything works
3. **Run the full backtesting** on your complete dataset
4. **Analyze the results** using the generated CSV files
5. **Adjust parameters** in product master based on results
6. **Iterate and improve** your forecasting configuration

For any issues or questions, check the log files in the output directory for detailed error messages. 