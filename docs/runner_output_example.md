# Forecasting Pipeline Runner - Output Example

This document shows what the output of the forecasting pipeline runner looks like, including console output, generated files, and their contents.

## Console Output

When you run the pipeline, you'll see detailed logging output like this:

```
2025-07-22 04:33:38 - forecaster.runner.pipeline - INFO - Starting forecasting pipeline
2025-07-22 04:33:38 - forecaster.runner.pipeline - INFO - Configuration: {
    'data_dir': 'forecaster/data/dummy', 
    'demand_file': 'sku_demand_daily.csv', 
    'product_master_file': 'product_master_daily.csv', 
    'output_dir': 'output', 
    'run_date': '2023-08-15', 
    'demand_frequency': 'd', 
    'batch_size': 5, 
    'max_workers': 2, 
    'validate_data': True, 
     
    'aggregation_enabled': True, 
    'forecasting_enabled': True, 
    'forecast_model': 'moving_average', 
    'log_level': 'INFO'
}
2025-07-22 04:33:38 - forecaster.runner.pipeline - INFO - Step 1: Loading and validating data
2025-07-22 04:33:38 - forecaster.runner.pipeline - INFO - Loading demand data from: forecaster/data/dummy/sku_demand_daily.csv
2025-07-22 04:33:38 - forecaster.runner.pipeline - INFO - Loaded 36500 demand records
2025-07-22 04:33:38 - forecaster.runner.pipeline - INFO - Loading product master data from: forecaster/data/dummy/product_master_daily.csv
2025-07-22 04:33:38 - forecaster.runner.pipeline - INFO - Loaded 50 product master records
2025-07-22 04:33:38 - forecaster.runner.pipeline - INFO - Validating product master coverage
2025-07-22 04:33:38 - forecaster.runner.pipeline - INFO - All product-location combinations covered in product master
2025-07-22 04:33:38 - forecaster.runner.pipeline - INFO - Validating demand completeness
2025-07-22 04:33:38 - forecaster.runner.pipeline - INFO - All demand entries are complete
2025-07-22 04:33:38 - forecaster.runner.pipeline - INFO - Step 2: Handling outliers
2025-07-22 04:33:38 - forecaster.runner.pipeline - INFO - Processing outliers
2025-07-22 04:33:38 - forecaster.runner.pipeline - INFO - Outlier processing completed: 35035 cleaned records
2025-07-22 04:33:38 - forecaster.runner.pipeline - INFO - Outlier insights saved to: output/outlier_insights.csv
2025-07-22 04:33:38 - forecaster.runner.pipeline - INFO - Outlier summary: {
    'total_outliers': 1465, 
    'products_with_outliers': 10, 
    'locations_with_outliers': 5, 
    'total_original_demand': 229588.33, 
    'total_replaced_demand': 212102.72, 
    'demand_reduction': 17485.61, 
    'outlier_methods': {'zscore': 1023, 'mad': 440, 'iqr': 2}, 
    'date_range': {
        'earliest': '2022-02-09 00:00:00', 
        'latest': '2023-12-29 00:00:00'
    }
}
2025-07-22 04:33:38 - forecaster.runner.pipeline - INFO - Step 3: Aggregating data
2025-07-22 04:33:38 - forecaster.runner.pipeline - INFO - Aggregating data for run date: 2023-08-15
2025-07-22 04:33:40 - forecaster.runner.pipeline - INFO - Aggregation completed: 4200 aggregated records
2025-07-22 04:33:40 - forecaster.runner.pipeline - INFO - Step 4: Generating forecasts
2025-07-22 04:33:40 - forecaster.runner.pipeline - INFO - Starting forecast generation
2025-07-22 04:33:40 - forecaster.runner.pipeline - INFO - Found 50 product-location combinations
2025-07-22 04:33:40 - forecaster.runner.pipeline - INFO - Using parallel processing
2025-07-22 04:33:41 - forecaster.runner.pipeline - INFO - Combined 50 forecasts from 10 batches
2025-07-22 04:33:41 - forecaster.runner.pipeline - INFO - Forecast generation completed: {
    'total_batches': 10, 
    'successful_batches': 10, 
    'failed_batches': 0, 
    'success_rate': 1.0, 
    'average_processing_time': 0, 
    'total_product_locations': 50
}
2025-07-22 04:33:41 - forecaster.runner.pipeline - INFO - Step 5: Saving results
2025-07-22 04:33:41 - forecaster.runner.pipeline - INFO - Outlier data saved to: output/demand_outliers_removed.csv
2025-07-22 04:33:41 - forecaster.runner.pipeline - INFO - Aggregated data saved to: output/demand_aggregated.csv
2025-07-22 04:33:41 - forecaster.runner.pipeline - INFO - Forecasts saved to: output/forecasts.csv
2025-07-22 04:33:41 - forecaster.runner.pipeline - INFO - Forecasting pipeline completed successfully
2025-07-22 04:33:41 - forecaster.runner.pipeline - INFO - Total execution time: 3.00 seconds
```

## Generated Files

The pipeline generates several output files in the `output/` directory:

### 1. Forecasts File (`forecasts.csv`)

This is the main output containing all generated forecasts:

```csv
product_id,location_id,forecast_date,model,window_length,horizon,forecast_values,forecast_mean
PROD_001,LOC_001,2023-08-15,moving_average,120,10,"[542.86, 542.86, 542.86, 542.86, 542.86, 542.86, 542.86, 542.86, 542.86, 542.86]",542.86
PROD_001,LOC_002,2023-08-15,moving_average,120,10,"[595.93, 595.93, 595.93, 595.93, 595.93, 595.93, 595.93, 595.93, 595.93, 595.93]",595.93
PROD_001,LOC_003,2023-08-15,moving_average,120,10,"[677.12, 677.12, 677.12, 677.12, 677.12, 677.12, 677.12, 677.12, 677.12, 677.12]",677.12
PROD_002,LOC_001,2023-08-15,moving_average,100,10,"[473.73, 473.73, 473.73, 473.73, 473.73, 473.73, 473.73, 473.73, 473.73, 473.73]",473.73
PROD_007,LOC_001,2023-08-15,moving_average,100,15,"[267.60, 267.60, 267.60, 267.60, 267.60, 267.60, 267.60, 267.60, 267.60, 267.60, 267.60, 267.60, 267.60, 267.60, 267.60]",267.60
```

**Columns:**
- `product_id`: Product identifier
- `location_id`: Location identifier  
- `forecast_date`: Date when forecast was generated
- `model`: Forecasting model used (e.g., "moving_average")
- `window_length`: Number of periods used for the moving average
- `horizon`: Number of periods forecasted into the future
- `forecast_values`: Array of forecast values for each period
- `forecast_mean`: Average of all forecast values

### 2. Aggregated Data File (`demand_aggregated.csv`)

Contains demand data aggregated into risk-period buckets:

```csv
product_id,location_id,product_category,bucket_start_date,bucket_end_date,bucket_size_days,demand_frequency,risk_period,total_demand,avg_demand,min_demand,max_demand,std_demand,demand_records,last_stock_level,bucket_completeness
PROD_001,LOC_001,ELECTRONICS,2022-01-04,2022-01-10,7,d,7,326.85,46.69,30.76,61.18,11.49,7,113.65,1.0
PROD_001,LOC_001,ELECTRONICS,2022-01-11,2022-01-17,7,d,7,310.82,44.40,25.63,63.46,11.62,7,149.81,1.0
PROD_001,LOC_001,ELECTRONICS,2022-01-18,2022-01-24,7,d,7,316.66,45.24,26.59,62.09,11.18,7,166.82,1.0
```

**Columns:**
- `product_id`, `location_id`, `product_category`: Product and location identifiers
- `bucket_start_date`, `bucket_end_date`: Start and end dates of the aggregation bucket
- `bucket_size_days`: Number of days in the bucket
- `demand_frequency`: Original demand frequency ('d' for daily)
- `risk_period`: Risk period in days (from product master)
- `total_demand`: Sum of all demand in the bucket
- `avg_demand`, `min_demand`, `max_demand`, `std_demand`: Statistical measures
- `demand_records`: Number of demand records in the bucket
- `last_stock_level`: Stock level at the end of the bucket
- `bucket_completeness`: Completeness ratio (1.0 = complete bucket)

### 3. Outlier Insights File (`outlier_insights.csv`)

Details about outliers that were detected and handled:

```csv
product_id,product_category,location_id,date,demand,stock_level,outlier_handled,outlier_method,outlier_threshold,original_demand
PROD_005,FOOD,LOC_004,2022-02-09,143.43,181.03,True,mad,1.2,148.07
PROD_002,CLOTHING,LOC_001,2022-02-25,93.45,105.01,True,zscore,1.5,95.26
PROD_006,FOOD,LOC_004,2022-03-02,143.69,143.23,True,mad,1.5,147.75
```

**Columns:**
- `product_id`, `product_category`, `location_id`: Product and location identifiers
- `date`: Date of the outlier
- `demand`: Demand value after outlier handling (capped)
- `stock_level`: Stock level on that date
- `outlier_handled`: Whether outlier was handled (True/False)
- `outlier_method`: Method used (zscore, mad, iqr, rolling)
- `outlier_threshold`: Threshold used for detection
- `original_demand`: Original demand value before capping

### 4. Cleaned Demand Data (`demand_outliers_removed.csv`)

The demand data after outlier handling (large file with all cleaned demand records).

### 5. Log Files (`forecast_run_YYYYMMDD_HHMMSS.log`)

Detailed execution logs with timestamps for debugging and monitoring.

## Summary Statistics

The pipeline provides comprehensive summary statistics:

### Outlier Summary
- **Total outliers detected**: 1,465
- **Products with outliers**: 10
- **Locations with outliers**: 5
- **Total original demand**: 229,588.33
- **Total replaced demand**: 212,102.72
- **Demand reduction**: 17,485.61 (7.6% reduction)
- **Outlier methods used**: zscore (1,023), mad (440), iqr (2)

### Processing Statistics
- **Total batches**: 10
- **Successful batches**: 10
- **Failed batches**: 0
- **Success rate**: 100%
- **Total product-location combinations**: 50
- **Total execution time**: 3.00 seconds

### Data Summary
- **Input demand records**: 36,500
- **Product master records**: 50
- **Outlier-cleaned records**: 35,035
- **Aggregated records**: 4,200
- **Generated forecasts**: 50

## Key Features

1. **Parallel Processing**: Supports both parallel and sequential processing
2. **Batch Processing**: Processes product-location combinations in configurable batches
3. **Comprehensive Logging**: Detailed logs for monitoring and debugging
4. **Multiple Output Formats**: CSV files for easy analysis and integration
5. **Statistical Insights**: Detailed outlier and processing statistics
6. **Configurable Parameters**: Window length, horizon, outlier methods, etc.
7. **Data Validation**: Ensures data quality and completeness
8. **Risk Period Aggregation**: Aggregates data into appropriate time buckets

## Usage Example

```python
from forecaster.runner import run_pipeline, RunnerConfig

# Run with default configuration
result = run_pipeline()

# Or with custom configuration
config = RunnerConfig(
    run_date=date(2023, 8, 15),
    batch_size=5,
    max_workers=2,
    forecast_model='moving_average'
)
result = run_pipeline(config)

print(f"Generated {result['data_summary']['forecast_records']} forecasts")
print(f"Execution time: {result['total_execution_time']:.2f} seconds")
```

This comprehensive output structure makes the forecasting pipeline suitable for production use with proper monitoring, debugging, and analysis capabilities. 