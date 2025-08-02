# Product Filtering Guide

## Overview

The Forecaster system now supports running the complete workflow for only certain products by modifying only the product master file. This allows you to:

- **Reduce memory usage** by loading only relevant demand data
- **Speed up processing** by analyzing only selected products
- **Focus analysis** on specific product categories or locations
- **Maintain data integrity** without modifying the original customer demand file

## How It Works

The system now filters customer demand data at load time based on the product-location combinations present in the product master file. This filtering is applied across all components:

1. **Data Validation** - Only validates products in the product master
2. **Backtesting** - Only processes products in the product master
3. **Safety Stock Calculation** - Uses results from filtered backtesting
4. **Simulation** - Only simulates products in the product master
5. **Web Interface** - Only displays products in the product master

## Usage

### Method 1: Modify the Product Master File

Simply edit `forecaster/data/customer_product_master.csv` to include only the products you want to analyze:

```csv
product_id,location_id,forecast_methods,risk_period,demand_frequency,lead_time,forecast_window_length,forecast_horizon
BPRN,WB,"prophet,arima",7,d,14,4,1
BPWP,WB,"prophet",7,d,14,4,1
BPRQ,WB,"arima",7,d,14,4,1
```

### Method 2: Create a Subset Product Master

Create a new product master file with only the desired products:

```bash
# Example: Create a subset for testing
head -5 forecaster/data/customer_product_master.csv > forecaster/data/product_master_subset.csv
```

Then run the workflow with the subset:

```bash
python run_complete_workflow.py --product-master-file product_master_subset.csv
```

## Benefits

### Memory Efficiency
- **Before**: Loads entire customer demand file (14,040 records in example)
- **After**: Loads only relevant records (3,345 records for 5 products = 76% reduction)

### Processing Speed
- **Faster validation** - Only validates selected products
- **Faster backtesting** - Only processes selected products
- **Faster simulation** - Only simulates selected products
- **Faster web interface** - Only loads selected products

### Data Integrity
- **No modification** of original customer demand file
- **Consistent filtering** across all components
- **Easy to switch** between different product sets

## Example Results

Based on the test with 5 products vs 21 products:

```
📊 Summary:
   Total demand records: 14,040
   Filtered demand records: 3,345
   Reduction: 10,695 records (76.2%)
   Total product-location combinations: 21
   Filtered product-location combinations: 5
   Reduction: 16 combinations
```

## Testing the Functionality

Run the test scripts to verify filtering works correctly:

```bash
# Test basic filtering
python test_filtering.py

# Test subset filtering
python test_filtering_subset.py
```

## Implementation Details

### Modified Components

1. **`forecaster/data/loader.py`**
   - Added `load_customer_demand_filtered()` method
   - Enhanced `load_customer_demand()` with filtering option

2. **`run_data_validation.py`**
   - Now loads demand data filtered by product master

3. **`forecaster/backtesting/unified_backtester.py`**
   - Now loads demand data filtered by product master

4. **`forecaster/simulation/data_loader.py`**
   - Now loads demand data filtered by product master

5. **`webapp/app.py`**
   - Now loads demand data filtered by product master

### Filtering Logic

The filtering works by:
1. Loading the product master file
2. Extracting unique product-location combinations
3. Filtering the customer demand data to include only those combinations
4. Processing only the filtered data throughout the workflow

## Best Practices

1. **Backup your original product master** before making changes
2. **Test with a small subset** first to verify results
3. **Use descriptive filenames** for different product master subsets
4. **Document your product selections** for reproducibility

## Troubleshooting

### Common Issues

1. **No demand data found for products**
   - Verify product-location combinations exist in customer demand file
   - Check for typos in product_id or location_id

2. **Unexpected filtering results**
   - Run test scripts to verify filtering logic
   - Check product master file format and content

3. **Memory usage still high**
   - Ensure product master file contains only desired products
   - Verify filtering is working with test scripts

### Verification Commands

```bash
# Check product master content
head -10 forecaster/data/customer_product_master.csv

# Check unique product-location combinations
python -c "
import pandas as pd
df = pd.read_csv('forecaster/data/customer_product_master.csv')
print(f'Total records: {len(df)}')
print(f'Unique combinations: {len(df[["product_id", "location_id"]].drop_duplicates())}')
"

# Test filtering
python test_filtering_subset.py
```

## Conclusion

The new product filtering functionality provides a powerful way to focus your analysis on specific products while maintaining data integrity and improving performance. Simply modify the product master file to control which products are included in the entire workflow. 