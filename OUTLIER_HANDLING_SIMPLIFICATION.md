# Outlier Handling Simplification

## Overview
This document summarizes the removal of the `--no-outliers` command line option and the simplification of outlier handling to use only the product master configuration.

## Changes Made

### 1. Removed `--no-outliers` Command Line Option

**Files Updated:**
- `run_complete_workflow.py`
- `run_unified_backtest.py`

**Changes:**
- Removed `--no-outliers` argument from argument parsers
- Removed `outlier_enabled` parameter from function signatures
- Removed conditional logic that checked `outlier_enabled` flag
- Removed `--no-outliers` flag from command construction

### 2. Removed `outlier_enabled` Configuration Parameter

**Files Updated:**
- `forecaster/backtesting/config.py`
- `forecaster/backtesting/unified_backtester.py`

**Changes:**
- Removed `outlier_enabled: bool = True` from `BacktestConfig` class
- Removed logging output that displayed outlier handling status
- Removed parameter passing for `outlier_enabled`

### 3. Updated Documentation

**Files Updated:**
- `docs/API.md`
- `CUSTOMER_DATA_GUIDE.md`
- `docs/runner_output_example.md`
- `UNIFIED_PIPELINE_REFACTOR.md`

**Changes:**
- Removed `outlier_enabled=True` from configuration examples
- Updated command examples to remove `--no-outliers` flag
- Cleaned up documentation to reflect simplified approach

## How Outlier Handling Works Now

### Product Master Configuration
Outlier handling is now controlled entirely through the product master file:

```csv
product_id,location_id,outlier_method,outlier_threshold
PROD_001,LOC_001,iqr,1.5
PROD_002,LOC_002,zscore,3.0
PROD_003,LOC_003,no,1.5
```

### Available Outlier Methods
- `iqr`: Interquartile Range method (default)
- `zscore`: Z-score method
- `mad`: Median Absolute Deviation method
- `rolling`: Rolling statistics method
- `no`: No outlier detection (preserves all data)

### Default Behavior
- If `outlier_method` is not specified in product master, defaults to `iqr`
- If `outlier_threshold` is not specified, defaults to `1.5`
- Outlier handling always runs during backtesting (no global disable)

## Benefits of This Change

1. **Simplified Configuration**: No need for command line flags
2. **Product-Specific Control**: Each product can have different outlier settings
3. **Consistent Behavior**: Outlier handling always runs, controlled by product master
4. **Reduced Complexity**: Fewer configuration options to manage
5. **Better Data Control**: Users can disable outliers for specific products using `outlier_method = 'no'`

## Migration Guide

### For Users Previously Using `--no-outliers`

**Before:**
```bash
python run_complete_workflow.py --no-outliers
```

**After:**
Set `outlier_method = 'no'` in your product master file for products where you don't want outlier detection:

```csv
product_id,location_id,outlier_method,outlier_threshold
PROD_001,LOC_001,no,1.5
```

### For Users with Custom Outlier Settings

No changes needed - your existing product master configurations will continue to work as before.

## Testing

The changes have been tested to ensure:
1. **Backward Compatibility**: Existing product master configurations work unchanged
2. **Functionality**: Outlier detection methods (iqr, zscore, mad, rolling, no) all work correctly
3. **Integration**: The complete workflow runs without issues
4. **Documentation**: All examples and guides are updated

## Impact on Pipeline

- **No Breaking Changes**: All existing functionality remains intact
- **Simplified Interface**: Fewer command line options to manage
- **Better Control**: Product-specific outlier handling through product master
- **Consistent Behavior**: Outlier handling always runs, controlled by product configuration

## Usage Recommendations

- Use `outlier_method = 'no'` in product master for products where:
  - All demand data should be preserved
  - Outliers represent real business events
  - Data quality is high and outliers are rare

- Use other methods (iqr, zscore, mad, rolling) for products where:
  - Outlier detection improves forecasting accuracy
  - Data quality issues are common
  - Outliers represent data errors rather than real demand 