# Data Validation System

## Overview

The data validation system has been completely separated from the forecasting pipeline and is now a standalone, comprehensive validation framework. This allows users to validate their data independently before running any forecasting operations.

## Architecture

### Core Components

```
forecaster/validation/
├── __init__.py              # Module exports
├── types.py                 # Data structures and types
├── validator.py             # Main orchestrator
├── schema_validator.py      # Schema validation
├── completeness_validator.py # Data completeness validation
├── quality_validator.py     # Data quality validation
└── coverage_validator.py    # Data coverage validation
```

### Validation Types

1. **Schema Validation**: Validates column structure, data types, and required fields
2. **Completeness Validation**: Checks for missing dates, frequency consistency, and data gaps
3. **Quality Validation**: Identifies negative values, outliers, and statistical anomalies
4. **Coverage Validation**: Ensures product-location combinations exist in both datasets

## Usage

### Standalone Validation

```bash
# Basic validation
python run_data_validation.py

# Custom data files
python run_data_validation.py --demand-file my_demand.csv --product-master-file my_product_master.csv

# Specific validation options
python run_data_validation.py --no-quality --no-completeness

# Strict mode (warnings treated as errors)
python run_data_validation.py --strict-mode

# Save validation report
python run_data_validation.py --output-report validation_report.json
```

### Integrated in Complete Workflow

```bash
# Validation is now Step 1 of the complete workflow
python run_complete_workflow.py
```

## Validation Features

### Schema Validation

**Demand Data Schema:**
- Required columns: `product_id`, `product_category`, `location_id`, `date`, `demand`, `stock_level`
- Data type validation for date, numeric, and string columns
- Missing column detection

**Product Master Schema:**
- Required columns: `product_id`, `location_id`, `product_category`, `demand_frequency`, `risk_period`, `leadtime`
- Valid frequency values: 'd', 'w', 'm'
- Risk period validation (reasonable limits)
- Duplicate product-location detection

### Completeness Validation

**Missing Date Detection:**
- Identifies missing dates in expected date ranges
- Frequency consistency checking
- Large data gap detection (>30 days)

**Date Continuity:**
- Future date detection
- Very old date detection (>10 years)
- Consecutive zero demand periods

### Quality Validation

**Negative Values:**
- Negative demand detection
- Negative stock level detection

**Extreme Values:**
- Statistical outlier detection using IQR method
- Extreme demand and stock level identification

**Statistical Anomalies:**
- Zero variance detection
- Very low variability detection
- Data consistency checks (demand > stock level)

### Coverage Validation

**Product-Location Coverage:**
- Missing combinations in product master
- Extra combinations in product master
- Orphaned products/locations

**Data Volume Coverage:**
- Low volume products (<10 records)
- Low volume locations (<10 records)
- Low volume combinations (<5 records)

## Validation Report

### Console Output

```
📊 COMPREHENSIVE DATA VALIDATION REPORT
================================================================================
✅ Overall Status: VALID
⏱️  Execution Time: 0.03 seconds
📈 Total Issues: 5
🚨 Critical Issues: 0
⚠️  Warnings: 5

📋 VALIDATION SUMMARY:
----------------------------------------
✅ PASS Demand Schema: 0 issues
✅ PASS Product Master Schema: 0 issues
✅ PASS Data Coverage: 0 issues
✅ PASS Data Quality: 3 issues
✅ PASS Data Completeness: 2 issues

🔍 DETAILED ISSUES:
----------------------------------------
📁 DATA QUALITY:
  1. ⚠️ Found 143 consecutive zero demand periods
  2. ⚠️ Found 2 records with extreme demand values
  3. ⚠️ Found 25 records where demand exceeds stock level

📁 DATA COMPLETENESS:
  1. ⚠️ Missing dates for IBWQ-WB (2023-08-25)
  2. ⚠️ Missing dates for RSWQ-WB (2023-08-25)

💡 RECOMMENDATIONS:
----------------------------------------
  1. Data validation passed successfully
```

### JSON Report

The validation system can generate detailed JSON reports:

```json
{
  "overall_valid": true,
  "total_issues": 5,
  "critical_issues": 0,
  "warnings": 5,
  "execution_time": 0.03,
  "demand_validation": {
    "is_valid": true,
    "issues": [],
    "summary": {
      "total_records": 1338,
      "required_columns_present": true
    }
  },
  "completeness_validation": {
    "is_valid": true,
    "issues": [
      {
        "severity": "warning",
        "category": "completeness",
        "message": "Missing dates for IBWQ-WB",
        "details": {
          "missing_dates": ["2023-08-25"],
          "missing_count": 1
        }
      }
    ]
  }
}
```

## Severity Levels

- **CRITICAL**: Must be fixed before proceeding (schema errors, missing required data)
- **ERROR**: Should be addressed (negative values, missing combinations)
- **WARNING**: Should be reviewed (outliers, missing dates, data gaps)
- **INFO**: Informational only (very old dates, low variability)

## Integration with Forecasting Pipeline

The forecasting pipeline now uses the new validation system internally:

```python
# In unified_pipeline.py
from ..validation import DataValidator

validator = DataValidator(log_level=self.config.log_level)
report = validator.validate_data(
    demand_data=self.demand_data,
    product_master_data=self.product_master_data,
    demand_frequency=self.config.demand_frequency
)
```

## Benefits

### For Users

1. **Early Detection**: Validate data before running expensive forecasting operations
2. **Comprehensive Coverage**: Multiple validation types catch different issues
3. **Detailed Reporting**: Clear, actionable feedback on data issues
4. **Flexible Usage**: Run validation independently or as part of workflow
5. **Strict Mode**: Option to treat warnings as errors for quality control

### For Developers

1. **Modular Design**: Easy to add new validation types
2. **Extensible**: Simple to extend with custom validation rules
3. **Reusable**: Validation components can be used independently
4. **Well-Documented**: Clear API and comprehensive documentation
5. **Type Safe**: Strong typing with dataclasses and enums

## Migration from Old System

The old validation logic has been completely replaced:

- **Removed**: Basic validation in `_validate_data()` method
- **Added**: Comprehensive validation system with detailed reporting
- **Enhanced**: Multiple validation types with severity levels
- **Improved**: Better error messages and actionable recommendations

## Future Enhancements

Potential improvements for the validation system:

1. **Custom Validation Rules**: User-defined validation rules
2. **Validation Profiles**: Predefined validation configurations
3. **Data Quality Scoring**: Quantitative quality metrics
4. **Automated Fixes**: Suggestions for automatic data corrections
5. **Validation History**: Track validation results over time
6. **Integration with Data Sources**: Direct validation of database connections 