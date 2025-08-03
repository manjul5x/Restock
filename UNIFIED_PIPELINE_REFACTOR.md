# Unified Pipeline Refactor

## Overview

This document describes the major refactor that consolidates the normal pipeline and enhanced pipeline into a single, unified approach. The unified pipeline combines the best aspects of both pipelines while providing a more maintainable and extensible architecture.

## Key Changes

### 1. **Single Pipeline Architecture**
- **Before**: Two separate pipelines (normal and enhanced) with different flows
- **After**: One unified pipeline that adapts based on forecasting method

### 2. **Method-Specific Parameter Optimization**
- **Before**: Fixed parameters or Prophet-only optimization
- **After**: Each forecasting method has its own optimization strategy

### 3. **Product Grouping by Method**
- **Before**: All products processed together regardless of method
- **After**: Products grouped by forecasting method for efficient processing

### 4. **Extensible Framework**
- **Before**: Hard-coded method handling
- **After**: Factory pattern for easy addition of new forecasting methods

## Architecture

### Core Components

#### 1. **ParameterOptimizer Framework**
```
forecaster/forecasting/parameter_optimizer.py
```

**Base Class**: `ParameterOptimizer`
- Abstract base class for all parameter optimizers
- Defines interface for parameter optimization and forecaster creation

**Method-Specific Optimizers**:
- `MovingAverageParameterOptimizer`: No optimization needed
- `ProphetParameterOptimizer`: Seasonality analysis + parameter optimization
- `ARIMAParameterOptimizer`: Auto-ARIMA parameter selection

**Factory**: `ParameterOptimizerFactory`
- Creates appropriate optimizer based on forecasting method
- Supports registration of new optimizers

#### 2. **Unified Pipeline**
```
forecaster/runner/unified_pipeline.py
```

**Key Features**:
- Works at daily level for forecasting (like enhanced pipeline)
- Groups products by forecasting method
- Uses method-specific parameter optimization
- Supports parallel processing within method groups

#### 3. **Unified Backtester**
```
forecaster/backtesting/unified_backtester.py
```

**Key Features**:
- Method-specific parameter optimization for backtesting
- Daily-level forecasting with risk period aggregation
- Product grouping by forecasting method
- Comprehensive accuracy metrics

#### 4. **Unified Runner**
```
run_unified_pipeline.py
```

**Key Features**:
- Single entry point for all forecasting needs
- Command-line interface with comprehensive options
- Automatic method detection and optimization

## Data Flow

### Unified Pipeline Flow
```
Daily Data → Method Grouping → Method-Specific Optimization → Daily Forecasting → Aggregation → Results
```

### Method-Specific Optimization

#### Moving Average
```
Product Master Parameters → No Optimization → Direct Forecaster Creation
```

#### Prophet
```
Daily Data → Seasonality Analysis → Parameter Optimization → Optimized Forecaster Creation
```

#### ARIMA
```
Daily Data → Auto-ARIMA Selection → Parameter Optimization → Optimized Forecaster Creation
```

## Benefits

### 1. **Simplified Architecture**
- Single pipeline to maintain instead of two
- Consistent interface across all forecasting methods
- Reduced code duplication

### 2. **Better Performance**
- Method-specific optimization reduces unnecessary computation
- Product grouping enables efficient parallel processing
- Optimized parameters improve forecast accuracy

### 3. **Extensibility**
- Easy to add new forecasting methods
- Factory pattern for clean extension
- Consistent parameter optimization interface

### 4. **Maintainability**
- Clear separation of concerns
- Method-specific logic isolated in optimizers
- Standardized interfaces

## Usage

### Basic Usage
```bash
# Run unified pipeline with default settings
python run_unified_pipeline.py

# Run with custom data files
python run_unified_pipeline.py --demand-file my_demand.csv --product-master-file my_product_master.csv

# Run with specific run date
python run_unified_pipeline.py --run-date 2024-01-15

# Run with custom processing settings
python run_unified_pipeline.py --batch-size 20 --max-workers 8
```

### Advanced Usage
```bash
# Disable outlier handling
python run_unified_pipeline.py

# Data validation only (no forecasting)
python run_unified_pipeline.py --no-forecasting

# Debug logging
python run_unified_pipeline.py --log-level DEBUG
```

## Configuration

### Product Master Requirements
The product master must include a `forecast_method` column specifying the forecasting method for each product:

```csv
product_id,location_id,forecast_method,forecast_window_length,forecast_horizon,...
PROD_001,LOC_001,moving_average,25,1,...
PROD_002,LOC_002,prophet,30,2,...
PROD_003,LOC_003,arima,20,1,...
```

### Supported Methods
- `moving_average`: Simple moving average forecasting
- `prophet`: Prophet with seasonality analysis
- `arima`: ARIMA with auto-parameter selection

## Migration Guide

### From Normal Pipeline
1. **Replace runner script**: Use `run_unified_pipeline.py` instead of `run_customer_backtest.py`
2. **Update product master**: Add `forecast_method` column
3. **Update output paths**: Results now go to `output/unified/`

### From Enhanced Pipeline
1. **Replace runner script**: Use `run_unified_pipeline.py` instead of `run_enhanced_backtest.py`
2. **No product master changes needed**: Enhanced pipeline already uses method-specific optimization
3. **Update output paths**: Results now go to `output/unified/`

### Backward Compatibility
- Old pipeline scripts still work but are deprecated
- Product master without `forecast_method` defaults to `moving_average`
- Output format remains compatible

## Adding New Forecasting Methods

### 1. Create Parameter Optimizer
```python
class NewMethodParameterOptimizer(ParameterOptimizer):
    def __init__(self):
        super().__init__("new_method")
    
    def optimize_parameters(self, data, product_record, base_parameters):
        # Implement method-specific optimization
        return optimized_parameters
    
    def create_forecaster(self, parameters):
        # Create and return forecaster instance
        return NewMethodForecaster(parameters)
```

### 2. Register with Factory
```python
ParameterOptimizerFactory.register_optimizer("new_method", NewMethodParameterOptimizer)
```

### 3. Update Product Master
Add `new_method` as a valid `forecast_method` value.

## Performance Considerations

### Optimization Strategies
- **Moving Average**: No optimization overhead
- **Prophet**: Seasonality analysis adds ~2-5 seconds per product
- **ARIMA**: Auto-parameter selection adds ~1-3 seconds per product

### Parallel Processing
- Products grouped by method for efficient parallel processing
- Each method group processed independently
- Configurable batch size and worker count

### Memory Usage
- Daily data kept in memory for optimization
- Method-specific optimizers may cache analysis results
- Configurable memory limits

## Testing

### Unit Tests
- Parameter optimizer tests for each method
- Pipeline integration tests
- Backtester accuracy tests

### Integration Tests
- End-to-end pipeline execution
- Method-specific optimization validation
- Performance benchmarking

## Future Enhancements

### Planned Features
1. **Hybrid Methods**: Combine multiple forecasting methods
2. **Dynamic Method Selection**: Auto-select best method based on data characteristics
3. **Advanced Optimization**: Machine learning-based parameter optimization
4. **Real-time Updates**: Incremental parameter optimization

### Extensibility Points
1. **Custom Optimizers**: User-defined parameter optimization strategies
2. **Custom Forecasters**: Integration with external forecasting libraries
3. **Custom Metrics**: User-defined accuracy and performance metrics

## Troubleshooting

### Common Issues

#### 1. **Unsupported Method Error**
```
ValueError: Unsupported forecasting method: 'unknown_method'
```
**Solution**: Add `forecast_method` column to product master with valid values.

#### 2. **Parameter Optimization Failure**
```
Error optimizing Prophet parameters: Insufficient data
```
**Solution**: Ensure sufficient historical data for parameter optimization.

#### 3. **Memory Issues**
```
MemoryError: Insufficient memory for parallel processing
```
**Solution**: Reduce `max_workers` or `batch_size` in configuration.

### Debug Mode
```bash
python run_unified_pipeline.py --log-level DEBUG
```

This provides detailed logging of:
- Parameter optimization steps
- Method grouping results
- Processing progress
- Error details

## Conclusion

The unified pipeline refactor provides a more maintainable, extensible, and efficient approach to forecasting. By consolidating the best aspects of both normal and enhanced pipelines, it offers:

- **Simplified maintenance** with a single pipeline
- **Better performance** through method-specific optimization
- **Easy extensibility** for new forecasting methods
- **Consistent interface** across all forecasting operations

The refactor maintains backward compatibility while providing a foundation for future enhancements and new forecasting methods. 