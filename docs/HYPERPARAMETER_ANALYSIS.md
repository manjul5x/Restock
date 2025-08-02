# Prophet Hyperparameter Analysis

## Overview

The hyperparameter analysis module provides comprehensive testing and optimization of Prophet forecasting model parameters. This analysis helps determine the optimal configuration for your specific demand forecasting scenarios by testing multiple parameter combinations and measuring their performance through backtesting.

## Key Features

### 🔍 Comprehensive Parameter Testing
- **Indian Market Features**: Tests Indian holidays, festival seasons, monsoon effects
- **Seasonality Settings**: Analyzes quarterly, monthly, and weekly seasonality
- **Model Flexibility**: Tests different changepoint and seasonality prior scales
- **Data Window Options**: Tests various window lengths and minimum data requirements

### 📊 Performance Metrics
- **MAPE (Mean Absolute Percentage Error)**: Primary accuracy metric
- **MAE (Mean Absolute Error)**: Absolute error measurement
- **RMSE (Root Mean Square Error)**: Penalizes larger errors more heavily

### 🎯 Analysis Components
- **Parameter Comparison**: Compares all tested combinations
- **Individual Parameter Effects**: Shows how each parameter affects performance
- **Best Parameters Summary**: Identifies optimal configuration
- **Performance Distribution**: Shows overall performance patterns

## Parameters Analyzed

### Core Prophet Parameters
| Parameter | Description | Tested Values |
|-----------|-------------|---------------|
| `changepoint_range` | Proportion of history for trend changepoints | 0.6, 0.8, 0.9 |
| `n_changepoints` | Number of changepoints to estimate | 15, 25, 35 |
| `changepoint_prior_scale` | Flexibility of changepoints | 0.01, 0.05, 0.1 |
| `seasonality_prior_scale` | Flexibility of seasonality | 5.0, 10.0, 15.0 |
| `holidays_prior_scale` | Flexibility of holiday effects | 5.0, 10.0, 15.0 |
| `seasonality_mode` | Additive or multiplicative seasonality | 'additive', 'multiplicative' |

### Indian Market Specific Features
| Parameter | Description | Tested Values |
|-----------|-------------|---------------|
| `include_indian_holidays` | Include Indian national holidays | True, False |
| `include_quarterly_effects` | Include quarterly seasonality | True, False |
| `include_monthly_effects` | Include monthly seasonality | True, False |
| `include_festival_seasons` | Include festival season effects | True, False |
| `include_monsoon_effect` | Include monsoon season effects | True, False |

### Data Processing Parameters
| Parameter | Description | Tested Values |
|-----------|-------------|---------------|
| `window_length` | Training data window size | 30, 60, 90, None |
| `min_data_points` | Minimum data points required | 10, 15, 20 |

## Usage

### Web Interface

1. **Navigate to Hyperparameter Analysis**
   - Go to the web interface
   - Click on "Hyperparameter Analysis" in the navigation

2. **Configure Analysis**
   - **Sample Size**: Number of product-location combinations to test (3-15)
   - **Max Combinations**: Maximum parameter combinations to test (20-200)

3. **Run Analysis**
   - Click "Run Analysis"
   - Wait for completion (5-15 minutes depending on configuration)

4. **Review Results**
   - **Best Parameters**: Optimal configuration found
   - **Analysis Summary**: Overall performance statistics
   - **Parameter Effects**: Individual parameter impact analysis
   - **Visualizations**: Four different plot types

### Command Line

```bash
# Run the example script
python examples/run_hyperparameter_analysis.py
```

### Programmatic Usage

```python
from forecaster.backtesting.hyperparameter_analyzer import run_hyperparameter_analysis
from forecaster.backtesting.config import BacktestConfig
from datetime import date

# Create configuration
config = BacktestConfig(
    data_dir="forecaster/data",
    demand_file="customer_demand.csv",
    product_master_file="customer_product_master.csv",
    output_dir="output/hyperparameter_analysis",
    
    analysis_start_date=date(2024, 1, 1),
    analysis_end_date=date(2025, 1, 1),
    demand_frequency="d",
    forecast_model="prophet",
    default_horizon=1,
    max_workers=4
)

# Run analysis
results = run_hyperparameter_analysis(config)

# Access results
best_parameters = results['analysis_results']['best_parameters']
summary_stats = results['analysis_results']['summary_stats']
plot_files = results['plot_files']
```

## Analysis Process

### 1. Data Preparation
- Loads customer demand and product master data
- Validates data quality and availability
- Samples product-location combinations for testing

### 2. Parameter Generation
- Creates comprehensive parameter combinations
- Uses stratified sampling to ensure diversity
- Limits combinations based on user configuration

### 3. Backtesting
- For each parameter combination:
  - Tests on multiple product-location combinations
  - Runs historical backtesting
  - Calculates performance metrics (MAPE, MAE, RMSE)

### 4. Analysis
- Aggregates results across all tests
- Identifies best performing parameters
- Analyzes individual parameter effects
- Generates comprehensive visualizations

## Visualization Types

### 1. Parameter Comparison
- **Histogram**: Distribution of MAPE scores across all combinations
- **Top 10 Chart**: Best performing parameter combinations
- **Insights**: Shows performance patterns and outliers

### 2. Parameter Effects
- **Box Plots**: Effect of individual parameters on MAPE
- **Comparison**: True vs False for boolean parameters
- **Insights**: Which parameters have the most impact

### 3. Best Parameters
- **Summary Table**: Optimal parameter configuration
- **Clear Display**: Easy-to-read parameter values
- **Insights**: What works best for your data

### 4. Performance Distribution
- **Metric Distributions**: MAPE, MAE, RMSE distributions
- **Statistical Summary**: Mean, median, standard deviation
- **Insights**: Overall performance characteristics

## Interpretation Guide

### Best Parameters
- **Primary Goal**: Lower MAPE is better
- **Secondary**: Consider MAE and RMSE
- **Practical**: Balance accuracy with model complexity

### Parameter Effects
- **Strong Effect**: Large difference between True/False values
- **Weak Effect**: Small difference between values
- **No Effect**: Similar performance across values

### Performance Distribution
- **Tight Distribution**: Consistent performance across combinations
- **Wide Distribution**: High variability, parameter choice matters
- **Skewed**: Most combinations perform poorly, few perform well

## Recommendations

### For Indian Market Data
1. **Always Test**: `include_indian_holidays` and `include_festival_seasons`
2. **Consider**: `include_monsoon_effect` for weather-sensitive products
3. **Experiment**: `seasonality_mode` (additive vs multiplicative)

### For Different Data Types
1. **High Seasonality**: Focus on seasonality parameters
2. **Trend Changes**: Optimize changepoint parameters
3. **Limited Data**: Test smaller window lengths
4. **Noisy Data**: Try different prior scales

### Performance Optimization
1. **Start Small**: Use 20-50 combinations for initial testing
2. **Scale Up**: Increase combinations for fine-tuning
3. **Focus Areas**: Concentrate on parameters with strong effects
4. **Validate**: Test best parameters on holdout data

## Troubleshooting

### Common Issues

**Analysis Takes Too Long**
- Reduce `max_combinations`
- Reduce `sample_size`
- Increase `max_workers` for parallel processing

**No Successful Results**
- Check data quality and availability
- Reduce `min_data_points` requirement
- Verify data format and columns

**Poor Performance**
- Check data preprocessing
- Verify date ranges and frequency
- Consider data quality issues

### Performance Tips

1. **Use Parallel Processing**: Set `max_workers` to CPU core count
2. **Start with Quick Tests**: Use small sample sizes initially
3. **Focus on Key Parameters**: Prioritize Indian market features
4. **Monitor Progress**: Check logs for detailed progress information

## Advanced Usage

### Custom Parameter Ranges
```python
from forecaster.backtesting.hyperparameter_analyzer import HyperparameterAnalyzer

analyzer = HyperparameterAnalyzer(config)

# Customize parameter ranges
custom_ranges = {
    'changepoint_range': [0.7, 0.8, 0.9],
    'n_changepoints': [20, 25, 30],
    'include_indian_holidays': [True, False],
    # Add more custom ranges...
}

# Generate custom combinations
combinations = analyzer._generate_custom_combinations(custom_ranges)
```

### Targeted Analysis
```python
# Focus on specific parameters
focused_params = {
    'include_indian_holidays': [True, False],
    'include_quarterly_effects': [True, False],
    'seasonality_mode': ['additive', 'multiplicative']
}

# Run focused analysis
results = analyzer.run_focused_analysis(focused_params)
```

## Integration with Forecasting Pipeline

The best parameters found through this analysis can be directly used in your forecasting pipeline:

```python
from forecaster.forecasting.prophet import create_prophet_forecaster

# Use best parameters from analysis
best_params = {
    'include_indian_holidays': True,
    'include_quarterly_effects': True,
    'seasonality_mode': 'multiplicative',
    # ... other best parameters
}

# Create forecaster with optimal parameters
forecaster = create_prophet_forecaster(best_params)
```

This ensures your production forecasting uses the optimal configuration for your specific data and business context. 