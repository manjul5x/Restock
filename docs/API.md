# Forecaster API Documentation

## Core Modules

### Backtesting (`forecaster.backtesting`)

#### `UnifiedBacktester` class
Main class for running backtesting simulations.

```python
from forecaster.backtesting import UnifiedBacktester, BacktestConfig

config = BacktestConfig(
    analysis_start_date=None,  # Will be calculated based on ss_window_length and first review date
analysis_end_date=None,     # Will be set to last review date
)
backtester = UnifiedBacktester(config)
results = backtester.run_unified_backtesting()
```

### Safety Stocks (`forecaster.safety_stocks`)

#### `SafetyStockCalculator` class
Calculates safety stock levels based on forecast errors.

```python
from forecaster.safety_stocks import SafetyStockCalculator
from data.loader import DataLoader

loader = DataLoader()
product_master_data = loader.load_product_master()
# forecast_comparison_data would be loaded from a CSV output from the backtester
calculator = SafetyStockCalculator(product_master_data)
results = calculator.calculate_safety_stocks(forecast_comparison_data)
```

### Simulation (`forecaster.simulation`)

#### `Simulator` class
Runs inventory simulation with different ordering policies.

```python
from forecaster.simulation import Simulator, SimulationConfig

config = SimulationConfig()
simulator = Simulator(config)
results = simulator.run_simulation()
```

### Data Loading (`data.loader`)

#### `DataLoader` class
Handles loading and validating all input data, with built-in caching and support for parallel processing. It is configured via `data/config/data_config.yaml`.

```python
from data.loader import DataLoader

# The DataLoader is a singleton in the main process
loader = DataLoader()

# Load product master and outflow (demand) data
product_master = loader.load_product_master()
outflow_data = loader.load_outflow(product_master=product_master) # Filtering is optional but recommended
```

### Web Interface (`webapp`)

#### Running the Web App
```shell
python webapp/app.py
```

## Data Schemas

### Demand Data Schema
Required columns and their types:
- `product_id` (str)
- `product_category` (str)
- `location_id` (str)
- `date` (datetime)
- `demand` (float)
- `stock_level` (float)
- `incoming_inventory` (float)

### Product Master Schema
Required columns and their types:
- `product_id` (str)
- `location_id` (str)
- `product_category` (str)
- `demand_frequency` (str): 'd', 'w', or 'm'
- `risk_period` (int)
- `forecast_window_length` (int)
- `forecast_horizon` (int)
- `leadtime` (int)

## Configuration

### BacktestConfig
Configuration for backtesting runs:
```python
config = BacktestConfig(
    analysis_start_date=date(2023, 1, 1),
    analysis_end_date=date(2023, 12, 31),
    demand_frequency="d",
    forecast_model="moving_average",
    default_horizon=2,
    batch_size=10,
    max_workers=4,
    validate_data=True,

    aggregation_enabled=True,
    log_level="INFO"
)
```

## Output Formats

### Backtest Results
```python
{
    'total_forecasts': int,
    'accuracy_metrics': pd.DataFrame,
    'forecast_comparison': pd.DataFrame,
    'processing_time': float
}
```

### Safety Stock Results
```python
pd.DataFrame({
    'product_id': str,
    'location_id': str,
    'review_date': datetime,
    'safety_stock': float,
    'errors': List[float],
    'error_count': int,
    'distribution_type': str
})
```

### Simulation Results
```python
{
    'summary': pd.DataFrame,  # Overall metrics
    'detailed_results': Dict[str, pd.DataFrame]  # Per product-location results
}
``` 