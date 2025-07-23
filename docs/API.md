# Forecaster API Documentation

## Core Modules

### Backtesting (`forecaster.backtesting`)

#### `Backtester` class
Main class for running backtesting simulations.

```python
from forecaster.backtesting import Backtester, BacktestConfig

config = BacktestConfig(
    data_dir="forecaster/data",
    demand_file="customer_demand.csv",
    product_master_file="customer_product_master.csv"
)
backtester = Backtester(config)
results = backtester.run()
```

### Safety Stocks (`forecaster.safety_stocks`)

#### `SafetyStockCalculator` class
Calculates safety stock levels based on forecast errors.

```python
from forecaster.safety_stocks import SafetyStockCalculator

calculator = SafetyStockCalculator(product_master_data)
results = calculator.calculate_safety_stocks(forecast_comparison_data)
```

### Simulation (`forecaster.simulation`)

#### `Simulator` class
Runs inventory simulation with different ordering policies.

```python
from forecaster.simulation import Simulator

simulator = Simulator(config)
results = simulator.run()
```

### Data Loading (`forecaster.data`)

#### `DemandDataLoader` class
Handles loading and validating demand data.

```python
from forecaster.data import DemandDataLoader

loader = DemandDataLoader()
demand_data = loader.load_csv("customer_demand.csv")
```

### Web Interface (`forecaster.webapp`)

#### Running the Web App
```python
from forecaster.webapp import app

if __name__ == "__main__":
    app.run(debug=True)
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
    data_dir="forecaster/data",
    demand_file="customer_demand.csv",
    product_master_file="customer_product_master.csv",
    output_dir="output/customer_backtest",
    historic_start_date=date(2022, 1, 1),
    analysis_start_date=date(2023, 1, 1),
    analysis_end_date=date(2023, 12, 31),
    demand_frequency="d",
    forecast_model="moving_average",
    default_horizon=2,
    batch_size=10,
    max_workers=4,
    validate_data=True,
    outlier_enabled=True,
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