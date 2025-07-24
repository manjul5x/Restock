# Inventory Forecasting & Simulation Suite

A comprehensive tool for demand forecasting, safety stock calculation, and inventory simulation with a web interface for visualization.

## Features

- **Demand Forecasting**
  - Multiple forecasting models (Moving Average, Prophet)
  - Outlier detection and handling
  - Parallel processing support
  - Backtesting capabilities

- **Safety Stock Calculation**
  - Dynamic safety stock levels
  - Multiple review periods support
  - Service level optimization
  - Error-based calculations

- **Inventory Simulation**
  - Order policy simulation
  - Lead time consideration
  - Stock level tracking
  - Performance metrics

- **Web Interface**
  - Interactive visualizations
  - Forecast comparison
  - Safety stock analysis
  - Inventory simulation results

## Quick Start

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/restock.git
   cd restock
   ```

2. **Set Up Environment with uv**
   ```bash
   # Install uv if you haven't already
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Install dependencies
   uv sync
   
   # For development with additional tools
   uv sync --extra dev
   ```

3. **Prepare Your Data**
   - Replace example data in `forecaster/data/` with your data:
     - `customer_demand.csv`: Your demand data
     - `customer_product_master.csv`: Your product master data
   - See [Customer Data Guide](CUSTOMER_DATA_GUIDE.md) for data format requirements

4. **Run the Pipeline**
   ```bash
   # Run backtesting
   uv run python run_customer_backtest.py

   # Calculate safety stocks
   uv run python run_safety_stock_calculation.py

   # Run simulation
   uv run python run_simulation.py

   # Start web interface
   uv run python webapp/app.py
   ```

   **Or use the Makefile shortcuts:**
   ```bash
   make run-backtest
   make run-safety-stocks
   make run-simulation
   make run-webapp
   ```

## Data Requirements

### Demand Data (`customer_demand.csv`)
- Required columns:
  - `product_id`: Product identifier
  - `product_category`: Product category
  - `location_id`: Location identifier
  - `date`: Date in YYYY-MM-DD format
  - `demand`: Demand quantity
  - `stock_level`: Stock level
  - `incoming_inventory`: Incoming inventory

### Product Master (`customer_product_master.csv`)
- Required columns:
  - `product_id`: Product identifier
  - `location_id`: Location identifier
  - `product_category`: Product category
  - `demand_frequency`: 'd' (daily), 'w' (weekly), or 'm' (monthly)
  - `risk_period`: Risk period length
  - `forecast_window_length`: Forecasting window length
  - `forecast_horizon`: Forecast horizon
  - `leadtime`: Lead time in days

## Development

### Code Quality
This project uses modern Python development tools:

- **uv**: Fast Python package manager and installer
- **black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking
- **pytest**: Testing framework
- **pre-commit**: Git hooks for code quality

### Development Commands
```bash
# Install development dependencies
make install-dev

# Format code
make format

# Run linting
make lint

# Run tests
make test

# Run all checks
make check

# Show all available commands
make help
```

### Pre-commit Setup
```bash
# Install pre-commit hooks
make dev
```

## Output Files

### Backtesting Results
- Location: `output/customer_backtest/`
  - `backtest_results.csv`: Summary of all forecasts
  - `accuracy_metrics.csv`: Accuracy metrics by product/location
  - `forecast_comparison.csv`: Detailed forecast vs actual comparison

### Safety Stock Results
- Location: `output/safety_stocks/`
  - `safety_stock_results.csv`: Calculated safety stock levels

### Simulation Results
- Location: `output/simulation/`
  - `simulation_summary.csv`: Overall simulation metrics
  - `detailed_results/`: Detailed simulation data by product/location

## Web Interface

The web interface provides interactive visualizations for:
- Demand patterns and forecasts
- Safety stock levels and review periods
- Inventory simulation results
- Performance metrics

Access at `http://localhost:5000` after starting the webapp.

## Documentation

- [Customer Data Guide](CUSTOMER_DATA_GUIDE.md): Detailed data requirements
- [Example Data](forecaster/data/dummy/): Sample data format
- [API Documentation](docs/API.md): Module documentation

## Requirements

- Python 3.9+
- See `requirements.txt` for package dependencies

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with Prophet by Facebook Research
- Visualization powered by Plotly
- Web interface using Flask
