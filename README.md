# Inventory Forecasting & Analysis System

A comprehensive, enterprise-grade inventory forecasting and analysis system with unified pipeline, advanced backtesting, safety stock optimization, and interactive web interface for real-time inventory insights.

## 🚀 Key Features

### 📊 **Unified Forecasting Pipeline**
- **Multiple Models**: Prophet, ARIMA, and Moving Average with automatic parameter optimization
- **Consistent Aggregation**: Unified approach across all forecasters for reliable results
- **Advanced Backtesting**: Comprehensive performance evaluation with detailed metrics
- **Data Validation**: Built-in data quality checks and completeness validation

### 📈 **Enhanced Analytics**
- **Safety Stock Optimization**: Dynamic calculation with multiple distribution methods
- **Inventory Simulation**: Realistic simulation with order policies and lead time consideration
- **Performance Metrics**: Service level, stockout rate, inventory days, and total holding costs
- **Visual Comparison**: Actual vs. simulated performance with highlighting

### 🎯 **Interactive Web Interface**
- **Real-time Filtering**: Dynamic filtering by product, location, and forecast method
- **Performance Highlighting**: Visual indicators showing improvements and areas for attention
- **Responsive Design**: Works seamlessly on desktop and mobile devices
- **Comprehensive Views**: Forecast visualization, safety stocks, simulation, and inventory comparison

### 🔧 **Enterprise Features**
- **Parallel Processing**: Optimized for large datasets with configurable workers
- **Data Validation**: Comprehensive quality checks and error reporting
- **Flexible Configuration**: Customizable parameters and review periods
- **Production Ready**: Robust error handling and logging

## 🎉 What's New

### Recent Major Improvements:
- ✅ **Unified Pipeline**: Consistent forecasting approach across all models
- ✅ **Performance Highlighting**: Visual indicators for metric improvements
- ✅ **Inventory Days**: More intuitive metric replacing inventory turns
- ✅ **Total Inventory Holding**: Complete cost analysis in units and currency
- ✅ **Streamlined Workflow**: Single command for complete analysis
- ✅ **Enhanced Web Interface**: Better UX with compact metric display
- ✅ **Data Validation**: Comprehensive data quality assurance
- ✅ **Default Optimization**: Sensible defaults for quick setup

## 📋 Quick Start

### 1. **Setup & Installation**
```bash
# Install UV (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone <your-repo-url>
cd Forecaster
uv sync
```

### 2. **Prepare Your Data**
Replace example data files with your own:
- `forecaster/data/customer_demand.csv` - Your demand data
- `forecaster/data/customer_product_master.csv` - Your product master data

See [Customer Data Guide](CUSTOMER_DATA_GUIDE.md) for detailed format requirements.

### 3. **Run Complete Analysis**
```bash
# Single command for complete workflow
python run_complete_workflow.py
```

This runs:
- Data validation
- Unified backtesting
- Safety stock calculation
- Inventory simulation
- Results generation

### 4. **Explore Results**
```bash
python webapp/run.py
```

Open `http://localhost:5001` to access the interactive web interface.

## 📊 Available Analysis Pages

### 📈 **Forecast Visualization**
- Historical vs. forecasted demand comparison
- Multi-model performance analysis
- Interactive filtering and zoom capabilities

### 🛡️ **Safety Stocks**
- Dynamic safety stock level analysis
- Distribution method comparison
- Trend analysis and optimization

### 🎯 **Simulation Visualization**
- Inventory simulation results
- Stock level tracking and analysis
- Performance metric visualization

### ⚖️ **Inventory Comparison** ⭐ **NEW!**
- **Actual vs. Simulated Performance** with visual highlighting
- **Key Metrics**:
  - Service Level & Stockout Rate
  - Inventory Days (lower is better)
  - Total Inventory Holding (units & cost)
  - Missed Demand & Stockout Days
  - Overstocking/Understocking percentages
- **Performance Indicators**: Green checkmarks for improvements, red X for areas needing attention

## 📁 Data Requirements

### Demand Data (`customer_demand.csv`)
Required columns:
- `product_id`: Product identifier
- `product_category`: Product category
- `location_id`: Location identifier
- `date`: Date in YYYY-MM-DD format
- `demand`: Demand quantity
- `stock_level`: Stock level
- `incoming_inventory`: Incoming inventory

### Product Master (`customer_product_master.csv`)
Required columns:
- `product_id`: Product identifier
- `location_id`: Location identifier
- `product_category`: Product category
- `demand_frequency`: 'd' (daily), 'w' (weekly), or 'm' (monthly)
- `risk_period`: Risk period length
- `forecast_window_length`: Forecasting window length
- `forecast_horizon`: Forecast horizon
- `forecast_method`: 'moving_average', 'prophet', 'arima'
- `service_level`: Service level percentage (0.0 to 1.0)
- `leadtime`: Lead time in demand frequency units
- `inventory_cost`: Unit cost of inventory

## 🔧 Configuration Options

### Analysis Period
- Set custom analysis start and end dates
- Ensure sufficient historical data (recommend 10+ months before analysis start)

### Review Dates (Safety Stocks)
- Default: 1st, 8th, 15th, 22nd of every month in 2024
- Fully customizable via command line arguments

### Parallel Processing
- Default: 8 workers for optimal performance
- Adjustable via `--max-workers` parameter

## 📚 Documentation

- **[Quick Start Guide](QUICK_START_GUIDE.md)** - Get up and running quickly
- **[Customer Data Guide](CUSTOMER_DATA_GUIDE.md)** - Detailed data format requirements
- **[Migration Guide](MIGRATION_GUIDE.md)** - Technical implementation details
- **[Data Validation System](DATA_VALIDATION_SYSTEM.md)** - Data quality assurance
- **[Unified Pipeline Refactor](UNIFIED_PIPELINE_REFACTOR.md)** - Architecture overview

## 🛠️ Development

### Project Structure
```
Forecaster/
├── forecaster/           # Core forecasting modules
│   ├── forecasting/     # Forecasting models and pipeline
│   ├── backtesting/     # Backtesting and evaluation
│   ├── safety_stocks/   # Safety stock calculations
│   ├── simulation/      # Inventory simulation
│   ├── validation/      # Data validation system
│   └── data/           # Data loading and processing
├── webapp/             # Interactive web interface
├── examples/           # Example scripts and usage
├── output/            # Generated results and visualizations
└── docs/              # Documentation and guides
```

### Running Tests
```bash
# Run all tests
uv run python -m pytest

# Run specific test categories
uv run python -m pytest forecaster/tests/ -v
```

### Development Setup
```bash
# Install development dependencies
uv sync --extra dev

# Run linting
uv run ruff check .

# Run type checking
uv run mypy forecaster/
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: Check the guides in the `docs/` directory
- **Issues**: Report bugs and feature requests via GitHub Issues
- **Data Issues**: Use the built-in data validation system for troubleshooting

---

**Ready to optimize your inventory?** Start with the [Quick Start Guide](QUICK_START_GUIDE.md) and transform your inventory management! 🚀
