# Quick Start Guide - Inventory Forecasting & Analysis System

Welcome to the enhanced Inventory Forecasting & Analysis System! This guide will help you get started quickly with the unified pipeline that provides comprehensive inventory insights.

## 🚀 Quick Overview

This system provides:
- **Unified Forecasting Pipeline** with Prophet, ARIMA, and Moving Average models
- **Advanced Backtesting** with performance metrics and visualization
- **Safety Stock Optimization** with multiple calculation methods
- **Inventory Simulation** with detailed performance analysis
- **Interactive Web Interface** with real-time filtering and visualization
- **Data Validation** to ensure data quality and completeness

## 📋 Prerequisites

- Python 3.8 or higher
- UV package manager (recommended) or pip
- Your inventory and demand data

## Step 1: Setup & Installation

### Quick Setup (Recommended)
```bash
# Clone the repository
git clone <your-repo-url>
cd Forecaster

# Run the setup script (installs uv, dependencies, and dev tools)
./setup.sh
```

### Manual Setup
```bash
# Install UV if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup the project
git clone <your-repo-url>
cd Forecaster
uv sync

# For development (optional)
uv sync --extra dev
```

## Step 2: Prepare Your Data

Replace the example data files with your own:

### Required Files:
- **`forecaster/data/customer_demand.csv`** - Your demand data
- **`forecaster/data/customer_product_master.csv`** - Your product master data

### Data Format Requirements

**customer_demand.csv** must have these columns:
- `product_id` - Product identifier
- `product_category` - Product category  
- `location_id` - Location identifier
- `date` - Date in YYYY-MM-DD format
- `demand` - Demand quantity
- `stock_level` - Stock level
- `incoming_inventory` - Incoming inventory (can be 0)

**customer_product_master.csv** must have these columns:
- `product_id` - Product identifier
- `location_id` - Location identifier
- `product_category` - Product category
- `demand_frequency` - 'd' (daily), 'w' (weekly), or 'm' (monthly)
- `risk_period` - Risk period as integer multiple of demand frequency (e.g., 7 for weekly = 7 weeks)
- `outlier_method` - Outlier detection method: 'iqr', 'zscore', 'mad', 'rolling' (optional, default: 'iqr')
- `outlier_threshold` - Outlier detection threshold (optional, default: 1.5)
- `forecast_window_length` - Forecasting window length in risk periods
- `forecast_horizon` - Forecast horizon in risk periods
- `forecast_method` - Forecasting method: 'moving_average', 'prophet', 'arima' (optional, default: 'moving_average')
- `distribution` - Safety stock distribution type: 'kde', 'normal' (optional, default: 'kde')
- `service_level` - Service level percentage 0.0 to 1.0 (optional, default: 0.95)
- `ss_window_length` - Rolling window length for safety stock calculation in demand frequency units (optional, default: 180)
- `leadtime` - Lead time in demand frequency units (e.g., 7 for weekly = 7 weeks)
- `inventory_cost` - Unit cost of inventory (optional, default: 0.0)
- `moq` - Minimum order quantity (optional, default: 1.0)

## Step 3: Run the Complete Workflow

### Single Command (Recommended)
```bash
uv run python run_complete_workflow.py
```

This single command runs:
1. **Data Validation** - Ensures data quality and completeness
2. **Unified Backtesting** - Tests forecasting models with performance metrics
3. **Safety Stock Calculation** - Optimizes inventory levels
4. **Inventory Simulation** - Simulates inventory performance
5. **Results Generation** - Creates all output files and visualizations

### Step-by-Step (For Customization)
```bash
# 1. Validate your data
uv run python run_data_validation.py

# 2. Run backtesting with unified approach
uv run python run_unified_backtest.py

# 3. Calculate safety stocks (uses default review dates)
uv run python run_safety_stock_calculation.py

# 4. Run simulation
uv run python run_simulation.py
```

### Using Makefile Shortcuts
```bash
# Run individual components
make run-backtest
make run-safety-stocks
make run-simulation
make run-webapp
```

## Step 4: Start the Web Interface

```bash
uv run python webapp/run.py
```

## Step 5: Explore Results

Open your browser and go to `http://localhost:5001`

### Available Pages:

#### 📊 **Forecast Visualization**
- View historical demand vs. forecasted demand
- Compare different forecasting methods (Prophet, ARIMA, Moving Average)
- Filter by product, location, and forecast method
- Interactive charts with zoom and pan capabilities

#### 📈 **Safety Stocks**
- Analyze safety stock levels across products and locations
- View safety stock trends and distributions
- Filter by product category and location
- Compare different calculation methods

#### 🎯 **Simulation Visualization**
- View inventory simulation results
- Analyze stock levels, stockouts, and performance metrics
- Filter by product, location, and forecast method
- Interactive time series charts

#### ⚖️ **Inventory Comparison** ⭐ **NEW!**
- **Actual vs. Simulated Performance** comparison
- **Performance Highlighting** - See which metrics improved
- **Key Metrics**:
  - Service Level & Stockout Rate
  - Inventory Days (replacing inventory turns)
  - Total Inventory Holding (units & cost)
  - Missed Demand & Stockout Days
  - Overstocking/Understocking percentages
- **Compact Display** - All metrics in one organized view
- **Visual Indicators** - Green checkmarks for improvements, red X for areas needing attention

## 🎯 Key Features

### Enhanced Metrics
- **Inventory Days**: Average inventory / Average daily demand (lower is better)
- **Total Inventory Holding**: On-hand + on-order inventory in units and cost
- **Performance Comparison**: Side-by-side actual vs. simulated metrics
- **Visual Highlighting**: Immediate identification of improvements

### Improved Workflow
- **Unified Pipeline**: Consistent approach across all forecasters
- **Parameter Optimization**: Automatic optimization for each forecasting method
- **Data Validation**: Comprehensive data quality checks
- **Default Settings**: Sensible defaults for quick setup

### Better User Experience
- **Auto-filtering**: Default selections for all pages
- **Performance Indicators**: Clear visual feedback on improvements
- **Responsive Design**: Works on desktop and mobile
- **Real-time Updates**: Dynamic filtering and visualization

## 📁 Output Structure

After running the workflow, you'll find results in:
```
output/complete_workflow/
├── backtesting/
│   ├── forecast_comparison.csv
│   ├── forecast_visualization_data.csv
│   └── backtest_results.csv
├── safety_stocks/
│   ├── safety_stock_results.csv
│   └── safety_stock_plots/
└── simulation/
    ├── simulation_summary.csv
    ├── detailed_results/
    └── simulation_plots/
```

## 🔧 Configuration Options

### Analysis Period
- Set your analysis start and end dates
- Ensure sufficient historical data (recommend 10+ months before analysis start)

### Review Dates (Safety Stocks)
- Default: 1st, 8th, 15th, 22nd of every month in 2024
- Customizable via command line arguments

### Parallel Processing
- Default: 8 workers for faster processing
- Adjustable via `--max-workers` parameter

## 🆘 Troubleshooting

### Common Issues:
1. **Data Format**: Ensure your CSV files match the required format
2. **Date Range**: Make sure you have sufficient historical data
3. **Memory**: For large datasets, consider reducing parallel workers
4. **Dependencies**: Ensure all packages are installed correctly

### Getting Help:
- Check the `MIGRATION_GUIDE.md` for detailed technical information
- Review `DATA_VALIDATION_SYSTEM.md` for data requirements
- See `UNIFIED_PIPELINE_REFACTOR.md` for architecture details

## 🎉 What's New

### Recent Improvements:
- ✅ **Unified Forecasting Pipeline** - Consistent approach across all models
- ✅ **Enhanced Web Interface** - Better visualization and user experience
- ✅ **Performance Highlighting** - Visual indicators for improvements
- ✅ **Inventory Days Metric** - More intuitive than inventory turns
- ✅ **Total Inventory Holding** - Complete inventory cost analysis
- ✅ **Data Validation** - Comprehensive data quality checks
- ✅ **Streamlined Workflow** - Single command for complete analysis
- ✅ **Better Documentation** - Comprehensive guides and examples

---

**Ready to get started?** Run `uv run python run_complete_workflow.py` and explore your inventory insights! 🚀

