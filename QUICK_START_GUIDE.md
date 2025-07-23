# Quick Start Guide - Inventory Comparison

Follow these simple steps to run the inventory forecasting pipeline and view the inventory comparison results.

## Step 1: Prepare Your Data
Replace the example data files with your own:
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

## Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

## Step 3: Run the Pipeline (3 commands)
```bash
# 1. Run backtesting
python run_customer_backtest.py #loki set your analysis start and end date here. Ask cursor how. 

# 2. Calculate safety stocks  
python run_safety_stock_calculation.py #loki pass your review s here. a list of the first of every month for the analysis period.

# 3. Run simulation
python run_simulation.py
```

## Step 4: Start the Web Interface
```bash
python webapp/app.py
```

## Step 5: Access Inventory Comparison
1. Open your web browser
2. Go to: `http://localhost:5001`
3. Click on **"Inventory Comparison"** in the navigation menu
4. The page will automatically load and display the comparison results

