# 🏢 Company Snowflake Setup Guide

## 📋 **Overview**
This guide explains how to configure the Restock application for your company's Snowflake environment.

## 🔧 **Configuration Steps**

### **Step 1: Copy Template Configuration**
```bash
# Copy the template to create your configuration
cp data/config/data_config_template.yaml data/config/data_config.yaml
```

### **Step 2: Update Snowflake Credentials**
Edit `data/config/data_config.yaml` and update the following sections:

```yaml
# Your company's Snowflake configuration
snowflake:
  account: "YOUR_COMPANY_ACCOUNT.us-east-1"  # e.g., "abc12345.us-east-1"
  user: "YOUR_USERNAME"                      # e.g., "john.doe"
  password: "YOUR_PASSWORD"                 # e.g., "secure_password_123"
  role: "YOUR_ROLE"                         # e.g., "ACCOUNTADMIN" or "ANALYST"
  warehouse: "YOUR_WAREHOUSE"               # e.g., "COMPUTE_WH" or "ANALYSIS_WH"
  database: "YOUR_DATABASE"                 # e.g., "PRODUCTION_DB"
  schema: "YOUR_SCHEMA"                     # e.g., "RESTOCK" or "INVENTORY"
```

### **Step 3: Update Table Names**
Update the table mappings to match your company's table names:

```yaml
snowflake_tables:
  # Input tables (read from) - Update these to your actual table names
  outflow: "YOUR_OUTFLOW_TABLE"           # e.g., "DEMAND_DATA" or "SALES_TRANSACTIONS"
  product_master: "YOUR_PRODUCT_MASTER_TABLE"  # e.g., "PRODUCT_MASTER" or "INVENTORY_ITEMS"
  
  # Output tables (write to) - These will be created automatically
  safety_stocks: "SAFETY_STOCKS_RESULTS"
  forecast_comparison: "FORECAST_COMPARISON_RESULTS"
  simulation_results: "SIMULATION_RESULTS"
  forecast_visualization: "FORECAST_VISUALIZATION_DATA"
  input_data_with_regressors: "INPUT_DATA_WITH_REGRESSORS"
```

## 🧪 **Testing Your Configuration**

### **Test 1: Connection Test**
```bash
python -c "
from data.access.snowflake_accessor import SnowflakeAccessor
import yaml

# Load your config
with open('data/config/data_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Test connection
accessor = SnowflakeAccessor(config['snowflake'])
result = accessor.validate_connection()
print('✅ Connection successful!' if result else '❌ Connection failed')
"
```

### **Test 2: Data Access Test**
```bash
python -c "
from data.loader import DataLoader

# Test data loading
loader = DataLoader()
product_master = loader.load_product_master()
outflow = loader.load_outflow()

print(f'✅ Product master: {len(product_master)} records')
print(f'✅ Outflow: {len(outflow)} records')
"
```

## 🔄 **Switching Between CSV and Snowflake**

### **Use Snowflake (Production)**
```yaml
storage:
  type: "snowflake"
```

### **Use CSV (Development/Testing)**
```yaml
storage:
  type: "csv"
```

## 🛡️ **Security Best Practices**

### **Option 1: Environment Variables (Recommended)**
Create a `.env` file:
```bash
# .env file
SNOWFLAKE_ACCOUNT=your_account.us-east-1
SNOWFLAKE_USER=your_username
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_ROLE=your_role
SNOWFLAKE_WAREHOUSE=your_warehouse
SNOWFLAKE_DATABASE=your_database
SNOWFLAKE_SCHEMA=your_schema
```

Then update `data_config.yaml`:
```yaml
snowflake:
  account: "${SNOWFLAKE_ACCOUNT}"
  user: "${SNOWFLAKE_USER}"
  password: "${SNOWFLAKE_PASSWORD}"
  role: "${SNOWFLAKE_ROLE}"
  warehouse: "${SNOWFLAKE_WAREHOUSE}"
  database: "${SNOWFLAKE_DATABASE}"
  schema: "${SNOWFLAKE_SCHEMA}"
```

### **Option 2: Separate Config Files**
```bash
# Development
cp data_config.yaml data_config_dev.yaml

# Production
cp data_config.yaml data_config_prod.yaml
```

## 📊 **Required Table Schemas**

### **Input Tables Required:**

#### **Outflow Table Schema:**
```sql
CREATE TABLE YOUR_OUTFLOW_TABLE (
    product_id VARCHAR,
    location_id VARCHAR,
    date DATE,
    demand NUMBER,
    product_name VARCHAR,
    location_name VARCHAR,
    voucher_type VARCHAR,
    net_amount NUMBER,
    transaction_id VARCHAR
);
```

#### **Product Master Table Schema:**
```sql
CREATE TABLE YOUR_PRODUCT_MASTER_TABLE (
    product_id VARCHAR,
    product_name VARCHAR,
    product_code VARCHAR,
    location_id VARCHAR,
    product_type_id VARCHAR,
    reorder_level NUMBER,
    bin_capacity NUMBER,
    status VARCHAR,
    created_date DATE,
    modified_date DATE,
    ss_window_length NUMBER
);
```

## 🚀 **Running the Application**

### **Data Validation**
```bash
python run_data_validation.py
```

### **Backtesting**
```bash
python run_backtesting.py
```

### **Safety Stock Calculation**
```bash
python run_safety_stock_calculation.py
```

### **Simulation**
```bash
python run_simulation.py
```

## 🔍 **Troubleshooting**

### **Common Issues:**

1. **Connection Failed**
   - Check account format: `account.us-east-1`
   - Verify user has access to warehouse/database/schema
   - Check network/firewall settings

2. **Table Not Found**
   - Verify table names in `snowflake_tables` section
   - Check case sensitivity (Snowflake is case-sensitive)
   - Ensure user has SELECT permissions

3. **Permission Denied**
   - Verify user role has necessary permissions
   - Check warehouse usage permissions
   - Ensure database/schema access

### **Debug Commands:**
```bash
# Test connection only
python -c "from data.access.snowflake_accessor import SnowflakeAccessor; import yaml; config = yaml.safe_load(open('data/config/data_config.yaml')); print('✅ Connected' if SnowflakeAccessor(config['snowflake']).validate_connection() else '❌ Failed')"

# Test data loading
python -c "from data.loader import DataLoader; loader = DataLoader(); print(f'Records: {len(loader.load_product_master())}')"
```

## 📞 **Support**

If you encounter issues:
1. Check the troubleshooting section above
2. Verify your Snowflake credentials and permissions
3. Ensure all required tables exist with correct schemas
4. Test connection using the debug commands

---

**🎉 Once configured, your application will automatically use Snowflake for all data operations!**
