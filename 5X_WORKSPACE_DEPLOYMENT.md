# 5X Workspace Deployment Guide

This guide shows how to deploy the restock forecasting project to 5X workspace using environment variables.

## Overview

The project has been updated to support environment variable-based configuration, following the same pattern as your other project (HRMS). This eliminates the need for config files and enables secure deployment to 5X workspace.

## Key Changes Made

### 1. **Environment Variable Configuration**
- Created `data/access/env_snowflake_config.py` - Environment-based Snowflake configuration
- Created `data/access/env_snowflake_accessor.py` - Environment-based data accessor
- Created `env_template.txt` - Template for environment variables

### 2. **Private Key Authentication**
- Uses IDE-style private key authentication (same as your other project)
- Eliminates need for username/password + MFA
- More secure and workspace-friendly

## Setup Steps

### Step 1: Set Environment Variables

Create a `.env` file in your project root (copy from `env_template.txt`):

```bash
# Copy the template
cp env_template.txt .env

# Edit with your actual values
nano .env
```

Fill in your environment variables:
```bash
# Snowflake Configuration
FIVEX_SNOWFLAKE_ACCOUNT=da37542.uae-north.azure
FIVEX_SNOWFLAKE_USER=manjul
FIVEX_SNOWFLAKE_ROLE=ACCOUNTADMIN
FIVEX_SNOWFLAKE_WAREHOUSE=COMPUTE_WH
FIVEX_SNOWFLAKE_DATABASE=RESTOCK_DB
FIVEX_SNOWFLAKE_SCHEMA=TRANSFORMATION

# Private Key Authentication (from your IDE setup)
FIVEX_SNOWFLAKE_PRIVATE_KEY_FILE=/path/to/your/rsa_key.p8
FIVEX_SNOWFLAKE_PRIVATE_KEY_FILE_PWD=your_key_passphrase
```

### Step 2: Test Environment Setup

Run the test script to verify everything is working:

```bash
python test_env_connection.py
```

This will test:
- ✅ Environment variables are set
- ✅ Configuration loading works
- ✅ Snowflake connection succeeds
- ✅ Data access works

### Step 3: Update Your Code to Use Environment Variables

Replace your current SnowflakeAccessor usage:

**Before (config file):**
```python
from data.access.snowflake_accessor import SnowflakeAccessor

# Load from config file
accessor = SnowflakeAccessor(connection_config)
```

**After (environment variables):**
```python
from data.access.env_snowflake_accessor import EnvSnowflakeAccessor

# Use environment variables
accessor = EnvSnowflakeAccessor(use_env_vars=True)
```

### Step 4: Update Main Pipeline Scripts

Update your main pipeline scripts to use environment variables:

```python
# In your main pipeline files
from data.access.env_snowflake_accessor import EnvSnowflakeAccessor

# Initialize with environment variables
accessor = EnvSnowflakeAccessor(use_env_vars=True)

# Use as before
data = accessor.read_data("YOUR_TABLE")
accessor.write_data(df, "YOUR_TABLE")
```

## Environment Variable Reference

| Variable | Description | Example |
|----------|-------------|---------|
| `FIVEX_SNOWFLAKE_ACCOUNT` | Snowflake account identifier | `da37542.uae-north.azure` |
| `FIVEX_SNOWFLAKE_USER` | Snowflake username | `manjul` |
| `FIVEX_SNOWFLAKE_ROLE` | Snowflake role | `ACCOUNTADMIN` |
| `FIVEX_SNOWFLAKE_WAREHOUSE` | Snowflake warehouse | `COMPUTE_WH` |
| `FIVEX_SNOWFLAKE_DATABASE` | Snowflake database | `RESTOCK_DB` |
| `FIVEX_SNOWFLAKE_SCHEMA` | Snowflake schema | `TRANSFORMATION` |
| `FIVEX_SNOWFLAKE_PRIVATE_KEY_FILE` | Path to private key file | `/path/to/rsa_key.p8` |
| `FIVEX_SNOWFLAKE_PRIVATE_KEY_FILE_PWD` | Private key passphrase | `your_passphrase` |

## Benefits of This Approach

### 1. **Workspace Compatibility**
- ✅ Works seamlessly with 5X workspace
- ✅ No config files to manage
- ✅ Environment variables are workspace-native

### 2. **Security**
- ✅ Private key authentication (more secure than passwords)
- ✅ No hardcoded credentials
- ✅ IDE integration

### 3. **Consistency**
- ✅ Same pattern as your other project (HRMS)
- ✅ Proven to work in 5X workspace
- ✅ Easy to maintain

### 4. **Deployment**
- ✅ No MFA required
- ✅ No manual credential entry
- ✅ Automated deployment ready

## Migration Checklist

- [ ] Set up environment variables in `.env` file
- [ ] Test connection with `test_env_connection.py`
- [ ] Update main pipeline scripts to use `EnvSnowflakeAccessor`
- [ ] Update any hardcoded connection references
- [ ] Test full pipeline with environment variables
- [ ] Deploy to 5X workspace

## Troubleshooting

### Common Issues

1. **Missing Environment Variables**
   ```bash
   # Check if variables are set
   echo $FIVEX_SNOWFLAKE_ACCOUNT
   ```

2. **Private Key Issues**
   ```bash
   # Verify key file exists and is readable
   ls -la /path/to/your/rsa_key.p8
   ```

3. **Connection Issues**
   ```bash
   # Run the test script
   python test_env_connection.py
   ```

### Getting Help

If you encounter issues:
1. Check the test script output
2. Verify environment variables are set correctly
3. Ensure private key file is accessible
4. Check Snowflake account permissions

## Next Steps

1. **Set up environment variables** following the template
2. **Test the connection** using the test script
3. **Update your pipeline code** to use environment variables
4. **Deploy to 5X workspace** with confidence!

This approach follows the exact same pattern as your successful HRMS project, so it should work seamlessly in the 5X workspace environment.


