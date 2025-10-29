#!/usr/bin/env python3
"""
Data Preprocessing Script - Environment Variable Version for 5X Workspace
Processes raw demand data and computes regressor features using environment variables.

This script:
1. Loads raw data from STAGE schema (OUTFLOW_BHASIN, PRODUCT_MASTER_BHASIN)
2. Computes regressor features using InputDataPrepper
3. Saves processed data to TRANSFORMATION schema (PROCESSED_DATA_WITH_REGRESSORS)

Usage:
    python run_data_preprocessing_env.py
"""

import sys
import os
import logging
from pathlib import Path
from datetime import datetime

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

# Ensure forecaster module is accessible
forecaster_path = Path(__file__).parent / "forecaster"
if str(forecaster_path) not in sys.path:
    sys.path.append(str(forecaster_path))

# Import environment-based components
from data.loader_env import EnvDataLoader
from data.input_data_prepper import InputDataPrepper

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main preprocessing function using environment variables."""
    print("🚀 Starting Data Preprocessing - Environment Variable Version")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Validate environment variables
        _validate_environment()
        
        # Initialize environment-based data loader
        print("📊 Initializing environment-based data loader...")
        loader = EnvDataLoader()
        
        # Load raw data from STAGE schema
        print("📥 Loading raw data from STAGE schema...")
        outflow_df = loader.load_outflow()
        product_master_df = loader.load_product_master()
        
        print(f"✅ Loaded {len(outflow_df)} rows from OUTFLOW_BHASIN")
        print(f"✅ Loaded {len(product_master_df)} rows from PRODUCT_MASTER_BHASIN")
        
        # Initialize InputDataPrepper
        print("🔧 Initializing InputDataPrepper...")
        prepper = InputDataPrepper()
        enabled_regressors = prepper.get_enabled_regressors()
        print(f"✅ Enabled regressors: {', '.join(enabled_regressors)}")
        
        # Process data with regressor features
        print("⚙️ Processing data and computing regressor features...")
        processed_df = prepper.prepare_data(outflow_df, product_master_df)
        
        print(f"✅ Processed data: {len(processed_df)} rows, {len(processed_df.columns)} columns")
        print(f"📊 Original columns: {len(outflow_df.columns)}")
        print(f"📊 New columns: {len(processed_df.columns) - len(outflow_df.columns)}")
        
        # Save processed data to TRANSFORMATION schema
        print("💾 Saving processed data to TRANSFORMATION schema...")
        loader.save_processed_data(processed_df, if_exists='replace')
        
        print("✅ Data preprocessing completed successfully!")
        print(f"📊 Final processed data: {len(processed_df)} rows, {len(processed_df.columns)} columns")
        print(f"🏁 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Data preprocessing failed: {e}")
        print(f"❌ Data preprocessing failed: {e}")
        return False


def _validate_environment():
    """Validate required environment variables"""
    required_vars = [
        'FIVEX_SNOWFLAKE_ACCOUNT',
        'FIVEX_SNOWFLAKE_USER',
        'FIVEX_SNOWFLAKE_DATABASE',
        'FIVEX_SNOWFLAKE_READ_SCHEMA',
        'FIVEX_SNOWFLAKE_WRITE_SCHEMA',
        'FIVEX_SNOWFLAKE_PRIVATE_KEY_FILE',
        'FIVEX_SNOWFLAKE_PRIVATE_KEY_FILE_PWD'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    print("✅ Environment variables validated")
    
    # Print schema configuration
    read_schema = os.getenv('FIVEX_SNOWFLAKE_READ_SCHEMA', 'STAGE')
    write_schema = os.getenv('FIVEX_SNOWFLAKE_WRITE_SCHEMA', 'TRANSFORMATION')
    print(f"📖 Read Schema: {read_schema}")
    print(f"📝 Write Schema: {write_schema}")


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
