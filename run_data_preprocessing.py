#!/usr/bin/env python3
"""
Data Preprocessing Script - Phase 2

This script uses the existing InputDataPrepper to process raw data and compute
all regressor features, then saves the processed data to Snowflake.

Data Flow:
Raw Data (Snowflake) → InputDataPrepper → Processed Data (Snowflake)
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

from data.input_data_prepper import InputDataPrepper
from data.loader import DataLoader
from data.config.paths import DataConfig
from data.access.snowflake_accessor import SnowflakeAccessor

def main():
    """Main preprocessing function."""
    
    print("🚀 Starting Data Preprocessing - Phase 2")
    print("=" * 60)
    
    try:
        # Step 1: Load configuration
        print("📋 Loading configuration...")
        data_config = DataConfig()
        config = data_config.config
        
        # Initialize Snowflake accessor
        snowflake_config = config['snowflake']
        snowflake_accessor = SnowflakeAccessor(snowflake_config)
        
        print("✅ Configuration loaded successfully")
        
        # Step 2: Load raw data from Snowflake
        print("📊 Loading raw data from Snowflake...")
        
        # Initialize data loader with config file path
        loader = DataLoader()
        
        # Load raw outflow data
        print("   Loading outflow data...")
        outflow_df = loader.load_outflow()
        print(f"   ✅ Loaded {len(outflow_df)} rows of outflow data")
        
        # Load product master data
        print("   Loading product master data...")
        product_master_df = loader.load_product_master()
        print(f"   ✅ Loaded {len(product_master_df)} rows of product master data")
        
        # Step 3: Initialize InputDataPrepper
        print("🔧 Initializing InputDataPrepper...")
        prepper = InputDataPrepper()
        
        # Show enabled regressors
        enabled_regressors = prepper.get_enabled_regressors()
        print(f"   ✅ Enabled regressors: {enabled_regressors}")
        
        # Step 4: Process data with InputDataPrepper
        print("⚙️ Processing data with InputDataPrepper...")
        print("   Computing regressor features...")
        
        processed_df = prepper.prepare_data(outflow_df, product_master_df)
        
        print(f"   ✅ Data processing completed")
        print(f"   📊 Original shape: {outflow_df.shape}")
        print(f"   📊 Processed shape: {processed_df.shape}")
        print(f"   📊 New columns: {set(processed_df.columns) - set(outflow_df.columns)}")
        
        # Step 5: Save processed data to Snowflake
        print("💾 Saving processed data to Snowflake...")
        
        # Get the target table name
        target_table = "PROCESSED_DATA_WITH_REGRESSORS"
        
        # Save to Snowflake (snowflake_accessor will handle UPPERCASE conversion)
        snowflake_accessor.write_data(processed_df, target_table, if_exists='replace')
        
        print(f"   ✅ Processed data saved to {target_table}")
        
        # Step 6: Verify the saved data
        print("🔍 Verifying saved data...")
        
        # Read back a sample to verify
        sample_df = snowflake_accessor.read_data(target_table, columns=['product_id', 'location_id', 'date', 'demand', 'outflow'])
        print(f"   ✅ Verification successful - {len(sample_df)} rows in target table")
        
        # Show sample of processed data
        print("\n📋 Sample of processed data:")
        print(sample_df.head())
        
        # Step 7: Summary
        print("\n🎉 Data Preprocessing Completed Successfully!")
        print("=" * 60)
        print(f"✅ Raw data loaded: {len(outflow_df)} rows")
        print(f"✅ Product master loaded: {len(product_master_df)} rows")
        print(f"✅ Regressor features computed: {len(enabled_regressors)} features")
        print(f"✅ Processed data saved: {len(processed_df)} rows")
        print(f"✅ Target table: {target_table}")
        print(f"✅ New columns added: {len(set(processed_df.columns) - set(outflow_df.columns))}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data preprocessing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🚀 Phase 2 completed! Ready for Phase 3...")
    else:
        print("\n🛑 Phase 2 failed. Please check the errors above.")
        sys.exit(1)
