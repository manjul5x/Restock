#!/usr/bin/env python3
"""
Complete Forecasting Pipeline - Environment Variable Version for 5X Workspace
Data Preprocessing + Future Forecasting with Multi-Schema Support

This script runs the complete forecasting pipeline in two phases:
1. Data Preprocessing: Processes raw data and computes regressor features
2. Future Forecasting: Generates forecasts using processed data with regressor features

The pipeline uses environment variables for configuration and supports:
- Multi-schema operations (read from STAGE, write to TRANSFORMATION)
- Private key authentication
- 5X workspace deployment

Usage:
    python run_complete_forecasting_pipeline_env.py [options]

Options:
    --forecast-date DATE     Date to forecast from (YYYY-MM-DD)
    --table-mode MODE        Table mode: 'truncate' or 'append'
    --verbose                Enable verbose logging
    --skip-preprocessing     Skip data preprocessing step (use existing processed data)
    --skip-forecasting       Skip future forecasting step (only run preprocessing)
"""

import sys
import argparse
import time
import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

# Import the individual pipeline components
from run_data_preprocessing_env import main as run_preprocessing
from run_future_forecasting_processed_env import ProcessedDataFutureForecaster

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CompleteForecastingPipelineEnv:
    """
    Complete forecasting pipeline using environment variables for 5X workspace deployment.
    
    This class provides a unified interface for running the complete forecasting workflow,
    ensuring proper sequencing and error handling between the two phases.
    """
    
    def __init__(self, 
                 forecast_date: Optional[str] = None,
                 table_mode: str = 'truncate',
                 verbose: bool = False,
                 skip_preprocessing: bool = False,
                 skip_forecasting: bool = False):
        """
        Initialize the complete forecasting pipeline.
        
        Args:
            forecast_date: Date to forecast from (YYYY-MM-DD format)
            table_mode: Table mode ('truncate' or 'append')
            verbose: Enable verbose logging
            skip_preprocessing: Skip data preprocessing step
            skip_forecasting: Skip future forecasting step
        """
        self.forecast_date = forecast_date or datetime.now().strftime('%Y-%m-%d')
        self.table_mode = table_mode
        self.verbose = verbose
        self.skip_preprocessing = skip_preprocessing
        self.skip_forecasting = skip_forecasting
        
        # Results storage
        self.preprocessing_results = None
        self.forecasting_results = None
        
        # Validate environment variables
        self._validate_environment()
        
        # Display initialization information (consistent with original)
        print("🚀 Complete Forecasting Pipeline Initialized (Environment Variables)")
        print("=" * 60)
        print(f"📅 Forecast Date: {self.forecast_date or 'Today'}")
        print(f"🗃️  Table Mode: {self.table_mode}")
        print(f"🔧 Skip Preprocessing: {self.skip_preprocessing}")
        print(f"🔮 Skip Forecasting: {self.skip_forecasting}")
        print(f"📝 Verbose Logging: {self.verbose}")
        print("=" * 60)
    
    def _validate_environment(self):
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
        
        if self.verbose:
            print("✅ Environment variables validated")
            logger.info("Environment variables validated successfully")
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete forecasting pipeline.
        
        Returns:
            Dictionary containing pipeline results
        """
        start_time = time.time()
        
        try:
            print(f"\n🚀 Starting Complete Forecasting Pipeline")
            print(f"📅 Forecast Date: {self.forecast_date}")
            print(f"🗄️ Table Mode: {self.table_mode}")
            print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            logger.info(f"Starting Complete Forecasting Pipeline - Environment Variable Version")
            logger.info(f"Forecast Date: {self.forecast_date}")
            logger.info(f"Table Mode: {self.table_mode}")
            logger.info(f"Skip Preprocessing: {self.skip_preprocessing}")
            logger.info(f"Skip Forecasting: {self.skip_forecasting}")
            
            # Phase 1: Data Preprocessing
            if not self.skip_preprocessing:
                print(f"\n📊 PHASE 1: DATA PREPROCESSING")
                preprocessing_start = time.time()
                
                try:
                    logger.info("Starting data preprocessing phase...")
                    preprocessing_success = run_preprocessing()
                    preprocessing_time = time.time() - preprocessing_start
                    
                    if preprocessing_success:
                        print(f"✅ Data preprocessing completed successfully in {preprocessing_time:.1f}s")
                        logger.info(f"Data preprocessing completed successfully in {preprocessing_time:.1f}s")
                        self.preprocessing_results = {
                            'status': 'success',
                            'execution_time_seconds': preprocessing_time,
                            'timestamp': datetime.now().isoformat()
                        }
                    else:
                        self.preprocessing_results = {
                            'status': 'failed',
                            'execution_time_seconds': preprocessing_time,
                            'timestamp': datetime.now().isoformat(),
                            'error': 'Preprocessing returned False'
                        }
                        print(f"❌ Data preprocessing failed after {preprocessing_time:.1f}s")
                        logger.error("Data preprocessing returned False")
                        raise Exception("Data preprocessing failed")
                        
                except Exception as e:
                    preprocessing_time = time.time() - preprocessing_start
                    self.preprocessing_results = {
                        'status': 'failed',
                        'execution_time_seconds': preprocessing_time,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    }
                    print(f"❌ Data preprocessing failed: {e}")
                    logger.error(f"Data preprocessing failed: {e}")
                    raise
            else:
                print(f"⏭️ Skipping data preprocessing (using existing processed data)")
                self.preprocessing_results = {
                    'status': 'skipped',
                    'execution_time_seconds': 0,
                    'timestamp': datetime.now().isoformat()
                }
            
            # Phase 2: Future Forecasting
            if not self.skip_forecasting:
                print(f"\n🔮 PHASE 2: FUTURE FORECASTING")
                forecasting_start = time.time()
                
                try:
                    logger.info("Starting future forecasting phase...")
                    # Initialize forecaster with environment variables
                    forecaster = ProcessedDataFutureForecaster(
                        forecast_date=self.forecast_date,
                        table_mode=self.table_mode,
                        use_env_vars=True
                    )
                    
                    # Run forecasting
                    self.forecasting_results = forecaster.run_forecasting()
                    forecasting_time = time.time() - forecasting_start
                    
                    if self.forecasting_results.get('status') == 'success':
                        print(f"✅ Future forecasting completed successfully in {forecasting_time:.1f}s")
                        print(f"📊 Products processed: {self.forecasting_results.get('successful_products', 0)}")
                        print(f"🔮 Predictions generated: {self.forecasting_results.get('predictions_generated', 0)}")
                        print(f"📈 Success rate: {self.forecasting_results.get('success_rate', 0):.1%}")
                        logger.info(f"Future forecasting completed successfully in {forecasting_time:.1f}s")
                        logger.info(f"Products processed: {self.forecasting_results.get('successful_products', 0)}")
                        logger.info(f"Predictions generated: {self.forecasting_results.get('predictions_generated', 0)}")
                    else:
                        print(f"❌ Future forecasting failed")
                        logger.error("Future forecasting failed")
                        raise Exception("Future forecasting failed")
                        
                except Exception as e:
                    forecasting_time = time.time() - forecasting_start
                    self.forecasting_results = {
                        'status': 'failed',
                        'execution_time_seconds': forecasting_time,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    }
                    print(f"❌ Future forecasting failed: {e}")
                    logger.error(f"Future forecasting failed: {e}")
                    raise
            else:
                print(f"⏭️ Skipping future forecasting")
                self.forecasting_results = {
                    'status': 'skipped',
                    'execution_time_seconds': 0,
                    'timestamp': datetime.now().isoformat()
                }
            
            # Calculate total execution time
            total_time = time.time() - start_time
            
            # Create success result
            result = {
                'pipeline_status': 'success',
                'total_execution_time_seconds': total_time,
                'preprocessing': self.preprocessing_results,
                'forecasting': self.forecasting_results,
                'timestamp': datetime.now().isoformat()
            }
            
            # Print final summary
            self._print_final_summary(result)
            
            return result
            
        except Exception as e:
            total_time = time.time() - start_time
            print(f"❌ Pipeline failed with error: {e}")
            return self._create_failure_result(start_time, f"Pipeline error: {e}")
    
    def _create_failure_result(self, start_time: float, error_message: str) -> Dict[str, Any]:
        """Create a failure result"""
        total_time = time.time() - start_time
        return {
            'pipeline_status': 'failed',
            'total_execution_time_seconds': total_time,
            'preprocessing': self.preprocessing_results or {'status': 'not_started'},
            'forecasting': self.forecasting_results or {'status': 'not_started'},
            'error': error_message,
            'timestamp': datetime.now().isoformat()
        }
    
    def _print_final_summary(self, results: Dict[str, Any]):
        """Print final pipeline summary."""
        print(f"\n🎉 COMPLETE FORECASTING PIPELINE SUMMARY")
        print("=" * 60)
        print(f"📅 Forecast Date: {self.forecast_date or 'Today'}")
        print(f"⏱️  Total Execution Time: {results['total_execution_time_seconds']:.1f}s ({results['total_execution_time_seconds']/60:.1f} minutes)")
        print(f"🕐 Completed At: {results['timestamp']}")
        
        # Preprocessing summary
        if self.preprocessing_results:
            prep_status = self.preprocessing_results['status']
            prep_time = self.preprocessing_results.get('execution_time_seconds', 0)
            print(f"\n📊 Data Preprocessing: {prep_status.upper()}")
            if prep_status == 'success':
                print(f"   ⏱️  Execution Time: {prep_time:.1f}s")
            elif prep_status == 'failed':
                print(f"   ❌ Failed after {prep_time:.1f}s")
            elif prep_status == 'skipped':
                print(f"   ⏭️  Skipped (using existing data)")
        
        # Forecasting summary
        if self.forecasting_results:
            forecast_status = self.forecasting_results.get('status', 'unknown')
            forecast_time = self.forecasting_results.get('execution_time_seconds', 0)
            print(f"\n🔮 Future Forecasting: {forecast_status.upper()}")
            if forecast_status == 'success':
                print(f"   ⏱️  Execution Time: {forecast_time:.1f}s")
                print(f"   📊 Products Processed: {self.forecasting_results.get('successful_products', 0)}")
                print(f"   🔮 Predictions Generated: {self.forecasting_results.get('predictions_generated', 0)}")
                print(f"   📈 Success Rate: {self.forecasting_results.get('success_rate', 0):.1f}%")
            elif forecast_status == 'failed':
                print(f"   ❌ Failed after {forecast_time:.1f}s")
            elif forecast_status == 'skipped':
                print(f"   ⏭️  Skipped")
        
        print("=" * 60)


def main():
    """Main function to run the complete forecasting pipeline."""
    parser = argparse.ArgumentParser(
        description='Complete Forecasting Pipeline - Environment Variable Version for 5X Workspace'
    )
    
    parser.add_argument(
        '--forecast-date',
        type=str,
        default=None,
        help='Date to forecast from (YYYY-MM-DD format). Defaults to today.'
    )
    
    parser.add_argument(
        '--table-mode',
        type=str,
        choices=['truncate', 'append'],
        default='truncate',
        help='Table mode: truncate (clear tables) or append (add to existing). Default: truncate'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--skip-preprocessing',
        action='store_true',
        help='Skip data preprocessing step (use existing processed data)'
    )
    
    parser.add_argument(
        '--skip-forecasting',
        action='store_true',
        help='Skip future forecasting step (only run preprocessing)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments (consistent with original)
    if args.skip_preprocessing and args.skip_forecasting:
        print("❌ Error: Cannot skip both preprocessing and forecasting. At least one phase must run.")
        sys.exit(1)
    
    try:
        # Initialize pipeline
        logger.info("Initializing Complete Forecasting Pipeline - Environment Variable Version")
        pipeline = CompleteForecastingPipelineEnv(
            forecast_date=args.forecast_date,
            table_mode=args.table_mode,
            verbose=args.verbose,
            skip_preprocessing=args.skip_preprocessing,
            skip_forecasting=args.skip_forecasting
        )
        
        # Execute pipeline
        results = pipeline.run_complete_pipeline()
        
        # Exit with appropriate code
        if results['pipeline_status'] == 'success':
            print(f"\n🎉 Pipeline completed successfully!")
            logger.info("Pipeline completed successfully")
            sys.exit(0)
        else:
            print(f"\n❌ Pipeline failed!")
            logger.error("Pipeline failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n⏹️  Pipeline interrupted by user")
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Pipeline failed with unexpected error: {e}")
        logger.error(f"Pipeline failed with unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
