#!/usr/bin/env python3
"""
Complete Forecasting Pipeline - Data Preprocessing + Future Forecasting

This script runs the complete forecasting pipeline in two phases:
1. Data Preprocessing: Processes raw data and computes regressor features
2. Future Forecasting: Generates forecasts using processed data with regressor features

The pipeline ensures that data preprocessing is completed successfully before
proceeding to future forecasting, providing a seamless end-to-end workflow.

Usage:
    python run_complete_forecasting_pipeline.py [options]

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
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

# Import the individual pipeline components
from run_data_preprocessing import main as run_preprocessing
from run_future_forecasting_processed import ProcessedDataFutureForecaster


class CompleteForecastingPipeline:
    """
    Complete forecasting pipeline that orchestrates data preprocessing and future forecasting.
    
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
            forecast_date: Date to forecast from (default: today)
            table_mode: 'truncate' to clear tables first, 'append' to add to existing data
            verbose: Enable verbose logging
            skip_preprocessing: Skip data preprocessing step
            skip_forecasting: Skip future forecasting step
        """
        self.forecast_date = forecast_date
        self.table_mode = table_mode
        self.verbose = verbose
        self.skip_preprocessing = skip_preprocessing
        self.skip_forecasting = skip_forecasting
        
        # Pipeline results
        self.preprocessing_results = None
        self.forecasting_results = None
        
        print("🚀 Complete Forecasting Pipeline Initialized")
        print("=" * 60)
        print(f"📅 Forecast Date: {self.forecast_date or 'Today'}")
        print(f"🗃️  Table Mode: {self.table_mode}")
        print(f"🔧 Skip Preprocessing: {self.skip_preprocessing}")
        print(f"🔮 Skip Forecasting: {self.skip_forecasting}")
        print("=" * 60)
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete forecasting pipeline.
        
        Returns:
            Dictionary with complete pipeline results and statistics
        """
        pipeline_start_time = time.time()
        print(f"\n🚀 Starting Complete Forecasting Pipeline at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Phase 1: Data Preprocessing
            if not self.skip_preprocessing:
                print(f"\n📊 PHASE 1: DATA PREPROCESSING")
                print("-" * 40)
                preprocessing_start = time.time()
                
                try:
                    # Run data preprocessing
                    preprocessing_success = run_preprocessing()
                    preprocessing_time = time.time() - preprocessing_start
                    
                    if preprocessing_success:
                        self.preprocessing_results = {
                            'status': 'success',
                            'execution_time_seconds': preprocessing_time,
                            'timestamp': datetime.now().isoformat()
                        }
                        print(f"✅ Data preprocessing completed successfully in {preprocessing_time:.1f}s")
                    else:
                        self.preprocessing_results = {
                            'status': 'failed',
                            'execution_time_seconds': preprocessing_time,
                            'timestamp': datetime.now().isoformat()
                        }
                        print(f"❌ Data preprocessing failed after {preprocessing_time:.1f}s")
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
                    raise
            else:
                print(f"\n⏭️  PHASE 1: SKIPPED (Data Preprocessing)")
                print("-" * 40)
                self.preprocessing_results = {
                    'status': 'skipped',
                    'execution_time_seconds': 0,
                    'timestamp': datetime.now().isoformat()
                }
                print("✅ Data preprocessing skipped - using existing processed data")
            
            # Phase 2: Future Forecasting
            if not self.skip_forecasting:
                print(f"\n🔮 PHASE 2: FUTURE FORECASTING")
                print("-" * 40)
                forecasting_start = time.time()
                
                try:
                    # Initialize and run future forecaster
                    forecaster = ProcessedDataFutureForecaster(
                        forecast_date=self.forecast_date,
                        table_mode=self.table_mode
                    )
                    
                    # Run future forecasting
                    self.forecasting_results = forecaster.run_forecasting()
                    forecasting_time = time.time() - forecasting_start
                    
                    print(f"✅ Future forecasting completed successfully in {forecasting_time:.1f}s")
                    
                except Exception as e:
                    forecasting_time = time.time() - forecasting_start
                    self.forecasting_results = {
                        'status': 'failed',
                        'execution_time_seconds': forecasting_time,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    }
                    print(f"❌ Future forecasting failed: {e}")
                    raise
            else:
                print(f"\n⏭️  PHASE 2: SKIPPED (Future Forecasting)")
                print("-" * 40)
                self.forecasting_results = {
                    'status': 'skipped',
                    'execution_time_seconds': 0,
                    'timestamp': datetime.now().isoformat()
                }
                print("✅ Future forecasting skipped")
            
            # Calculate total execution time
            total_time = time.time() - pipeline_start_time
            
            # Compile final results
            final_results = {
                'pipeline_status': 'success',
                'total_execution_time_seconds': total_time,
                'preprocessing': self.preprocessing_results,
                'forecasting': self.forecasting_results,
                'timestamp': datetime.now().isoformat()
            }
            
            # Print final summary
            self._print_final_summary(final_results)
            
            return final_results
            
        except Exception as e:
            total_time = time.time() - pipeline_start_time
            print(f"\n❌ Complete pipeline failed after {total_time:.1f}s: {e}")
            
            return {
                'pipeline_status': 'failed',
                'total_execution_time_seconds': total_time,
                'preprocessing': self.preprocessing_results,
                'forecasting': self.forecasting_results,
                'error': str(e),
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
        description='Complete Forecasting Pipeline - Data Preprocessing + Future Forecasting',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run complete pipeline (preprocessing + forecasting)
    python run_complete_forecasting_pipeline.py
    
    # Run with specific forecast date
    python run_complete_forecasting_pipeline.py --forecast-date 2025-10-17
    
    # Run in append mode (don't clear existing data)
    python run_complete_forecasting_pipeline.py --table-mode append
    
    # Only run preprocessing (skip forecasting)
    python run_complete_forecasting_pipeline.py --skip-forecasting
    
    # Only run forecasting (skip preprocessing, use existing processed data)
    python run_complete_forecasting_pipeline.py --skip-preprocessing
    
    # Run with verbose logging
    python run_complete_forecasting_pipeline.py --verbose
        """
    )
    
    parser.add_argument('--forecast-date', type=str, 
                       help='Date to forecast from (YYYY-MM-DD). Default: today')
    parser.add_argument('--table-mode', choices=['truncate', 'append'], default='truncate',
                       help='Table mode: truncate (clear tables first) or append (add to existing data). Default: truncate')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--skip-preprocessing', action='store_true',
                       help='Skip data preprocessing step (use existing processed data)')
    parser.add_argument('--skip-forecasting', action='store_true',
                       help='Skip future forecasting step (only run preprocessing)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.skip_preprocessing and args.skip_forecasting:
        print("❌ Error: Cannot skip both preprocessing and forecasting. At least one phase must run.")
        sys.exit(1)
    
    # Initialize and run complete pipeline
    pipeline = CompleteForecastingPipeline(
        forecast_date=args.forecast_date,
        table_mode=args.table_mode,
        verbose=args.verbose,
        skip_preprocessing=args.skip_preprocessing,
        skip_forecasting=args.skip_forecasting
    )
    
    try:
        results = pipeline.run_complete_pipeline()
        
        # Exit with appropriate code
        if results['pipeline_status'] == 'success':
            print(f"\n🎉 Pipeline completed successfully!")
            sys.exit(0)
        else:
            print(f"\n❌ Pipeline failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n⏹️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Pipeline failed with unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
