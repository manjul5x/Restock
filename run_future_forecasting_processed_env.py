#!/usr/bin/env python3
"""
Future Forecasting Script - Environment Variable Version for 5X Workspace
Generates forecasts using processed data with regressor features.

This script:
1. Loads processed data from TRANSFORMATION schema (PROCESSED_DATA_WITH_REGRESSORS)
2. Loads product master from STAGE schema (PRODUCT_MASTER_BHASIN)
3. Generates forecasts for each product using moving average
4. Saves forecasts to TRANSFORMATION schema (FUTURE_PREDICTIONS_RESULTS)

Usage:
    python run_future_forecasting_processed_env.py [options]

Options:
    --forecast-date DATE     Date to forecast from (YYYY-MM-DD)
    --table-mode MODE        Table mode: 'truncate' or 'append'
    --verbose                Enable verbose logging
"""

import sys
import os
import argparse
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

# Import environment-based components
from data.loader_env import EnvDataLoader

# Handle ProductMasterSchema import with fallback
try:
    from forecaster.validation.product_master_schema import ProductMasterSchema
except ImportError:
    # Create a simple fallback class with the required method
    class ProductMasterSchema:
        @staticmethod
        def get_risk_period_days(frequency: str, risk_period: int) -> int:
            """
            Convert frequency and risk period to total days
            
            Args:
                frequency: 'd', 'w', or 'm'
                risk_period: Integer multiple of frequency
                
            Returns:
                Total days
            """
            if frequency == "d":
                return risk_period
            elif frequency == "w":
                return risk_period * 7
            elif frequency == "m":
                return risk_period * 30
            else:
                raise ValueError(f"Unknown frequency: {frequency}")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProcessedDataFutureForecaster:
    """
    Future forecaster using processed data with regressor features.
    Environment variable version for 5X workspace deployment.
    """
    
    def __init__(self, 
                 forecast_date: Optional[str] = None,
                 table_mode: str = 'truncate',
                 use_env_vars: bool = True):
        """
        Initialize the future forecaster.
        
        Args:
            forecast_date: Date to forecast from (YYYY-MM-DD format)
            table_mode: Table mode ('truncate' or 'append')
            use_env_vars: Use environment variables for configuration
        """
        self.forecast_date = forecast_date or datetime.now().strftime('%Y-%m-%d')
        self.table_mode = table_mode
        self.use_env_vars = use_env_vars
        
        # Initialize data loader
        if use_env_vars:
            self.data_loader = EnvDataLoader()
        else:
            raise ValueError("This forecaster requires use_env_vars=True for 5X workspace deployment")
        
        # Results storage
        self.future_predictions: List[Dict[str, Any]] = []
        
        # Validate environment variables
        if use_env_vars:
            self._validate_environment()
    
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
        
        logger.info("✅ Environment variables validated")
    
    def run_forecasting(self) -> Dict[str, Any]:
        """
        Run the complete future forecasting pipeline.
        
        Returns:
            Dictionary containing forecasting results
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"🚀 Starting Future Forecasting - Environment Variable Version")
            logger.info(f"📅 Forecast Date: {self.forecast_date}")
            logger.info(f"🗄️ Table Mode: {self.table_mode}")
            
            # Load product master from STAGE schema
            logger.info("📥 Loading product master from STAGE schema...")
            product_master_df = self.data_loader.load_product_master()
            logger.info(f"✅ Loaded {len(product_master_df)} products")
            
            # Debug: Print product master columns
            logger.info(f"Product master columns: {list(product_master_df.columns)}")
            logger.info(f"First few rows:\n{product_master_df.head()}")
            
            # Validate product master columns
            self._validate_product_master_columns(product_master_df)
            
            # Load processed data from TRANSFORMATION schema
            logger.info("📥 Loading processed data from TRANSFORMATION schema...")
            processed_data = self.data_loader.load_processed_data()
            logger.info(f"✅ Loaded {len(processed_data)} rows of processed data")
            logger.info(f"Processed data columns: {list(processed_data.columns)}")
            logger.info(f"First few rows of processed data:\n{processed_data.head()}")
            
            # Ensure date column is properly converted to datetime
            if 'date' in processed_data.columns:
                logger.info("Converting date column to datetime...")
                processed_data['date'] = pd.to_datetime(processed_data['date'], errors='coerce')
                logger.info(f"Date column converted. Sample dates: {processed_data['date'].head()}")
                
                # Check for any NaT (Not a Time) values
                nat_count = processed_data['date'].isna().sum()
                if nat_count > 0:
                    logger.warning(f"Found {nat_count} invalid date values that were converted to NaT")
            else:
                logger.warning("No 'date' column found in processed data")
            
            # Process each product
            successful_products = 0
            total_products = len(product_master_df)
            
            for idx, product_record in product_master_df.iterrows():
                try:
                    # Validate product configuration
                    if not self._validate_product_config(product_record):
                        try:
                            product_id = product_record['product_id']
                        except KeyError:
                            product_id = 'unknown'
                        logger.warning(f"⚠️ Skipping product {product_id} - invalid configuration")
                        continue
                    
                    # Process product
                    self._process_product_enhanced(product_record, processed_data)
                    successful_products += 1
                    
                    if (idx + 1) % 50 == 0:
                        logger.info(f"📊 Processed {idx + 1}/{total_products} products")
                        
                except Exception as e:
                    try:
                        product_id = product_record['product_id']
                    except KeyError:
                        product_id = 'unknown'
                    logger.error(f"❌ Failed to process product {product_id}: {e}")
                    continue
            
            # Save results to TRANSFORMATION schema
            logger.info("💾 Saving forecast results to TRANSFORMATION schema...")
            self._save_results()
            
            # Calculate metrics
            execution_time = (datetime.now() - start_time).total_seconds()
            success_rate = successful_products / total_products if total_products > 0 else 0
            
            results = {
                'status': 'success',
                'forecast_date': self.forecast_date,
                'products_processed': total_products,
                'successful_products': successful_products,
                'predictions_generated': len(self.future_predictions),
                'execution_time_seconds': execution_time,
                'success_rate': success_rate,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"✅ Future forecasting completed successfully!")
            logger.info(f"📊 Products processed: {successful_products}/{total_products}")
            logger.info(f"🔮 Predictions generated: {len(self.future_predictions)}")
            logger.info(f"📈 Success rate: {success_rate:.1%}")
            logger.info(f"⏰ Execution time: {execution_time:.1f}s")
            
            return results
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"❌ Future forecasting failed: {e}")
            
            return {
                'status': 'failed',
                'forecast_date': self.forecast_date,
                'products_processed': 0,
                'successful_products': 0,
                'predictions_generated': 0,
                'execution_time_seconds': execution_time,
                'success_rate': 0.0,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _validate_product_master_columns(self, df: pd.DataFrame):
        """Validate product master DataFrame has required columns"""
        required_columns = ['product_id', 'location_id', 'demand_frequency', 'risk_period']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Product master missing required columns: {missing_columns}")
        
        logger.info("✅ Product master columns validated")
    
    def _validate_product_config(self, product_record: pd.Series) -> bool:
        """Validate product configuration"""
        try:
            # Debug: Print available columns and first few values
            logger.info(f"Available columns: {list(product_record.index)}")
            logger.info(f"Product record type: {type(product_record)}")
            logger.info(f"Product record values: {product_record.to_dict()}")
            
            product_id = product_record['product_id']
            location_id = product_record['location_id']
            demand_frequency = product_record['demand_frequency']
            risk_period = product_record['risk_period']
            
            # Validate demand frequency
            if demand_frequency not in ['d', 'w', 'm']:
                logger.warning(f"Invalid demand_frequency '{demand_frequency}' for product {product_id}")
                return False
            
            # Validate risk period
            if not isinstance(risk_period, (int, float)) or risk_period <= 0:
                logger.warning(f"Invalid risk_period '{risk_period}' for product {product_id}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating product config: {e}")
            return False
    
    def _process_product_enhanced(self, product_record: pd.Series, processed_data: pd.DataFrame):
        """Process individual product with enhanced forecasting"""
        try:
            logger.info(f"Processing product: {product_record['product_id']}")
            
            product_id = product_record['product_id']
            location_id = product_record['location_id']
            demand_frequency = product_record['demand_frequency']
            risk_period = product_record['risk_period']
            
            logger.info(f"Product config: {product_id}, {location_id}, {demand_frequency}, {risk_period}")
            
            # Calculate risk period in days
            risk_period_days = ProductMasterSchema.get_risk_period_days(demand_frequency, risk_period)
            
            # Filter data for this product
            product_filter = (
                (processed_data['product_id'] == product_id) &
                (processed_data['location_id'] == location_id)
            )
            product_data = processed_data[product_filter].copy()
            
            if len(product_data) == 0:
                logger.warning(f"No data found for product {product_id} at location {location_id}")
                return
            
            # Log data types for debugging
            logger.info(f"Product {product_id} data types: {product_data.dtypes.to_dict()}")
            if 'date' in product_data.columns:
                logger.info(f"Date column sample: {product_data['date'].head()}")
                logger.info(f"Date column type: {type(product_data['date'].iloc[0]) if len(product_data) > 0 else 'No data'}")
            
            # Filter to historical data only (before forecast date)
            try:
                forecast_date_dt = pd.to_datetime(self.forecast_date)
                
                # Ensure date column is datetime for comparison
                if 'date' in product_data.columns:
                    product_data['date'] = pd.to_datetime(product_data['date'], errors='coerce')
                
                # Filter out any NaT values and get historical data
                valid_dates = product_data['date'].notna()
                historical_data = product_data[valid_dates & (product_data['date'] < forecast_date_dt)].copy()
                
                training_data = historical_data
                
            except Exception as e:
                logger.error(f"Error filtering historical data for product {product_id}: {e}")
                logger.error(f"Date column info: {product_data['date'].dtype if 'date' in product_data.columns else 'No date column'}")
                return
            
            if len(training_data) == 0:
                logger.warning(f"No historical data found for product {product_id} before {self.forecast_date}")
                return
            
            # Generate risk-period aggregated forecast
            risk_period_forecast = self._generate_risk_period_forecast(
                training_data, risk_period_days, demand_frequency
            )
            
            # Calculate daily forecast
            daily_forecast = risk_period_forecast / risk_period_days
            
            # Create prediction record
            prediction_record = {
                'product_id': product_id,
                'location_id': location_id,
                'forecasted_on': self.forecast_date,
                'forecast_period_start': self.forecast_date,
                'forecast_period_end': (forecast_date_dt + pd.Timedelta(days=risk_period_days)).strftime('%Y-%m-%d'),
                'risk_period_days': risk_period_days,
                'demand_frequency': demand_frequency,
                'forecast_method': 'moving_average',
                'predicted_outflow_total': risk_period_forecast,
                'predicted_outflow_daily': daily_forecast,
                'training_period_start': training_data['date'].min().strftime('%Y-%m-%d'),
                'training_period_end': training_data['date'].max().strftime('%Y-%m-%d'),
                'training_data_points': len(training_data)
            }
            
            self.future_predictions.append(prediction_record)
            
        except Exception as e:
            logger.error(f"Error processing product {product_record['product_id']}: {e}")
            raise
    
    def _generate_risk_period_forecast(self, training_data: pd.DataFrame, 
                                     risk_period_days: int, demand_frequency: str) -> float:
        """Generate risk-period aggregated forecast using moving average"""
        try:
            # Calculate historical outflow average
            outflow_average = training_data['outflow'].mean()
            
            # The outflow already represents risk-period aggregated value
            # No need to multiply by risk_period_days (avoids double-counting)
            risk_period_forecast = outflow_average
            
            return risk_period_forecast
            
        except Exception as e:
            logger.error(f"Error generating forecast: {e}")
            return 0.0
    
    def _save_results(self):
        """Save future predictions to TRANSFORMATION schema"""
        try:
            if not self.future_predictions:
                logger.warning("No predictions to save")
                return
            
            # Convert to DataFrame
            predictions_df = pd.DataFrame(self.future_predictions)
            
            # Clear table if truncate mode
            if self.table_mode == 'truncate':
                self._clear_table()
                if_exists_mode = 'replace'
            else:
                if_exists_mode = 'append'
            
            # Save to TRANSFORMATION schema
            self.data_loader.save_future_predictions(predictions_df, if_exists=if_exists_mode)
            
            logger.info(f"✅ Saved {len(predictions_df)} predictions to TRANSFORMATION schema")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise
    
    def _clear_table(self):
        """Clear the future predictions table"""
        try:
            table_name = os.getenv("FIVEX_SNOWFLAKE_FUTURE_PREDICTIONS_TABLE", "FUTURE_PREDICTIONS_RESULTS")
            write_schema = os.getenv("FIVEX_SNOWFLAKE_WRITE_SCHEMA", "TRANSFORMATION")
            
            query = f"TRUNCATE TABLE {write_schema}.{table_name}"
            self.data_loader.accessor.execute_query(query)
            
            logger.info(f"✅ Cleared table {write_schema}.{table_name}")
            
        except Exception as e:
            logger.warning(f"Failed to clear table: {e}")


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description='Future Forecasting - Environment Variable Version for 5X Workspace'
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
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize forecaster
        forecaster = ProcessedDataFutureForecaster(
            forecast_date=args.forecast_date,
            table_mode=args.table_mode,
            use_env_vars=True
        )
        
        # Run forecasting
        results = forecaster.run_forecasting()
        
        # Exit with appropriate code
        if results['status'] == 'success':
            sys.exit(0)
        else:
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Forecasting failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
