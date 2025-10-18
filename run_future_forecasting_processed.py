#!/usr/bin/env python3
"""
Future Forecasting Script - Using Processed Data with Regressor Features

This script generates future predictions using the processed data with regressor features
from the PROCESSED_DATA_WITH_REGRESSORS table. It leverages the rich feature set
computed by InputDataPrepper for improved forecasting accuracy.

Key Features:
- Uses processed data with regressor features (outflow, rp_lag, half_rp_lag, season, etc.)
- Leverages forward-looking outflow aggregation for better accuracy
- Uses same product configurations as backtesting
- Enhanced forecasting with temporal and seasonal patterns

Key Changes from Original:
1. Data Source: Uses processed_data table instead of raw outflow
2. Target Variable: Uses 'outflow' column instead of 'demand'
3. Regressor Features: Leverages all computed regressor features
4. Enhanced Logic: Incorporates temporal patterns and seasonality
"""

import sys
import argparse
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import warnings
import time

# Add forecaster package to path
sys.path.append(str(Path(__file__).parent))

try:
    from data.loader import DataLoader
    from forecaster.validation.product_master_schema import ProductMasterSchema
    from forecaster.forecasting.moving_average import MovingAverageModel
    from forecaster.utils.logger import configure_workflow_logging, get_logger
except ImportError as e:
    print(f"Error importing required modules: {e}")
    sys.exit(1)


def validate_production_columns(product_master_df: pd.DataFrame) -> None:
    """
    Validate that production data has required columns.
    
    Args:
        product_master_df: Product master DataFrame
        
    Raises:
        ValueError: If required columns are missing
    """
    required_columns = ['product_id', 'location_id', 'demand_frequency', 'risk_period']
    missing_columns = [col for col in required_columns if col not in product_master_df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns in product master: {missing_columns}")


class ProcessedDataFutureForecaster:
    """
    Future forecaster that uses processed data with regressor features.
    
    This forecaster leverages the rich feature set computed by InputDataPrepper
    to provide more accurate predictions using temporal patterns, seasonality,
    and forward-looking aggregation.
    """
    
    def __init__(self, forecast_date: str = None, data_loader: DataLoader = None, table_mode: str = 'truncate'):
        """
        Initialize the future forecaster.
        
        Args:
            forecast_date: Date to forecast from (default: today)
            data_loader: DataLoader instance (optional)
            table_mode: 'truncate' to clear table first, 'append' to add to existing data
        """
        self.forecast_date = forecast_date or date.today().strftime('%Y-%m-%d')
        self.data_loader = data_loader or DataLoader()
        self.future_predictions = []
        self.table_mode = table_mode
        
        # Setup logging
        self.logger = get_logger('processed_future_forecaster')
        
        self.logger.info(f"Initialized ProcessedDataFutureForecaster for forecast date: {self.forecast_date}")
        self.logger.info(f"Table mode: {table_mode}")
    
    def run_forecasting(self) -> Dict[str, Any]:
        """
        Run the complete future forecasting pipeline using processed data.
        
        Returns:
            Dictionary with forecasting results and statistics
        """
        start_time = time.time()
        self.logger.info("🚀 Starting processed data future forecasting pipeline")
        
        try:
            # Load product master
            self.logger.info("Loading product master...")
            product_master_df = self.data_loader.load_product_master()
            self.logger.info(f"Loaded {len(product_master_df)} products")
            
            # Validate production columns
            validate_production_columns(product_master_df)
            
            # Load processed data with regressor features
            self.logger.info("Loading processed data with regressor features...")
            data_loading_start = time.time()
            processed_data = self._load_processed_data(product_master_df)
            data_loading_time = time.time() - data_loading_start
            self.logger.info(f"Loaded {len(processed_data)} processed data records in {data_loading_time:.2f}s")
            print(f"📊 Data loading completed in {data_loading_time:.2f}s")
            
            # Process each product
            successful_products = 0
            skipped_products = 0
            processing_start_time = time.time()
            
            self.logger.info(f"Processing {len(product_master_df)} products...")
            
            for idx, product_record in product_master_df.iterrows():
                if idx % 50 == 0:  # Progress update every 50 products
                    elapsed = time.time() - processing_start_time
                    print(f"⏱️  Processed {idx}/{len(product_master_df)} products in {elapsed:.1f}s")
                
                try:
                    # Validate product configuration
                    if not self._validate_product_config(product_record):
                        skipped_products += 1
                        continue
                    
                    # Check if product has processed data
                    product_id = product_record['product_id']
                    location_id = product_record['location_id']
                    product_data = processed_data[
                        (processed_data['product_id'] == product_id) &
                        (processed_data['location_id'] == location_id)
                    ]
                    
                    if product_data.empty:
                        self.logger.debug(f"Skipping product {product_id} at {location_id} - no processed data found")
                        skipped_products += 1
                        continue
                    
                    # Check for valid outflow data (non-zero)
                    if product_data['outflow'].sum() == 0:
                        self.logger.debug(f"Skipping product {product_id} at {location_id} - no non-zero outflow data")
                        skipped_products += 1
                        continue
                    
                    # Process the product with enhanced forecasting
                    self._process_product_enhanced(product_record, product_data)
                    successful_products += 1
                    
                except Exception as e:
                    self.logger.error(f"Failed to process product {product_record['product_id']} at {product_record['location_id']}: {e}")
                    skipped_products += 1
                    continue
            
            processing_time = time.time() - processing_start_time
            self.logger.info(f"Successfully processed {successful_products} products out of {len(product_master_df)} (skipped: {skipped_products})")
            print(f"⏱️  Processing completed in {processing_time:.1f}s")
            print(f"✅ Successfully processed: {successful_products} products")
            print(f"❌ Skipped: {skipped_products} products")
            
            # Save results
            self._save_results()
            
            total_time = time.time() - start_time
            self.logger.info(f"Processed data future forecasting completed. Generated {len(self.future_predictions)} predictions")
            
            print(f"\n🎯 FINAL RESULTS:")
            print(f"⏱️  Total execution time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
            print(f"📊 Products processed: {successful_products}/{len(product_master_df)}")
            print(f"🔮 Predictions generated: {len(self.future_predictions)}")
            print(f"📈 Success rate: {(successful_products/len(product_master_df)*100):.1f}%")
            
            return {
                'status': 'success',
                'forecast_date': self.forecast_date,
                'products_processed': len(product_master_df),
                'successful_products': successful_products,
                'skipped_products': skipped_products,
                'predictions_generated': len(self.future_predictions),
                'execution_time_seconds': total_time,
                'success_rate': successful_products/len(product_master_df)*100
            }
            
        except Exception as e:
            self.logger.error(f"Processed data future forecasting pipeline failed: {e}")
            raise
    
    def _load_processed_data(self, product_master_df: pd.DataFrame) -> pd.DataFrame:
        """
        Load processed data with regressor features.
        
        Args:
            product_master_df: Product master DataFrame for filtering
            
        Returns:
            Processed data DataFrame with regressor features
        """
        try:
            # Load processed data from Snowflake
            processed_data = self.data_loader.accessor.read_data(
                table_name=self.data_loader.config['snowflake_tables']['processed_data']
            )
            
            # Convert column names to lowercase for pipeline compatibility
            processed_data.columns = processed_data.columns.str.lower()
            
            # Convert date column to datetime
            if 'date' in processed_data.columns:
                processed_data['date'] = pd.to_datetime(processed_data['date'])
            
            # Filter by product master for efficiency
            product_locations = set(
                product_master_df[["product_id", "location_id"]].apply(tuple, axis=1)
            )
            processed_data = processed_data[
                processed_data[["product_id", "location_id"]].apply(tuple, axis=1).isin(product_locations)
            ]
            
            self.logger.info(f"Filtered processed data to {len(processed_data)} records for {len(product_locations)} product-location combinations")
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Failed to load processed data: {e}")
            raise
    
    def _validate_product_config(self, product_record: Dict[str, Any]) -> bool:
        """
        Validate product configuration.
        
        Args:
            product_record: Product configuration record
            
        Returns:
            True if valid, False otherwise
        """
        demand_frequency = product_record.get('demand_frequency')
        risk_period = product_record.get('risk_period')
        
        if demand_frequency is None or demand_frequency not in ['d', 'w', 'm']:
            self.logger.debug(f"Invalid demand_frequency: {demand_frequency}")
            return False
        
        if risk_period is None or not isinstance(risk_period, (int, float)) or risk_period <= 0:
            self.logger.debug(f"Invalid risk_period: {risk_period}")
            return False
        
        return True
    
    def _process_product_enhanced(self, product_record: Dict[str, Any], product_data: pd.DataFrame):
        """
        Process a single product with enhanced forecasting using regressor features.
        
        Args:
            product_record: Product configuration
            product_data: Processed data with regressor features
        """
        product_id = product_record['product_id']
        location_id = product_record['location_id']
        
        if product_data.empty:
            self.logger.warning(f"No processed data found for product {product_id} at {location_id}")
            return
        
        # Check for valid outflow data
        if product_data['outflow'].sum() == 0:
            self.logger.warning(f"No non-zero outflow data for product {product_id} at {location_id}")
            return
        
        # Sort by date
        product_data = product_data.sort_values('date').reset_index(drop=True)
        
        # Get product configuration
        demand_frequency = product_record.get('demand_frequency')
        risk_period = product_record.get('risk_period')
        
        # Calculate risk period days
        try:
            risk_period_days = ProductMasterSchema.get_risk_period_days(
                demand_frequency,
                risk_period
            )
        except Exception as e:
            self.logger.warning(f"Failed to calculate risk period for product {product_id} at {location_id}: {e}. Using 7 days as fallback.")
            risk_period_days = 7
        
        # Generate future dates for risk period
        future_dates = self._generate_future_dates(risk_period_days, demand_frequency)
        
        # Prepare training data (historical data up to forecast date)
        forecast_date_dt = pd.to_datetime(self.forecast_date)
        training_data = product_data[product_data['date'] < forecast_date_dt].copy()
        
        if training_data.empty:
            self.logger.warning(f"No training data before {self.forecast_date} for product {product_id} at {location_id}")
            return
        
        # Check for valid training data
        if training_data['outflow'].sum() == 0:
            self.logger.warning(f"All training data is zero for product {product_id} at {location_id}")
            return
        
        # Generate risk-period aggregated forecast (single forecast per product)
        risk_period_forecast = self._generate_risk_period_forecast(
            training_data, 
            risk_period_days,
            demand_frequency
        )
        
        # Add single aggregated prediction to results
        prediction_record = {
            'product_id': product_id,
            'location_id': location_id,
            'forecasted_on': self.forecast_date,
            'forecast_period_start': self.forecast_date,
            'forecast_period_end': (pd.to_datetime(self.forecast_date) + timedelta(days=risk_period_days)).strftime('%Y-%m-%d'),
            'risk_period_days': risk_period_days,
            'demand_frequency': demand_frequency,
            'forecast_method': 'moving_average_aggregated',
            'predicted_outflow_total': round(risk_period_forecast, 2),
            'predicted_outflow_daily': round(risk_period_forecast / risk_period_days, 2) if risk_period_days > 0 else 0,
            'training_period_start': training_data['date'].min(),
            'training_period_end': training_data['date'].max(),
            'training_data_points': len(training_data)
        }
        
        self.future_predictions.append(prediction_record)
        
        self.logger.debug(f"Generated {len(future_dates)} enhanced predictions for product {product_id} at {location_id}")
    
    def _generate_risk_period_forecast(self, 
                                      training_data: pd.DataFrame, 
                                      risk_period_days: int,
                                      demand_frequency: str) -> float:
        """
        Generate risk-period aggregated forecast (single forecast per product).
        
        This solves the decimal forecasting issue by providing a single aggregated
        forecast for the entire risk period instead of daily decimal forecasts.
        
        Args:
            training_data: Historical training data
            risk_period_days: Number of days in risk period
            demand_frequency: 'd' for daily, 'w' for weekly, 'm' for monthly
            
        Returns:
            Single aggregated forecast value for the entire risk period
        """
        # Calculate historical outflow average
        # NOTE: outflow already represents the sum over risk period (120 days)
        # So we should NOT multiply by risk_period_days again!
        outflow_average = training_data['outflow'].mean()
        
        # The outflow average is already the risk-period aggregated value
        # No need to multiply by risk_period_days as that would be double-counting
        risk_period_forecast = outflow_average
        
        self.logger.debug(f"Risk period forecast: {outflow_average:.2f} (outflow already represents {risk_period_days}-day sum)")
        
        return risk_period_forecast
    
    def _generate_future_dates(self, risk_period_days: int, demand_frequency: str) -> List[date]:
        """
        Generate future dates for the risk period.
        
        Args:
            risk_period_days: Number of days in risk period
            demand_frequency: 'd' for daily, 'w' for weekly, 'm' for monthly
            
        Returns:
            List of future dates
        """
        # Convert to pandas frequency
        freq_map = {
            'd': 'D',
            'w': 'W', 
            'm': 'M'
        }
        
        freq = freq_map.get(demand_frequency, 'D')
        
        # Calculate end date
        end_date = pd.to_datetime(self.forecast_date) + timedelta(days=risk_period_days)
        
        # Generate date range
        date_range = pd.date_range(
            start=self.forecast_date,
            end=end_date,
            freq=freq
        )
        
        # Convert to date objects and filter out past dates
        future_dates = [d.date() for d in date_range if d.date() > pd.to_datetime(self.forecast_date).date()]
        
        return future_dates
    
    def _save_results(self):
        """Save future predictions to Snowflake with table mode handling."""
        if not self.future_predictions:
            self.logger.warning("No predictions to save")
            return
        
        try:
            # Convert to DataFrame
            predictions_df = pd.DataFrame(self.future_predictions)
            
            # Convert date columns to string for Snowflake compatibility
            date_columns = ['forecasted_on', 'forecast_period_start', 'forecast_period_end', 'training_period_start', 'training_period_end']
            for col in date_columns:
                if col in predictions_df.columns:
                    predictions_df[col] = predictions_df[col].astype(str)
            
            # Handle table mode
            table_name = self.data_loader.config['snowflake_tables']['future_predictions']
            
            if self.table_mode == 'truncate':
                self.logger.info(f"Truncating table {table_name} before saving new data...")
                print(f"🧹 Clearing {table_name} table...")
                self._clear_table(table_name)
                if_exists_mode = 'replace'
            else:  # append mode
                self.logger.info(f"Appending to existing table {table_name}...")
                print(f"📝 Appending to {table_name} table...")
                if_exists_mode = 'append'
            
            # Save to Snowflake
            save_start = time.time()
            self.data_loader.accessor.write_data(predictions_df, table_name, if_exists=if_exists_mode)
            save_time = time.time() - save_start
            
            self.logger.info(f"Saved {len(predictions_df)} future predictions to {table_name} in {save_time:.2f}s")
            print(f"✅ Saved {len(predictions_df)} predictions to Snowflake table: {table_name} in {save_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Failed to save predictions: {e}")
            raise
    
    def _clear_table(self, table_name: str):
        """Clear the Snowflake table."""
        try:
            # Get connection and clear table
            conn = self.data_loader.accessor._get_connection()
            cursor = conn.cursor()
            cursor.execute(f"DELETE FROM {table_name}")
            cursor.close()
            conn.close()
            self.logger.info(f"Successfully cleared table {table_name}")
        except Exception as e:
            if "does not exist" in str(e).lower() or "not found" in str(e).lower():
                self.logger.info(f"Table {table_name} does not exist yet - this is normal for first run")
                print(f"✅ Table {table_name} does not exist yet - this is normal for first run")
            else:
                self.logger.warning(f"Could not clear table {table_name}: {e}")
                print(f"⚠️ Warning: Could not clear table {table_name}: {e}")


def main():
    """Main function to run the processed data future forecasting pipeline."""
    parser = argparse.ArgumentParser(description='Processed Data Future Forecasting Pipeline')
    parser.add_argument('--forecast-date', type=str, help='Date to forecast from (YYYY-MM-DD)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--table-mode', choices=['truncate', 'append'], default='truncate', 
                       help='Table mode: truncate (clear table first) or append (add to existing data)')
    
    args = parser.parse_args()
    
    # Setup logging
    configure_workflow_logging('processed_future_forecasting')
    
    # Initialize and run forecaster
    forecaster = ProcessedDataFutureForecaster(
        forecast_date=args.forecast_date, 
        table_mode=args.table_mode
    )
    
    try:
        results = forecaster.run_forecasting()
        
        print(f"\n🎉 Processed Data Future Forecasting Completed Successfully!")
        print(f"📅 Forecast Date: {results['forecast_date']}")
        print(f"📊 Products Processed: {results['successful_products']}/{results['products_processed']}")
        print(f"🔮 Predictions Generated: {results['predictions_generated']}")
        print(f"⏱️  Execution Time: {results['execution_time_seconds']:.1f}s")
        print(f"📈 Success Rate: {results['success_rate']:.1f}%")
        
    except Exception as e:
        print(f"❌ Future forecasting failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
