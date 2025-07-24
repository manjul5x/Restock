"""
Main pipeline for orchestrating the forecasting process.
"""

import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, date
from pathlib import Path
import time

from .config import RunnerConfig, BatchConfig
from .parallel import ParallelProcessor, process_batch_with_logging
from ..data.loader import load_csv, load_product_master_daily, load_product_master_weekly, validate_product_master_coverage
from ..data.demand_validator import DemandValidator
from ..data.aggregator import DemandAggregator
from ..outlier.handler import OutlierHandler
from ..forecasting.moving_average import MovingAverageForecaster
from ..forecasting.prophet import ProphetForecaster
from ..utils.logger import ForecasterLogger


class ForecastingPipeline:
    """Main pipeline for orchestrating the forecasting process."""
    
    def __init__(self, config: RunnerConfig):
        self.config = config
        self.logger = ForecasterLogger(__name__, config.log_level, config.log_file)
        self.parallel_processor = ParallelProcessor(config)
        
        # Initialize components
        self.demand_data = None
        self.product_master_data = None
        self.outlier_data = None
        self.aggregated_data = None
        self.forecasts = None
        
        # Set run date if not provided
        if self.config.run_date is None:
            self._set_default_run_date()
    
    def _set_default_run_date(self):
        """Set the run date to the latest date in the demand data."""
        try:
            demand_file = self.config.get_demand_file_path()
            if demand_file.exists():
                # Load just the date column to find the latest date
                date_df = pd.read_csv(demand_file, usecols=['date'])
                date_df['date'] = pd.to_datetime(date_df['date'])
                latest_date = date_df['date'].max().date()
                self.config.run_date = latest_date
                self.logger.info(f"Set run date to latest date in data: {latest_date}")
            else:
                raise FileNotFoundError(f"Demand file not found: {demand_file}")
        except Exception as e:
            self.logger.error(f"Failed to set default run date: {e}")
            raise
    
    def run(self) -> Dict[str, Any]:
        """
        Run the complete forecasting pipeline.
        
        Returns:
            Dictionary with pipeline results and statistics
        """
        start_time = time.time()
        self.logger.info("Starting forecasting pipeline")
        self.logger.info(f"Configuration: {self.config.to_dict()}")
        
        try:
            # Step 1: Load and validate data
            self.logger.info("Step 1: Loading and validating data")
            self._load_data()
            
            if self.config.validate_data:
                self._validate_data()
            
            # Step 2: Handle outliers
            if self.config.outlier_enabled:
                self.logger.info("Step 2: Handling outliers")
                self._handle_outliers()
            
            # Step 3: Aggregate data
            if self.config.aggregation_enabled:
                self.logger.info("Step 3: Aggregating data")
                self._aggregate_data()
            
            # Step 4: Generate forecasts
            if self.config.forecasting_enabled:
                self.logger.info("Step 4: Generating forecasts")
                self._generate_forecasts()
            
            # Step 5: Save results
            self.logger.info("Step 5: Saving results")
            self._save_results()
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Generate summary
            summary = self._generate_summary(total_time)
            
            self.logger.info("Forecasting pipeline completed successfully")
            self.logger.info(f"Total execution time: {total_time:.2f} seconds")
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise
    
    def _load_data(self):
        """Load demand and product master data."""
        # Load demand data
        demand_file = self.config.get_demand_file_path()
        self.logger.info(f"Loading demand data from: {demand_file}")
        self.demand_data = load_csv(demand_file)
        self.logger.info(f"Loaded {len(self.demand_data)} demand records")
        
        # Load product master data
        product_master_file = self.config.get_product_master_file_path()
        self.logger.info(f"Loading product master data from: {product_master_file}")
        
        if self.config.demand_frequency == 'd':
            self.product_master_data = load_product_master_daily(product_master_file)
        else:
            self.product_master_data = load_product_master_weekly(product_master_file)
        
        self.logger.info(f"Loaded {len(self.product_master_data)} product master records")
    
    def _validate_data(self):
        """Validate data quality and coverage."""
        # Validate product master coverage
        if self.config.validate_coverage:
            self.logger.info("Validating product master coverage")
            coverage_report = validate_product_master_coverage(self.demand_data, self.product_master_data)
            if coverage_report['missing_combinations']:
                self.logger.warning(f"Found {len(coverage_report['missing_combinations'])} missing product-location combinations")
            else:
                self.logger.info("All product-location combinations covered in product master")
        
        # Validate demand completeness
        if self.config.validate_completeness:
            self.logger.info("Validating demand completeness")
            validator = DemandValidator()
            frequency = "daily" if self.config.demand_frequency == "d" else "weekly"
            completeness_report = validator.validate_demand_completeness(frequency)
            
            if completeness_report['invalid_combinations'] > 0:
                self.logger.warning(f"Found {completeness_report['invalid_combinations']} invalid combinations")
                self.logger.warning(f"Missing dates total: {completeness_report['missing_dates_total']}")
            else:
                self.logger.info("All demand entries are complete")
    
    def _handle_outliers(self):
        """Handle outliers in demand data."""
        self.logger.info("Processing outliers")
        
        # Create outlier handler
        outlier_handler = OutlierHandler()
        
        # Process outliers using the handler's method
        frequency = "daily" if self.config.demand_frequency == "d" else "weekly"
        cleaned_df, outlier_df = outlier_handler.process_demand_outliers(
            frequency=frequency,
            default_method="iqr",
            default_threshold=1.5
        )
        
        # Use cleaned data as the outlier-processed data
        self.outlier_data = cleaned_df
        self.logger.info(f"Outlier processing completed: {len(self.outlier_data)} cleaned records")
        
        # Save outlier insights if requested
        if self.config.outlier_output_insights and len(outlier_df) > 0:
            outlier_summary = outlier_handler.get_outlier_summary(outlier_df)
            insights_file = self.config.output_dir / "outlier_insights.csv"
            outlier_df.to_csv(insights_file, index=False)
            self.logger.info(f"Outlier insights saved to: {insights_file}")
            self.logger.info(f"Outlier summary: {outlier_summary}")
    
    def _aggregate_data(self):
        """Aggregate demand data into risk period buckets."""
        self.logger.info(f"Aggregating data for run date: {self.config.run_date}")
        
        # Use outlier data if available, otherwise use original demand data
        input_data = self.outlier_data if self.outlier_data is not None else self.demand_data
        
        # Create aggregator
        aggregator = DemandAggregator()
        
        # Aggregate data
        frequency = "daily" if self.config.demand_frequency == "d" else "weekly"
        # Convert date to datetime for comparison
        cutoff_datetime = datetime.combine(self.config.run_date, datetime.min.time())
        self.aggregated_data = aggregator.create_risk_period_buckets(
            cutoff_datetime,
            frequency
        )
        
        self.logger.info(f"Aggregation completed: {len(self.aggregated_data)} aggregated records")
    
    def _generate_forecasts(self):
        """Generate forecasts using parallel processing."""
        self.logger.info("Starting forecast generation")
        
        # Get unique product-location combinations
        product_locations = self.product_master_data[['product_id', 'location_id']].drop_duplicates()
        product_location_tuples = list(product_locations.itertuples(index=False, name=None))
        
        self.logger.info(f"Found {len(product_location_tuples)} product-location combinations")
        
        # Create batches
        batches = self.parallel_processor.create_batches(product_location_tuples)
        
        # Process batches (parallel or sequential)
        if self.config.max_workers > 1:
            self.logger.info("Using parallel processing")
            batch_results = self.parallel_processor.process_batches_parallel(
                batches, 
                self._process_forecast_batch
            )
        else:
            self.logger.info("Using sequential processing")
            batch_results = self.parallel_processor.process_batches_sequential(
                batches, 
                self._process_forecast_batch
            )
        
        # Combine results
        self.forecasts = self._combine_forecast_results(batch_results)
        
        # Get processing statistics
        stats = self.parallel_processor.get_processing_stats(batch_results)
        self.logger.info(f"Forecast generation completed: {stats}")
    
    def _process_forecast_batch(self, batch: BatchConfig) -> List[Dict[str, Any]]:
        """Process a single batch of product-location combinations."""
        batch_forecasts = []
        
        # Use aggregated data for forecasting, mapping columns correctly
        input_data = self.aggregated_data if self.aggregated_data is not None else self.outlier_data
        
        # If using aggregated data, map the columns to what the forecaster expects
        if input_data is self.aggregated_data and len(input_data) > 0:
            # Create a copy with mapped columns
            input_data = input_data.copy()
            input_data['date'] = input_data['bucket_start_date']
            input_data['demand'] = input_data['total_demand']
        
        for product_id, location_id in batch.product_locations:
            try:
                # Get product master record
                product_record = self.product_master_data[
                    (self.product_master_data['product_id'] == product_id) & 
                    (self.product_master_data['location_id'] == location_id)
                ].iloc[0]
                
                # Filter data for this product-location
                product_data = input_data[
                    (input_data['product_id'] == product_id) & 
                    (input_data['location_id'] == location_id)
                ].copy()
                
                if len(product_data) > 0:
                    self.logger.debug(f"Processing {product_id}-{location_id} with {len(product_data)} records")
                    
                    # Get forecast method from product master (fallback to config if not specified)
                    forecast_method = product_record.get('forecast_method', self.config.forecast_model)
                    
                    # Create forecaster based on product-specific method
                    if forecast_method == 'moving_average':
                        forecaster = MovingAverageForecaster(
                            window_length=product_record['forecast_window_length'] * product_record['risk_period'],
                            horizon=product_record['forecast_horizon']
                        )
                    elif forecast_method == 'prophet':
                        forecaster = ProphetForecaster(
                            window_length=product_record['forecast_window_length'] * product_record['risk_period'],
                            horizon=product_record['forecast_horizon'],
                            yearly_seasonality=True,
                            weekly_seasonality=True,
                            daily_seasonality=False
                        )
                    elif forecast_method == 'arima':
                        # TODO: Implement ARIMA forecaster
                        self.logger.warning(f"ARIMA forecasting not yet implemented for {product_id}-{location_id}, falling back to moving_average")
                        forecaster = MovingAverageForecaster(
                            window_length=product_record['forecast_window_length'] * product_record['risk_period'],
                            horizon=product_record['forecast_horizon']
                        )
                    else:
                        raise ValueError(f"Unsupported forecast method: {forecast_method}")
                    
                    # Fit the model first
                    forecaster.fit(product_data)
                    
                    # Generate forecast
                    forecast_series = forecaster.forecast(steps=product_record['forecast_horizon'])
                    
                    # Create forecast result
                    forecast_result = {
                        'product_id': product_id,
                        'location_id': location_id,
                        'forecast_date': self.config.run_date,
                        'model': forecast_method,
                        'window_length': product_record['forecast_window_length'] * product_record['risk_period'],
                        'horizon': product_record['forecast_horizon'],
                        'forecast_values': forecast_series.tolist(),
                        'forecast_mean': forecast_series.mean()
                    }
                    
                    batch_forecasts.append(forecast_result)
                    self.logger.debug(f"Successfully forecasted {product_id}-{location_id}")
                else:
                    self.logger.debug(f"No data for {product_id}-{location_id}")
                
            except Exception as e:
                self.logger.error(f"Failed to forecast for {product_id}-{location_id}: {e}")
                # Continue with other combinations in the batch
        
        return batch_forecasts
    
    def _combine_forecast_results(self, batch_results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Combine forecast results from all batches."""
        all_forecasts = []
        
        for batch_result in batch_results:
            if batch_result['status'] == 'success' and 'result' in batch_result:
                # Handle both parallel and sequential processing results
                result_data = batch_result['result']
                
                # For parallel processing: result_data = {'processing_time': ..., 'data': [...]}
                # For sequential processing: result_data = [...] (direct list of forecasts)
                if isinstance(result_data, dict) and 'data' in result_data:
                    # Parallel processing result
                    batch_forecasts = result_data['data']
                elif isinstance(result_data, list):
                    # Sequential processing result
                    batch_forecasts = result_data
                else:
                    self.logger.warning(f"Unexpected result structure for batch {batch_result.get('batch_id', 'unknown')}")
                    continue
                
                if batch_forecasts:
                    all_forecasts.extend(batch_forecasts)
                    self.logger.debug(f"Added {len(batch_forecasts)} forecasts from batch {batch_result.get('batch_id', 'unknown')}")
        
        if all_forecasts:
            # Convert to DataFrame
            forecasts_df = pd.DataFrame(all_forecasts)
            self.logger.info(f"Combined {len(forecasts_df)} forecasts from {len(batch_results)} batches")
            return forecasts_df
        else:
            self.logger.warning("No forecasts generated")
            return pd.DataFrame()
    
    def _save_results(self):
        """Save pipeline results to files."""
        # Save outlier data
        if self.outlier_data is not None:
            outlier_file = self.config.get_outlier_output_path()
            self.outlier_data.to_csv(outlier_file, index=False)
            self.logger.info(f"Outlier data saved to: {outlier_file}")
        
        # Save aggregated data
        if self.aggregated_data is not None:
            aggregated_file = self.config.get_aggregated_output_path()
            self.aggregated_data.to_csv(aggregated_file, index=False)
            self.logger.info(f"Aggregated data saved to: {aggregated_file}")
        
        # Save forecasts
        if self.forecasts is not None and len(self.forecasts) > 0:
            forecast_file = self.config.get_forecast_output_path()
            self.forecasts.to_csv(forecast_file, index=False)
            self.logger.info(f"Forecasts saved to: {forecast_file}")
    
    def _generate_summary(self, total_time: float) -> Dict[str, Any]:
        """Generate a summary of the pipeline execution."""
        summary = {
            'pipeline_status': 'completed',
            'total_execution_time': total_time,
            'run_date': self.config.run_date.isoformat() if self.config.run_date else None,
            'demand_frequency': self.config.demand_frequency,
            'forecast_model': self.config.forecast_model,
            'data_summary': {
                'demand_records': len(self.demand_data) if self.demand_data is not None else 0,
                'product_master_records': len(self.product_master_data) if self.product_master_data is not None else 0,
                'outlier_records': len(self.outlier_data) if self.outlier_data is not None else 0,
                'aggregated_records': len(self.aggregated_data) if self.aggregated_data is not None else 0,
                'forecast_records': len(self.forecasts) if self.forecasts is not None else 0
            },
            'output_files': {
                'outlier_data': str(self.config.get_outlier_output_path()),
                'aggregated_data': str(self.config.get_aggregated_output_path()),
                'forecasts': str(self.config.get_forecast_output_path()),
                'log_file': str(self.config.log_file)
            }
        }
        
        return summary


def run_pipeline(config: Optional[RunnerConfig] = None) -> Dict[str, Any]:
    """
    Convenience function to run the forecasting pipeline.
    
    Args:
        config: Pipeline configuration (uses default if None)
    
    Returns:
        Pipeline execution summary
    """
    if config is None:
        config = RunnerConfig()
    
    pipeline = ForecastingPipeline(config)
    return pipeline.run()
