"""
Main backtesting module for historical forecasting simulation.
"""

import pandas as pd
import numpy as np
from datetime import date, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from tqdm import tqdm

from .config import BacktestConfig
from ..data.loader import DemandDataLoader
from ..data.demand_validator import DemandValidator
from ..outlier.handler import OutlierHandler
from ..data.aggregator import DemandAggregator
from ..forecasting.moving_average import MovingAverageForecaster
from ..forecasting.prophet import ProphetForecaster
from ..forecasting.arima import ARIMAForecaster
from ..forecasting.base import calculate_forecast_metrics
from ..utils.logger import ForecasterLogger
from ..data.product_master_schema import ProductMasterSchema


class Backtester:
    """
    Main backtesting class for historical forecasting simulation.
    Works at risk period level instead of daily level.
    """
    
    def __init__(self, config: BacktestConfig):
        """Initialize the backtester with configuration."""
        self.config = config
        self.logger = ForecasterLogger("backtester", config.log_level, config.log_file)
        
        # Data storage
        self.demand_data: Optional[pd.DataFrame] = None
        self.product_master_data: Optional[pd.DataFrame] = None
        self.outlier_data: Optional[pd.DataFrame] = None
        
        # Results storage
        self.backtest_results: List[Dict[str, Any]] = []
        self.accuracy_metrics: List[Dict[str, Any]] = []
        self.forecast_comparisons: List[Dict[str, Any]] = []
        
        # Validate configuration
        if not self.config.validate_dates():
            raise ValueError("Invalid date configuration: historic_start_date <= analysis_start_date <= analysis_end_date")
    
    def _print_initial_summary(self):
        """Print initial summary of the backtesting configuration and data."""
        print("\n" + "="*60)
        print("BACKTESTING SUMMARY")
        print("="*60)
        print(f"Data Configuration:")
        print(f"  • Demand file: {self.config.demand_file}")
        print(f"  • Product master file: {self.config.product_master_file}")
        print(f"  • Total demand records: {len(self.demand_data):,}")
        print(f"  • Total product-location combinations: {len(self.product_master_data):,}")
        
        print(f"\nDate Configuration:")
        print(f"  • Historic start: {self.config.historic_start_date}")
        print(f"  • Analysis period: {self.config.analysis_start_date} to {self.config.analysis_end_date}")
        print(f"  • Analysis dates: {len(self.config.get_analysis_dates())} dates")
        
        print(f"\nForecasting Configuration:")
        print(f"  • Forecast model: {self.config.forecast_model}")
        print(f"  • Default horizon: {self.config.default_horizon} risk periods")
        print(f"  • Demand frequency: {self.config.demand_frequency}")
        print(f"  • Outlier handling: {'Enabled' if self.config.outlier_enabled else 'Disabled'}")
        print(f"  • Aggregation: {'Enabled' if self.config.aggregation_enabled else 'Disabled'}")
        
        print(f"\nProcessing Configuration:")
        print(f"  • Batch size: {self.config.batch_size}")
        print(f"  • Max workers: {self.config.max_workers}")
        print(f"  • Expected total forecasts: {len(self.config.get_analysis_dates()) * len(self.product_master_data):,}")
        print("="*60 + "\n")

    def run(self) -> Dict:
        """
        Run the complete backtesting process.
        
        Returns:
            Dictionary with backtesting summary and results
        """
        start_time = time.time()
        self.logger.info("Starting backtesting process")
        self.logger.info(f"Configuration: {self.config.__dict__}")
        
        try:
            # Step 1: Load and validate data
            self._load_data()
            
            # Print initial summary
            self._print_initial_summary()
            
            # Step 2: Handle outliers across the entire analysis period
            self._handle_outliers()
            
            # Step 3: Run backtesting for each analysis date
            self._run_backtesting()
            
            # Step 4: Calculate accuracy metrics
            self._calculate_accuracy_metrics()
            
            # Step 5: Save results
            self._save_results()
            
            # Step 6: Generate summary
            total_time = time.time() - start_time
            summary = self._generate_summary(total_time)
            
            self.logger.info("Backtesting completed successfully")
            return summary
            
        except Exception as e:
            self.logger.error(f"Backtesting failed: {e}")
            raise
    
    def _load_data(self):
        """Load and validate demand and product master data."""
        self.logger.info("Step 1: Loading data")
        
        # Load data with correct data directory
        loader = DemandDataLoader(data_dir=self.config.data_dir)
        
        # Load demand data
        self.demand_data = loader.load_csv(self.config.demand_file)
        self.logger.info(f"Loaded {len(self.demand_data)} demand records")
        
        # Load product master data
        self.product_master_data = loader.load_csv(self.config.product_master_file)
        self.logger.info(f"Loaded {len(self.product_master_data)} product master records")
        
        # Validate demand completeness
        validator = DemandValidator()
        # Convert frequency format: 'd' -> 'daily', 'w' -> 'weekly', 'm' -> 'monthly'
        frequency_map = {'d': 'daily', 'w': 'weekly', 'm': 'monthly'}
        validation_frequency = frequency_map.get(self.config.demand_frequency, 'daily')
        completeness_result = validator.validate_demand_completeness_with_data(
            self.demand_data, self.product_master_data, validation_frequency
        )
        
        if completeness_result.get('missing_dates_total', 0) > 0:
            self.logger.warning(f"Demand completeness issues: {completeness_result['issues']}")
    
    def _handle_outliers(self):
        """Handle outliers across the entire analysis period."""
        self.logger.info("Step 2: Handling outliers")
        
        historic_end = self.config.analysis_end_date
        # Convert date columns to date for comparison
        filtered_data = self.demand_data[
            (pd.to_datetime(self.demand_data['date']).dt.date >= self.config.historic_start_date) &
            (pd.to_datetime(self.demand_data['date']).dt.date <= historic_end)
        ].copy()
        
        self.logger.info(f"Processing outliers for {len(filtered_data)} records")
        
        outlier_handler = OutlierHandler()
        outlier_result = outlier_handler.process_demand_outliers_with_data(
            filtered_data, 
            self.product_master_data
        )
        
        self.outlier_data = outlier_result['cleaned_data']
        outlier_summary = outlier_result['summary']
        
        self.logger.info(f"Outlier processing completed: {len(self.outlier_data)} cleaned records")
        self.logger.info(f"Outlier summary: {outlier_summary}")
    
    def _get_cutoff_date(self, analysis_date: date) -> date:
        """
        Get the cutoff date as the closest date in the data before the analysis date.
        
        Args:
            analysis_date: The analysis date
            
        Returns:
            The cutoff date (closest date in data before analysis_date)
        """
        # Get all unique dates in the outlier data
        available_dates = pd.to_datetime(self.outlier_data['date']).dt.date.unique()
        available_dates = sorted(available_dates)
        
        # Find the closest date before or equal to analysis_date
        cutoff_date = None
        for available_date in reversed(available_dates):
            if available_date <= analysis_date:
                cutoff_date = available_date
                break
        
        if cutoff_date is None:
            raise ValueError(f"No data available before analysis date {analysis_date}")
        
        return cutoff_date
    
    def _run_backtesting(self):
        """Run backtesting for each analysis date."""
        self.logger.info("Step 3: Running backtesting")
        
        analysis_dates = self.config.get_analysis_dates()
        self.logger.info(f"Running backtesting for {len(analysis_dates)} analysis dates")
        
        # Get all product-location combinations
        product_locations = list(zip(self.product_master_data['product_id'], self.product_master_data['location_id']))
        self.logger.info(f"Processing {len(product_locations)} product-location combinations")
        
        # Initialize results storage
        self.backtest_results = []
        
        if self.config.max_workers > 1:
            # Parallel processing
            self.logger.info(f"Using parallel processing with {self.config.max_workers} workers")
            self._run_backtesting_parallel(analysis_dates, product_locations)
        else:
            # Sequential processing
            self.logger.info("Using sequential processing")
            self._run_backtesting_sequential(analysis_dates, product_locations)
    
    def _run_backtesting_sequential(self, analysis_dates: List[date], product_locations: List[Tuple[str, str]]):
        """Run backtesting sequentially."""
        total_forecasts = 0
        
        # Progress bar for analysis dates
        with tqdm(total=len(analysis_dates), desc="Processing analysis dates", unit="date") as pbar:
            for analysis_date in analysis_dates:
                try:
                    # Get cutoff date (closest date in data before analysis_date)
                    cutoff_date = self._get_cutoff_date(analysis_date)
                    
                    # Aggregate data up to cutoff date
                    aggregated_data = self._aggregate_data_for_date(cutoff_date)
                    
                    if aggregated_data is not None and len(aggregated_data) > 0:
                        # Generate forecasts for this date
                        forecasts = self._generate_forecasts_for_date(analysis_date, cutoff_date, aggregated_data, product_locations)
                        self.backtest_results.extend(forecasts)
                        total_forecasts += len(forecasts)
                        self.logger.info(f"Generated {len(forecasts)} forecasts for {analysis_date}")
                    else:
                        self.logger.warning(f"No aggregated data available for {analysis_date}")
                        
                except Exception as e:
                    self.logger.error(f"Error processing {analysis_date}: {e}")
                
                pbar.update(1)
                pbar.set_postfix({
                    'Date': str(analysis_date),
                    'Forecasts': total_forecasts
                })
    
    def _run_backtesting_parallel(self, analysis_dates: List[date], product_locations: List[Tuple[str, str]]):
        """Run backtesting in parallel."""
        # Prepare arguments for parallel processing
        parallel_args = []
        for analysis_date in analysis_dates:
            cutoff_date = self._get_cutoff_date(analysis_date)
            parallel_args.append((analysis_date, cutoff_date, product_locations))
        
        total_forecasts = 0
        batch_size = self.config.batch_size
        
        # Progress bar for analysis dates
        with tqdm(total=len(analysis_dates), desc="Processing analysis dates (parallel)", unit="date") as pbar:
            for i in range(0, len(parallel_args), batch_size):
                batch = parallel_args[i:i + batch_size]
                
                # Process batch in parallel
                with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
                    # Submit batch of tasks
                    future_to_date = {
                        executor.submit(self._process_single_date, args): args[0] 
                        for args in batch
                    }
                    
                    # Collect results as they complete
                    for future in as_completed(future_to_date):
                        analysis_date = future_to_date[future]
                        try:
                            forecasts = future.result()
                            if forecasts:
                                self.backtest_results.extend(forecasts)
                                total_forecasts += len(forecasts)
                                self.logger.info(f"Generated {len(forecasts)} forecasts for {analysis_date}")
                        except Exception as e:
                            self.logger.error(f"Error processing {analysis_date}: {e}")
                        
                        pbar.update(1)
                        pbar.set_postfix({
                            'Date': str(analysis_date),
                            'Forecasts': total_forecasts
                        })
    
    def _process_single_date(self, args):
        """Process a single analysis date (for parallel processing)."""
        analysis_date, cutoff_date, product_locations = args
        
        try:
            # Aggregate data up to cutoff date
            aggregated_data = self._aggregate_data_for_date(cutoff_date)
            
            if aggregated_data is not None and len(aggregated_data) > 0:
                # Generate forecasts for this date
                forecasts = self._generate_forecasts_for_date(analysis_date, cutoff_date, aggregated_data, product_locations)
                return forecasts
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"Error processing date {analysis_date}: {e}")
            return []
    
    def _aggregate_data_for_date(self, cutoff_date: date) -> Optional[pd.DataFrame]:
        """Aggregate data up to the given cutoff date."""
        try:
            # Filter outlier data up to cutoff date
            filtered_data = self.outlier_data[
                pd.to_datetime(self.outlier_data['date']).dt.date <= cutoff_date
            ].copy()
            
            if len(filtered_data) == 0:
                return None
            
            # Aggregate data
            aggregator = DemandAggregator()
            aggregated_data = aggregator.create_risk_period_buckets_with_data(
                filtered_data,
                self.product_master_data,
                cutoff_date
            )
            
            return aggregated_data
            
        except Exception as e:
            self.logger.error(f"Error aggregating data for cutoff date {cutoff_date}: {e}")
            return None
    
    def _generate_forecasts_for_date(
        self, 
        analysis_date: date, 
        cutoff_date: date, 
        aggregated_data: pd.DataFrame, 
        product_locations: List[Tuple[str, str]]
    ) -> List[Dict[str, Any]]:
        """Generate forecasts for a specific analysis date at risk period level."""
        forecasts = []
        
        # Map aggregated data columns for forecasting
        input_data = aggregated_data.copy()
        input_data['date'] = input_data['bucket_start_date']
        input_data['demand'] = input_data['total_demand']
        
        for product_id, location_id in product_locations:
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
                    # Use default horizon for backtesting (in risk periods)
                    horizon = self.config.default_horizon
                    
                    # Get forecast method from product master (fallback to config if not specified)
                    forecast_method = product_record.get('forecast_method', self.config.forecast_model)
                    
                    # Create forecaster based on product-specific method
                    if forecast_method == 'moving_average':
                        forecaster = MovingAverageForecaster(
                            window_length=product_record['forecast_window_length'],
                            horizon=horizon
                        )
                    elif forecast_method == 'prophet':
                        forecaster = ProphetForecaster(
                            window_length=product_record['forecast_window_length']
                        )
                    elif forecast_method == 'arima':
                        forecaster = ARIMAForecaster(
                            window_length=product_record['forecast_window_length'],
                            horizon=horizon,
                            auto_arima=True,
                            max_p=3,
                            max_d=2,
                            max_q=3,
                            seasonal=False,
                            min_data_points=5
                        )
                    else:
                        raise ValueError(f"Unsupported forecast method: {forecast_method}")
                    
                    # Fit the model
                    forecaster.fit(product_data)
                    
                    # Generate forecast
                    forecast_series = forecaster.forecast(steps=horizon)
                    
                    # Create forecast result
                    forecast_result = {
                        'analysis_date': analysis_date,
                        'cutoff_date': cutoff_date,
                        'product_id': product_id,
                        'location_id': location_id,
                        'model': forecast_method,
                        'window_length': product_record['forecast_window_length'],
                        'horizon': horizon,
                        'forecast_values': forecast_series.tolist(),
                        'forecast_mean': forecast_series.mean(),
                        'risk_period': product_record['risk_period'],
                        'demand_frequency': product_record['demand_frequency']
                    }
                    
                    forecasts.append(forecast_result)
                    
            except Exception as e:
                self.logger.error(f"Failed to forecast for {product_id}-{location_id} on {analysis_date}: {e}")
                continue
        
        return forecasts
    
    def _calculate_accuracy_metrics(self):
        """Calculate accuracy metrics for all forecasts at risk period level."""
        self.logger.info("Step 4: Calculating accuracy metrics")
        
        if not self.backtest_results:
            self.logger.warning("No backtest results to calculate metrics for")
            self.accuracy_metrics = []
            return
        
        self.accuracy_metrics = []
        
        # Progress bar for accuracy calculation
        with tqdm(total=len(self.backtest_results), desc="Calculating accuracy metrics", unit="forecast") as pbar:
            for forecast in self.backtest_results:
                analysis_date = forecast['analysis_date']
                product_id = forecast['product_id']
                location_id = forecast['location_id']
                horizon = forecast['horizon']
                forecast_values = forecast['forecast_values']
                risk_period = forecast['risk_period']
                demand_frequency = forecast['demand_frequency']
                
                # Get actual aggregated demands for comparison
                actual_demands = self._get_actual_aggregated_demands(
                    analysis_date, product_id, location_id, horizon, risk_period, demand_frequency
                )
                
                if actual_demands is not None and len(actual_demands) == horizon:
                    # Filter out NaN values for metrics calculation
                    valid_indices = [i for i, val in enumerate(actual_demands) if not pd.isna(val)]
                    
                    if len(valid_indices) > 0:
                        # Use only valid data for metrics calculation
                        valid_actual = [actual_demands[i] for i in valid_indices]
                        valid_forecast = [forecast_values[i] for i in valid_indices]
                        
                        # Calculate metrics
                        metrics = calculate_forecast_metrics(pd.Series(valid_actual), pd.Series(valid_forecast))
                    
                        # Create accuracy result
                        accuracy_result = {
                            'analysis_date': analysis_date,
                            'product_id': product_id,
                            'location_id': location_id,
                            'horizon': horizon,
                            'mae': metrics['mae'],
                            'mape': metrics['mape'],
                            'rmse': metrics['rmse'],
                            'bias': metrics['bias'],
                            'actual_demands': actual_demands,
                            'forecast_values': forecast_values,
                            'risk_period': risk_period,
                            'demand_frequency': demand_frequency,
                            'valid_periods': len(valid_indices)
                        }
                        
                        self.accuracy_metrics.append(accuracy_result)
                        
                        # Generate forecast comparisons
                        self._generate_forecast_comparisons(
                            analysis_date, product_id, location_id, 
                            actual_demands, forecast_values, 
                            risk_period, demand_frequency
                        )
                    else:
                        self.logger.warning(f"No valid actual demand data for {product_id}-{location_id} on {analysis_date}")
                else:
                    self.logger.warning(f"No actual demand data available for {product_id}-{location_id} on {analysis_date}")
                
                pbar.update(1)
        
        self.logger.info(f"Calculated accuracy metrics for {len(self.accuracy_metrics)} forecasts")
    
    def _get_actual_aggregated_demands(
        self, 
        analysis_date: date, 
        product_id: str, 
        location_id: str, 
        horizon: int,
        risk_period: int,
        demand_frequency: str
    ) -> Optional[List[float]]:
        """Get actual aggregated demand values for comparison with forecasts."""
        try:
            # Calculate risk period size in days
            risk_period_days = int(ProductMasterSchema.get_risk_period_days(demand_frequency, risk_period))
            
            # Calculate the start and end dates for the forecast horizon
            # Each horizon step represents one risk period
            start_date = analysis_date  # First forecast bucket starts on analysis date
            end_date = start_date + timedelta(days=(horizon * risk_period_days) - 1)
            
            # Get daily demand data for the forecast period
            daily_data = self.demand_data[
                (pd.to_datetime(self.demand_data['date']).dt.date >= start_date) &
                (pd.to_datetime(self.demand_data['date']).dt.date <= end_date) &
                (self.demand_data['product_id'] == product_id) &
                (self.demand_data['location_id'] == location_id)
            ].sort_values('date')
            
            if len(daily_data) == 0:
                return None
            
            # Aggregate daily data into risk period buckets
            aggregated_demands = []
            current_start = start_date
            
            for step in range(horizon):
                step_end = current_start + timedelta(days=risk_period_days - 1)
                
                # Get daily data for this risk period
                step_data = daily_data[
                    (pd.to_datetime(daily_data['date']).dt.date >= current_start) &
                    (pd.to_datetime(daily_data['date']).dt.date <= step_end)
                ]
                
                if len(step_data) == risk_period_days:  # Complete risk period
                    total_demand = step_data['demand'].sum()
                    aggregated_demands.append(total_demand)
                else:
                    # Incomplete risk period - use NaN
                    aggregated_demands.append(float('nan'))
                
                current_start = step_end + timedelta(days=1)
            
            return aggregated_demands
                
        except Exception as e:
            self.logger.error(f"Error getting actual demands for {product_id}-{location_id}: {e}")
            return None
    
    def _generate_forecast_comparisons(
        self,
        analysis_date: date,
        product_id: str,
        location_id: str,
        actual_demands: List[float],
        forecast_values: List[float],
        risk_period: int,
        demand_frequency: str
    ):
        """Generate detailed forecast comparison data."""
        try:
            # Calculate risk period size in days
            risk_period_days = int(ProductMasterSchema.get_risk_period_days(demand_frequency, risk_period))
            
            # Calculate risk period dates
            start_date = analysis_date  # First forecast bucket starts on analysis date
            
            for step in range(len(actual_demands)):
                risk_period_start = start_date + timedelta(days=step * risk_period_days)
                risk_period_end = risk_period_start + timedelta(days=risk_period_days - 1)
                
                actual_demand = actual_demands[step]
                forecast_demand = forecast_values[step]
                
                # Handle NaN values
                if pd.isna(actual_demand):
                    error = float('nan')
                    absolute_error = float('nan')
                    percentage_error = float('nan')
                else:
                    error = actual_demand - forecast_demand
                    absolute_error = abs(error)
                    percentage_error = (error / actual_demand * 100) if actual_demand != 0 else 0
                
                comparison = {
                    'analysis_date': analysis_date,
                    'risk_period_start': risk_period_start,
                    'risk_period_end': risk_period_end,
                    'product_id': product_id,
                    'location_id': location_id,
                    'step': step + 1,
                    'actual_demand': actual_demand,
                    'forecast_demand': forecast_demand,
                    'error': error,
                    'absolute_error': absolute_error,
                    'percentage_error': percentage_error,
                    'risk_period': risk_period,
                    'demand_frequency': demand_frequency
                }
                
                self.forecast_comparisons.append(comparison)
                
        except Exception as e:
            self.logger.error(f"Error generating forecast comparisons for {product_id}-{location_id}: {e}")
    
    def _save_results(self):
        """Save all backtesting results to files."""
        self.logger.info("Step 5: Saving results")
        
        # Save backtest results
        if self.backtest_results:
            results_df = pd.DataFrame(self.backtest_results)
            results_path = self.config.get_backtest_results_path()
            results_df.to_csv(results_path, index=False)
            self.logger.info(f"Backtest results saved to: {results_path}")
        
        # Save accuracy metrics
        if self.accuracy_metrics:
            metrics_df = pd.DataFrame(self.accuracy_metrics)
            metrics_path = self.config.get_accuracy_metrics_path()
            metrics_df.to_csv(metrics_path, index=False)
            self.logger.info(f"Accuracy metrics saved to: {metrics_path}")
        
        # Save forecast comparisons
        if self.forecast_comparisons:
            comparison_df = pd.DataFrame(self.forecast_comparisons)
            comparison_path = self.config.get_forecast_comparison_path()
            comparison_df.to_csv(comparison_path, index=False)
            self.logger.info(f"Forecast comparisons saved to: {comparison_path}")
        
        # Create and save enhanced forecast visualization data
        self._create_forecast_visualization_data()
    
    def _generate_summary(self, total_time: float) -> Dict[str, Any]:
        """Generate a summary of the backtesting execution."""
        summary = {
            'backtest_status': 'completed',
            'total_execution_time': total_time,
            'config': {
                'historic_start_date': self.config.historic_start_date.isoformat(),
                'analysis_start_date': self.config.analysis_start_date.isoformat(),
                'analysis_end_date': self.config.analysis_end_date.isoformat(),
                'demand_frequency': self.config.demand_frequency,
                'forecast_model': self.config.forecast_model,
                'default_horizon': self.config.default_horizon
            },
            'data_summary': {
                'demand_records': len(self.demand_data) if self.demand_data is not None else 0,
                'product_master_records': len(self.product_master_data) if self.product_master_data is not None else 0,
                'outlier_records': len(self.outlier_data) if self.outlier_data is not None else 0,
                'backtest_results': len(self.backtest_results),
                'accuracy_metrics': len(self.accuracy_metrics),
                'forecast_comparisons': len(self.forecast_comparisons)
            },
            'output_files': {
                'backtest_results': str(self.config.get_backtest_results_path()),
                'accuracy_metrics': str(self.config.get_accuracy_metrics_path()),
                'forecast_comparison': str(self.config.get_forecast_comparison_path()),
                'log_file': str(self.config.log_file)
            }
        }
        
        # Add accuracy summary if metrics available
        if self.accuracy_metrics:
            metrics_df = pd.DataFrame(self.accuracy_metrics)
            summary['accuracy_summary'] = {
                'mean_mae': metrics_df['mae'].mean(),
                'mean_mape': metrics_df['mape'].mean(),
                'mean_rmse': metrics_df['rmse'].mean(),
                'mean_bias': metrics_df['bias'].mean(),
                'total_forecasts': len(self.backtest_results)  # Use backtest_results instead of metrics_df
            }
        else:
            # Add basic summary even if no accuracy metrics
            summary['accuracy_summary'] = {
                'mean_mae': 0.0,
                'mean_mape': 0.0,
                'mean_rmse': 0.0,
                'mean_bias': 0.0,
                'total_forecasts': len(self.backtest_results)  # Always show total forecasts
            }
        
        return summary 
    
    def _create_forecast_visualization_data(self):
        """Create enhanced forecast visualization data with historical and forecast data."""
        self.logger.info("Creating enhanced forecast visualization data")
        
        if not self.backtest_results:
            self.logger.warning("No backtest results to create visualization data for")
            return
        
        enhanced_data = []
        
        # Group backtest results by analysis date and product-location
        for forecast in self.backtest_results:
            try:
                analysis_date = forecast['analysis_date']
                product_id = forecast['product_id']
                location_id = forecast['location_id']
                
                self.logger.debug(f"Processing {product_id}-{location_id} on {analysis_date}")
                
                # Get product configuration
                product_record = self.product_master_data[
                    (self.product_master_data['product_id'] == product_id) & 
                    (self.product_master_data['location_id'] == location_id)
                ].iloc[0]
                
                risk_period = product_record['risk_period']
                demand_frequency = product_record['demand_frequency']
                window_length = product_record['forecast_window_length']
                
                # Calculate cutoff date (analysis_date - 1 day)
                cutoff_date = analysis_date - timedelta(days=1)
                
                # Get historical aggregated data used for forecasting
                historical_data = self._get_historical_aggregated_data_for_visualization(
                    cutoff_date, product_id, location_id
                )
                
                if historical_data is None or len(historical_data) == 0:
                    self.logger.warning(f"No historical data for {product_id}-{location_id} on {analysis_date}")
                    continue
                
                # Get forecast horizon data from accuracy metrics
                horizon_data = self._get_forecast_horizon_data_for_visualization(
                    analysis_date, product_id, location_id
                )
                
                if horizon_data.empty:
                    self.logger.warning(f"No horizon data for {product_id}-{location_id} on {analysis_date}")
                    continue
                
                # Create enhanced record
                enhanced_record = {
                    'analysis_date': analysis_date,
                    'product_id': product_id,
                    'location_id': location_id,
                    'risk_period': risk_period,
                    'demand_frequency': demand_frequency,
                    'window_length': window_length,
                    'historical_bucket_start_dates': [str(d) for d in historical_data['bucket_start_date'].tolist()],
                    'historical_demands': historical_data['total_demand'].tolist(),
                    'forecast_horizon_start_dates': [str(d) for d in horizon_data['start_date'].tolist()],
                    'forecast_horizon_actual_demands': horizon_data['actual_demand'].tolist(),
                    'forecast_horizon_forecast_demands': horizon_data['forecast_demand'].tolist(),
                    'forecast_horizon_errors': horizon_data['error'].tolist()
                }
                
                enhanced_data.append(enhanced_record)
                
            except Exception as e:
                self.logger.error(f"Error processing forecast for {product_id}-{location_id} on {analysis_date}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Create DataFrame and save
        if enhanced_data:
            enhanced_df = pd.DataFrame(enhanced_data)
            visualization_path = self.config.get_forecast_visualization_path()
            enhanced_df.to_csv(visualization_path, index=False)
            self.logger.info(f"Enhanced forecast visualization data saved to: {visualization_path}")
        else:
            self.logger.warning("No enhanced forecast visualization data created")
    
    def _get_historical_aggregated_data_for_visualization(self, cutoff_date: date, product_id: str, location_id: str):
        """Get historical aggregated data used for forecasting for visualization."""
        try:
            # Filter outlier data up to cutoff date
            filtered_data = self.outlier_data[
                pd.to_datetime(self.outlier_data['date']).dt.date <= cutoff_date
            ].copy()
            
            if len(filtered_data) == 0:
                return None
            
            # Aggregate data
            aggregator = DemandAggregator()
            aggregated_data = aggregator.create_risk_period_buckets_with_data(
                filtered_data,
                self.product_master_data,
                cutoff_date
            )
            
            # Filter for specific product-location
            product_data = aggregated_data[
                (aggregated_data['product_id'] == product_id) & 
                (aggregated_data['location_id'] == location_id)
            ].copy()
            
            # Sort by bucket start date
            product_data = product_data.sort_values('bucket_start_date')
            
            return product_data
            
        except Exception as e:
            self.logger.error(f"Error getting historical data for visualization: {e}")
            return None
    
    def _get_forecast_horizon_data_for_visualization(self, analysis_date: date, product_id: str, location_id: str):
        """Get forecast horizon data from accuracy metrics for visualization."""
        try:
            # Find matching accuracy metric
            matching_metric = None
            for metric in self.accuracy_metrics:
                if (metric['analysis_date'] == analysis_date and 
                    metric['product_id'] == product_id and 
                    metric['location_id'] == location_id):
                    matching_metric = metric
                    break
            
            if matching_metric is None:
                return pd.DataFrame()
            
            # Parse actual_demands and forecast_values from string representation
            actual_demands_str = matching_metric['actual_demands']
            forecast_values_str = matching_metric['forecast_values']
            
            # Convert string representations to actual lists
            if isinstance(actual_demands_str, str):
                # Remove np.float64 wrapper and convert to float
                actual_demands_str = actual_demands_str.replace('np.float64(', '').replace(')', '')
                actual_demands = [float(x.strip()) for x in actual_demands_str.strip('[]').split(',')]
            else:
                actual_demands = actual_demands_str
                
            if isinstance(forecast_values_str, str):
                forecast_values = [float(x.strip()) for x in forecast_values_str.strip('[]').split(',')]
            else:
                forecast_values = forecast_values_str
            
            risk_period = matching_metric['risk_period']
            demand_frequency = matching_metric['demand_frequency']
            
            # Calculate risk period size in days
            risk_period_days = int(ProductMasterSchema.get_risk_period_days(demand_frequency, risk_period))
            
            # Calculate risk period dates
            start_date = analysis_date  # First forecast bucket starts on analysis date
            
            horizon_data = []
            for step in range(len(actual_demands)):
                risk_period_start = start_date + timedelta(days=step * risk_period_days)
                actual_demand = actual_demands[step]
                forecast_demand = forecast_values[step]
                error = actual_demand - forecast_demand
                
                horizon_data.append({
                    'start_date': risk_period_start,
                    'actual_demand': actual_demand,
                    'forecast_demand': forecast_demand,
                    'error': error
                })
            
            return pd.DataFrame(horizon_data)
            
        except Exception as e:
            self.logger.error(f"Error getting horizon data for visualization: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame() 