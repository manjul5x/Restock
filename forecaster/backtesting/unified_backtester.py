"""
Unified Backtesting Module

This module provides unified backtesting functionality that:
1. Uses method-specific parameter optimization (once per product)
2. Groups products by forecasting method
3. Works at daily level for forecasting
4. Aggregates results at risk_period level
5. Supports multiple forecasting methods with extensible framework
6. Supports multiple forecast methods per product-location
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
import json

from .config import BacktestConfig
from ..data.loader import DemandDataLoader
from ..data.demand_validator import DemandValidator
from ..outlier.handler import OutlierHandler
from ..data.aggregator import DemandAggregator
from ..forecasting.parameter_optimizer import ParameterOptimizerFactory
from ..forecasting.core_engine import CoreForecastingEngine
from ..forecasting.base import calculate_forecast_metrics
from ..utils.logger import ForecasterLogger
from ..data.product_master_schema import ProductMasterSchema


class UnifiedBacktester:
    """
    Unified backtesting class that supports multiple forecasting methods
    with method-specific parameter optimization (once per product).
    """
    
    def __init__(self, config: BacktestConfig):
        """Initialize the unified backtester with configuration."""
        self.config = config
        self.logger = ForecasterLogger(
            "unified_backtester", config.log_level, config.log_file
        )
        
        # Data storage
        self.demand_data: Optional[pd.DataFrame] = None
        self.product_master_data: Optional[pd.DataFrame] = None
        self.expanded_product_master_data: Optional[pd.DataFrame] = None  # Expanded for methods
        self.outlier_data: Optional[pd.DataFrame] = None
        
        # Results storage
        self.backtest_results: List[Dict[str, Any]] = []
        self.accuracy_metrics: List[Dict[str, Any]] = []
        self.forecast_comparisons: List[Dict[str, Any]] = []
        
        # Product grouping by forecasting method
        self.product_groups = {}
        
        # Core forecasting engine
        self.forecasting_engine = CoreForecastingEngine()
        
        # Optimized parameters cache (set once per product)
        self.optimized_parameters_cache = {}
        
        # Validate configuration
        if not self.config.validate_dates():
            raise ValueError(
                "Invalid date configuration: historic_start_date <= analysis_start_date <= analysis_end_date"
            )
    
    def run(self) -> Dict:
        """
        Run the complete unified backtesting process.
        
        Returns:
            Dictionary with backtesting summary and results
        """
        start_time = time.time()
        self.logger.info("Starting unified backtesting process")
        self.logger.info(f"Configuration: {self.config.__dict__}")
        
        try:
            # Step 1: Load and validate data
            self._load_data()
            
            # Step 1.5: Expand product master for multiple methods
            self._expand_product_master_for_methods()
            
            # Print initial summary
            self._print_initial_summary()
            
            # Step 2: Group products by forecasting method
            self._group_products_by_method()
            
            # Step 3: Handle outliers across the entire analysis period
            self._handle_outliers()
            
            # Step 4: Optimize parameters once for each product-location-method
            self._optimize_parameters_once()
            
            # Step 5: Run backtesting for each analysis date
            self._run_unified_backtesting()
            
            # Step 6: Calculate accuracy metrics
            self._calculate_accuracy_metrics()
            
            # Step 7: Save results
            self._save_results()
            
            # Generate summary
            total_time = time.time() - start_time
            summary = self._generate_summary(total_time)
            
            self.logger.info("Unified backtesting completed successfully")
            return summary
            
        except Exception as e:
            self.logger.error(f"Unified backtesting failed: {e}")
            raise
    
    def _load_data(self):
        """Load and validate demand and product master data."""
        self.logger.info("Step 1: Loading and validating data")
        
        # Load demand data
        demand_loader = DemandDataLoader(self.config.data_dir)
        self.demand_data = demand_loader.load_csv(self.config.demand_file)
        
        # Load product master data
        self.product_master_data = demand_loader.load_csv(self.config.product_master_file)
        
        # Validate product master schema
        ProductMasterSchema.validate_dataframe(self.product_master_data)
        self.product_master_data = ProductMasterSchema.standardize_dataframe(self.product_master_data)
        
        self.logger.info(f"Loaded {len(self.demand_data)} demand records")
        self.logger.info(f"Loaded {len(self.product_master_data)} product master records")
    
    def _expand_product_master_for_methods(self):
        """Expand product master to create separate rows for each forecast method."""
        self.logger.info("Step 1.5: Expanding product master for multiple methods")
        
        self.expanded_product_master_data = ProductMasterSchema.expand_product_master_for_methods(
            self.product_master_data
        )
        
        self.logger.info(f"Expanded to {len(self.expanded_product_master_data)} product-method combinations")
        
        # Log method distribution
        method_counts = self.expanded_product_master_data['forecast_method'].value_counts()
        for method, count in method_counts.items():
            self.logger.info(f"Method '{method}': {count} product-location combinations")
    
    def _print_initial_summary(self):
        """Print initial summary of the backtesting configuration and data."""
        print("\n" + "="*60)
        print("UNIFIED BACKTESTING SUMMARY")
        print("="*60)
        print(f"Data Configuration:")
        print(f"  • Demand file: {self.config.demand_file}")
        print(f"  • Product master file: {self.config.product_master_file}")
        print(f"  • Total demand records: {len(self.demand_data):,}")
        print(f"  • Original product-location combinations: {len(self.product_master_data):,}")
        print(f"  • Expanded product-method combinations: {len(self.expanded_product_master_data):,}")
        
        print(f"\nDate Configuration:")
        print(f"  • Historic start: {self.config.historic_start_date}")
        print(f"  • Analysis period: {self.config.analysis_start_date} to {self.config.analysis_end_date}")
        print(f"  • Analysis dates: {len(self.config.get_analysis_dates())} dates")
        
        print(f"\nForecasting Configuration:")
        print(f"  • Default horizon: {self.config.default_horizon} risk periods")
        print(f"  • Demand frequency: {self.config.demand_frequency}")
        print(f"  • Outlier handling: {'Enabled' if self.config.outlier_enabled else 'Disabled'}")
        print(f"  • Aggregation: {'Enabled' if self.config.aggregation_enabled else 'Disabled'}")
        
        print(f"\nProcessing Configuration:")
        print(f"  • Batch size: {self.config.batch_size}")
        print(f"  • Max workers: {self.config.max_workers}")
        print(f"  • Expected total forecasts: {len(self.config.get_analysis_dates()) * len(self.expanded_product_master_data):,}")
        print("="*60 + "\n")
    
    def _group_products_by_method(self):
        """Group products by their forecasting method for efficient processing."""
        self.logger.info("Step 2: Grouping products by forecasting method")
        
        # Group by forecasting method using expanded product master
        for _, product_record in self.expanded_product_master_data.iterrows():
            method = product_record['forecast_method']
            
            if method not in self.product_groups:
                self.product_groups[method] = []
            
            self.product_groups[method].append({
                'product_id': product_record['product_id'],
                'location_id': product_record['location_id'],
                'forecast_method': method,
                'product_record': product_record
            })
        
        # Log grouping results
        for method, products in self.product_groups.items():
            self.logger.info(f"Method '{method}': {len(products)} product-method combinations")
    
    def _handle_outliers(self):
        """Handle outliers across the entire analysis period."""
        self.logger.info("Step 3: Handling outliers")
        
        # Create outlier handler
        outlier_handler = OutlierHandler()
        
        # Process outliers using the actual loaded data
        result = outlier_handler.process_demand_outliers_with_data(
            self.demand_data, 
            self.product_master_data,
            default_method="iqr",
            default_threshold=1.5
        )
        
        # Use cleaned data as the outlier-processed data
        self.outlier_data = result['cleaned_data']
        self.logger.info(f"Outlier processing completed: {len(self.outlier_data)} cleaned records")
    
    def _optimize_parameters_once(self):
        """Step 4: Optimize parameters once for each product-location-method using all available data."""
        self.logger.info("Step 4: Optimizing parameters once for each product-location-method")
        
        for method_name, products in self.product_groups.items():
            self.logger.info(f"Optimizing parameters for {len(products)} products with method: {method_name}")
            
            for product_info in products:
                product_record = product_info['product_record']
                product_id = product_record['product_id']
                location_id = product_record['location_id']
                forecast_method = product_record['forecast_method']
                
                # Get all available data for this product-location
                all_data = self._get_all_data_for_product_location(product_id, location_id)
                
                if len(all_data) == 0:
                    self.logger.warning(f"No data found for {product_id}-{location_id}, skipping parameter optimization")
                    continue
                
                # Get base parameters from product master
                base_parameters = {
                    'window_length': product_record.get('forecast_window_length') * product_record.get('risk_period', 1),
                    'horizon': product_record.get('forecast_horizon', self.config.default_horizon),
                    'risk_period_days': product_record.get('risk_period')
                }
                
                # Optimize parameters using all available data
                try:
                    optimizer = ParameterOptimizerFactory.get_optimizer(forecast_method)
                    optimized_parameters = optimizer.optimize_parameters(
                        all_data, product_record, base_parameters
                    )
                    
                    # Cache the optimized parameters with method included
                    product_key = f"{product_id}_{location_id}_{forecast_method}"
                    self.optimized_parameters_cache[product_key] = optimized_parameters
                    
                    self.logger.debug(f"Optimized parameters for {product_id}-{location_id}-{forecast_method}: {optimized_parameters}")
                    
                except Exception as e:
                    self.logger.error(f"Error optimizing parameters for {product_id}-{location_id}-{forecast_method}: {e}")
                    continue
        
        self.logger.info(f"Parameter optimization completed for {len(self.optimized_parameters_cache)} product-location-method combinations")
    
    def _get_all_data_for_product_location(self, product_id: str, location_id: str) -> pd.DataFrame:
        """Get all available data for a product-location combination."""
        # Use outlier data if available, otherwise use original demand data
        input_data = self.outlier_data if self.outlier_data is not None else self.demand_data
        
        # Filter for this product-location
        product_data = input_data[
            (input_data['product_id'] == product_id) & 
            (input_data['location_id'] == location_id)
        ]
        
        return product_data
    
    def _run_unified_backtesting(self):
        """Step 5: Run backtesting for each analysis date."""
        self.logger.info("Step 5: Running unified backtesting")
        
        # Get analysis dates
        analysis_dates = self.config.get_analysis_dates()
        
        # Create product-location-method combinations for processing
        product_locations = []
        for method_name, products in self.product_groups.items():
            for product_info in products:
                product_locations.append(product_info)
        
        self.logger.info(f"Running backtesting for {len(analysis_dates)} dates and {len(product_locations)} product-method combinations")
        
        # Run backtesting (sequential or parallel)
        if self.config.max_workers > 1:
            self._run_unified_backtesting_parallel(analysis_dates, product_locations)
        else:
            self._run_unified_backtesting_sequential(analysis_dates, product_locations)
    
    def _run_unified_backtesting_sequential(
        self, analysis_dates: List[date], product_locations: List[Dict]
    ):
        """Run backtesting sequentially."""
        self.logger.info("Running backtesting sequentially")
        
        for analysis_date in tqdm(analysis_dates, desc="Analysis dates"):
            cutoff_date = self._get_cutoff_date(analysis_date)
            
            for product_info in product_locations:
                result = self._run_unified_forecast_for_product_location(
                    analysis_date, cutoff_date, 
                    product_info['product_id'], product_info['location_id'], 
                    product_info['product_record']
                )
                
                if result:
                    self.backtest_results.append(result)
    
    def _run_unified_backtesting_parallel(
        self, analysis_dates: List[date], product_locations: List[Dict]
    ):
        """Run backtesting in parallel."""
        self.logger.info(f"Running backtesting in parallel with {self.config.max_workers} workers")
        
        # Create tasks
        tasks = []
        for analysis_date in analysis_dates:
            cutoff_date = self._get_cutoff_date(analysis_date)
            
            for product_info in product_locations:
                task = {
                    'analysis_date': analysis_date,
                    'cutoff_date': cutoff_date,
                    'product_id': product_info['product_id'],
                    'location_id': product_info['location_id'],
                    'product_record': product_info['product_record']
                }
                tasks.append(task)
        
        # Process tasks in parallel
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = [executor.submit(self._process_single_backtest_task, task) for task in tasks]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing tasks"):
                result = future.result()
                if result:
                    # Extract comparisons from result and add to forecast_comparisons list
                    if 'comparisons' in result:
                        comparisons = result.pop('comparisons')  # Remove from result
                        if comparisons:
                            self.forecast_comparisons.extend(comparisons)
                    
                    self.backtest_results.append(result)
    
    def _process_single_backtest_task(self, task: Dict) -> Optional[Dict[str, Any]]:
        """Process a single backtest task."""
        return self._run_unified_forecast_for_product_location(
            task['analysis_date'], task['cutoff_date'],
            task['product_id'], task['location_id'], task['product_record']
        )
    
    def _run_unified_forecast_for_product_location(
        self, analysis_date: date, cutoff_date: date, 
        product_id: str, location_id: str, product_record: pd.Series
    ) -> Optional[Dict[str, Any]]:
        """Run unified forecast for a single product-location-method combination."""
        try:
            # Get daily data for this product-location up to cutoff date
            daily_data = self._get_daily_data_for_product_location(product_id, location_id, cutoff_date)
            
            if len(daily_data) == 0:
                self.logger.warning(f"No data found for {product_id}-{location_id} up to {cutoff_date}")
                return None
            
            # Get forecast method from product record
            forecast_method = product_record['forecast_method']
            
            # Get optimized parameters for this product-location-method
            product_key = f"{product_id}_{location_id}_{forecast_method}"
            optimized_parameters = self.optimized_parameters_cache.get(product_key, {})
            
            # Generate forecast using the core forecasting engine
            horizon = product_record.get('forecast_horizon', self.config.default_horizon)
            forecast_result = self.forecasting_engine.generate_forecast(
                daily_data, product_record, analysis_date, horizon, optimized_parameters
            )
            
            if not forecast_result or 'forecast_values' not in forecast_result:
                self.logger.warning(f"No forecast generated for {product_id}-{location_id}-{forecast_method}")
                return None
            
            # Get actual aggregated demands for comparison
            actual_demands = self._get_actual_aggregated_demand(
                analysis_date, product_id, location_id, product_record
            )
            
            if not actual_demands:
                self.logger.warning(f"No actual demands found for {product_id}-{location_id} on {analysis_date}")
                return None
            
            # Calculate forecast metrics
            forecast_values = forecast_result['forecast_values']
            metrics = calculate_forecast_metrics(actual_demands, forecast_values)
            
            # Create backtest result
            result = {
                'analysis_date': analysis_date,
                'product_id': product_id,
                'location_id': location_id,
                'forecast_method': forecast_method,  # Include method in results
                'risk_period': product_record.get('risk_period'),
                'demand_frequency': product_record.get('demand_frequency'),
                'forecast_values': forecast_values,
                'actual_demands': actual_demands,
                'metrics': metrics,
                'parameters': optimized_parameters,
                'first_date_used': forecast_result.get('first_date_used')  # Include first date used
            }
            
            # Generate forecast comparisons for visualization
            # Convert forecast_values to list if it's not already
            if not isinstance(forecast_values, list):
                forecast_values = [forecast_values]
            

            
            comparisons = self._generate_forecast_comparisons(
                analysis_date, product_id, location_id, forecast_method,
                actual_demands, forecast_values,
                product_record.get('risk_period'), product_record.get('demand_frequency'),
                forecast_result.get('first_date_used')  # Pass first_date_used
            )
            
            # Add comparisons to the result so they can be collected later
            result['comparisons'] = comparisons
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in backtest for {product_id}-{location_id}-{forecast_method}: {e}")
            return None
    
    def _get_daily_data_for_product_location(
        self, product_id: str, location_id: str, cutoff_date: date
    ) -> pd.DataFrame:
        """Get daily data for a product-location up to cutoff date."""
        # Use outlier data if available, otherwise use original demand data
        input_data = self.outlier_data if self.outlier_data is not None else self.demand_data
        
        # Filter for this product-location up to cutoff date
        product_data = input_data[
            (input_data['product_id'] == product_id) & 
            (input_data['location_id'] == location_id) &
            (input_data['date'] <= cutoff_date)
        ]
        
        return product_data.sort_values('date')
    
    def _get_actual_aggregated_demand(
        self, analysis_date: date, product_id: str, location_id: str, product_record: pd.Series
    ) -> Optional[List[float]]:
        """Get actual aggregated demands for comparison."""
        try:
            # Use outlier data if available, otherwise use original demand data
            input_data = self.outlier_data if self.outlier_data is not None else self.demand_data
            
            # Get parameters
            risk_period = product_record.get('risk_period')
            demand_frequency = product_record.get('demand_frequency')
            horizon = product_record.get('forecast_horizon', self.config.default_horizon)
            
            # Calculate risk period in days
            if demand_frequency == 'd':
                risk_period_days = risk_period
            elif demand_frequency == 'w':
                risk_period_days = risk_period * 7
            elif demand_frequency == 'm':
                risk_period_days = risk_period * 30
            else:
                raise ValueError(f"Invalid demand frequency: {demand_frequency}")
            
            # Get actual demands for each risk period
            actual_demands = []
            for i in range(horizon):
                period_start = analysis_date + timedelta(days=i * risk_period_days)
                period_end = analysis_date + timedelta(days=(i + 1) * risk_period_days)
                
                # Filter data for this period
                period_data = input_data[
                    (input_data['product_id'] == product_id) &
                    (input_data['location_id'] == location_id) &
                    (input_data['date'] >= period_start) &
                    (input_data['date'] < period_end)
                ]
                
                # Aggregate demand for this period
                if len(period_data) > 0:
                    period_demand = period_data['demand'].sum()
                else:
                    period_demand = 0.0
                
                actual_demands.append(period_demand)
            
            return actual_demands
            
        except Exception as e:
            self.logger.error(f"Error getting actual demands for {product_id}-{location_id}: {e}")
            return None
    
    def _get_cutoff_date(self, analysis_date: date) -> date:
        """Get cutoff date for forecasting (exclusive)."""
        return analysis_date
    
    def _calculate_accuracy_metrics(self):
        """Step 6: Calculate accuracy metrics."""
        self.logger.info("Step 6: Calculating accuracy metrics")
        
        for result in self.backtest_results:
            metrics = result['metrics']
            
            accuracy_record = {
                'analysis_date': result['analysis_date'],
                'product_id': result['product_id'],
                'location_id': result['location_id'],
                'forecast_method': result['forecast_method'],  # Include method
                'mae': metrics.get('mae', 0),
                'mape': metrics.get('mape', 0),
                'rmse': metrics.get('rmse', 0),
                'bias': metrics.get('bias', 0),
                'demand_frequency': result['demand_frequency']
            }
            
            self.accuracy_metrics.append(accuracy_record)
        
        self.logger.info(f"Calculated accuracy metrics for {len(self.accuracy_metrics)} forecasts")
    
    def _generate_forecast_comparisons(
        self, analysis_date: date, product_id: str, location_id: str, forecast_method: str,
        actual_demands: List[float], forecast_values: List[float],
        risk_period: int, demand_frequency: str, first_date_used: Optional[date] = None
    ):
        """Generate forecast comparison data for visualization."""
        self.logger.debug(f"Generating comparisons for {product_id}-{location_id}-{forecast_method}: {len(actual_demands)} actual, {len(forecast_values)} forecast")
        
        # Convert numpy arrays to regular Python types
        actual_demands = [float(actual) for actual in actual_demands]
        forecast_values = [float(forecast) for forecast in forecast_values]
        
        # Create comparison records
        comparisons = []
        for i, (actual, forecast) in enumerate(zip(actual_demands, forecast_values)):
            # Calculate risk period start and end to avoid overlap
            if i == 0:
                # First step: start on analysis date
                risk_period_start = analysis_date
            else:
                # Subsequent steps: start the day after the previous period ends
                risk_period_start = analysis_date + timedelta(days=i * risk_period) + timedelta(days=1)
            
            risk_period_end = analysis_date + timedelta(days=(i + 1) * risk_period)
            
            comparison = {
                'analysis_date': analysis_date,
                'product_id': product_id,
                'location_id': location_id,
                'forecast_method': forecast_method,  # Include method in comparisons
                'risk_period': risk_period,  # Use the actual risk_period from product master
                'step': i + 1,  # Add 'step' column for simulation compatibility
                'risk_period_start': risk_period_start,
                'risk_period_end': risk_period_end,
                'actual_demand': actual,
                'forecast_demand': forecast,
                'forecast_error': actual - forecast,
                'absolute_error': abs(actual - forecast),
                'percentage_error': ((actual - forecast) / actual * 100) if actual != 0 else 0,
                'demand_frequency': demand_frequency,
                'first_date_used': first_date_used  # Include first date of data used
            }
            
            comparisons.append(comparison)
        
        return comparisons
    
    def _save_results(self):
        """Save backtesting results."""
        self.logger.info("Step 7: Saving results")
        
        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save backtest results
        if self.backtest_results:
            results_file = Path(self.config.output_dir) / "backtest_results.csv"
            results_df = pd.DataFrame(self.backtest_results)
            results_df.to_csv(results_file, index=False)
            self.logger.info(f"Backtest results saved to: {results_file}")
        
        # Save accuracy metrics
        if self.accuracy_metrics:
            metrics_file = Path(self.config.output_dir) / "accuracy_metrics.csv"
            metrics_df = pd.DataFrame(self.accuracy_metrics)
            metrics_df.to_csv(metrics_file, index=False)
            self.logger.info(f"Accuracy metrics saved to: {metrics_file}")
        
        # Save forecast comparisons
        if self.forecast_comparisons:
            comparison_file = Path(self.config.output_dir) / "forecast_comparison.csv"
            comparison_df = pd.DataFrame(self.forecast_comparisons)
            comparison_df.to_csv(comparison_file, index=False)
            self.logger.info(f"Forecast comparisons saved to: {comparison_file}")
            
            # Generate forecast visualization data
            self._generate_forecast_visualization_data(comparison_df)
    
    def _generate_forecast_visualization_data(self, comparison_df: pd.DataFrame):
        """Generate forecast visualization data with historical and forecast information."""
        self.logger.info("Generating forecast visualization data")
        
        # Load customer demand data for historical calculations
        demand_df = self.demand_data.copy()
        demand_df['date'] = pd.to_datetime(demand_df['date']).dt.date
        
        # Group by product, location, forecast method, AND analysis date
        grouped_data = comparison_df.groupby(['product_id', 'location_id', 'forecast_method', 'analysis_date']).agg({
            'risk_period': 'first',
            'demand_frequency': 'first',
            'first_date_used': 'first'
        }).reset_index()
        
        # Get product master data for additional fields
        product_master_df = self.product_master_data.copy()
        
        visualization_data = []
        
        for _, row in grouped_data.iterrows():
            product_id = row['product_id']
            location_id = row['location_id']
            forecast_method = row['forecast_method']
            analysis_date = row['analysis_date']
            risk_period = row['risk_period']
            demand_frequency = row['demand_frequency']
            first_date_used = row['first_date_used']
            
            # Get product master record for window_length
            product_record = product_master_df[
                (product_master_df['product_id'] == product_id) & 
                (product_master_df['location_id'] == location_id)
            ].iloc[0]
            window_length = product_record.get('forecast_window_length', 25) * product_record.get('risk_period', 1)
            
            # Generate historical bucket start dates
            historical_bucket_start_dates = self._generate_historical_bucket_dates(
                analysis_date, first_date_used, risk_period, demand_frequency
            )
            
            # Calculate historical demands for each bucket
            historical_demands = self._calculate_historical_demands(
                demand_df, product_id, location_id, historical_bucket_start_dates, risk_period, demand_frequency
            )
            
            # Get forecast horizon data from comparison file
            forecast_data = comparison_df[
                (comparison_df['product_id'] == product_id) &
                (comparison_df['location_id'] == location_id) &
                (comparison_df['forecast_method'] == forecast_method) &
                (comparison_df['analysis_date'] == analysis_date)
            ].sort_values('step')
            
            forecast_horizon_start_dates = forecast_data['risk_period_start'].tolist()
            forecast_horizon_actual_demands = forecast_data['actual_demand'].tolist()
            forecast_horizon_forecast_demands = forecast_data['forecast_demand'].tolist()
            forecast_horizon_errors = forecast_data['forecast_error'].tolist()
            
            visualization_record = {
                'analysis_date': analysis_date,
                'product_id': product_id,
                'location_id': location_id,
                'forecast_method': forecast_method,  # Add forecast_method column
                'risk_period': risk_period,
                'demand_frequency': demand_frequency,
                'window_length': window_length,
                'historical_bucket_start_dates': historical_bucket_start_dates,
                'historical_demands': historical_demands,
                'forecast_horizon_start_dates': forecast_horizon_start_dates,
                'forecast_horizon_actual_demands': forecast_horizon_actual_demands,
                'forecast_horizon_forecast_demands': forecast_horizon_forecast_demands,
                'forecast_horizon_errors': forecast_horizon_errors
            }
            
            visualization_data.append(visualization_record)
        
        # Save forecast visualization data
        if visualization_data:
            visualization_file = Path(self.config.output_dir) / "forecast_visualization_data.csv"
            visualization_df = pd.DataFrame(visualization_data)
            
            # Convert datetime objects and numpy types to more readable formats
            for col in ['historical_bucket_start_dates', 'forecast_horizon_start_dates']:
                if col in visualization_df.columns:
                    visualization_df[col] = visualization_df[col].apply(
                        lambda x: [d.strftime('%Y-%m-%d') if isinstance(d, date) else str(d) for d in x]
                    )
            
            for col in ['historical_demands', 'forecast_horizon_actual_demands', 'forecast_horizon_forecast_demands', 'forecast_horizon_errors']:
                if col in visualization_df.columns:
                    visualization_df[col] = visualization_df[col].apply(
                        lambda x: [float(val) for val in x]
                    )
            
            visualization_df.to_csv(visualization_file, index=False)
            self.logger.info(f"Forecast visualization data saved to: {visualization_file}")
    
    def _generate_historical_bucket_dates(
        self, analysis_date: date, first_date_used: date, risk_period: int, demand_frequency: str
    ) -> List[date]:
        """Generate historical bucket start dates going backwards from analysis date."""
        bucket_dates = []
        
        # Start one demand frequency before analysis date
        if demand_frequency == 'd':
            start_date = analysis_date - timedelta(days=1)
        elif demand_frequency == 'w':
            start_date = analysis_date - timedelta(weeks=1)
        elif demand_frequency == 'm':
            # Approximate month as 30 days
            start_date = analysis_date - timedelta(days=30)
        else:
            start_date = analysis_date - timedelta(days=1)
        
        # Generate bucket start dates going backwards
        current_date = start_date
        while current_date >= first_date_used:
            bucket_dates.append(current_date)
            
            # Move backwards by risk period (converted to days)
            if demand_frequency == 'd':
                current_date = current_date - timedelta(days=risk_period)
            elif demand_frequency == 'w':
                current_date = current_date - timedelta(days=risk_period * 7)
            elif demand_frequency == 'm':
                current_date = current_date - timedelta(days=risk_period * 30)
            else:
                current_date = current_date - timedelta(days=risk_period)
        
        # Reverse to get chronological order and exclude dates before first_date_used
        bucket_dates = [d for d in reversed(bucket_dates) if d >= first_date_used]
        return bucket_dates
    
    def _calculate_historical_demands(
        self, demand_df: pd.DataFrame, product_id: str, location_id: str, 
        bucket_start_dates: List[date], risk_period: int, demand_frequency: str
    ) -> List[float]:
        """Calculate historical demands for each bucket."""
        historical_demands = []
        
        for i, bucket_start in enumerate(bucket_start_dates):
            # Calculate bucket size in days based on demand frequency and risk period
            if demand_frequency == 'd':
                bucket_size_days = risk_period
            elif demand_frequency == 'w':
                bucket_size_days = risk_period * 7
            elif demand_frequency == 'm':
                bucket_size_days = risk_period * 30
            else:
                bucket_size_days = risk_period
            
            # Calculate bucket end date
            bucket_end = bucket_start + timedelta(days=bucket_size_days - 1)
            
            # Filter demand data for this bucket
            bucket_demand = demand_df[
                (demand_df['product_id'] == product_id) &
                (demand_df['location_id'] == location_id) &
                (demand_df['date'] >= bucket_start) &
                (demand_df['date'] <= bucket_end)
            ]
            
            # Sum the demand for this bucket
            total_demand = bucket_demand['demand'].sum()
            historical_demands.append(total_demand)
        
        return historical_demands
    
    def _generate_summary(self, total_time: float) -> Dict[str, Any]:
        """Generate backtesting summary."""
        summary = {
            'backtesting_type': 'unified',
            'execution_time': total_time,
            'analysis_period': {
                'start': self.config.analysis_start_date,
                'end': self.config.analysis_end_date,
                'dates': len(self.config.get_analysis_dates())
            },
            'data_summary': {
                'total_demand_records': len(self.demand_data),
                'original_product_records': len(self.product_master_data),
                'expanded_product_records': len(self.expanded_product_master_data),
                'outlier_records': len(self.outlier_data) if self.outlier_data is not None else 0
            },
            'results_summary': {
                'total_forecasts': len(self.backtest_results),
                'total_comparisons': len(self.forecast_comparisons),
                'products_optimized': len(self.optimized_parameters_cache)
            },
            'product_groups': {method: len(products) for method, products in self.product_groups.items()},
            'configuration': {
                'demand_frequency': self.config.demand_frequency,
                'max_workers': self.config.max_workers,
                'batch_size': self.config.batch_size
            }
        }
        
        return summary


def run_unified_backtest(config: BacktestConfig) -> Dict[str, Any]:
    """Run unified backtesting with the given configuration."""
    backtester = UnifiedBacktester(config)
    return backtester.run() 