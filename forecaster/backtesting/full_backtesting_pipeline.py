"""
Full Backtesting Pipeline - Aggregated Risk-Period Logic

This is a completely new backtesting pipeline that replaces the legacy system.
It provides aggregated risk-period logic, flexible regressors, and chunked persistence.

Key Features:
- Forward-looking outflow aggregation (risk-period windows)
- Flexible regressor system (lag, season, week-of-month, recency)
- Parallel execution with per-product data passing
- Chunked persistence for crash-safety and resume capability
- New forecasting engine with MovingAverage and configurable Prophet
"""

import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from tqdm import tqdm
import math

try:
    from data.loader import DataLoader
    from data.input_data_prepper import InputDataPrepper
except ImportError:
    # Fallback for when running from forecaster package
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from data.loader import DataLoader
    from data.input_data_prepper import InputDataPrepper

from forecaster.validation.product_master_schema import ProductMasterSchema
from forecaster.outlier.handler import OutlierHandler
from forecaster.forecasting.engine import ForecastingEngine 
from forecaster.utils.pipeline_decorators import pipeline_step
from forecaster.validation.product_master_schema import ProductMasterSchema



class FullBacktestingPipeline:
    """
    Main pipeline orchestrator for the new backtesting system.
    
    This pipeline implements aggregated risk-period logic where:
    - outflow = forward-looking sum of raw demand over risk period window
    - regressors include lag, seasonality, week-of-month, and recency features
    - execution is parallelized by product with chunked persistence
    """
    
    def __init__(self, 
                 analysis_start_date: date,
                 analysis_end_date: date,
                 demand_frequency: str,
                 max_workers: int = 8,
                 run_id: str = None,
                 resume: bool = False,
                 log_level: str = "INFO"):
        """
        Initialize the pipeline.
        
        Args:
            analysis_start_date: Start date for analysis
            analysis_end_date: End date for analysis  
            demand_frequency: 'd', 'w', or 'm'
            max_workers: Maximum parallel workers
            run_id: Unique run identifier
            resume: Whether to resume existing run
            log_level: Logging level
        """
        self.analysis_start_date = analysis_start_date
        self.analysis_end_date = analysis_end_date
        self.demand_frequency = demand_frequency
        self.max_workers = max_workers
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.resume = resume
        self.log_level = log_level
        
        # Initialize components
        self.data_loader = DataLoader()
        self.outlier_handler = OutlierHandler(self.data_loader)
        self.forecasting_engine = ForecastingEngine() 
        self.input_data_prepper = InputDataPrepper()
        
        # Data storage
        self.product_master_df = None
        self.outflow_data = None
        self.product_data_cache = {}
        
        # Chunked persistence state
        self.chunk_idx = 0
        self.buffer_comparisons = []
        self.buffer_components = []
        self.buffer_comparison_rows = 0
        self.buffer_component_rows = 0
        self.last_flush_time = time.time()
        self.buffer_flush_threshold = float('inf')  # Only flush by time, not by row count
        self.buffer_flush_interval = 300  # seconds (5 minutes)
        
        # Get logger
        from forecaster.utils.logger import get_logger
        self.logger = get_logger(__name__, level=log_level)
        
        # Initialize start time for execution timing
        self.start_time = None
        
        self.logger.info(f"Pipeline initialized for {analysis_start_date} to {analysis_end_date}")
        self.logger.info(f"Demand frequency: {demand_frequency}, Max workers: {max_workers}")
        self.logger.info(f"Run ID: {self.run_id}, Resume: {resume}")
    
    def execute(self) -> Dict[str, Any]:
        """
        Execute the complete backtesting pipeline.
        
        Returns:
            Dictionary with execution results and metadata
        """
        try:
            # Set start time for execution timing
            self.start_time = time.time()
            
            self.logger.log_workflow_step("Starting full backtesting pipeline", 1, 6)
            
            # Execute all pipeline steps (logging and timing handled by decorators)
            self._load_and_validate_data()
            # self._expand_product_master_by_methods()  # Skipping for testing
            self._handle_outliers()
            self._prep_input_data()
            self._execute_parallel_backtesting()
            self._finalize_results()
            
            self.logger.log_workflow_step("Pipeline completed successfully", 6, 6)
            
            execution_time = time.time() - self.start_time
            
            # Log performance summary
            self.logger.info(f"Pipeline execution completed in {execution_time:.2f} seconds")
            if hasattr(self, 'performance_metrics'):
                self.logger.info(f"Performance metrics: {self.performance_metrics}")
            
            # Save performance profile if enabled
            if hasattr(self, 'profile_enabled') and self.profile_enabled:
                self._save_performance_profile()
            
            return {
                'status': 'success',
                'run_id': self.run_id,
                'products_processed': len(self.product_master_df),
                'analysis_period': f"{self.analysis_start_date} to {self.analysis_end_date}",
                'execution_time': execution_time
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            raise
    
    @pipeline_step("Loading and validating data", 1, 6)
    def _load_and_validate_data(self):
        """Load and validate product master and outflow data."""
        
        # Load product master
        self.product_master_df = self.data_loader.load_product_master()
        self.logger.info(f"Loaded product master: {len(self.product_master_df)} records")
        
        # Validate product master schema
        # TODO: need to check what's going on here. Do we need to validate? Is it expensive?
        ProductMasterSchema.validate_dataframe(self.product_master_df)
        self.product_master_df = ProductMasterSchema.standardize_dataframe(self.product_master_df)
        self.logger.info("Product master validated and standardized")
        
        # Load outflow data (restricted to products in product master)
        self.outflow_data = self.data_loader.load_outflow(product_master=self.product_master_df)
        
        self.logger.info(f"Loaded outflow data: {len(self.outflow_data)} records")
    
    @pipeline_step("Expanding product master by forecast methods", 2, 6)
    def _expand_product_master_by_methods(self):
        """Expand product master to create separate rows for each forecast method."""
        
        self.product_master_df = ProductMasterSchema.expand_product_master_for_methods(self.product_master_df)
        self.logger.info(f"Expanded product master: {len(self.product_master_df)} method combinations")
    
    @pipeline_step("Processing outliers", 3, 6)
    def _handle_outliers(self):
        """Handle outliers in demand data."""
        
        result = self.outlier_handler.process_demand_outliers_with_data(
            demand_df=self.outflow_data,
            product_master_df=self.product_master_df
        )
        
        # Use cleaned data for further processing
        if 'cleaned_data' in result and len(result['cleaned_data']) > 0:
            self.outflow_data = result['cleaned_data']
            self.logger.info(f"Outliers processed, using cleaned data: {len(self.outflow_data)} records")
        else:
            self.logger.info("No outliers found, using original data")
    
    @pipeline_step("Preparing input data with regressors", 4, 6)
    def _prep_input_data(self):
        """Prepare input data with aggregated outflow and regressors using InputDataPrepper."""
        
        # Use InputDataPrepper to compute all regressors for the entire dataset
        self.outflow_data = self.input_data_prepper.prepare_data(
            df=self.outflow_data,
            product_master_df=self.product_master_df
        )
        
        self.logger.info(f"Prepared input data: {len(self.outflow_data)} records with regressors")
        
        # Save the prepared input data with regressors to CSV
        try:
            self.data_loader.save_input_data_with_regressors(self.outflow_data)
            self.logger.info(f"Saved input data with regressors to CSV: {len(self.outflow_data)} records")
        except Exception as e:
            self.logger.warning(f"Failed to save input data with regressors: {e}")
            # Don't fail the pipeline if saving fails
    
    # Reindexing to contiguous dates removed - regressor functions handle this efficiently
    
    # Old regressor computation methods removed - now handled by InputDataPrepper
    
    @pipeline_step("Executing parallel backtesting", 5, 6)
    def _execute_parallel_backtesting(self):
        """Execute parallel backtesting for all products."""
        
        # Build tasks efficiently
        tasks = self._build_product_tasks()
        self.logger.info(f"Built {len(tasks)} product tasks")
        
        # Handle case where no tasks were built
        if not tasks:
            self.logger.warning("No tasks to execute - skipping parallel execution")
            return
        
        # Execute in parallel
        self._execute_tasks_parallel(tasks)
    
    def _build_product_tasks(self) -> List[Dict[str, Any]]:
        """Build product tasks efficiently by filtering data once per product-location combination."""
        self.logger.info("Building product tasks efficiently...")
        
        # Build tasks efficiently - filter data once per product-location
        tasks = []
        
        for _, product_record in tqdm(self.product_master_df.iterrows(), total=len(self.product_master_df), desc="Building tasks", leave=False):
            product_id = product_record['product_id']
            location_id = product_record['location_id']

            # Filter data for this product-location
            product_data = self.outflow_data[
                (self.outflow_data['product_id'] == product_id) &
                (self.outflow_data['location_id'] == location_id)
            ].copy()

            if len(product_data) > 0:
                # Sort by date and include all columns
                product_data = product_data.sort_values('date').reset_index(drop=True)
                
                # Parse forecast methods safely
                forecast_methods = self._parse_forecast_methods(product_record['forecast_methods'])
                
                # Create a task for each forecast method
                for method in forecast_methods:
                    task = {
                        'product_id': product_id,
                        'location_id': location_id,
                        'forecast_method': method,
                        'product_record': product_record.to_dict(),
                        'product_data': product_data,  # Shared data
                        'analysis_dates': self._get_analysis_dates_for_product(product_data)
                    }
                    tasks.append(task)
        
        self.logger.info(f"Built {len(tasks)} tasks")
        return tasks
    
    def _parse_forecast_methods(self, forecast_method) -> List[str]:
        """
        Safely parse forecast_method into a list of method names.
        
        Handles:
        - String: "prophet" -> ["prophet"]
        - Comma-separated string: "prophet,moving_average" -> ["prophet", "moving_average"]
        - List: ["prophet", "moving_average"] -> ["prophet", "moving_average"]
        - Empty/None: -> []
        """
        if forecast_method is None:
            return []
        
        if isinstance(forecast_method, str):
            if not forecast_method.strip():
                return []
            # Split by comma and strip whitespace
            return [method.strip() for method in forecast_method.split(',') if method.strip()]
        
        elif isinstance(forecast_method, list):
            # Convert list elements to strings and filter out empty ones
            return [str(method).strip() for method in forecast_method if method and str(method).strip()]
        
        else:
            # Try to convert to string as fallback
            try:
                method_str = str(forecast_method).strip()
                return [method_str] if method_str else []
            except:
                return []
    
    def _get_analysis_dates_for_product(self, product_data: pd.DataFrame) -> List[date]:
        """Get analysis dates for a specific product, bounded by product_data and analysis period."""
        # Find min/max date in product_data, fallback to analysis period if empty
        if not product_data.empty and 'date' in product_data.columns:
            data_min = product_data['date'].min()
            data_max = product_data['date'].max()
        else:
            data_min = self.analysis_start_date
            data_max = self.analysis_end_date

        # Use the intersection of product_data and analysis period
        # Convert analysis dates to datetime for comparison with data dates
        analysis_start_dt = pd.Timestamp(self.analysis_start_date)
        analysis_end_dt = pd.Timestamp(self.analysis_end_date)
        
        start_date = max(analysis_start_dt, data_min)
        end_date = min(analysis_end_dt, data_max)

        # Defensive: if start_date > end_date, return empty list
        if start_date > end_date:
            self.logger.warning(
                f"Start date ({start_date}) is after end date ({end_date})"
                f"{product_data['product_id'].iloc[0]}_{product_data['location_id'].iloc[0]}."
                "Returning empty analysis dates."
            )
            return []

        dates = pd.date_range(
            start=start_date,
            end=end_date,
            freq=self.demand_frequency.upper()
        ).date.tolist()

        return dates
    
    def _execute_tasks_parallel(self, tasks: List[Dict[str, Any]]):
        """Execute tasks in parallel using ProcessPoolExecutor."""
        self.logger.info(f"Starting parallel execution with {self.max_workers} workers")
        
        # Initialize failure tracking
        failed_tasks = []
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self._process_product_task, task): task 
                for task in tasks
            }
            
            # Process completed futures with tqdm progress bar
            completed_count = 0
            successful_count = 0
            
            # Create progress bar
            with tqdm(total=len(tasks), desc="Processing tasks", unit="task") as pbar:
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    completed_count += 1
                    
                    try:
                        result = future.result()
                        if result:
                            forecast_comparisons, prophet_components = result
                            
                            # Add to buffers
                            self.buffer_comparisons.extend(forecast_comparisons)
                            self.buffer_components.extend(prophet_components)
                            self.buffer_comparison_rows += len(forecast_comparisons)
                            self.buffer_component_rows += len(prophet_components)
                            
                            # Check if we should flush buffers
                            self._check_and_flush_buffers()
                            
                            successful_count += 1
                            
                    except Exception as e:
                        # Track failed task details
                        failed_task_info = {
                            'product_id': task['product_id'],
                            'location_id': task['location_id'],
                            'forecast_method': task['forecast_method'],
                            'error_message': str(e),
                            'timestamp': datetime.now().isoformat(),
                            'run_id': self.run_id
                        }
                        failed_tasks.append(failed_task_info)
                        
                        self.logger.error(f"Task failed for {task['product_id']}_{task['location_id']} "
                                       f"with method {task['forecast_method']}: {e}")
                    
                    # Update progress bar with current status
                    pbar.set_postfix({
                        'Success': successful_count,
                        'Failed': len(failed_tasks),
                        'Total': completed_count
                    })
                    pbar.update(1)
                    
                    # Log progress every 10 tasks for log files
                    if completed_count % 10 == 0:
                        self.logger.debug(f"Completed {completed_count}/{len(tasks)} tasks "
                                       f"(Success: {successful_count}, Failed: {len(failed_tasks)})")
        
        # Final flush of any remaining data
        self._flush_all_buffers()
        
        # Save failure information to file
        if failed_tasks:
            self._save_failure_log(failed_tasks)
        
        self.logger.info(f"Parallel execution completed: {completed_count} tasks processed "
                        f"(Success: {successful_count}, Failed: {len(failed_tasks)})")
    
    def _process_product_task(self, task: Dict[str, Any]) -> Optional[Tuple[List, List]]:
        """Process a single product task (runs in worker process)."""
        try:
            product_record = task['product_record']
            product_data = task['product_data']
            analysis_dates = task['analysis_dates']
            forecast_method = task['forecast_method']
            
            # Initialize result containers
            forecast_comparisons = []
            prophet_components = []
            
            # Resolve time parameters from product record
            risk_period_days = ProductMasterSchema.get_risk_period_days(
                product_record.get('demand_frequency'),
                product_record.get('risk_period')
            )
            forecast_window_days = product_record.get('forecast_window_length')
            horizon_days = product_record.get('forecast_horizon')
            

            
            # Process each analysis date
            for analysis_date in analysis_dates:
                # Convert analysis_date to datetime for Timedelta operations
                analysis_datetime = pd.Timestamp(analysis_date)
                
                # Compute cutoff dates
                cutoff_date = analysis_datetime - pd.Timedelta(days=risk_period_days)
                window_cutoff = max(
                    product_data['date'].min(),
                    analysis_datetime - pd.Timedelta(days=forecast_window_days)
                )
                
                # Filter training data
                training_data = product_data[
                    (product_data['date'] <= cutoff_date) & 
                    (product_data['date'] >= window_cutoff)
                ].copy()
                
                # Drop identifier columns, keep outflow and regressors
                training_data = training_data.drop(['product_id', 'location_id'], axis=1, errors='ignore')
                
                # Create future data frame
                future_end_date = analysis_datetime + pd.Timedelta(days=horizon_days)
                future_data = product_data[
                    (product_data['date'] >= analysis_datetime) & 
                    (product_data['date'] <= future_end_date)
                ].copy()
                                
                # Drop identifier columns, keep outflow and regressors
                future_data = future_data.drop(['product_id', 'location_id'], axis=1, errors='ignore')
                
                # Generate forecast using new engine
                try:
                    forecast_comparison_df, prophet_components_df = self.forecasting_engine.generate_forecast(
                        forecast_method, training_data, product_record, future_data
                    )
                    
                    # Add back product_id and location_id to forecast_comparison_df and prophet_components_df
                    for df in [forecast_comparison_df, prophet_components_df]:
                        df['product_id'] = product_record.get('product_id')
                        df['location_id'] = product_record.get('location_id')

                    # Convert DataFrames to lists of dicts for serialization
                    forecast_comparisons.extend(forecast_comparison_df.to_dict('records'))
                    prophet_components.extend(prophet_components_df.to_dict('records'))
                    
                    
                except Exception as e:
                    # Log forecast generation error but continue with other dates
                    self.logger.warning(f"Forecast generation failed for {product_record.get('product_id')}_{product_record.get('location_id')} on {analysis_date} with method {forecast_method}: {e}")
                    continue
            
            return forecast_comparisons, prophet_components
            
        except Exception as e:
            # Log error but don't crash the worker
            return None
    
    def _check_and_flush_buffers(self):
        """Check if buffers should be flushed and flush if needed."""
        now = time.time()
        
        # Flush if time threshold reached (every 5 minutes)
        if (now - self.last_flush_time) >= self.buffer_flush_interval:
            
            self._flush_comparison_buffer()
            self._flush_component_buffer()
    
    def _flush_comparison_buffer(self):
        """Flush comparison buffer to chunk file."""
        if not self.buffer_comparisons:
            return
        
        try:
            df = pd.DataFrame(self.buffer_comparisons)
            self.data_loader.save_results_chunk(
                df=df,
                category='backtesting',
                base_filename='forecast_comparison',
                run_id=self.run_id,
                chunk_idx=self.chunk_idx
            )
            
            self.logger.debug(f"Flushed comparison chunk {self.chunk_idx}: {len(df)} rows")
            self.buffer_comparisons.clear()
            self.buffer_comparison_rows = 0
            
        except Exception as e:
            self.logger.error(f"Failed to flush comparison buffer: {e}")
    
    def _flush_component_buffer(self):
        """Flush component buffer to chunk file."""
        if not self.buffer_components:
            return
        
        try:
            df = pd.DataFrame(self.buffer_components)
            self.data_loader.save_results_chunk(
                df=df,
                category='backtesting',
                base_filename='prophet_components',
                run_id=self.run_id,
                chunk_idx=self.chunk_idx
            )
            
            self.logger.debug(f"Flushed component chunk {self.chunk_idx}: {len(df)} rows")
            self.buffer_components.clear()
            self.buffer_component_rows = 0
            
            # Increment chunk index for next flush
            self.chunk_idx += 1
            self.last_flush_time = time.time()
            
        except Exception as e:
            self.logger.error(f"Failed to flush component buffer: {e}")
    
    def _flush_all_buffers(self):
        """Flush all remaining buffers."""
        self._flush_comparison_buffer()
        self._flush_component_buffer()
    
    def _save_failure_log(self, failed_tasks: List[Dict[str, Any]]) -> None:
        """
        Save failed task information to a CSV file for analysis.
        
        Args:
            failed_tasks: List of dictionaries containing failure information
        """
        try:
            # Create failure log DataFrame
            failure_df = pd.DataFrame(failed_tasks)
            
            # Save to failure log file
            failure_log_path = Path(self.data_loader.config['paths']['output_dir']) / 'backtesting' / f'failure_log_{self.run_id}.csv'
            failure_log_path.parent.mkdir(parents=True, exist_ok=True)
            
            failure_df.to_csv(failure_log_path, index=False)
            
            self.logger.info(f"Saved failure log with {len(failed_tasks)} failed tasks to {failure_log_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save failure log: {e}")
            # Don't raise - this is not critical for the main pipeline
    
    def enable_profiling(self, profile_enabled: bool = True) -> None:
        """
        Enable or disable performance profiling.
        
        Args:
            profile_enabled: Whether to enable profiling
        """
        self.profile_enabled = profile_enabled
        if profile_enabled:
            self.logger.info("Performance profiling enabled")
    
    def _save_performance_profile(self) -> None:
        """Save performance profile data to file."""
        try:
            import cProfile
            import pstats
            from io import StringIO
            
            # Create performance summary
            profile_data = {
                'run_id': self.run_id,
                'execution_time': time.time() - self.start_time,
                'total_products': len(self.product_master_df),
                'max_workers': self.max_workers,
                'demand_frequency': self.demand_frequency,
                'analysis_period': f"{self.analysis_start_date} to {self.analysis_end_date}",
                'timestamp': datetime.now().isoformat()
            }
            
            # Save profile data
            profile_path = Path(self.data_loader.config['paths']['output_dir']) / 'backtesting' / f'performance_profile_{self.run_id}.json'
            profile_path.parent.mkdir(parents=True, exist_ok=True)
            
            import json
            with open(profile_path, 'w') as f:
                json.dump(profile_data, f, indent=2)
            
            self.logger.info(f"Performance profile saved to {profile_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save performance profile: {e}")
            # Don't raise - this is not critical for the main pipeline
    
    @pipeline_step("Finalizing results from chunks", 6, 6)
    def _finalize_results(self):
        """Finalize results by consolidating all chunks."""
        
        try:
            # Finalize forecast comparison
            self.data_loader.finalize_results_from_chunks(
                category='backtesting',
                base_filename='forecast_comparison',
                run_id=self.run_id
            )
            
            # Finalize prophet components
            self.data_loader.finalize_results_from_chunks(
                category='backtesting',
                base_filename='prophet_components',
                run_id=self.run_id
            )
            
            self.logger.info("Results finalized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to finalize results: {e}")
            raise


def full_backtesting_pipeline(analysis_start_date: date,
                            analysis_end_date: date,
                            demand_frequency: str,
                            max_workers: int = 8,
                            run_id: str = None,
                            resume: bool = False,
                            log_level: str = "INFO",
                            profile: bool = False) -> Dict[str, Any]:
    """
    Main entry point for the full backtesting pipeline.
    
    Args:
        analysis_start_date: Start date for analysis
        analysis_end_date: End date for analysis
        demand_frequency: 'd', 'w', or 'm'
        max_workers: Maximum parallel workers
        run_id: Unique run identifier
        resume: Whether to resume existing run
        log_level: Logging level
        profile: Whether to enable performance profiling
        
    Returns:
        Dictionary with execution results
    """
    pipeline = FullBacktestingPipeline(
        analysis_start_date=analysis_start_date,
        analysis_end_date=analysis_end_date,
        demand_frequency=demand_frequency,
        max_workers=max_workers,
        run_id=run_id,
        resume=resume,
        log_level=log_level
    )
    
    # Enable profiling if requested
    if profile:
        pipeline.enable_profiling(True)
    
    return pipeline.execute()
