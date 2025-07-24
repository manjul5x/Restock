"""
Enhanced Backtesting Module with Best Model Parameters

This module provides enhanced backtesting functionality that:
1. Uses best model parameters from seasonality analysis
2. Runs forecasts on daily data
3. Aggregates results at risk_period level
4. Saves results for forecast visualization
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
from ..forecasting.prophet import ProphetForecaster, create_prophet_forecaster
from ..forecasting.moving_average import MovingAverageForecaster
from ..forecasting.arima import ARIMAForecaster
from ..forecasting.base import calculate_forecast_metrics
from ..utils.logger import ForecasterLogger
from ..data.product_master_schema import ProductMasterSchema
from ..forecasting.seasonality_analyzer import SeasonalityAnalyzer


class EnhancedBacktester:
    """
    Enhanced backtesting class that uses best model parameters from seasonality analysis.
    Runs forecasts on daily data and aggregates results at risk_period level.
    """

    def __init__(self, config: BacktestConfig):
        """Initialize the enhanced backtester with configuration."""
        self.config = config
        self.logger = ForecasterLogger(
            "enhanced_backtester", config.log_level, config.log_file
        )

        # Data storage
        self.demand_data: Optional[pd.DataFrame] = None
        self.product_master_data: Optional[pd.DataFrame] = None
        self.outlier_data: Optional[pd.DataFrame] = None

        # Results storage
        self.backtest_results: List[Dict[str, Any]] = []
        self.accuracy_metrics: List[Dict[str, Any]] = []
        self.forecast_comparisons: List[Dict[str, Any]] = []
        self.seasonality_analysis_results: Dict[str, Any] = {}

        # Seasonality analyzer
        self.seasonality_analyzer = SeasonalityAnalyzer()

        # Validate configuration
        if not self.config.validate_dates():
            raise ValueError(
                "Invalid date configuration: historic_start_date <= analysis_start_date <= analysis_end_date"
            )

    def run(self) -> Dict:
        """
        Run the complete enhanced backtesting process.

        Returns:
            Dictionary with backtesting summary and results
        """
        start_time = time.time()
        self.logger.info("Starting enhanced backtesting process")
        self.logger.info(f"Configuration: {self.config.__dict__}")

        try:
            # Step 1: Load and validate data
            self._load_data()

            # Print initial summary
            self._print_initial_summary()

            # Step 2: Handle outliers across the entire analysis period
            self._handle_outliers()

            # Step 3: Run enhanced backtesting for each analysis date
            self._run_enhanced_backtesting()

            # Step 4: Calculate accuracy metrics
            self._calculate_accuracy_metrics()

            # Step 5: Save results
            self._save_results()

            # Step 6: Generate summary
            total_time = time.time() - start_time
            summary = self._generate_summary(total_time)

            self.logger.info("Enhanced backtesting completed successfully")
            return summary

        except Exception as e:
            self.logger.error(f"Enhanced backtesting failed: {e}")
            # Return a minimal summary even on error
            return {
                "total_time": time.time() - start_time,
                "total_forecasts": 0,
                "successful_forecasts": 0,
                "error": str(e),
            }

    def _load_data(self):
        """Load and validate data."""
        self.logger.info("Step 1: Loading and validating data")

        # Load demand data (daily frequency for enhanced backtesting)
        data_loader = DemandDataLoader()
        self.demand_data = data_loader.load_customer_demand()

        # Load product master data
        self.product_master_data = data_loader.load_customer_product_master()

        # Validate data
        if self.config.validate_data:
            validator = DemandValidator()
            validation_results = validator.validate_demand_completeness_with_data(
                self.demand_data, self.product_master_data, "daily"
            )
            if not validation_results.get("is_valid", True):
                self.logger.warning(
                    f"Data validation issues: {validation_results.get('issues', [])}"
                )

        self.logger.info(f"Loaded {len(self.demand_data)} demand records")
        self.logger.info(
            f"Loaded {len(self.product_master_data)} product master records"
        )

    def _print_initial_summary(self):
        """Print initial data summary."""
        self.logger.info("Data Summary:")
        self.logger.info(
            f"  Date range: {self.demand_data['date'].min()} to {self.demand_data['date'].max()}"
        )
        self.logger.info(f"  Products: {self.demand_data['product_id'].nunique()}")
        self.logger.info(f"  Locations: {self.demand_data['location_id'].nunique()}")
        self.logger.info(f"  Total demand records: {len(self.demand_data)}")

    def _handle_outliers(self):
        """Handle outliers across the entire analysis period."""
        if not self.config.outlier_enabled:
            self.logger.info("Outlier handling disabled")
            return

        self.logger.info("Step 2: Handling outliers")

        outlier_handler = OutlierHandler()
        outlier_results = outlier_handler.process_demand_outliers_with_data(
            self.demand_data, self.product_master_data
        )
        self.outlier_data = outlier_results["cleaned_data"]

        self.logger.info(
            f"Outlier handling completed. Processed {len(self.outlier_data)} records"
        )

    def _run_enhanced_backtesting(self):
        """Run enhanced backtesting for each analysis date."""
        self.logger.info("Step 3: Running enhanced backtesting")

        analysis_dates = self.config.get_analysis_dates()
        self.logger.info(
            f"Running enhanced backtesting for {len(analysis_dates)} analysis dates"
        )

        # Get all product-location combinations
        product_locations = list(
            zip(
                self.product_master_data["product_id"],
                self.product_master_data["location_id"],
            )
        )
        self.logger.info(
            f"Processing {len(product_locations)} product-location combinations"
        )

        # Initialize results storage
        self.backtest_results = []

        if self.config.max_workers > 1:
            # Parallel processing
            self.logger.info(
                f"Using parallel processing with {self.config.max_workers} workers"
            )
            self._run_enhanced_backtesting_parallel(analysis_dates, product_locations)
        else:
            # Sequential processing
            self.logger.info("Using sequential processing")
            self._run_enhanced_backtesting_sequential(analysis_dates, product_locations)

    def _run_enhanced_backtesting_sequential(
        self, analysis_dates: List[date], product_locations: List[Tuple[str, str]]
    ):
        """Run enhanced backtesting sequentially."""
        total_forecasts = 0

        # Progress bar for analysis dates
        with tqdm(
            total=len(analysis_dates), desc="Processing analysis dates", unit="date"
        ) as pbar:
            for analysis_date in analysis_dates:
                try:
                    # Get cutoff date (closest date in data before analysis_date)
                    cutoff_date = self._get_cutoff_date(analysis_date)

                    # Process each product-location combination
                    for product_id, location_id in product_locations:
                        try:
                            # Run enhanced forecast for this product-location
                            forecast_result = (
                                self._run_enhanced_forecast_for_product_location(
                                    analysis_date, cutoff_date, product_id, location_id
                                )
                            )

                            if forecast_result:
                                self.backtest_results.append(forecast_result)
                                total_forecasts += 1

                        except Exception as e:
                            self.logger.error(
                                f"Error processing {product_id}-{location_id} on {analysis_date}: {e}"
                            )
                            continue

                    self.logger.info(f"Generated forecasts for {analysis_date}")

                except Exception as e:
                    self.logger.error(f"Error processing {analysis_date}: {e}")

                pbar.update(1)
                pbar.set_postfix(
                    {"Date": str(analysis_date), "Forecasts": total_forecasts}
                )

    def _run_enhanced_backtesting_parallel(
        self, analysis_dates: List[date], product_locations: List[Tuple[str, str]]
    ):
        """Run enhanced backtesting in parallel."""
        # Prepare arguments for parallel processing
        args_list = []
        for analysis_date in analysis_dates:
            cutoff_date = self._get_cutoff_date(analysis_date)
            for product_id, location_id in product_locations:
                args_list.append((analysis_date, cutoff_date, product_id, location_id))

        # Process in parallel
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = [
                executor.submit(self._run_enhanced_forecast_for_product_location, *args)
                for args in args_list
            ]

            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Processing forecasts"
            ):
                try:
                    result = future.result()
                    if result:
                        self.backtest_results.append(result)
                except Exception as e:
                    self.logger.error(f"Error in parallel processing: {e}")

    def _run_enhanced_forecast_for_product_location(
        self, analysis_date: date, cutoff_date: date, product_id: str, location_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Run enhanced forecast for a specific product-location using best model parameters.

        Args:
            analysis_date: Date of analysis
            cutoff_date: Date to cut off historical data
            product_id: Product ID
            location_id: Location ID

        Returns:
            Dictionary with forecast results or None if failed
        """
        try:
            # Get product master record
            product_record = self.product_master_data[
                (self.product_master_data["product_id"] == product_id)
                & (self.product_master_data["location_id"] == location_id)
            ].iloc[0]

            # Get daily data for this product-location up to cutoff date
            daily_data = self._get_daily_data_for_product_location(
                product_id, location_id, cutoff_date
            )

            if len(daily_data) < 10:  # Minimum data requirement
                self.logger.debug(
                    f"Insufficient data for {product_id}-{location_id}: {len(daily_data)} points"
                )
                return None

            # Step 1: Run seasonality analysis to get best model parameters
            best_parameters = self._get_best_model_parameters(
                daily_data, product_record
            )

            # Ensure window_length is handled correctly
            if "window_length" in best_parameters:
                if best_parameters["window_length"] is not None:
                    try:
                        best_parameters["window_length"] = int(
                            best_parameters["window_length"]
                        )
                    except (ValueError, TypeError):
                        best_parameters["window_length"] = (
                            None  # Use all available data
                        )
                # If window_length is None, keep it as None to use all available data

            # Add horizon parameter for Prophet
            try:
                best_parameters["horizon"] = int(product_record["risk_period"])
            except (ValueError, TypeError):
                best_parameters["horizon"] = 7  # Default fallback

            # Step 2: Create forecaster with best parameters
            forecaster = self._create_forecaster_with_best_parameters(
                best_parameters, product_record
            )

            # Step 3: Fit model on daily data
            forecaster.fit(daily_data)

            # Step 4: Generate daily forecast
            daily_forecast = forecaster.forecast(steps=product_record["risk_period"])

            # Step 5: Aggregate daily forecast to risk period level
            aggregated_forecast = self._aggregate_daily_forecast_to_risk_period(
                daily_forecast, product_record
            )

            # Step 6: Get actual aggregated demand for comparison
            actual_aggregated_demand = self._get_actual_aggregated_demand(
                analysis_date, product_id, location_id, product_record
            )

            # Step 7: Create forecast result
            forecast_result = {
                "analysis_date": analysis_date,
                "cutoff_date": cutoff_date,
                "product_id": product_id,
                "location_id": location_id,
                "model": "prophet_enhanced",
                "best_parameters": best_parameters,
                "daily_forecast_values": daily_forecast.tolist(),
                "aggregated_forecast_values": aggregated_forecast,
                "actual_aggregated_demand": actual_aggregated_demand,
                "risk_period": product_record["risk_period"],
                "demand_frequency": product_record["demand_frequency"],
                "forecast_mean": (
                    np.mean(aggregated_forecast) if aggregated_forecast else None
                ),
                "data_points_used": len(daily_data),
            }

            return forecast_result

        except Exception as e:
            self.logger.error(
                f"Failed to run enhanced forecast for {product_id}-{location_id} on {analysis_date}: {e}"
            )
            return None

    def _get_daily_data_for_product_location(
        self, product_id: str, location_id: str, cutoff_date: date
    ) -> pd.DataFrame:
        """Get daily data for a specific product-location up to cutoff date."""
        # Use outlier-handled data if available, otherwise use original data
        data_source = (
            self.outlier_data if self.outlier_data is not None else self.demand_data
        )

        # Filter data
        product_data = data_source[
            (data_source["product_id"] == product_id)
            & (data_source["location_id"] == location_id)
            & (pd.to_datetime(data_source["date"]) < pd.Timestamp(cutoff_date))
        ].copy()

        # Sort by date and ensure proper format
        product_data = product_data.sort_values("date").reset_index(drop=True)

        return product_data

    def _get_best_model_parameters(
        self, daily_data: pd.DataFrame, product_record: pd.Series
    ) -> Dict[str, Any]:
        """
        Get best model parameters using seasonality analysis.

        Args:
            daily_data: Daily demand data
            product_record: Product master record

        Returns:
            Dictionary of best model parameters
        """
        try:
            # Create a temporary Prophet forecaster for seasonality analysis
            temp_forecaster = ProphetForecaster(
                window_length=None  # Use entire available data
            )

            # Fit the model
            temp_forecaster.fit(daily_data)

            # Get seasonality analysis results
            seasonality_analysis = temp_forecaster.get_seasonality_analysis()

            if seasonality_analysis and "recommendations" in seasonality_analysis:
                # Use best model parameters from seasonality analysis
                best_parameters = seasonality_analysis["recommendations"].get(
                    "best_model_parameters", {}
                )

                # Merge with default parameters
                default_parameters = {
                    "changepoint_range": 0.8,
                    "n_changepoints": 25,
                    "changepoint_prior_scale": 0.05,
                    "seasonality_prior_scale": 10.0,
                    "holidays_prior_scale": 10.0,
                    "seasonality_mode": "multiplicative",
                    "weekly_seasonality": True,
                    "daily_seasonality": False,
                    "include_indian_holidays": True,
                    "include_regional_holidays": False,
                    "include_quarterly_effects": True,
                    "include_monthly_effects": True,
                    "include_festival_seasons": True,
                    "include_monsoon_effect": False,
                    "min_data_points": 60,
                    "window_length": None,  # Use None to indicate use all available data
                }

                # Update with best parameters from seasonality analysis
                best_parameters = {**default_parameters, **best_parameters}

                self.logger.debug(
                    f"Using best parameters for {product_record['product_id']}: {best_parameters}"
                )
                return best_parameters
            else:
                # Fallback to default parameters if no seasonality analysis available
                self.logger.warning(
                    f"No seasonality analysis available for {product_record['product_id']}, using default parameters"
                )
                return {
                    "changepoint_range": 0.8,
                    "n_changepoints": 25,
                    "changepoint_prior_scale": 0.05,
                    "seasonality_prior_scale": 10.0,
                    "holidays_prior_scale": 10.0,
                    "seasonality_mode": "multiplicative",
                    "weekly_seasonality": True,
                    "daily_seasonality": False,
                    "include_indian_holidays": True,
                    "include_regional_holidays": False,
                    "include_quarterly_effects": True,
                    "include_monthly_effects": True,
                    "include_festival_seasons": True,
                    "include_monsoon_effect": False,
                    "min_data_points": 60,
                    "window_length": None,  # Use None to indicate use all available data
                }

        except Exception as e:
            self.logger.error(f"Error getting best model parameters: {e}")
            # Return default parameters as fallback
            return {
                "changepoint_range": 0.8,
                "n_changepoints": 25,
                "changepoint_prior_scale": 0.05,
                "seasonality_prior_scale": 10.0,
                "holidays_prior_scale": 10.0,
                "seasonality_mode": "multiplicative",
                "weekly_seasonality": True,
                "daily_seasonality": False,
                "include_indian_holidays": True,
                "include_regional_holidays": False,
                "include_quarterly_effects": True,
                "include_monthly_effects": True,
                "include_festival_seasons": True,
                "include_monsoon_effect": False,
                "min_data_points": 60,
                "window_length": None,  # Use None to indicate use all available data
            }

    def _create_forecaster_with_best_parameters(
        self, best_parameters: Dict[str, Any], product_record: pd.Series
    ) -> ProphetForecaster:
        """Create Prophet forecaster with best model parameters."""
        return create_prophet_forecaster(best_parameters)

    def _aggregate_daily_forecast_to_risk_period(
        self, daily_forecast: pd.Series, product_record: pd.Series
    ) -> List[float]:
        """
        Aggregate daily forecast to risk period level.

        Args:
            daily_forecast: Daily forecast series
            product_record: Product master record

        Returns:
            List of aggregated forecast values
        """
        risk_period = product_record["risk_period"]

        # Convert daily forecast to list
        daily_values = daily_forecast.tolist()

        # Aggregate into risk period buckets
        aggregated_values = []
        for i in range(0, len(daily_values), risk_period):
            bucket_values = daily_values[i : i + risk_period]
            aggregated_value = sum(bucket_values)
            aggregated_values.append(aggregated_value)

        return aggregated_values

    def _get_actual_aggregated_demand(
        self,
        analysis_date: date,
        product_id: str,
        location_id: str,
        product_record: pd.Series,
    ) -> Optional[List[float]]:
        """
        Get actual aggregated demand for comparison.

        Args:
            analysis_date: Date of analysis
            product_id: Product ID
            location_id: Location ID
            product_record: Product master record

        Returns:
            List of actual aggregated demand values or None
        """
        try:
            risk_period = int(product_record["risk_period"])

            # Get actual data for the forecast horizon
            start_date = analysis_date
            end_date = analysis_date + timedelta(days=risk_period)

            # Use outlier-handled data if available
            data_source = (
                self.outlier_data if self.outlier_data is not None else self.demand_data
            )

            actual_data = data_source[
                (data_source["product_id"] == product_id)
                & (data_source["location_id"] == location_id)
                & (pd.to_datetime(data_source["date"]) >= pd.Timestamp(start_date))
                & (pd.to_datetime(data_source["date"]) < pd.Timestamp(end_date))
            ].copy()

            if len(actual_data) == 0:
                return None

            # Sort by date
            actual_data = actual_data.sort_values("date")

            # Aggregate by risk period
            actual_data["date"] = pd.to_datetime(actual_data["date"])
            actual_data["bucket"] = (
                actual_data["date"] - pd.Timestamp(start_date)
            ).dt.days // risk_period

            aggregated_actual = actual_data.groupby("bucket")["demand"].sum().tolist()

            return aggregated_actual

        except Exception as e:
            self.logger.error(f"Error getting actual aggregated demand: {e}")
            return None

    def _get_cutoff_date(self, analysis_date: date) -> date:
        """Get cutoff date (closest date in data before analysis_date)."""
        # Convert analysis_date to timestamp for comparison
        analysis_timestamp = pd.Timestamp(analysis_date)

        # Get all dates in the data
        all_dates = pd.to_datetime(self.demand_data["date"]).unique()

        # Find the closest date before analysis_date
        valid_dates = all_dates[all_dates < analysis_timestamp]

        if len(valid_dates) == 0:
            raise ValueError(f"No data available before {analysis_date}")

        # Get the closest date
        cutoff_date = valid_dates.max()

        return cutoff_date.date()

    def _calculate_accuracy_metrics(self):
        """Calculate accuracy metrics for all forecasts."""
        self.logger.info("Step 4: Calculating accuracy metrics")

        for result in self.backtest_results:
            if result.get("actual_aggregated_demand") and result.get(
                "aggregated_forecast_values"
            ):
                actual = result["actual_aggregated_demand"]
                forecast = result["aggregated_forecast_values"]

                # Calculate metrics
                metrics = calculate_forecast_metrics(actual, forecast)

                # Add to result
                result["metrics"] = metrics
                self.accuracy_metrics.append(
                    {
                        "analysis_date": result["analysis_date"],
                        "product_id": result["product_id"],
                        "location_id": result["location_id"],
                        "mape": metrics.get("mape"),
                        "mae": metrics.get("mae"),
                        "rmse": metrics.get("rmse"),
                        "mape_percent": metrics.get("mape_percent"),
                    }
                )

                # Generate forecast comparisons for this result
                self._generate_forecast_comparisons(
                    result["analysis_date"],
                    result["product_id"],
                    result["location_id"],
                    actual,
                    forecast,
                    result["risk_period"],
                    result["demand_frequency"],
                )

    def _generate_forecast_comparisons(
        self,
        analysis_date: date,
        product_id: str,
        location_id: str,
        actual_demands: List[float],
        forecast_values: List[float],
        risk_period: int,
        demand_frequency: str,
    ):
        """Generate detailed forecast comparison data."""
        try:
            # Calculate risk period size in days
            risk_period_days = int(risk_period)  # Convert to regular int for timedelta

            # Calculate risk period dates
            start_date = analysis_date  # First forecast bucket starts on analysis date

            for step in range(len(actual_demands)):
                risk_period_start = start_date + timedelta(days=step * risk_period_days)
                risk_period_end = risk_period_start + timedelta(
                    days=risk_period_days - 1
                )

                actual_demand = actual_demands[step]
                forecast_demand = forecast_values[step]

                # Handle NaN values
                if pd.isna(actual_demand):
                    error = float("nan")
                    absolute_error = float("nan")
                    percentage_error = float("nan")
                else:
                    error = actual_demand - forecast_demand
                    absolute_error = abs(error)
                    percentage_error = (
                        (error / actual_demand * 100) if actual_demand != 0 else 0
                    )

                comparison = {
                    "analysis_date": analysis_date,
                    "risk_period_start": risk_period_start,
                    "risk_period_end": risk_period_end,
                    "product_id": product_id,
                    "location_id": location_id,
                    "step": step + 1,
                    "actual_demand": actual_demand,
                    "forecast_demand": forecast_demand,
                    "error": error,
                    "absolute_error": absolute_error,
                    "percentage_error": percentage_error,
                    "risk_period": risk_period,
                    "demand_frequency": demand_frequency,
                }

                self.forecast_comparisons.append(comparison)

        except Exception as e:
            self.logger.error(
                f"Error generating forecast comparisons for {product_id}-{location_id}: {e}"
            )

    def _save_results(self):
        """Save backtest results."""
        self.logger.info("Step 5: Saving results")

        # Create output directory
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save backtest results
        results_df = pd.DataFrame(self.backtest_results)
        results_file = output_dir / "enhanced_backtest_results.csv"
        results_df.to_csv(results_file, index=False)

        # Save accuracy metrics
        if self.accuracy_metrics:
            metrics_df = pd.DataFrame(self.accuracy_metrics)
            metrics_file = output_dir / "enhanced_accuracy_metrics.csv"
            metrics_df.to_csv(metrics_file, index=False)

        # Save forecast comparisons
        if self.forecast_comparisons:
            comparison_df = pd.DataFrame(self.forecast_comparisons)
            comparison_file = output_dir / "forecast_comparison.csv"
            comparison_df.to_csv(comparison_file, index=False)
            self.logger.info(f"Forecast comparisons saved to {comparison_file}")

        # Save forecast visualization data
        self._save_forecast_visualization_data()

        self.logger.info(f"Results saved to {output_dir}")

    def _save_forecast_visualization_data(self):
        """Save data for forecast visualization in the same format as regular backtest."""
        visualization_data = []

        for result in self.backtest_results:
            if result.get("aggregated_forecast_values"):
                # Get historical data for this product-location
                historical_data = self._get_historical_data_for_visualization(
                    result["product_id"], result["location_id"], result["analysis_date"]
                )

                # Get forecast horizon data
                forecast_horizon_data = (
                    self._get_forecast_horizon_data_for_visualization(
                        result["product_id"],
                        result["location_id"],
                        result["analysis_date"],
                        result["risk_period"],
                    )
                )

                # Create visualization entry in regular format
                viz_entry = {
                    "analysis_date": result["analysis_date"],
                    "product_id": result["product_id"],
                    "location_id": result["location_id"],
                    "risk_period": result["risk_period"],
                    "demand_frequency": result["demand_frequency"],
                    "window_length": result.get("window_length", 25),
                    "historical_bucket_start_dates": json.dumps(
                        historical_data["dates"]
                    ),
                    "historical_demands": json.dumps(historical_data["demands"]),
                    "forecast_horizon_start_dates": json.dumps(
                        forecast_horizon_data["dates"]
                    ),
                    "forecast_horizon_actual_demands": json.dumps(
                        forecast_horizon_data["actual_demands"]
                    ),
                    "forecast_horizon_forecast_demands": json.dumps(
                        forecast_horizon_data["forecast_demands"]
                    ),
                    "forecast_horizon_errors": json.dumps(
                        forecast_horizon_data["errors"]
                    ),
                }

                visualization_data.append(viz_entry)

        if visualization_data:
            viz_df = pd.DataFrame(visualization_data)
            viz_file = (
                Path(self.config.output_dir)
                / "enhanced_forecast_visualization_data.csv"
            )
            viz_df.to_csv(viz_file, index=False)
            self.logger.info(f"Forecast visualization data saved to {viz_file}")

    def _generate_summary(self, total_time: float) -> Dict[str, Any]:
        """Generate summary of the enhanced backtesting run."""
        summary = {
            "total_time": total_time,
            "total_forecasts": len(self.backtest_results),
            "successful_forecasts": len(
                [
                    r
                    for r in self.backtest_results
                    if r.get("aggregated_forecast_values")
                ]
            ),
            "analysis_dates": len(
                set(r["analysis_date"] for r in self.backtest_results)
            ),
            "products": len(set(r["product_id"] for r in self.backtest_results)),
            "locations": len(set(r["location_id"] for r in self.backtest_results)),
            "output_files": [
                str(Path(self.config.output_dir) / "enhanced_backtest_results.csv"),
                str(Path(self.config.output_dir) / "enhanced_accuracy_metrics.csv"),
                str(Path(self.config.output_dir) / "forecast_comparison.csv"),
                str(
                    Path(self.config.output_dir)
                    / "enhanced_forecast_visualization_data.csv"
                ),
            ],
        }

        # Add accuracy summary if metrics available
        if self.accuracy_metrics:
            mape_values = [
                m["mape"] for m in self.accuracy_metrics if m["mape"] is not None
            ]
            if mape_values:
                summary["avg_mape"] = np.mean(mape_values)
                summary["median_mape"] = np.median(mape_values)
                summary["min_mape"] = np.min(mape_values)
                summary["max_mape"] = np.max(mape_values)

        self.logger.info("Enhanced Backtesting Summary:")
        self.logger.info(f"  Total time: {total_time:.2f} seconds")
        self.logger.info(f"  Total forecasts: {summary['total_forecasts']}")
        self.logger.info(f"  Successful forecasts: {summary['successful_forecasts']}")
        if "avg_mape" in summary:
            self.logger.info(f"  Average MAPE: {summary['avg_mape']:.2f}%")

        return summary

    def _get_historical_data_for_visualization(
        self, product_id: str, location_id: str, analysis_date: date
    ) -> Dict[str, List]:
        """Get historical data for visualization in the same format as regular backtest."""
        try:
            # Get product record for risk period
            product_record = self.product_master_data[
                (self.product_master_data["product_id"] == product_id)
                & (self.product_master_data["location_id"] == location_id)
            ].iloc[0]

            # Get ALL historical data up to analysis date (use entire available data)
            historical_data = self.demand_data[
                (self.demand_data["product_id"] == product_id)
                & (self.demand_data["location_id"] == location_id)
                & (self.demand_data["date"] < analysis_date)
            ].sort_values("date")

            # Use all available historical data (no window limitation)

            # Aggregate to risk period buckets
            risk_period = product_record.get("risk_period_days", 11)
            historical_dates = []
            historical_demands = []

            # Group by risk period and calculate bucket start dates and demands
            for i in range(0, len(historical_data), risk_period):
                bucket_data = historical_data.iloc[i : i + risk_period]
                if not bucket_data.empty:
                    bucket_start_date = bucket_data["date"].min()
                    bucket_demand = bucket_data["demand"].sum()
                    historical_dates.append(bucket_start_date.strftime("%Y-%m-%d"))
                    historical_demands.append(float(bucket_demand))

            return {"dates": historical_dates, "demands": historical_demands}

        except Exception as e:
            self.logger.warning(
                f"Error getting historical data for {product_id}-{location_id}: {e}"
            )
            return {"dates": [], "demands": []}

    def _get_forecast_horizon_data_for_visualization(
        self, product_id: str, location_id: str, analysis_date: date, risk_period: int
    ) -> Dict[str, List]:
        """Get forecast horizon data for visualization in the same format as regular backtest."""
        try:
            # Get actual demand for the forecast horizon
            forecast_horizon_start = analysis_date
            forecast_horizon_end = analysis_date + timedelta(days=int(risk_period))

            actual_data = self.demand_data[
                (self.demand_data["product_id"] == product_id)
                & (self.demand_data["location_id"] == location_id)
                & (self.demand_data["date"] >= forecast_horizon_start)
                & (self.demand_data["date"] < forecast_horizon_end)
            ].sort_values("date")

            # Get forecast data from the result
            forecast_values = []
            for result in self.backtest_results:
                if (
                    result["product_id"] == product_id
                    and result["location_id"] == location_id
                    and result["analysis_date"] == analysis_date
                ):
                    forecast_values = result.get("aggregated_forecast_values", [])
                    break

            # Prepare data in regular format
            forecast_dates = [analysis_date.strftime("%Y-%m-%d")]
            actual_demands = []
            forecast_demands = []
            errors = []

            # Aggregate actual demand for the risk period
            if not actual_data.empty:
                actual_demand = actual_data["demand"].sum()
                actual_demands.append(float(actual_demand))
            else:
                actual_demands.append(None)

            # Get forecast demand
            if forecast_values:
                forecast_demand = sum(forecast_values)
                forecast_demands.append(float(forecast_demand))
            else:
                forecast_demands.append(None)

            # Calculate error
            if actual_demands[0] is not None and forecast_demands[0] is not None:
                error = forecast_demands[0] - actual_demands[0]
                errors.append(float(error))
            else:
                errors.append(None)

            return {
                "dates": forecast_dates,
                "actual_demands": actual_demands,
                "forecast_demands": forecast_demands,
                "errors": errors,
            }

        except Exception as e:
            self.logger.warning(
                f"Error getting forecast horizon data for {product_id}-{location_id}: {e}"
            )
            return {
                "dates": [],
                "actual_demands": [],
                "forecast_demands": [],
                "errors": [],
            }


def run_enhanced_backtest(config: BacktestConfig) -> Dict[str, Any]:
    """
    Run enhanced backtesting with best model parameters.

    Args:
        config: Backtest configuration

    Returns:
        Dictionary with backtesting results
    """
    backtester = EnhancedBacktester(config)
    result = backtester.run()
    return result
