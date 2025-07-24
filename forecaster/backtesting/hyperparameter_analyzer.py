"""
Hyperparameter analysis module for Prophet forecasting models.

This module provides comprehensive analysis of Prophet hyperparameters through
backtesting to determine optimal configurations for different scenarios.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import date, timedelta
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
import logging
import sys
import os

from .config import BacktestConfig
from ..data.loader import DemandDataLoader
from ..forecasting.prophet import ProphetForecaster, create_prophet_forecaster
from ..forecasting.base import calculate_forecast_metrics
from ..utils.logger import ForecasterLogger


class HyperparameterAnalyzer:
    """
    Comprehensive hyperparameter analysis for Prophet forecasting models.
    """

    def __init__(self, config: BacktestConfig):
        """Initialize the hyperparameter analyzer."""
        self.config = config
        self.logger = ForecasterLogger(
            "hyperparameter_analyzer", config.log_level, config.log_file
        )

        # Data storage
        self.demand_data: Optional[pd.DataFrame] = None
        self.product_master_data: Optional[pd.DataFrame] = None

        # Results storage
        self.analysis_results: List[Dict[str, Any]] = []
        self.parameter_combinations: List[Dict[str, Any]] = []
        self.best_parameters: Dict[str, Any] = {}

    def load_data(self):
        """Load demand and product master data."""
        self.logger.info("Loading data for hyperparameter analysis")

        # Load demand data using config data directory
        data_loader = DemandDataLoader(self.config.data_dir)

        # Load demand data - try customer demand first, then fallback to dummy data
        try:
            self.demand_data = data_loader.load_customer_demand()
        except FileNotFoundError:
            # Fallback to dummy data
            self.demand_data = data_loader.load_dummy_data("daily")

        # Load product master data (use daily frequency)
        self.product_master_data = data_loader.load_product_master_daily()

        self.logger.info(
            f"Loaded {len(self.demand_data)} demand records and {len(self.product_master_data)} product records"
        )

    def generate_parameter_combinations(self) -> List[Dict[str, Any]]:
        """
        Generate comprehensive parameter combinations for testing.

        Returns:
            List of parameter dictionaries to test
        """
        # Define parameter ranges for comprehensive testing
        parameter_ranges = {
            # Core Prophet parameters
            "changepoint_range": [0.6, 0.8, 0.9],
            "n_changepoints": [15, 25, 35],
            "changepoint_prior_scale": [0.01, 0.05, 0.1],
            "seasonality_prior_scale": [5.0, 10.0, 15.0],
            "holidays_prior_scale": [5.0, 10.0, 15.0],
            "seasonality_mode": ["additive", "multiplicative"],
            # Seasonality settings
            "weekly_seasonality": [True, False],
            "daily_seasonality": [False],  # Usually not needed for daily data
            # Indian market specific features
            "include_indian_holidays": [True, False],
            "include_quarterly_effects": [True, False],
            "include_monthly_effects": [True, False],
            "include_festival_seasons": [True, False],
            "include_monsoon_effect": [True, False],
            # Data window settings
            "window_length": [30, 60, 90, None],  # None means use all available data
            "min_data_points": [10, 15, 20],
            # Required parameter for validation
            "horizon": [1],  # Prophet doesn't use horizon but validation requires it
        }

        # Generate all combinations
        keys = parameter_ranges.keys()
        combinations = []

        for values in itertools.product(*parameter_ranges.values()):
            param_dict = dict(zip(keys, values))
            combinations.append(param_dict)

        self.logger.info(f"Generated {len(combinations)} parameter combinations")
        return combinations

    def run_single_backtest(
        self,
        product_id: str,
        location_id: str,
        parameters: Dict[str, Any],
        analysis_dates: List[date],
    ) -> Dict[str, Any]:
        """
        Run backtesting for a single product-location with specific parameters.

        Args:
            product_id: Product ID to test
            location_id: Location ID to test
            parameters: Prophet parameters to test
            analysis_dates: List of dates to analyze

        Returns:
            Dictionary with backtesting results
        """
        try:
            # Filter data for this product-location
            product_data = self.demand_data[
                (self.demand_data["product_id"] == product_id)
                & (self.demand_data["location_id"] == location_id)
            ].copy()

            if len(product_data) < parameters.get("min_data_points", 10):
                self.logger.warning(
                    f"Insufficient data for {product_id}-{location_id}: {len(product_data)} points"
                )
                return {
                    "product_id": product_id,
                    "location_id": location_id,
                    "parameters": parameters,
                    "error": "Insufficient data points",
                    "metrics": None,
                }

            # Sort by date
            product_data = product_data.sort_values("date").reset_index(drop=True)

            # Run backtesting for each analysis date
            all_forecasts = []
            all_actuals = []

            for analysis_date in analysis_dates:
                # Filter historical data up to analysis date
                historical_data = product_data[
                    product_data["date"] <= analysis_date
                ].copy()

                if len(historical_data) < parameters.get("min_data_points", 10):
                    continue

                # Create and fit forecaster
                forecaster = create_prophet_forecaster(parameters)

                try:
                    forecaster.fit(historical_data)

                    # Generate forecast for next period
                    forecast = forecaster.forecast(steps=1)

                    if len(forecast) > 0:
                        # Get actual value for next period
                        next_date = analysis_date + timedelta(days=1)
                        actual_data = product_data[product_data["date"] == next_date]

                        if len(actual_data) > 0:
                            actual_value = actual_data["demand"].iloc[0]
                            forecast_value = forecast.iloc[0]

                            all_forecasts.append(forecast_value)
                            all_actuals.append(actual_value)
                        else:
                            self.logger.debug(f"No actual data found for {next_date}")
                    else:
                        self.logger.debug(f"No forecast generated for {analysis_date}")

                except Exception as e:
                    self.logger.warning(
                        f"Failed to forecast for {product_id}-{location_id} on {analysis_date}: {str(e)}"
                    )

            if len(all_forecasts) == 0:
                self.logger.warning(
                    f"No successful forecasts for {product_id}-{location_id}"
                )
                return {
                    "product_id": product_id,
                    "location_id": location_id,
                    "parameters": parameters,
                    "error": "No successful forecasts generated",
                    "metrics": None,
                }

            self.logger.info(
                f"Generated {len(all_forecasts)} forecasts for {product_id}-{location_id}"
            )

            # Calculate metrics
            metrics = calculate_forecast_metrics(
                actual=pd.Series(all_actuals), forecast=pd.Series(all_forecasts)
            )

            return {
                "product_id": product_id,
                "location_id": location_id,
                "parameters": parameters,
                "error": None,
                "metrics": metrics,
                "num_forecasts": len(all_forecasts),
            }

        except Exception as e:
            return {
                "product_id": product_id,
                "location_id": location_id,
                "parameters": parameters,
                "error": str(e),
                "metrics": None,
            }

    def _suppress_logs(self):
        """Suppress verbose logs from Prophet and Stan."""
        # Suppress warnings
        warnings.filterwarnings("ignore")

        # Suppress Prophet and Stan logs
        logging.getLogger("prophet").setLevel(logging.ERROR)
        logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
        logging.getLogger("stanpy").setLevel(logging.ERROR)

        # Store original loggers
        self._original_prophet_logger = logging.getLogger("prophet")
        self._original_cmdstanpy_logger = logging.getLogger("cmdstanpy")
        self._original_stanpy_logger = logging.getLogger("stanpy")

    def _restore_logs(self):
        """Restore original logging behavior."""
        # Restore loggers if they were modified
        if hasattr(self, "_original_prophet_logger"):
            self._original_prophet_logger.setLevel(logging.INFO)
        if hasattr(self, "_original_cmdstanpy_logger"):
            self._original_cmdstanpy_logger.setLevel(logging.INFO)
        if hasattr(self, "_original_stanpy_logger"):
            self._original_stanpy_logger.setLevel(logging.INFO)

    def run_comprehensive_analysis(
        self, sample_size: int = 5, max_combinations: int = 50
    ) -> Dict[str, Any]:
        """
        Run comprehensive hyperparameter analysis.

        Args:
            sample_size: Number of product-location combinations to sample
            max_combinations: Maximum number of parameter combinations to test

        Returns:
            Dictionary with analysis results
        """
        self.logger.info("Starting comprehensive hyperparameter analysis")

        # Load data if not already loaded
        if self.demand_data is None:
            self.load_data()

        # Generate parameter combinations
        all_combinations = self.generate_parameter_combinations()

        # Limit combinations if specified
        if max_combinations and len(all_combinations) > max_combinations:
            # Use stratified sampling to ensure good coverage
            all_combinations = self._stratified_sample_combinations(
                all_combinations, max_combinations
            )

        # Sample product-location combinations
        product_locations = self._sample_product_locations(sample_size)

        # Generate analysis dates
        analysis_dates = self.config.get_analysis_dates()
        self.logger.info(
            f"Generated {len(analysis_dates)} analysis dates: {analysis_dates[:5]}..."
        )

        self.logger.info(
            f"Testing {len(all_combinations)} parameter combinations on {len(product_locations)} product-locations"
        )

        # Suppress verbose logs
        self._suppress_logs()

        try:
            # Run analysis
            results = []
            total_tests = len(all_combinations) * len(product_locations)

            print(f"Running {total_tests} tests...")
            print(
                f"Testing {len(all_combinations)} parameter combinations on {len(product_locations)} product-locations"
            )

            # Create progress bar for overall analysis
            print(f"\nStarting analysis with {total_tests} total tests...")
            with tqdm(
                total=total_tests,
                desc="Hyperparameter Analysis",
                unit="test",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
                position=0,
                leave=True,
            ) as pbar:
                successful_tests = 0
                failed_tests = 0

                for i, params in enumerate(all_combinations):
                    for j, (product_id, location_id) in enumerate(product_locations):
                        # Force refresh the progress bar
                        pbar.refresh()

                        result = self.run_single_backtest(
                            product_id, location_id, params, analysis_dates
                        )
                        results.append(result)

                        # Track success/failure
                        if (
                            result.get("error") is None
                            and result.get("metrics") is not None
                        ):
                            successful_tests += 1
                        else:
                            failed_tests += 1

                        # Update progress bar with overall stats
                        pbar.set_postfix(
                            {
                                "Success": successful_tests,
                                "Failed": failed_tests,
                                "Success Rate": (
                                    f"{successful_tests/(successful_tests+failed_tests)*100:.1f}%"
                                    if (successful_tests + failed_tests) > 0
                                    else "0%"
                                ),
                            }
                        )
                        pbar.update(1)
                        pbar.refresh()  # Force refresh after update

                        # Debug: Print first few results
                        if len(results) <= 5:
                            print(
                                f"Result {len(results)}: {result.get('error', 'No error')}"
                            )
        finally:
            # Restore logs
            self._restore_logs()

        # Analyze results
        analysis_summary = self._analyze_results(results)

        # Print final summary
        successful_count = len(
            [
                r
                for r in results
                if r.get("error") is None and r.get("metrics") is not None
            ]
        )
        failed_count = len(
            [
                r
                for r in results
                if r.get("error") is not None or r.get("metrics") is None
            ]
        )

        print(f"\n✅ Analysis completed!")
        print(f"📊 Final Results:")
        print(f"   - Total tests: {len(results)}")
        print(f"   - Successful tests: {successful_count}")
        print(f"   - Failed tests: {failed_count}")
        print(
            f"   - Success rate: {successful_count/len(results)*100:.1f}%"
            if len(results) > 0
            else "   - Success rate: 0%"
        )

        self.analysis_results = results
        self.parameter_combinations = all_combinations
        self.best_parameters = analysis_summary.get("best_parameters", {})

        return analysis_summary

    def _stratified_sample_combinations(
        self, combinations: List[Dict[str, Any]], max_size: int
    ) -> List[Dict[str, Any]]:
        """
        Stratified sampling of parameter combinations to ensure good coverage.

        Args:
            combinations: All parameter combinations
            max_size: Maximum number to sample

        Returns:
            Sampled combinations
        """
        # Group by key parameters to ensure diversity
        grouped = {}
        for combo in combinations:
            # Create a key based on key parameters
            key = (
                combo.get("seasonality_mode", "additive"),
                combo.get("include_indian_holidays", True),
                combo.get("include_quarterly_effects", True),
                combo.get("window_length", None),
            )

            if key not in grouped:
                grouped[key] = []
            grouped[key].append(combo)

        # Sample from each group
        sampled = []
        samples_per_group = max_size // len(grouped)

        for group_combos in grouped.values():
            if len(group_combos) <= samples_per_group:
                sampled.extend(group_combos)
            else:
                # Randomly sample from this group
                indices = np.random.choice(
                    len(group_combos), samples_per_group, replace=False
                )
                sampled.extend([group_combos[i] for i in indices])

        # Add any remaining slots with random combinations
        remaining = max_size - len(sampled)
        if remaining > 0:
            all_remaining = [c for c in combinations if c not in sampled]
            if all_remaining:
                additional = np.random.choice(
                    all_remaining, min(remaining, len(all_remaining)), replace=False
                )
                sampled.extend(additional)

        return sampled[:max_size]

    def _sample_product_locations(self, sample_size: int) -> List[Tuple[str, str]]:
        """
        Sample product-location combinations for analysis.

        Args:
            sample_size: Number of combinations to sample

        Returns:
            List of (product_id, location_id) tuples
        """
        # Get unique product-location combinations with sufficient data
        combinations = []

        for _, row in self.product_master_data.iterrows():
            product_id = row["product_id"]
            location_id = row["location_id"]

            # Check if we have sufficient data
            product_data = self.demand_data[
                (self.demand_data["product_id"] == product_id)
                & (self.demand_data["location_id"] == location_id)
            ]

            if len(product_data) >= 30:  # Minimum data requirement
                combinations.append((product_id, location_id))

        # Sample if we have more than needed
        if len(combinations) > sample_size:
            indices = np.random.choice(len(combinations), sample_size, replace=False)
            combinations = [combinations[i] for i in indices]

        return combinations

    def _analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze backtesting results to find optimal parameters.

        Args:
            results: List of backtesting results

        Returns:
            Analysis summary
        """
        # Filter successful results
        successful_results = [
            r for r in results if r["error"] is None and r["metrics"] is not None
        ]

        if len(successful_results) == 0:
            return {
                "error": "No successful backtesting results",
                "best_parameters": {},
                "parameter_analysis": {},
                "summary_stats": {},
            }

        # Calculate aggregate metrics for each parameter combination
        param_metrics = {}

        for result in successful_results:
            param_key = self._create_parameter_key(result["parameters"])

            if param_key not in param_metrics:
                param_metrics[param_key] = {
                    "parameters": result["parameters"],
                    "mape_scores": [],
                    "mae_scores": [],
                    "rmse_scores": [],
                    "num_tests": 0,
                }

            metrics = result["metrics"]
            param_metrics[param_key]["mape_scores"].append(
                metrics.get("mape", float("inf"))
            )
            param_metrics[param_key]["mae_scores"].append(
                metrics.get("mae", float("inf"))
            )
            param_metrics[param_key]["rmse_scores"].append(
                metrics.get("rmse", float("inf"))
            )
            param_metrics[param_key]["num_tests"] += 1

        # Calculate average metrics for each parameter combination
        for param_key, data in param_metrics.items():
            data["avg_mape"] = np.mean(data["mape_scores"])
            data["avg_mae"] = np.mean(data["mae_scores"])
            data["avg_rmse"] = np.mean(data["rmse_scores"])
            data["std_mape"] = np.std(data["mape_scores"])

        # Find best parameters (lowest MAPE)
        best_param_key = min(
            param_metrics.keys(), key=lambda k: param_metrics[k]["avg_mape"]
        )
        best_parameters = param_metrics[best_param_key]["parameters"]

        # Analyze individual parameter effects
        parameter_analysis = self._analyze_individual_parameters(param_metrics)

        # Summary statistics
        summary_stats = {
            "total_tests": len(results),
            "successful_tests": len(successful_results),
            "success_rate": len(successful_results) / len(results),
            "best_mape": param_metrics[best_param_key]["avg_mape"],
            "best_parameters": best_parameters,
        }

        return {
            "best_parameters": best_parameters,
            "parameter_analysis": parameter_analysis,
            "summary_stats": summary_stats,
            "all_param_metrics": param_metrics,
        }

    def _create_parameter_key(self, parameters: Dict[str, Any]) -> str:
        """Create a unique key for parameter combination."""
        # Sort parameters to ensure consistent keys
        sorted_params = sorted(parameters.items())
        return json.dumps(sorted_params, sort_keys=True)

    def _analyze_individual_parameters(
        self, param_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze the effect of individual parameters on forecast accuracy.

        Args:
            param_metrics: Dictionary of parameter metrics

        Returns:
            Analysis of individual parameter effects
        """
        analysis = {}

        # Key parameters to analyze
        key_params = [
            "include_indian_holidays",
            "include_quarterly_effects",
            "include_monthly_effects",
            "include_festival_seasons",
            "include_monsoon_effect",
            "seasonality_mode",
            "weekly_seasonality",
            "window_length",
            "changepoint_prior_scale",
            "seasonality_prior_scale",
        ]

        for param in key_params:
            param_values = {}

            for param_key, data in param_metrics.items():
                param_value = data["parameters"].get(param)
                if param_value not in param_values:
                    param_values[param_value] = []
                param_values[param_value].append(data["avg_mape"])

            # Calculate average MAPE for each parameter value
            param_analysis = {}
            for value, mape_scores in param_values.items():
                param_analysis[str(value)] = {
                    "avg_mape": np.mean(mape_scores),
                    "std_mape": np.std(mape_scores),
                    "count": len(mape_scores),
                }

            analysis[param] = param_analysis

        return analysis

    def generate_visualization_plots(
        self, output_dir: str = "output/hyperparameter_analysis"
    ) -> Dict[str, str]:
        """
        Generate comprehensive visualization plots for hyperparameter analysis.

        Args:
            output_dir: Directory to save plots

        Returns:
            Dictionary with plot file paths
        """
        if not self.analysis_results:
            raise ValueError("No analysis results available. Run analysis first.")

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Set up plotting style
        plt.style.use("default")
        sns.set_palette("husl")

        plot_files = {}

        # 1. Parameter Performance Comparison
        plot_files["parameter_comparison"] = self._create_parameter_comparison_plot(
            output_dir
        )

        # 2. Individual Parameter Effects
        plot_files["parameter_effects"] = self._create_parameter_effects_plot(
            output_dir
        )

        # 3. Best Parameters Summary
        plot_files["best_parameters"] = self._create_best_parameters_plot(output_dir)

        # 4. Performance Distribution
        plot_files["performance_distribution"] = (
            self._create_performance_distribution_plot(output_dir)
        )

        return plot_files

    def _create_parameter_comparison_plot(self, output_dir: str) -> str:
        """Create plot comparing different parameter combinations."""
        # Get successful results
        successful_results = [
            r
            for r in self.analysis_results
            if r["error"] is None and r["metrics"] is not None
        ]

        if not successful_results:
            return ""

        # Extract MAPE scores and parameter summaries
        mape_scores = []
        param_summaries = []

        for result in successful_results:
            mape = result["metrics"].get("mape", float("inf"))
            if mape != float("inf"):
                mape_scores.append(mape)

                # Create parameter summary
                params = result["parameters"]
                summary = (
                    f"Holidays:{params.get('include_indian_holidays', False)}, "
                    f"Quarterly:{params.get('include_quarterly_effects', False)}, "
                    f"Mode:{params.get('seasonality_mode', 'additive')}"
                )
                param_summaries.append(summary)

        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Top plot: MAPE distribution
        ax1.hist(mape_scores, bins=30, alpha=0.7, color="skyblue", edgecolor="black")
        ax1.axvline(
            np.mean(mape_scores),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(mape_scores):.2f}%",
        )
        ax1.axvline(
            np.median(mape_scores),
            color="orange",
            linestyle="--",
            label=f"Median: {np.median(mape_scores):.2f}%",
        )
        ax1.set_xlabel("MAPE (%)")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Distribution of MAPE Scores Across Parameter Combinations")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Bottom plot: Top 10 best parameter combinations
        if len(mape_scores) > 0:
            # Get top 10 combinations
            sorted_indices = np.argsort(mape_scores)[:10]
            top_mape = [mape_scores[i] for i in sorted_indices]
            top_summaries = [param_summaries[i] for i in sorted_indices]

            bars = ax2.barh(range(len(top_mape)), top_mape, color="lightgreen")
            ax2.set_yticks(range(len(top_mape)))
            ax2.set_yticklabels(top_summaries, fontsize=8)
            ax2.set_xlabel("MAPE (%)")
            ax2.set_title("Top 10 Parameter Combinations by MAPE")

            # Add value labels on bars
            for i, (bar, mape) in enumerate(zip(bars, top_mape)):
                ax2.text(
                    bar.get_width() + 0.1,
                    bar.get_y() + bar.get_height() / 2,
                    f"{mape:.2f}%",
                    va="center",
                    fontsize=8,
                )

        plt.tight_layout()

        # Save plot
        plot_path = f"{output_dir}/parameter_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        return plot_path

    def _create_parameter_effects_plot(self, output_dir: str) -> str:
        """Create plot showing effects of individual parameters."""
        if not hasattr(self, "best_parameters") or not self.best_parameters:
            return ""

        # Key parameters to analyze
        key_params = [
            "include_indian_holidays",
            "include_quarterly_effects",
            "include_monthly_effects",
            "include_festival_seasons",
            "include_monsoon_effect",
            "seasonality_mode",
            "weekly_seasonality",
        ]

        # Create subplots
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()

        for i, param in enumerate(key_params):
            if i >= len(axes):
                break

            ax = axes[i]

            # Get parameter values and their MAPE scores
            param_values = {}
            for result in self.analysis_results:
                if result["error"] is None and result["metrics"] is not None:
                    value = result["parameters"].get(param, False)
                    mape = result["metrics"].get("mape", float("inf"))
                    if mape != float("inf"):
                        if value not in param_values:
                            param_values[value] = []
                        param_values[value].append(mape)

            if param_values:
                # Create box plot
                labels = [str(k) for k in param_values.keys()]
                data = list(param_values.values())

                bp = ax.boxplot(data, labels=labels, patch_artist=True)

                # Color the boxes
                colors = ["lightblue", "lightgreen"]
                for patch, color in zip(bp["boxes"], colors[: len(bp["boxes"])]):
                    patch.set_facecolor(color)

                ax.set_title(f"Effect of {param}")
                ax.set_ylabel("MAPE (%)")
                ax.grid(True, alpha=0.3)

        # Remove empty subplots
        for i in range(len(key_params), len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()

        # Save plot
        plot_path = f"{output_dir}/parameter_effects.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        return plot_path

    def _create_best_parameters_plot(self, output_dir: str) -> str:
        """Create plot showing the best parameter configuration."""
        if not hasattr(self, "best_parameters") or not self.best_parameters:
            return ""

        # Create a summary table
        fig, ax = plt.subplots(figsize=(10, 6))

        # Prepare data for table
        param_names = list(self.best_parameters.keys())
        param_values = [str(self.best_parameters[k]) for k in param_names]

        # Create table
        table_data = [[name, value] for name, value in zip(param_names, param_values)]

        table = ax.table(
            cellText=table_data,
            colLabels=["Parameter", "Best Value"],
            cellLoc="left",
            loc="center",
        )

        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        # Color header
        for i in range(2):
            table[(0, i)].set_facecolor("#4CAF50")
            table[(0, i)].set_text_props(weight="bold", color="white")

        # Color alternating rows
        for i in range(1, len(table_data) + 1):
            for j in range(2):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor("#f0f0f0")

        ax.set_title(
            "Best Prophet Parameters from Hyperparameter Analysis",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        ax.axis("off")

        # Save plot
        plot_path = f"{output_dir}/best_parameters.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        return plot_path

    def _create_performance_distribution_plot(self, output_dir: str) -> str:
        """Create plot showing performance distribution across different metrics."""
        # Get successful results
        successful_results = [
            r
            for r in self.analysis_results
            if r["error"] is None and r["metrics"] is not None
        ]

        if not successful_results:
            return ""

        # Extract metrics
        mape_scores = [
            r["metrics"].get("mape", float("inf")) for r in successful_results
        ]
        mae_scores = [r["metrics"].get("mae", float("inf")) for r in successful_results]
        rmse_scores = [
            r["metrics"].get("rmse", float("inf")) for r in successful_results
        ]

        # Filter out infinite values
        mape_scores = [x for x in mape_scores if x != float("inf")]
        mae_scores = [x for x in mae_scores if x != float("inf")]
        rmse_scores = [x for x in rmse_scores if x != float("inf")]

        if not mape_scores:
            return ""

        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # MAPE distribution
        axes[0].hist(
            mape_scores, bins=20, alpha=0.7, color="skyblue", edgecolor="black"
        )
        axes[0].axvline(
            np.mean(mape_scores),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(mape_scores):.2f}%",
        )
        axes[0].set_xlabel("MAPE (%)")
        axes[0].set_ylabel("Frequency")
        axes[0].set_title("MAPE Distribution")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # MAE distribution
        if mae_scores:
            axes[1].hist(
                mae_scores, bins=20, alpha=0.7, color="lightgreen", edgecolor="black"
            )
            axes[1].axvline(
                np.mean(mae_scores),
                color="red",
                linestyle="--",
                label=f"Mean: {np.mean(mae_scores):.2f}",
            )
            axes[1].set_xlabel("MAE")
            axes[1].set_ylabel("Frequency")
            axes[1].set_title("MAE Distribution")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        # RMSE distribution
        if rmse_scores:
            axes[2].hist(
                rmse_scores, bins=20, alpha=0.7, color="lightcoral", edgecolor="black"
            )
            axes[2].axvline(
                np.mean(rmse_scores),
                color="red",
                linestyle="--",
                label=f"Mean: {np.mean(rmse_scores):.2f}",
            )
            axes[2].set_xlabel("RMSE")
            axes[2].set_ylabel("Frequency")
            axes[2].set_title("RMSE Distribution")
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_path = f"{output_dir}/performance_distribution.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        return plot_path


def run_hyperparameter_analysis(config: BacktestConfig = None) -> Dict[str, Any]:
    """
    Convenience function to run hyperparameter analysis.

    Args:
        config: BacktestConfig instance (optional)

    Returns:
        Analysis results
    """
    if config is None:
        config = BacktestConfig()

    analyzer = HyperparameterAnalyzer(config)
    results = analyzer.run_comprehensive_analysis()

    # Generate visualizations
    plot_files = analyzer.generate_visualization_plots()

    return {"analysis_results": results, "plot_files": plot_files, "analyzer": analyzer}
