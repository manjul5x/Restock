#!/usr/bin/env python3
"""
Flask web application for demand visualization.
Provides a nice UI for exploring demand data with filtering and aggregation.
"""

from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import seaborn as sns
import io
import base64
from datetime import date, timedelta
import sys
import math
import json
from pathlib import Path
from scipy.stats import gaussian_kde

# Add the forecaster package to the path
sys.path.append(str(Path(__file__).parent.parent))

from data.loader import DataLoader
from forecaster.utils import DemandVisualizer

app = Flask(__name__)

# Global variables to store data
loader = DataLoader()
visualizer = None


def initialize_data():
    """Initialize data loader and visualizer"""
    global visualizer
    if visualizer is None:
        try:
            # Load product master first
            product_master_data = loader.load_product_master()
            
            # Load customer demand data filtered by product master
            daily_data = loader.load_outflow(product_master=product_master_data)
            visualizer = DemandVisualizer(daily_data)
        except Exception as e:
            print(f"Warning: Could not initialize data loader: {e}")
            # Create empty visualizer with dummy data
            visualizer = None


def load_complete_workflow_data(data_type):
    """
    Load data from the complete workflow output directory using DataLoader.

    Args:
        data_type: Type of data to load ('backtesting', 'safety_stocks', 'simulation', 'forecast_visualization')

    Returns:
        DataFrame with the loaded data, or None if not found
    """
    try:
        if data_type == "backtesting":
            filename = loader.config['paths']['output_files']['forecast_comparison']
            path = loader.get_output_path("backtesting", filename)
            if path.exists():
                data = pd.read_csv(path)
                data["analysis_date"] = pd.to_datetime(data["analysis_date"])
                return data

        elif data_type == "forecast_visualization":
            filename = loader.config['paths']['output_files']['forecast_visualization']
            path = loader.get_output_path("backtesting", filename)
            if path.exists():
                data = pd.read_csv(path)
                data["analysis_date"] = pd.to_datetime(data["analysis_date"])
                return data

        elif data_type == "safety_stocks":
            filename = loader.config['paths']['output_files']['safety_stocks']
            path = loader.get_output_path("safety_stocks", filename)
            if path.exists():
                data = pd.read_csv(path)
                data["review_date"] = pd.to_datetime(data["review_date"])
                data["errors"] = data["errors"].apply(
                    lambda x: [float(e) for e in x.strip('[]').split(',')] if pd.notna(x) and x.strip('[]') else []
                )
                return data

        elif data_type == "simulation":
            filename = loader.config['paths']['output_files']['simulation_results']
            path = loader.get_output_path("simulation", filename)
            if path.exists():
                return pd.read_csv(path)

    except Exception as e:
        print(f"Error loading {data_type} data: {e}")

    return None


@app.route("/")
def index():
    """Main page with the visualization interface"""
    initialize_data()

    # Get available filter options
    if visualizer is not None:
        filters = visualizer.get_available_filters()
    else:
        # Provide empty filters if visualizer is not available
        filters = {"locations": [], "categories": [], "products": []}

    return render_template(
        "index.html",
        locations=filters["locations"],
        categories=filters["categories"],
        products=filters["products"],
    )


@app.route("/generate_plot", methods=["POST"])
def generate_plot():
    """Generate and return a plot based on user selections"""
    initialize_data()

    try:
        # Get form data
        plot_type = request.form.get("plot_type", "demand_trend")
        locations = request.form.getlist("locations")
        categories = request.form.getlist("categories")
        products = request.form.getlist("products")
        time_aggregation = request.form.get("time_aggregation", "daily")
        start_date_str = request.form.get("start_date", "")
        end_date_str = request.form.get("end_date", "")

        # Convert empty lists to None for the visualizer
        locations = locations if locations else None
        categories = categories if categories else None
        products = products if products else None

        # Parse dates
        start_date = None
        end_date = None
        if start_date_str:
            start_date = date.fromisoformat(start_date_str)
        if end_date_str:
            end_date = date.fromisoformat(end_date_str)

        if visualizer is None:
            return jsonify({"error": "Data not available"})

        # Generate plot
        fig = visualizer.generate_plot(
            plot_type=plot_type,
            locations=locations,
            categories=categories,
            products=products,
            time_aggregation=time_aggregation,
            start_date=start_date,
            end_date=end_date,
        )

        # Convert plot to base64 string
        img = io.BytesIO()
        fig.savefig(img, format="png", bbox_inches="tight", dpi=300)
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        plt.close(fig)

        return jsonify({"plot_url": plot_url})

    except Exception as e:
        import traceback

        print(f"Error in generate_plot: {str(e)}")
        return jsonify({"error": f"Failed to generate plot: {str(e)}"})


@app.route("/get_data_summary")
def get_data_summary():
    """Get summary statistics for the data"""
    initialize_data()

    if visualizer is None:
        return jsonify({"error": "Data not available"})

    try:
        summary = visualizer.get_data_summary()
        return jsonify(summary)
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/get_date_range")
def get_date_range():
    """Get the available date range for the data"""
    initialize_data()

    if visualizer is None:
        return jsonify({"error": "Data not available"})

    try:
        # Get the date range from the visualizer's data
        data = visualizer.data
        if data is not None and 'date' in data.columns:
            min_date = data['date'].min().strftime('%Y-%m-%d')
            max_date = data['date'].max().strftime('%Y-%m-%d')
            return jsonify({
                "start_date": min_date,
                "end_date": max_date
            })
        else:
            return jsonify({"error": "No date column found in data"})
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/safety_stocks")
def safety_stocks():
    """Safety stock visualization page"""
    try:
        # Load safety stock data from complete workflow
        safety_stock_data = load_complete_workflow_data("safety_stocks")

        if safety_stock_data is None:
            return render_template(
                "safety_stocks.html",
                error="Safety stock results not found. Please run the complete workflow first.",
            )

        # Get available filter options
        products = sorted(safety_stock_data["product_id"].dropna().unique().tolist())
        locations = sorted(safety_stock_data["location_id"].dropna().unique().tolist())
        review_dates = sorted(
            safety_stock_data["review_date"].dt.strftime("%Y-%m-%d").dropna().unique().tolist()
        )
        forecast_methods = sorted(
            safety_stock_data["forecast_method"].dropna().unique().tolist()
        )

        # Set default filter options
        default_product = products[0] if products else ""
        default_location = locations[0] if locations else ""
        default_review_date = review_dates[0] if review_dates else ""
        default_forecast_method = (
            "All Methods"
            if "All Methods" in forecast_methods
            else (forecast_methods[0] if forecast_methods else "")
        )

        return render_template(
            "safety_stocks.html",
            products=products,
            locations=locations,
            review_dates=review_dates,
            forecast_methods=forecast_methods,
            default_product=default_product,
            default_location=default_location,
            default_review_date=default_review_date,
            default_forecast_method=default_forecast_method,
            error=None,
        )

    except Exception as e:
        return render_template(
            "safety_stocks.html", error=f"Error loading safety stock data: {str(e)}"
        )


@app.route("/forecast_visualization")
def forecast_visualization():
    """Forecast visualization page"""
    try:
        # Load forecast visualization data from complete workflow
        forecast_data = load_complete_workflow_data("forecast_visualization")

        if forecast_data is None:
            return render_template(
                "forecast_visualization.html",
                error="Forecast visualization data not found. Please run the complete workflow first.",
            )

        # Get available filter options
        products = sorted(forecast_data["product_id"].dropna().unique().tolist())
        locations = sorted(forecast_data["location_id"].dropna().unique().tolist())
        analysis_dates = sorted(
            forecast_data["analysis_date"].dt.strftime("%Y-%m-%d").dropna().unique().tolist()
        )
        forecast_methods = sorted(forecast_data["forecast_method"].dropna().unique().tolist())

        # Set default filter options
        default_product = products[0] if products else ""
        default_location = locations[0] if locations else ""
        default_analysis_date = analysis_dates[0] if analysis_dates else ""
        default_forecast_method = (
            "All Methods"
            if "All Methods" in forecast_methods
            else (forecast_methods[0] if forecast_methods else "")
        )

        return render_template(
            "forecast_visualization.html",
            products=products,
            locations=locations,
            analysis_dates=analysis_dates,
            forecast_methods=forecast_methods,
            default_product=default_product,
            default_location=default_location,
            default_analysis_date=default_analysis_date,
            default_forecast_method=default_forecast_method,
            error=None,
        )

    except Exception as e:
        return render_template(
            "forecast_visualization.html",
            error=f"Error loading forecast data: {str(e)}",
        )


@app.route("/simulation_visualization")
def simulation_visualization():
    """Inventory simulation visualization page"""
    try:
        # Load simulation data from complete workflow
        simulation_data = load_complete_workflow_data("simulation")

        if simulation_data is None:
            return render_template(
                "simulation_visualization.html",
                error="Simulation data not found. Please run the complete workflow first.",
            )

        # Get unique products and locations for filter dropdowns
        products = ['all'] + sorted(simulation_data["product_id"].dropna().unique().tolist())
        locations = ['all'] + sorted(simulation_data["location_id"].dropna().unique().tolist())
        forecast_methods = sorted(simulation_data["forecast_method"].dropna().unique().tolist())
        if 'aggregated' not in forecast_methods:
            forecast_methods.append('aggregated')

        # Set default filter options
        default_product = products[0] if products else ""
        default_location = locations[0] if locations else ""
        default_forecast_method = (
            "All Methods"
            if "All Methods" in forecast_methods
            else (forecast_methods[0] if forecast_methods else "")
        )

        return render_template(
            "simulation_visualization.html",
            products=products,
            locations=locations,
            forecast_methods=forecast_methods,
            default_product=default_product,
            default_location=default_location,
            default_forecast_method=default_forecast_method,
            error=None,
        )
    except Exception as e:
        return render_template(
            "simulation_visualization.html",
            products=[],
            locations=[],
            forecast_methods=[],
            error=f"Error loading data: {str(e)}",
        )


@app.route("/inventory_comparison")
def inventory_comparison():
    """Inventory comparison page"""
    try:
        # Load simulation summary data to get available options
        filename = loader.config['paths']['output_files']['simulation_results']
        summary_path = loader.get_output_path("simulation", filename)
        if not summary_path.exists():
            return render_template(
                "inventory_comparison.html",
                error="Simulation data not found. Please run the complete workflow first.",
            )

        summary_data = pd.read_csv(summary_path)

        # Get available filter options
        products = sorted(summary_data["product_id"].dropna().unique().tolist())
        locations = sorted(summary_data["location_id"].dropna().unique().tolist())
        forecast_methods = sorted(summary_data["forecast_method"].dropna().unique().tolist())

        return render_template(
            "inventory_comparison.html",
            products=products,
            locations=locations,
            forecast_methods=forecast_methods,
            error=None,
        )

    except Exception as e:
        return render_template(
            "inventory_comparison.html",
            error=f"Error loading comparison data: {str(e)}",
        )


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.strftime("%Y-%m-%d")
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
        return obj


def calculate_actual_metrics(actual_data, product_master=None):
    """Calculate actual inventory metrics from customer demand data"""
    try:
        # Load customer demand data
        customer_demand = loader.load_outflow()

        if customer_demand.empty:
            return {}

        # Calculate basic metrics
        total_demand = customer_demand["demand"].sum()
        avg_daily_demand = customer_demand["demand"].mean()
        total_periods = len(customer_demand)

        # Calculate inventory metrics if stock_level is available
        inventory_metrics = {}
        if "stock_level" in customer_demand.columns:
            avg_stock_level = customer_demand["stock_level"].mean()
            min_stock_level = customer_demand["stock_level"].min()
            max_stock_level = customer_demand["stock_level"].max()

            inventory_metrics = {
                "avg_stock_level": avg_stock_level,
                "min_stock_level": min_stock_level,
                "max_stock_level": max_stock_level,
            }

        return {
            "total_demand": total_demand,
            "avg_daily_demand": avg_daily_demand,
            "total_periods": total_periods,
            **inventory_metrics,
        }

    except Exception as e:
        print(f"Error calculating actual metrics: {e}")
        return {}


from forecaster.utils.inventory_metrics_calculator import InventoryMetricsCalculator
from data.exceptions import DataAccessError

@app.route("/get_comparison_data", methods=["POST"])
def get_comparison_data():
    """Generate comprehensive inventory comparison data comparing actual vs simulated inventory levels"""
    try:
        # Get form data
        products = request.form.getlist("products")
        locations = request.form.getlist("locations")
        forecast_methods = request.form.getlist("forecast_methods")
        
        # Create filters dict
        filters = {}
        if products:
            filters['products'] = products
        if locations:
            filters['locations'] = locations
        if forecast_methods:
            filters['forecast_methods'] = forecast_methods
        
        # Load data using existing standard loader
        try:
            combined_detailed = loader.load_simulation_detailed_data(filters)
        except DataAccessError as e:
            return jsonify({"error": str(e)})
        
        # Calculate metrics using standardized calculator
        calculator = InventoryMetricsCalculator(data_loader=loader)
        comparison_results = []
        
        # Group by product-location-method
        for (product_id, location_id, forecast_method), group in combined_detailed.groupby(
            ["product_id", "location_id", "forecast_method"]
        ):
            if len(group) == 0:
                continue
                
            # Get product cost from existing product master data
            product_master = loader.load_product_master()
            product_record = product_master[
                (product_master["product_id"] == product_id) &
                (product_master["location_id"] == location_id)
            ]
            inventory_cost = float(product_record.iloc[0].get("inventory_cost", 0.0)) if len(product_record) > 0 else 0.0
            
            # Calculate all metrics at once
            metrics = calculator.calculate_all_metrics(group, inventory_cost)
            
            # Create comparison result
            comparison_result = {
                "product_id": str(product_id),
                "location_id": str(location_id),
                "forecast_method": str(forecast_method),
                **metrics
            }
            
            comparison_results.append(comparison_result)
        
        # Calculate overall metrics
        if comparison_results:
            total_actual_inventory_cost = sum(r["actual_total_inventory_cost"] for r in comparison_results)
            total_simulated_inventory_cost = sum(r["simulated_total_inventory_cost"] for r in comparison_results)
            inventory_difference = total_actual_inventory_cost - total_simulated_inventory_cost
            inventory_difference_percentage = (inventory_difference / total_actual_inventory_cost * 100) if total_actual_inventory_cost > 0 else 0
            
            overall_metrics = {
                "avg_actual_service_level": round(
                    sum(r["actual_service_level"] for r in comparison_results)
                    / len(comparison_results),
                    2,
                ),
                "avg_simulated_service_level": round(
                    sum(r["simulated_service_level"] for r in comparison_results)
                    / len(comparison_results),
                    2,
                ),
                "avg_actual_stockout_rate": round(
                    sum(r["actual_stockout_rate"] for r in comparison_results)
                    / len(comparison_results),
                    2,
                ),
                "avg_simulated_stockout_rate": round(
                    sum(r["simulated_stockout_rate"] for r in comparison_results)
                    / len(comparison_results),
                    2,
                ),
                "avg_actual_turnover_ratio": round(
                    sum(r["actual_turnover_ratio"] for r in comparison_results)
                    / len(comparison_results),
                    2,
                ),
                "avg_simulated_turnover_ratio": round(
                    sum(r["simulated_turnover_ratio"] for r in comparison_results)
                    / len(comparison_results),
                    2,
                ),
                "avg_inventory_difference_percentage": round(
                    inventory_difference_percentage,
                    2,
                ),
                "avg_service_level_difference": round(
                    sum(r["service_level_difference"] for r in comparison_results)
                    / len(comparison_results),
                    2,
                ),
                "avg_stockout_rate_difference": round(
                    sum(r["stockout_rate_difference"] for r in comparison_results)
                    / len(comparison_results),
                    2,
                ),
                "avg_turnover_ratio_difference": round(
                    sum(r["turnover_ratio_difference"] for r in comparison_results)
                    / len(comparison_results),
                    2,
                ),
                
                # Surplus stock metrics
                "avg_actual_surplus_stock_percentage": round(
                    sum(r["actual_surplus_stock_percentage"] for r in comparison_results)
                    / len(comparison_results),
                    2,
                ),
                "avg_actual_availability_percentage": round(
                    sum(r["actual_availability_percentage"] for r in comparison_results)
                    / len(comparison_results),
                    2,
                ),
                "avg_simulated_availability_percentage": round(
                    sum(r["simulated_availability_percentage"] for r in comparison_results)
                    / len(comparison_results),
                    2,
                ),
                "avg_availability_percentage_difference": round(
                    sum(r["availability_percentage_difference"] for r in comparison_results)
                    / len(comparison_results),
                    2,
                ),
                "total_actual_stockout_days": sum(
                    r["actual_stockout_days"] for r in comparison_results
                ),
                "total_simulated_stockout_days": sum(
                    r["simulated_stockout_days"] for r in comparison_results
                ),
                "total_actual_missed_demand": round(
                    sum(r["actual_missed_demand"] for r in comparison_results), 0
                ),
                "total_simulated_missed_demand": round(
                    sum(r["simulated_missed_demand"] for r in comparison_results), 0
                ),
                "total_actual_inventory_units": round(
                    sum(r["actual_total_inventory_units"] for r in comparison_results),
                    0,
                ),
                "total_simulated_inventory_units": round(
                    sum(r["simulated_total_inventory_units"] for r in comparison_results),
                    0,
                ),
                "total_actual_inventory_cost": round(
                    total_actual_inventory_cost, 2
                ),
                "total_simulated_inventory_cost": round(
                    total_simulated_inventory_cost,
                    2,
                ),
                "total_products": len(comparison_results),
                
                # Turnover ratio metrics
                # Calculate weighted average turnover ratios based on total inventory cost
                "avg_actual_turnover_ratio": round(
                    sum(r["actual_turnover_ratio"] * r["actual_total_inventory_cost"] for r in comparison_results)
                    / sum(r["actual_total_inventory_cost"] for r in comparison_results)
                    if sum(r["actual_total_inventory_cost"] for r in comparison_results) > 0 else 0,
                    2,
                ),
                "avg_simulated_turnover_ratio": round(
                    sum(r["simulated_turnover_ratio"] * r["simulated_total_inventory_cost"] for r in comparison_results)
                    / sum(r["simulated_total_inventory_cost"] for r in comparison_results)
                    if sum(r["simulated_total_inventory_cost"] for r in comparison_results) > 0 else 0,
                    2,
                ),
                "avg_turnover_ratio_difference": round(
                    (sum(r["simulated_turnover_ratio"] * r["simulated_total_inventory_cost"] for r in comparison_results)
                    / sum(r["simulated_total_inventory_cost"] for r in comparison_results)
                    if sum(r["simulated_total_inventory_cost"] for r in comparison_results) > 0 else 0) -
                    (sum(r["actual_turnover_ratio"] * r["actual_total_inventory_cost"] for r in comparison_results)
                    / sum(r["actual_total_inventory_cost"] for r in comparison_results)
                    if sum(r["actual_total_inventory_cost"] for r in comparison_results) > 0 else 0),
                    2,
                ),
            }
        else:
            overall_metrics = {}
        
        return jsonify({
            "comparison_results": comparison_results,
            "overall_metrics": overall_metrics,
        })
        
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/get_safety_stock_plot", methods=["POST"])
def get_safety_stock_plot():
    """Generate safety stock distribution plot"""
    try:
        # Get form data
        product = request.form.get("product")
        location = request.form.get("location")
        review_date = request.form.get("review_date")
        forecast_method = request.form.get("forecast_method")

        # Load safety stock data
        safety_stock_data = load_complete_workflow_data("safety_stocks")

        if safety_stock_data is None:
            return jsonify({"error": "Safety stock data not available"})

        # Apply filters
        filtered_data = safety_stock_data.copy()

        if product:
            filtered_data = filtered_data[filtered_data["product_id"] == product]
        if location:
            filtered_data = filtered_data[filtered_data["location_id"] == location]
        if review_date:
            filtered_data = filtered_data[
                filtered_data["review_date"].dt.strftime("%Y-%m-%d") == review_date
            ]
        if forecast_method and forecast_method != "All":
            filtered_data = filtered_data[
                filtered_data["forecast_method"] == forecast_method
            ]

        if filtered_data.empty:
            return jsonify({"error": "No data matches the selected filters"})

        # Get the specific record for the selected filters
        if len(filtered_data) > 1:
            # If multiple records, use the first one
            record = filtered_data.iloc[0]
        else:
            record = filtered_data.iloc[0]

        # Extract errors and safety stock value
        errors = record["errors"]
        safety_stock_value = record["safety_stock"]
        distribution_type = record.get("distribution", "kde")
        service_level = record.get("service_level", 0.95)

        if not errors:
            return jsonify(
                {"error": "No error data available for the selected filters"}
            )

        # Create the plot
        fig, ax1 = plt.subplots(figsize=(12, 8))

        # Create histogram of forecast errors
        n, bins, patches = ax1.hist(
            errors,
            bins=30,
            alpha=0.7,
            color="lightblue",
            edgecolor="black",
            label="Error Count",
        )
        ax1.set_xlabel("Forecast Error", fontsize=12)
        ax1.set_ylabel("Error Count", color="blue", fontsize=12)
        ax1.tick_params(axis="y", labelcolor="blue")

        # Create KDE curve
        kde = gaussian_kde(errors)
        x_range = np.linspace(min(errors), max(errors), 200)
        kde_values = kde(x_range)

        # Scale KDE to match histogram scale
        kde_scaled = kde_values * (max(n) / max(kde_values)) * 0.3

        # Create second y-axis for KDE
        ax2 = ax1.twinx()
        ax2.plot(x_range, kde_scaled, "b-", linewidth=2, label="KDE Density")
        ax2.set_ylabel("Density", color="red", fontsize=12)
        ax2.tick_params(axis="y", labelcolor="red")

        # Add safety stock line
        ax1.axvline(
            safety_stock_value,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Safety Stock ({safety_stock_value:.2f})",
        )

        # Set title
        title = f"Safety Stock Distribution\n{product} at {location}"
        if review_date:
            title += f" - {review_date}"
        if forecast_method and forecast_method != "All":
            title += f" ({forecast_method})"

        ax1.set_title(title, fontsize=14, fontweight="bold")

        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        # Add grid
        ax1.grid(True, alpha=0.3)

        plt.tight_layout()

        # Convert plot to base64 string
        img = io.BytesIO()
        fig.savefig(img, format="png", bbox_inches="tight", dpi=300)
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        plt.close(fig)

        # Calculate statistics
        error_count = len(errors)
        mean_error = np.mean(errors)
        std_error = np.std(errors)

        return jsonify(
            {
                "plot_url": plot_url,
                "safety_stock_value": float(safety_stock_value),
                "error_count": error_count,
                "distribution_type": distribution_type,
                "service_level": float(service_level),
                "mean_error": float(mean_error),
                "std_error": float(std_error),
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/get_simulation_plot", methods=["POST"])
def get_simulation_plot():
    """Generate simulation inventory chart"""
    try:
        # Get form data
        products = request.form.getlist("products")
        locations = request.form.getlist("locations")
        forecast_methods = request.form.getlist("forecast_methods")

        # Load simulation summary data for metrics
        filename = loader.config['paths']['output_files']['simulation_results']
        summary_file = loader.get_output_path("simulation", filename)
        if not summary_file.exists():
            return jsonify({"error": "Simulation summary data not available"})

        summary_data = pd.read_csv(summary_file)

        # Apply filters to summary data
        if products:
            summary_data = summary_data[summary_data["product_id"].isin(products)]
        if locations:
            summary_data = summary_data[summary_data["location_id"].isin(locations)]
        if forecast_methods:
            summary_data = summary_data[
                summary_data["forecast_method"].isin(forecast_methods)
            ]

        # Load detailed simulation data for plotting
        detailed_dir = loader.get_output_path("simulation/detailed_results", "")
        if not detailed_dir.exists():
            return jsonify({"error": "Detailed simulation data not available"})

        # Find matching simulation files
        all_data = []
        print(f"Looking for simulation files in: {detailed_dir}")
        print(f"Selected filters - Products: {products}, Locations: {locations}, Methods: {forecast_methods}")
        
        for file_path in detailed_dir.glob("*_simulation.csv"):
            print(f"\nProcessing file: {file_path}")
            # Extract product, location, and method from filename
            filename = file_path.stem  # e.g., "RSWQ_WB_moving_average_simulation" or "all_all_aggregated_simulation"
            parts = filename.split("_")
            
            # Special handling for aggregated results
            if filename.startswith("all_all_aggregated"):
                product_id = "all"
                location_id = "all"
                forecast_method = "aggregated"
                print("Found aggregated results file")
            elif len(parts) >= 4:
                product_id = parts[0]
                location_id = parts[1]

                # Handle forecast method - it could be multiple parts (e.g., "moving_average")
                # Find where "simulation" starts and take everything before it
                simulation_index = -1
                for i, part in enumerate(parts):
                    if part == "simulation":
                        simulation_index = i
                        break

                if simulation_index > 2:
                    forecast_method = "_".join(parts[2:simulation_index])
                else:
                    forecast_method = parts[2] if len(parts) > 2 else "unknown"
            else:
                print(f"Skipping file with insufficient parts: {filename}")
                continue

            print(f"Extracted - Product: {product_id}, Location: {location_id}, Method: {forecast_method}")

            # Apply filters, handling 'all' selections
            if products:
                if 'all' not in products and product_id not in products:
                    print(f"Skipping - product {product_id} not in selected products {products}")
                    continue
                if 'all' in products and product_id != 'all':
                    print(f"'all' selected but found individual product {product_id} - skipping")
                    continue
            if locations:
                if 'all' not in locations and location_id not in locations:
                    print(f"Skipping - location {location_id} not in selected locations {locations}")
                    continue
                if 'all' in locations and location_id != 'all':
                    print(f"'all' selected but found individual location {location_id} - skipping")
                    continue
            if forecast_methods and forecast_method not in forecast_methods:
                print(f"Skipping - method {forecast_method} not in selected methods {forecast_methods}")
                continue

            # Load the data
            try:
                print(f"Loading data from: {file_path}")
                data = pd.read_csv(file_path)
                data["date"] = pd.to_datetime(data["date"])
                data["product_id"] = product_id
                data["location_id"] = location_id
                data["forecast_method"] = forecast_method
                all_data.append(data)
                print("Successfully loaded data")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue

        if not all_data:
            return jsonify({"error": "No data matches the selected filters"})

        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)

        # Create the simulation chart
        fig, ax = plt.subplots(figsize=(20, 8))

        # Get display mode from request
        display_mode = request.form.get("display_mode", "units")
        
        # Plot for each product-location-method combination
        for (product_id, location_id, forecast_method), group in combined_data.groupby(
            ["product_id", "location_id", "forecast_method"]
        ):
            # Sort by date
            group = group.sort_values("date")
            
            # Determine which columns to use based on display mode
            column_mapping = {
                "inventory_on_hand": "inventory_on_hand_cost" if display_mode == "cost" else "inventory_on_hand",
                "actual_demand": "actual_demand_cost" if display_mode == "cost" else "actual_demand",
                "safety_stock": "safety_stock_cost" if display_mode == "cost" else "safety_stock",
                "FRSP": "FRSP_cost" if display_mode == "cost" else "FRSP",
                "net_stock": "net_stock_cost" if display_mode == "cost" else "net_stock",
                "rolling_max_inventory": "rolling_max_inventory_cost" if display_mode == "cost" else "rolling_max_inventory",
                "incoming_inventory": "incoming_inventory_cost" if display_mode == "cost" else "incoming_inventory",
                "actual_inventory": "actual_inventory_cost" if display_mode == "cost" else "actual_inventory",
                "order_placed": "order_placed_cost" if display_mode == "cost" else "order_placed"
            }

            # Plot simulated stock on hand (blue line)
            ax.plot(
                group["date"],
                group[column_mapping["inventory_on_hand"]],
                label="Stock on Hand",
                linewidth=3,
                alpha=0.9,
                color="#1f77b4",
            )

            # Plot actual demand (orange line)
            ax.plot(
                group["date"],
                group[column_mapping["actual_demand"]],
                label="Actual Demand",
                linewidth=2,
                alpha=0.8,
                linestyle="--",
                color="#ff7f0e",
            )

            # Plot safety stock (red dashed line)
            ax.plot(
                group["date"],
                group[column_mapping["safety_stock"]],
                label="Safety Stock",
                linewidth=2,
                alpha=0.7,
                linestyle=":",
                color="#d62728",
            )

            # # Plot FRSP (purple dashed line)
            # ax.plot(
            #     group["date"],
            #     group[column_mapping["FRSP"]],
            #     label="FRSP (Forecast over Risk Period)",
            #     linewidth=2,
            #     alpha=0.7,
            #     linestyle="--",
            #     color="#9467bd",
            # )

            # # Plot net stock (green line)
            # ax.plot(
            #     group["date"],
            #     group[column_mapping["net_stock"]],
            #     label="Net Stock Position",
            #     linewidth=2,
            #     alpha=0.7,
            #     color="#2ca02c",
            # )

            # Plot rolling max inventory (brown dotted line)
            ax.plot(
                group["date"],
                group[column_mapping["rolling_max_inventory"]],
                label="Rolling Max Inventory",
                linewidth=1.5,
                alpha=0.3,
                color="#8B4513",  # Saddle brown
                linestyle=":",
            )

            # # Plot incoming inventory (gold stars)
            # incoming_dates = group[group[column_mapping["incoming_inventory"]] > 0]["date"]
            # incoming_values = group[group[column_mapping["incoming_inventory"]] > 0][column_mapping["incoming_inventory"]]
            # if not incoming_dates.empty:
            #     ax.scatter(
            #         incoming_dates,
            #         incoming_values,
            #         marker="*",
            #         s=200,  # Size of the stars
            #         alpha=0.8,
            #         label="Incoming Inventory",
            #         color="#daa520",  # Golden color
            #         zorder=5  # Ensure stars are drawn on top
            #     )

            # Plot actual inventory as shaded area (light blue)
            ax.fill_between(
                group["date"],
                group[column_mapping["actual_inventory"]],
                alpha=0.2,
                label="Actual Inventory",
                color="#aec7e8",
            )

            # Plot orders placed as vertical bars (cyan)
            order_dates = group[group[column_mapping["order_placed"]] > 0]["date"]
            order_values = group[group[column_mapping["order_placed"]] > 0][column_mapping["order_placed"]]
            if not order_dates.empty:
                ax.bar(
                    order_dates,
                    order_values,
                    alpha=0.7,
                    label="Orders Placed",
                    width=1,
                    color="#17becf",
                )

        # Customize the plot
        ax.set_title("Inventory Simulation", fontsize=18, fontweight="bold", pad=20)
        ax.set_xlabel("Date", fontsize=14, fontweight="bold")
        y_label = "Cost ($)" if display_mode == "cost" else "Quantity (Units)"
        ax.set_ylabel(y_label, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.4, linestyle="-", linewidth=0.5)
        ax.legend(loc="upper right", fontsize=10)

        # Add vertical line for first order arrival (simulation start + lead time)
        if len(combined_data) > 0:
            first_date = pd.to_datetime(combined_data['date'].min())
            # Get lead time from the first group (assuming same lead time for all)
            first_group = list(combined_data.groupby(['product_id', 'location_id', 'forecast_method']))[0][1]
            leadtime = first_group['leadtime'].iloc[0] if 'leadtime' in first_group.columns else 0
            
            if leadtime > 0:
                arrival_date = first_date + pd.Timedelta(days=leadtime)
                ax.axvline(x=arrival_date, color='grey', linestyle='--', alpha=0.5, linewidth=2, 
                          label=f'First Order Arrival (Lead Time: {leadtime} days)')

        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        # Set background color
        ax.set_facecolor("#f8f9fa")
        fig.patch.set_facecolor("white")

        plt.tight_layout()

        # Convert plot to base64 string
        img = io.BytesIO()
        fig.savefig(img, format="png", bbox_inches="tight", dpi=300)
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        plt.close(fig)

        # Calculate summary statistics from the summary data
        if len(summary_data) > 0:
            summary_stats = {
                "total_products": len(summary_data["product_id"].unique()),
                "avg_service_level": float(
                    summary_data["service_level"].mean()
                ),  # Already in decimal format
                "avg_stockout_rate": float(
                    summary_data["stockout_rate"].mean()
                ),  # Already in decimal format
                "avg_inventory_turns": float(summary_data["inventory_turns"].mean()),
                "avg_on_hand": float(summary_data["avg_on_hand"].mean()),
            }
        else:
            summary_stats = {
                "total_products": 0,
                "avg_service_level": 0.0,
                "avg_stockout_rate": 0.0,
                "avg_inventory_turns": 0.0,
                "avg_on_hand": 0.0,
            }

        return jsonify({"plot_url": plot_url, "summary_stats": summary_stats})

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/get_forecast_visualization_plot", methods=["POST"])
def get_forecast_visualization_plot():
    """Generate forecast visualization plot using forecast_visualization_data"""
    try:
        # Get form data
        product = request.form.get("product")
        location = request.form.get("location")
        analysis_date = request.form.get("analysis_date")
        forecast_method = request.form.get("forecast_method")

        # Load forecast visualization data
        forecast_data = load_complete_workflow_data("forecast_visualization")

        if forecast_data is None:
            return jsonify({"error": "Forecast visualization data not available"})

        # Apply filters
        filtered_data = forecast_data.copy()

        if product:
            filtered_data = filtered_data[filtered_data["product_id"] == product]
        if location:
            filtered_data = filtered_data[filtered_data["location_id"] == location]
        if analysis_date:
            filtered_data = filtered_data[
                filtered_data["analysis_date"].dt.strftime("%Y-%m-%d") == analysis_date
            ]
        if forecast_method and forecast_method != "All":
            filtered_data = filtered_data[
                filtered_data["forecast_method"] == forecast_method
            ]

        if filtered_data.empty:
            return jsonify({"error": "No data matches the selected filters"})

        # Get the first row for the selected filters
        row = filtered_data.iloc[0]

        # Parse the data from string format
        import ast

        # Parse historical data
        historical_dates = ast.literal_eval(row["historical_bucket_start_dates"])
        historical_demands = ast.literal_eval(row["historical_demands"])

        # Parse forecast period data
        forecast_dates = ast.literal_eval(row["forecast_horizon_start_dates"])
        actual_demands = ast.literal_eval(row["forecast_horizon_actual_demands"])
        forecast_demands = ast.literal_eval(row["forecast_horizon_forecast_demands"])

        # Convert dates to datetime
        historical_dates = [pd.to_datetime(date) for date in historical_dates]
        forecast_dates = [pd.to_datetime(date) for date in forecast_dates]

        # Create the plot
        fig, ax = plt.subplots(figsize=(15, 8))

        # Plot historical demand as bars
        ax.bar(
            historical_dates,
            historical_demands,
            alpha=0.7,
            color="lightblue",
            label="Historical Demand",
            width=8,
        )

        # Plot actual demand in forecast period as bars
        ax.bar(
            forecast_dates,
            actual_demands,
            alpha=0.7,
            color="green",
            label="Actual Demand (Forecast Period)",
            width=8,
        )

        # Plot forecasted demand as a line
        ax.plot(
            forecast_dates,
            forecast_demands,
            marker="o",
            linewidth=3,
            markersize=8,
            color="red",
            label="Forecasted Demand",
        )

        # Customize the plot
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Demand", fontsize=12)

        # Create title with or without forecast method
        if forecast_method and forecast_method != "All":
            title = f"Forecast Visualization - {product} at {location}\nAnalysis Date: {analysis_date} - Method: {forecast_method}"
        else:
            title = f"Forecast Visualization - {product} at {location}\nAnalysis Date: {analysis_date}"

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)

        # Add text annotations for key metrics
        total_historical = sum(historical_demands)
        total_actual = sum(actual_demands)
        total_forecast = sum(forecast_demands)

        # Calculate forecast accuracy metrics
        errors = [abs(f - a) for f, a in zip(forecast_demands, actual_demands)]
        mae = sum(errors) / len(errors) if errors else 0
        mape = (
            sum(
                [abs(f - a) / a * 100 for f, a in zip(forecast_demands, actual_demands)]
            )
            / len(forecast_demands)
            if actual_demands
            else 0
        )

        # Add text box with metrics
        textstr = f"Historical Total: {total_historical:,.0f}\nActual Total: {total_actual:,.0f}\nForecast Total: {total_forecast:,.0f}\nMAE: {mae:.2f}\nMAPE: {mape:.2f}%"
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
        ax.text(
            0.02,
            0.98,
            textstr,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=props,
        )

        plt.tight_layout()

        # Convert plot to base64 string
        img = io.BytesIO()
        fig.savefig(img, format="png", bbox_inches="tight", dpi=300)
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        plt.close(fig)

        return jsonify(
            {
                "plot_url": plot_url,
                "mae": float(mae),
                "mape": float(mape),
                "total_forecasts": len(forecast_demands),
                "historical_points": len(historical_demands),
                "forecast_horizon": len(forecast_demands),
                "risk_period": int(row["risk_period"]),
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)})


# ============================================================================
# PARAMETER OPTIMIZATION ROUTES
# ============================================================================

@app.route("/parameter_optimization")
def parameter_optimization():
    """Parameter optimization page for Prophet forecasting"""
    try:
        # Load available products and locations from product master
        product_master = loader.load_product_master()
        
        # Get unique product-location combinations
        product_locations = product_master[['product_id', 'location_id']].drop_duplicates()
        
        # Load regressor configuration
        regressor_config = loader.load_regressor_config()
        
        # Load holiday data
        holiday_data = loader.load_holiday_data()
        
        # Get unique holidays for display
        unique_holidays = holiday_data['holiday'].dropna().unique().tolist() if not holiday_data.empty else []
        
        return render_template(
            "parameter_optimization.html",
            product_locations=product_locations.to_dict('records'),
            regressor_config=regressor_config,
            unique_holidays=unique_holidays
        )
        
    except Exception as e:
        return render_template(
            "parameter_optimization.html",
            product_locations=[],
            regressor_config={},
            unique_holidays=[],
            error=str(e)
        )


@app.route("/api/run_forecast", methods=["POST"])
def run_forecast():
    """API endpoint to run forecast with custom parameters"""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({"success": False, "error": "No data provided"})
        
        # Extract parameters
        product_id = data.get('product_id')
        location_id = data.get('location_id')
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        configuration = data.get('configuration', {})
        
        if not all([product_id, location_id, start_date, end_date]):
            return jsonify({"success": False, "error": "Missing required parameters"})
        
        # Add risk_period_days to configuration from product master
        try:
            from forecaster.validation.product_master_schema import ProductMasterSchema
            
            # Load product master to get risk_period and demand_frequency
            product_master = loader.load_product_master()
            product_record = product_master[
                (product_master['product_id'] == product_id) & 
                (product_master['location_id'] == location_id)
            ].iloc[0].to_dict()
            
            # Calculate risk_period_days
            risk_period_days = ProductMasterSchema.get_risk_period_days(
                product_record.get('demand_frequency'),
                product_record.get('risk_period')
            )
            
            # Add to configuration
            configuration['risk_period_days'] = risk_period_days
            
        except Exception as e:
            print(f"Warning: Could not add risk_period_days to configuration: {e}")
            # Continue without risk_period_days if there's an error
        
        # Create optimization task and execute forecast
        result = execute_optimization_forecast(
            product_id, location_id, start_date, end_date, configuration
        )
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify({"success": False, "error": result['error']})
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/get_optimization_date_range", methods=["POST"])
def get_optimization_date_range():
    """API endpoint to get available date range for a product-location optimization"""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({"success": False, "error": "No data provided"})
        
        # Extract parameters
        product_id = data.get('product_id')
        location_id = data.get('location_id')
        
        if not all([product_id, location_id]):
            return jsonify({"success": False, "error": "Missing required parameters"})
        
        # Get available date range
        result = get_available_date_range(product_id, location_id)
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify({"success": False, "error": result['error']})
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/save_configuration", methods=["POST"])
def save_configuration():
    """API endpoint to save configuration to main JSON file"""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({"success": False, "error": "No data provided"})
        
        # Extract parameters
        product_id = data.get('product_id')
        location_id = data.get('location_id')
        configuration = data.get('configuration', {})
        
        if not all([product_id, location_id]):
            return jsonify({"success": False, "error": "Missing required parameters"})
        
        # Update main configuration file
        result = update_main_configuration(product_id, location_id, configuration)
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify({"success": False, "error": result['error']})
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


# ============================================================================
# PARAMETER OPTIMIZATION HELPER FUNCTIONS
# ============================================================================

def create_optimization_task(product_id: str, location_id: str, start_date: str, end_date: str) -> dict:
    """
    Create an optimization task structure matching the existing pipeline.
    
    Args:
        product_id: Product identifier
        location_id: Location identifier
        start_date: Start date for forecast (YYYY-MM-DD)
        end_date: End date for forecast (YYYY-MM-DD)
        
    Returns:
        Task dictionary with required structure
    """
    try:
        # Load input data with regressors
        input_data = loader.load_input_data_with_regressors()
        
        # Filter data for the specific product-location
        product_data = input_data[
            (input_data['product_id'] == product_id) & 
            (input_data['location_id'] == location_id)
        ].copy()
        
        if product_data.empty:
            raise ValueError(f"No data found for {product_id} at {location_id}")
        
        # Convert date column to datetime
        product_data['date'] = pd.to_datetime(product_data['date'])
        
        # Sort by date
        product_data = product_data.sort_values('date')
        if len(product_data) < 25:
            raise ValueError(f"Insufficient data for {product_id} at {location_id}")
        
        # Load product master to get product configuration
        product_master = loader.load_product_master()
        product_record = product_master[
            (product_master['product_id'] == product_id) & 
            (product_master['location_id'] == location_id)
        ].iloc[0].to_dict()
        
        # Create task structure matching existing pipeline
        task = {
            'product_record': product_record,
            'product_data': product_data,
            'analysis_dates': [end_date],  # Use end_date as analysis date for optimization
            'forecast_method': 'prophet',
            'optimising_parameters': True,
            'start_date': start_date,      # Store start_date for reference
            'end_date': end_date           # Store end_date for reference
        }
        
        return task
        
    except Exception as e:
        raise RuntimeError(f"Failed to create optimization task: {e}")


def execute_optimization_forecast(product_id: str, location_id: str, start_date: str, end_date: str, configuration: dict) -> dict:
    """
    Execute forecast with custom parameters using the existing pipeline.
    
    Args:
        product_id: Product identifier
        location_id: Location identifier
        start_date: Start date for forecast (YYYY-MM-DD)
        end_date: End date for forecast (YYYY-MM-DD)
        configuration: Custom Prophet configuration
        
    Returns:
        Dictionary with forecast results and components, including historical context
    """
    try:
        # Create optimization task
        task = create_optimization_task(product_id, location_id, start_date, end_date)
        
        # Create extended futures table covering the full date range
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Get the base futures data from the task
        base_futures = task['product_data'].copy()
        
        # Create extended futures table
        extended_futures = create_extended_futures_table(
            base_futures, start_dt, end_dt
        )
        
        # Update task with extended futures
        task['future_data'] = extended_futures
        
        from forecaster.backtesting.full_backtesting_pipeline import FullBacktestingPipeline

        # Call the static method directly
        result = FullBacktestingPipeline.process_product_task(task, configuration=configuration)
        if result is None:
            raise RuntimeError("Forecasting failed in process_product_task")
        _, plotting_data = result  # We only need the plotting data (components_df)
        
        # Create historical data from product_data before the start date
        historical_data = task['product_data'].copy()
        historical_data = historical_data[historical_data['date'] < start_dt].copy()
        
        # Prepare historical data for plotting (similar structure to forecast data)
        if not historical_data.empty:
            # Sort by date to ensure chronological order
            historical_data = historical_data.sort_values('date')
            
            # Create historical DataFrame with the same structure as plotting_data
            historical_df = pd.DataFrame()
            
            # Add the date column (ds) and actual demand (y)
            historical_df['ds'] = historical_data['date'].dt.strftime('%Y-%m-%d')
            historical_df['y'] = historical_data['outflow']
            
            # Add all other columns from plotting_data with NaN values
            for col in plotting_data.columns:
                if col not in ['ds', 'y']:
                    historical_df[col] = np.nan
            
            # Ensure column order matches plotting_data
            historical_df = historical_df[plotting_data.columns]
            
            # Combine historical and forecast data
            combined_data = pd.concat([historical_df, plotting_data], ignore_index=True)
            
            # Sort by date to ensure chronological order
            combined_data['ds'] = pd.to_datetime(combined_data['ds'])
            combined_data = combined_data.sort_values('ds').reset_index(drop=True)
            
            # Convert back to string format for JSON serialization
            combined_data['ds'] = combined_data['ds'].dt.strftime('%Y-%m-%d')
        else:
            # No historical data available, just use forecast data
            combined_data = plotting_data.copy()
        
        # Replace NaN values with None (null in JSON) before conversion
        combined_data_clean = combined_data.replace({np.nan: None})
        
        results = {
            'success': True,
            'plotting_data': combined_data_clean.to_dict('records'),
            'analysis_date': task['analysis_dates'][0],  # Already a string, no need for isoformat()
            'date_range': {
                'start': start_date,
                'end': end_date
            },
            'historical_context': {
                'has_historical_data': len(historical_data) > 0,
                'historical_data_points': len(historical_data),
                'historical_start_date': historical_data['date'].min().strftime('%Y-%m-%d') if not historical_data.empty else None,
                'historical_end_date': historical_data['date'].max().strftime('%Y-%m-%d') if not historical_data.empty else None
            },
            'risk_period_info': {
                'risk_period_days': configuration.get('risk_period_days'),
                'forecast_start_date': start_date,
                'context_boundary_date': (pd.to_datetime(start_date) - pd.Timedelta(days=configuration.get('risk_period_days', 0))).strftime('%Y-%m-%d') if configuration.get('risk_period_days') else None
            }
        }
        
        # Debug logging
        print(f"Risk period days: {configuration.get('risk_period_days')}")
        print(f"Forecast start date: {start_date}")
        print(f"Context boundary date: {(pd.to_datetime(start_date) - pd.Timedelta(days=configuration.get('risk_period_days', 0))).strftime('%Y-%m-%d') if configuration.get('risk_period_days') else None}")
        print(f"Risk period info: {results['risk_period_info']}")
        
        return results
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def create_extended_futures_table(base_futures: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    """
    Create extended futures table covering the full user-specified date range.
    
    Args:
        base_futures: Base futures data from the task
        start_date: Start date for extended forecast
        end_date: End date for extended forecast
        
    Returns:
        Extended futures DataFrame
    """
    try:
        # Create date range
        # Filter base_futures (product_data) for dates between start_date and end_date inclusive
        mask = (base_futures['date'] >= pd.to_datetime(start_date)) & (base_futures['date'] <= pd.to_datetime(end_date))
        extended_futures = base_futures.loc[mask].copy().reset_index(drop=True)
        
        return extended_futures
        
    except Exception as e:
        raise RuntimeError(f"Failed to create extended futures table: {e}")


def get_available_date_range(product_id: str, location_id: str) -> dict:
    """
    Get the available date range for a specific product-location.
    
    Args:
        product_id: Product identifier
        location_id: Location identifier
        
    Returns:
        Dictionary with start_date and end_date (risk period after min non-NA date to last date with non-NA outflow)
    """
    try:
        # Load input data with regressors
        input_data = loader.load_input_data_with_regressors()
        
        # Filter data for the specific product-location
        product_data = input_data[
            (input_data['product_id'] == product_id) & 
            (input_data['location_id'] == location_id)
        ].copy()
        
        if product_data.empty:
            raise ValueError(f"No data found for {product_id} at {location_id}")
        
        # Convert date column to datetime
        product_data['date'] = pd.to_datetime(product_data['date'])
        
        # Sort by date
        product_data = product_data.sort_values('date')
        
        # Find the first date with positive outflow
        positive_outflow_data = product_data[product_data['outflow'] > 0]
        if positive_outflow_data.empty:
            raise ValueError(f"No positive outflow data found for {product_id} at {location_id}")
        
        # Calculate start date: first date with positive outflow
        start_date = positive_outflow_data.iloc[0]['date']

        # Calculate risk period in days
        from forecaster.validation.product_master_schema import ProductMasterSchema
        product_master = loader.load_product_master()
        product_record = product_master[
            (product_master['product_id'] == product_id) & 
            (product_master['location_id'] == location_id)
        ].iloc[0].to_dict()
        
        risk_period_days = ProductMasterSchema.get_risk_period_days(
            product_record.get('demand_frequency'),
            product_record.get('risk_period')
        )
        
        # Find the last date with non-NA outflow
        non_na_outflow_data = product_data[product_data['outflow'].notna()]
        if non_na_outflow_data.empty:
            raise ValueError(f"No non-NA outflow data found for {product_id} at {location_id}")
        end_date = non_na_outflow_data.iloc[-1]['date']  # Last date with non-NA outflow
        
        start_date = start_date + pd.Timedelta(days=risk_period_days+1)
        
        # Ensure start date is before end date and we have enough data points
        if start_date >= end_date:
            raise ValueError(f"Insufficient positive outflow data for {product_id} at {location_id}. Need data points between {start_date} and {end_date}")
        
        return {
            'success': True,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'total_data_points': len(product_data)
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def update_main_configuration(product_id: str, location_id: str, configuration: dict) -> dict:
    """
    Update the main JSON configuration file for a specific product-location.
    
    Args:
        product_id: Product identifier
        location_id: Location identifier
        configuration: Configuration to save
        
    Returns:
        Dictionary with success status
    """
    try:
        # Load current configuration
        from data.config.paths import get_input_file_path
        config_path = get_input_file_path('prophet_parameters')
        
        print(f"Config path: {config_path}")
        print(f"Config path exists: {Path(config_path).exists()}")
        
        if not Path(config_path).exists():
            # Create empty configuration if file doesn't exist
            print("Creating new configuration file")
            all_configs = {}
        else:
            print("Loading existing configuration file")
            with open(config_path, 'r') as f:
                all_configs = json.load(f)
            print(f"Loaded config keys: {list(all_configs.keys())}")
        
        # Create key for product-location
        key = str((product_id, location_id))
        print(f"Updating configuration for key: {key}")
        print(f"Configuration to save: {configuration}")
        
        # Remove cross_validation parameter before saving
        save_config = configuration.copy()
        save_config.pop('cross_validation', None)  # Remove if exists, ignore if doesn't
        
        # Update configuration for this product-location
        all_configs[key] = save_config
        
        # Save updated configuration
        print(f"Saving configuration to: {config_path}")
        with open(config_path, 'w') as f:
            json.dump(all_configs, f, indent=2)
        
        print("Configuration saved successfully")
        return {
            'success': True,
            'message': f'Configuration saved for {product_id} at {location_id}'
        }
        
    except Exception as e:
        print(f"Error saving configuration: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }


@app.route("/seasonality_analysis")
def seasonality_analysis():
    """Seasonality analysis page"""
    initialize_data()

    # Get available filter options
    if visualizer is not None:
        filters = visualizer.get_available_filters()
        locations = filters["locations"]
        categories = filters["categories"]
        products = filters["products"]
    else:
        # Provide empty filters if visualizer is not available
        locations = []
        categories = []
        products = []

    return render_template(
        "seasonality_analysis.html",
        locations=locations,
        categories=categories,
        products=products,
    )


@app.route("/run_seasonality_analysis", methods=["POST"])
def run_seasonality_analysis_route():
    """Run seasonality analysis and return results"""
    try:
        # Get form data
        products = request.form.getlist("products")
        locations = request.form.getlist("locations")
        categories = request.form.getlist("categories")
        analysis_type = request.form.get("analysis_type", "decomposition")

        # Initialize data loader
        customer_demand = loader.load_outflow()

        # Filter data
        if products:
            customer_demand = customer_demand[
                customer_demand["product_id"].isin(products)
            ]
        if locations:
            customer_demand = customer_demand[
                customer_demand["location_id"].isin(locations)
            ]
        if categories:
            customer_demand = customer_demand[
                customer_demand["product_category"].isin(categories)
            ]

        if customer_demand.empty:
            return jsonify(
                {
                    "success": False,
                    "error": "No data available for the selected filters",
                }
            )

        # Prepare time series data for Prophet
        customer_demand["date"] = pd.to_datetime(customer_demand["date"])
        customer_demand = customer_demand.sort_values("date")

        # Import Prophet forecaster and seasonality analyzer
        from forecaster.forecasting.prophet import ProphetForecaster
        from forecaster.forecasting.seasonality_analyzer import SeasonalityAnalyzer

        # Create and fit Prophet model
        forecaster = ProphetForecaster(
            weekly_seasonality=True,
            daily_seasonality=False,
            include_monthly_effects=True,
            include_quarterly_effects=True,
            include_festival_seasons=True,
            include_monsoon_effect=True,
        )

        # Fit the model with original data (ProphetForecaster handles conversion internally)
        forecaster.fit(customer_demand)

        # Get the fitted model
        model = forecaster.model

        # Get the Prophet-formatted data that was used for fitting
        prophet_data = forecaster._prepare_data_for_prophet(customer_demand)

        # Perform seasonality analysis
        analyzer = SeasonalityAnalyzer()
        analysis_results = analyzer.analyze_seasonality_components(
            customer_demand, model, prophet_data
        )

        # Get optimal components and recommendations
        optimal_components = analyzer.get_optimal_components(analysis_results)

        # Generate forecast data for comparison
        forecast_data = generate_forecast_comparison(
            customer_demand, optimal_components, analysis_results
        )

        # Prepare response data
        response_data = {
            "success": True,
            "results": {
                "summary": analysis_results.get("summary", {}),
                "seasonalities": analysis_results.get("seasonalities", {}),
                "recommendations": analysis_results.get("recommendations", {}),
                "model_parameters": optimal_components.get("model_parameters", {}),
                "forecast_data": forecast_data,
            },
        }

        return jsonify(response_data)

    except Exception as e:
        import traceback

        print(f"Error in seasonality analysis: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify(
            {"success": False, "error": f"Failed to run seasonality analysis: {str(e)}"}
        )


def generate_forecast_comparison(
    customer_demand, optimal_components, analysis_results=None
):
    """
    Generate forecast comparison data for default vs optimized parameters.

    Args:
        customer_demand: Filtered customer demand data
        optimal_components: Optimal components from seasonality analysis
        analysis_results: Full analysis results containing recommendations

    Returns:
        Dictionary with forecast data for both default and optimized parameters
    """
    try:
        # Prepare data for Prophet - use original daily granularity
        customer_demand["date"] = pd.to_datetime(customer_demand["date"])
        customer_demand = customer_demand.sort_values("date")

        # Create default forecaster
        from forecaster.forecasting.prophet import ProphetForecaster

        default_forecaster = ProphetForecaster(
            weekly_seasonality=True,
            daily_seasonality=False,
            include_monthly_effects=True,
            include_quarterly_effects=True,
            include_festival_seasons=True,
            include_monsoon_effect=True,
        )

        # Fit default model with original daily granularity data
        default_forecaster.fit(customer_demand)

        # Historical data - use the data that Prophet actually used for fitting
        # Get the Prophet-formatted data that was used for fitting
        prophet_data_default = default_forecaster._prepare_data_for_prophet(
            customer_demand
        )
        historical_dates = prophet_data_default["ds"].dt.strftime("%Y-%m-%d").tolist()
        historical_demands = prophet_data_default["y"].tolist()

        # Create optimized forecaster with recommended parameters
        # Get best model parameters from analysis results or fallback to regularization settings
        best_model_params = {}
        if analysis_results and "recommendations" in analysis_results:
            best_model_params = analysis_results["recommendations"].get(
                "best_model_parameters", {}
            )

        if not best_model_params:
            # Fallback to regularization settings if best_model_params not available
            reg_settings = optimal_components.get("regularization_settings", {})
            best_model_params = {
                "changepoint_prior_scale": reg_settings.get(
                    "changepoint_prior_scale", 0.05
                ),
                "seasonality_prior_scale": reg_settings.get(
                    "seasonality_prior_scale", 10.0
                ),
                "holidays_prior_scale": reg_settings.get("holidays_prior_scale", 10.0),
            }

        optimized_forecaster = ProphetForecaster(
            changepoint_prior_scale=best_model_params.get(
                "changepoint_prior_scale", 0.05
            ),
            seasonality_prior_scale=best_model_params.get(
                "seasonality_prior_scale", 10.0
            ),
            holidays_prior_scale=best_model_params.get("holidays_prior_scale", 10.0),
            weekly_seasonality=best_model_params.get("weekly_seasonality", True),
            daily_seasonality=best_model_params.get("daily_seasonality", False),
            include_monthly_effects=best_model_params.get(
                "include_monthly_effects", True
            ),
            include_quarterly_effects=best_model_params.get(
                "include_quarterly_effects", True
            ),
            include_festival_seasons=best_model_params.get(
                "include_festival_seasons", True
            ),
            include_monsoon_effect=best_model_params.get(
                "include_monsoon_effect", True
            ),
        )

        # Fit optimized model with original daily granularity data
        optimized_forecaster.fit(customer_demand)

        # Generate 4-month forecasts
        forecast_steps = 120  # 4 months * 30 days

        # Default forecast
        default_forecast = default_forecaster.forecast(steps=forecast_steps)
        default_forecast_dates = (
            pd.date_range(
                start=customer_demand["date"].max() + pd.Timedelta(days=1),
                periods=forecast_steps,
                freq="D",
            )
            .strftime("%Y-%m-%d")
            .tolist()
        )
        default_forecast_values = default_forecast.tolist()

        # Optimized forecast
        optimized_forecast = optimized_forecaster.forecast(steps=forecast_steps)
        optimized_forecast_dates = (
            pd.date_range(
                start=customer_demand["date"].max() + pd.Timedelta(days=1),
                periods=forecast_steps,
                freq="D",
            )
            .strftime("%Y-%m-%d")
            .tolist()
        )
        optimized_forecast_values = optimized_forecast.tolist()

        return {
            "historical_dates": historical_dates,
            "historical_demands": historical_demands,
            "default_forecast_dates": default_forecast_dates,
            "default_forecast_values": default_forecast_values,
            "optimized_forecast_dates": optimized_forecast_dates,
            "optimized_forecast_values": optimized_forecast_values,
        }

    except Exception as e:
        print(f"Error generating forecast comparison: {str(e)}")
        return None


@app.route("/get_stock_status_plot", methods=["POST"])
def get_stock_status_plot():
    """Generate stock status plot showing understock and overstock percentages over time"""
    try:
        # Load aggregated simulation data
        detailed_dir = loader.get_output_path("simulation/detailed_results", "")
        aggregated_file = detailed_dir / "all_all_aggregated_simulation.csv"
        
        if not aggregated_file.exists():
            return jsonify({"error": "Aggregated simulation data not found"})
            
        # Load the data
        data = pd.read_csv(aggregated_file)
        data["date"] = pd.to_datetime(data["date"])
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(20, 6))
        
        # Plot actual understock percentage
        ax.plot(
            data["date"],
            data["understock_percentage"],
            label="Actual Understock %",
            color="red",
            linewidth=2,
            alpha=0.8
        )

        # Plot simulated understock percentage
        ax.plot(
            data["date"],
            data["simulated_understock_percentage"],
            label="Simulated Understock %",
            color="green",
            linewidth=2,
            alpha=0.8
        )
        
        # Plot overstock percentage
        ax.plot(
            data["date"],
            data["overstock_percentage"],
            label="Overstock %",
            color="blue",
            linewidth=2,
            alpha=0.8
        )
        
        # Customize the plot
        ax.set_title("Stock Status Over Time (All Products)", fontsize=16, fontweight="bold")
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Percentage", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=10)
        
        # Add horizontal lines at key percentages
        ax.axhline(y=20, color='gray', linestyle='--', alpha=0.3)  # 20% reference line
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.3)  # 50% reference line
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        
        # Set y-axis to percentage range with some padding
        ax.set_ylim(-5, max(max(data["understock_percentage"].max(), 
                              data["simulated_understock_percentage"].max(),
                              data["overstock_percentage"].max()) + 5, 100))
        
        plt.tight_layout()
        
        # Convert plot to base64 string
        img = io.BytesIO()
        fig.savefig(img, format="png", bbox_inches="tight", dpi=300)
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        
        plt.close(fig)
        
        return jsonify({"plot_url": plot_url})
        
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/get_demand_analysis_plot", methods=["POST"])
def get_demand_analysis_plot():
    """Generate demand analysis chart showing actual demand, orders placed, and forecasted demand"""
    try:
        # Get form data
        products = request.form.getlist("products")
        locations = request.form.getlist("locations")
        forecast_methods = request.form.getlist("forecast_methods")

        # Load simulation detailed data
        detailed_dir = loader.get_output_path("simulation/detailed_results", "")
        if not detailed_dir.exists():
            return jsonify({"error": "Detailed simulation data not available"})

        # Load forecast comparison data for first risk period forecasts
        filename = loader.config['paths']['output_files']['forecast_comparison']
        forecast_comparison_file = loader.get_output_path("backtesting", filename)
        if not forecast_comparison_file.exists():
            return jsonify({"error": "Forecast comparison data not available"})

        forecast_data = pd.read_csv(forecast_comparison_file)
        # Filter for first risk period (step=1)
        first_risk_forecasts = forecast_data[forecast_data["step"] == 1].copy()
        first_risk_forecasts["analysis_date"] = pd.to_datetime(
            first_risk_forecasts["analysis_date"]
        )
        
        # Note: We no longer filter forecast data to simulation period
        # This allows the demand analysis plot to show all available data
        print("Demand analysis plot will show all available forecast data")

        # Find matching simulation files
        all_simulation_data = []
        for file_path in detailed_dir.glob("*_simulation.csv"):
            # Extract product, location, and method from filename
            filename = file_path.stem
            parts = filename.split("_")
            if len(parts) >= 4:
                product_id = parts[0]
                location_id = parts[1]

                # Handle forecast method - it could be multiple parts
                simulation_index = -1
                for i, part in enumerate(parts):
                    if part == "simulation":
                        simulation_index = i
                        break

                if simulation_index > 2:
                    forecast_method = "_".join(parts[2:simulation_index])
                else:
                    forecast_method = parts[2] if len(parts) > 2 else "unknown"

                # Apply filters
                if products and product_id not in products:
                    continue
                if locations and location_id not in locations:
                    continue
                if forecast_methods and forecast_method not in forecast_methods:
                    continue

                # Load the simulation data
                try:
                    data = pd.read_csv(file_path)
                    data["date"] = pd.to_datetime(data["date"])
                    data["product_id"] = product_id
                    data["location_id"] = location_id
                    data["forecast_method"] = forecast_method
                    all_simulation_data.append(data)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    continue

        if not all_simulation_data:
            return jsonify({"error": "No simulation data matches the selected filters"})

        # Combine all simulation data
        combined_simulation_data = pd.concat(all_simulation_data, ignore_index=True)

        # Create the demand analysis chart
        fig, ax = plt.subplots(figsize=(20, 6))

        # Plot for each product-location-method combination
        for (
            product_id,
            location_id,
            forecast_method,
        ), group in combined_simulation_data.groupby(
            ["product_id", "location_id", "forecast_method"]
        ):
            # Sort by date
            group = group.sort_values("date")

            # Plot actual daily demand as bars (blue bars)
            ax.bar(
                group["date"],
                group["actual_demand"],
                label="Actual Demand",
                alpha=0.7,
                color="blue",
                width=1,
            )

            # Plot orders placed as vertical bars (red bars)
            order_dates = group[group["order_placed"] > 0]["date"]
            order_values = group[group["order_placed"] > 0]["order_placed"]
            if not order_dates.empty:
                ax.bar(
                    order_dates,
                    order_values,
                    alpha=0.6,
                    label="Orders Placed",
                    width=1,
                    color="red",
                )

            # Get forecast data for this product-location-method
            forecast_filter = (
                (first_risk_forecasts["product_id"] == product_id)
                & (first_risk_forecasts["location_id"] == location_id)
                & (first_risk_forecasts["forecast_method"] == forecast_method)
            )

            if forecast_filter.any():
                forecast_group = first_risk_forecasts[forecast_filter].sort_values("analysis_date")

                # Plot first risk period forecasted demand (green line without markers)
                ax.plot(
                    forecast_group["analysis_date"],
                    forecast_group["forecast_demand"],
                    label="Forecast",
                    linewidth=2,
                    alpha=0.8,
                    color="green",
                )

                # Plot first risk period actual demand (orange line without markers)
                ax.plot(
                    forecast_group["analysis_date"],
                    forecast_group["actual_demand"],
                    label="First Risk Period Actual",
                    linewidth=2,
                    alpha=0.8,
                    color="orange",
                    linestyle="--",
                )

        # Customize the plot
        title = "Demand Analysis: Actual vs Forecasted vs Orders (All Available Data)"
        
        ax.set_title(
            title,
            fontsize=16,
            fontweight="bold",
        )
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Quantity", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=10)

        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        plt.tight_layout()

        # Convert plot to base64 string
        img = io.BytesIO()
        fig.savefig(img, format="png", bbox_inches="tight", dpi=300)
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        plt.close(fig)

        return jsonify({"plot_url": plot_url})

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
