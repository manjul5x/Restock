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


@app.route('/favicon.ico')
def favicon():
    """Serve a simple favicon to avoid 404 errors"""
    from flask import abort
    # Return a 204 No Content response instead of 404
    return '', 204


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

@app.route("/insights")
def insights():
    """Insights page with demand classification analysis"""
    initialize_data()
    return render_template("insights.html")


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


# ============================================================================
# INSIGHTS HELPER FUNCTIONS (Independent from other tabs)
# ============================================================================

def insights_calculate_analysis_period_dates(insights_analysis_period, insights_custom_start=None, insights_custom_end=None):
    """Calculate start and end dates for insights analysis period"""
    try:
        insights_end_date_calc = pd.Timestamp.now().normalize()
        insights_analysis_start_date = None
        insights_analysis_end_date = insights_end_date_calc
        
        if insights_analysis_period == 'custom':
            if insights_custom_start:
                insights_analysis_start_date = pd.to_datetime(insights_custom_start)
            if insights_custom_end:
                insights_analysis_end_date = pd.to_datetime(insights_custom_end)
        elif insights_analysis_period == 'last_3_months':
            insights_analysis_start_date = insights_end_date_calc - pd.DateOffset(months=3)
        elif insights_analysis_period == 'last_6_months':
            insights_analysis_start_date = insights_end_date_calc - pd.DateOffset(months=6)
        elif insights_analysis_period == 'last_12_months':
            insights_analysis_start_date = insights_end_date_calc - pd.DateOffset(months=12)
        elif insights_analysis_period == 'last_24_months':
            insights_analysis_start_date = insights_end_date_calc - pd.DateOffset(months=24)
        else:
            # Default to last 6 months for insights
            insights_analysis_start_date = insights_end_date_calc - pd.DateOffset(months=6)
        
        return insights_analysis_start_date, insights_analysis_end_date
        
    except Exception as insights_error:
        print(f"Error calculating insights analysis period dates: {insights_error}")
        # Return default 6 months for insights
        insights_end_date_calc = pd.Timestamp.now().normalize()
        return insights_end_date_calc - pd.DateOffset(months=6), insights_end_date_calc


def insights_load_master_table_by_type(insights_master_table_type="original"):
    """Load product master table based on type selection for insights"""
    import os
    try:
        if insights_master_table_type == "selected":
            # Try to load selected products master table for insights
            try:
                insights_selected_path = os.path.join(os.path.dirname(__file__), 'selected_master_table.csv')
                if os.path.exists(insights_selected_path):
                    insights_product_master_data = pd.read_csv(insights_selected_path)
                    print(f"✅ Loaded selected master table for insights: {len(insights_product_master_data)} products")
                else:
                    print("⚠️ Selected master table not found for insights, using original")
                    insights_product_master_data = loader.load_product_master()
            except Exception as insights_selected_error:
                print(f"⚠️ Error loading selected master table for insights: {insights_selected_error}")
                insights_product_master_data = loader.load_product_master()
        else:
            # Load original master table for insights
            insights_product_master_data = loader.load_product_master()
        
        return insights_product_master_data
        
    except Exception as insights_error:
        print(f"Error loading master table for insights: {insights_error}")
        # Return empty dataframe as fallback for insights
        return pd.DataFrame()


def insights_apply_standard_filters(insights_data, insights_location_filter, insights_category_filter, insights_product_id_filter, insights_product_master):
    """Apply standard filters to insights data"""
    try:
        insights_filtered_data = insights_data.copy()
        
        # Apply location filter for insights
        if insights_location_filter and insights_location_filter.strip():
            insights_locations = [loc.strip() for loc in insights_location_filter.split(',') if loc.strip()]
            if insights_locations and 'location_id' in insights_filtered_data.columns:
                insights_filtered_data = insights_filtered_data[insights_filtered_data['location_id'].isin(insights_locations)]
                print(f"🔽 After location filter for insights: {len(insights_filtered_data)} rows")
        
        # Apply category filter for insights
        if insights_category_filter and insights_category_filter.strip():
            insights_categories = [cat.strip() for cat in insights_category_filter.split(',') if cat.strip()]
            if insights_categories:
                if 'product_category' in insights_filtered_data.columns:
                    # Use category from demand data for insights
                    insights_filtered_data = insights_filtered_data[insights_filtered_data['product_category'].isin(insights_categories)]
                    print(f"🔽 After category filter (from demand data) for insights: {len(insights_filtered_data)} rows")
                elif not insights_product_master.empty and 'product_category' in insights_product_master.columns:
                    # Use category from product master for insights
                    insights_category_products = insights_product_master[insights_product_master['product_category'].isin(insights_categories)]['product_id'].unique()
                    insights_filtered_data = insights_filtered_data[insights_filtered_data['product_id'].isin(insights_category_products)]
                    print(f"🔽 After category filter (from master) for insights: {len(insights_filtered_data)} rows")
        
        # Apply product ID filter for insights
        if insights_product_id_filter and insights_product_id_filter.strip():
            insights_product_ids = [pid.strip() for pid in insights_product_id_filter.split(',') if pid.strip()]
            if insights_product_ids and 'product_id' in insights_filtered_data.columns:
                insights_filtered_data = insights_filtered_data[insights_filtered_data['product_id'].isin(insights_product_ids)]
                print(f"🔽 After product ID filter for insights: {len(insights_filtered_data)} rows")
        
        return insights_filtered_data
        
    except Exception as insights_error:
        print(f"Error applying standard filters for insights: {insights_error}")
        return insights_data


def insights_calculate_otif_metrics(insights_data, insights_otif_x_days=3):
    """Calculate OTIF (On Time In Full) metrics for insights using real sales data"""
    import os
    
    try:
        # Try multiple possible paths for the sales details file for insights
        insights_possible_paths = [
            'data/customer_data/fct_sales_details.csv',
            '../data/customer_data/fct_sales_details.csv',
            '../../data/customer_data/fct_sales_details.csv',
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'customer_data', 'fct_sales_details.csv')
        ]
        
        insights_sales_file_path = None
        for insights_path in insights_possible_paths:
            if os.path.exists(insights_path):
                insights_sales_file_path = insights_path
                break
        
        if not insights_sales_file_path:
            print("⚠️ OTIF Warning for insights: fct_sales_details.csv file not found in any expected location")
            print(f"Checked paths: {insights_possible_paths}")
            return {
                'otif_value_percentage': 'Data Not Available',
                'otif_units_percentage': 'Data Not Available', 
                'time_delay_percentage': 'Data Not Available',
                'qty_shortage_percentage': 'Data Not Available',
                'both_issues_percentage': 'Data Not Available',
                'avg_delay_days': 'Data Not Available'
            }
        
        # Load sales data for insights
        print(f"📊 Loading OTIF sales data for insights from {insights_sales_file_path}")
        insights_df = pd.read_csv(insights_sales_file_path)
        
        # Convert date columns for insights
        insights_df['FULFILLMENT_DATE'] = pd.to_datetime(insights_df['FULFILLMENT_DATE'])
        insights_df['DYNAMIC_EXPECTED_DATE'] = pd.to_datetime(insights_df['DYNAMIC_EXPECTED_DATE'])
        
        # Add X days tolerance to expected date for insights
        insights_df['DYNAMIC_EXPECTED_DATE'] = insights_df['DYNAMIC_EXPECTED_DATE'] + pd.Timedelta(days=insights_otif_x_days)
        
        # Calculate OTIF conditions for insights
        # OTIF = On Time (within expected date + X days) AND In Full (fulfilled >= booked)
        insights_otif_mask = (insights_df['FULFILLED_UNITS'] >= insights_df['BOOKED_UNITS']) & \
                           (insights_df['FULFILLMENT_DATE'] <= insights_df['DYNAMIC_EXPECTED_DATE'])
        
        # Calculate issue types for insights
        # 1. Time delay but correct quantity
        insights_time_delay_mask = (insights_df['FULFILLED_UNITS'] >= insights_df['BOOKED_UNITS']) & \
                                 (insights_df['FULFILLMENT_DATE'] > insights_df['DYNAMIC_EXPECTED_DATE'])
        
        # 2. Quantity shortage but on time
        insights_qty_shortage_mask = (insights_df['FULFILLED_UNITS'] < insights_df['BOOKED_UNITS']) & \
                                   (insights_df['FULFILLMENT_DATE'] <= insights_df['DYNAMIC_EXPECTED_DATE'])
        
        # 3. Both quantity and time issues
        insights_both_issues_mask = (insights_df['FULFILLED_UNITS'] < insights_df['BOOKED_UNITS']) & \
                                  (insights_df['FULFILLMENT_DATE'] > insights_df['DYNAMIC_EXPECTED_DATE'])
        
        # Calculate aggregated OTIF metrics for insights
        insights_total_sales = insights_df['TOTAL_BOOKED_SALES'].sum()
        insights_total_units = insights_df['BOOKED_UNITS'].sum()
        
        insights_otif_value = (
            insights_df.loc[insights_otif_mask, 'TOTAL_BOOKED_SALES'].sum() / insights_total_sales * 100
        ) if insights_total_sales > 0 else 0
        
        insights_otif_units = (
            insights_df.loc[insights_otif_mask, 'BOOKED_UNITS'].sum() / insights_total_units * 100
        ) if insights_total_units > 0 else 0
        
        # Calculate additional delivery performance metrics for insights
        insights_time_delay_percentage = (
            insights_df.loc[insights_time_delay_mask, 'TOTAL_BOOKED_SALES'].sum() / insights_total_sales * 100
        ) if insights_total_sales > 0 else 0
        
        insights_qty_shortage_percentage = (
            insights_df.loc[insights_qty_shortage_mask, 'TOTAL_BOOKED_SALES'].sum() / insights_total_sales * 100
        ) if insights_total_sales > 0 else 0
        
        insights_both_issues_percentage = (
            insights_df.loc[insights_both_issues_mask, 'TOTAL_BOOKED_SALES'].sum() / insights_total_sales * 100
        ) if insights_total_sales > 0 else 0
        
        # Calculate average delay days for time delay cases for insights
        if insights_time_delay_mask.any():
            insights_delay_days = (insights_df.loc[insights_time_delay_mask, 'FULFILLMENT_DATE'] - 
                                 insights_df.loc[insights_time_delay_mask, 'DYNAMIC_EXPECTED_DATE']).dt.days
            insights_avg_delay_days = insights_delay_days.mean()
        else:
            insights_avg_delay_days = 0
        
        print(f"✅ OTIF Value for insights: {insights_otif_value:.1f}%")
        print(f"✅ OTIF Units for insights: {insights_otif_units:.1f}%")
        print(f"⚠️ Time Delay for insights: {insights_time_delay_percentage:.1f}%")
        
        return {
            'otif_value_percentage': round(insights_otif_value, 1),
            'otif_units_percentage': round(insights_otif_units, 1),
            'time_delay_percentage': round(insights_time_delay_percentage, 1),
            'qty_shortage_percentage': round(insights_qty_shortage_percentage, 1),
            'both_issues_percentage': round(insights_both_issues_percentage, 1),
            'avg_delay_days': round(insights_avg_delay_days, 1)
        }
        
    except Exception as insights_error:
        print(f"❌ Error calculating OTIF metrics for insights: {insights_error}")
        return {
            'otif_value_percentage': 'Data Not Available',
            'otif_units_percentage': 'Data Not Available',
            'time_delay_percentage': 'Data Not Available',
            'qty_shortage_percentage': 'Data Not Available',
            'both_issues_percentage': 'Data Not Available',
            'avg_delay_days': 'Data Not Available'
        }


# ============================================================================
# INSIGHTS API ROUTES (Independent from other tabs)
# ============================================================================

@app.route("/get_demand_classification_plot", methods=["POST"])
def get_demand_classification_plot():
    """Generate demand classification plot using actual data"""
    try:
        from forecaster.insights.demand_classification import demand_classification
        
        # Get parameters
        analysis_period = request.form.get("analysis_period", "last_12_months")
        group_by = request.form.get("group_by", "product_id")
        time_unit = request.form.get("time_unit", "day")
        show_revenue = request.form.get("show_revenue", "false").lower() == "true"
        location_filter = request.form.get("location_filter", "")
        category_filter = request.form.get("category_filter", "")
        product_id_filter = request.form.get("product_id_filter", "").strip()
        include_zero_demand = request.form.get("include_zero_demand", "true").lower() == "true"
        pareto_lines = request.form.get("pareto_lines", "50,80,90")
        show_values_with_percent = request.form.get("show_values_with_percent", "true").lower() == "true"
        master_table_type = request.form.get("master_table_type", "original")

        print(f"🔍 Insights Classification Parameters:")
        print(f"  Analysis Period: {analysis_period}")
        print(f"  Location Filter: {location_filter}")
        print(f"  Category Filter: {category_filter}")
        print(f"  Product ID Filter: {product_id_filter}")
        print(f"  Master Table Type: {master_table_type}")
        
        # Load data using existing loader
        demand_data = loader.load_outflow()
        
        if demand_data.empty:
            return jsonify({
                "success": False,
                "error": "No demand data available"
            })
        
        print(f"📊 Loaded demand data: {len(demand_data)} rows")
        
        # Load master table
        if master_table_type == "selected":
            try:
                import os
                selected_path = os.path.join(os.path.dirname(__file__), 'selected_master_table.csv')
                if os.path.exists(selected_path):
                    product_master = pd.read_csv(selected_path)
                    print(f"✅ Loaded selected master table: {len(product_master)} products")
                else:
                    print("⚠️ Selected master table not found, using original")
                    product_master = loader.load_product_master()
            except Exception as e:
                print(f"⚠️ Error loading selected master table: {e}")
                product_master = loader.load_product_master()
        else:
            product_master = loader.load_product_master()
        
        # Calculate analysis period dates
        end_date_calc = pd.Timestamp.now().normalize()
        start_date_calc = None
        
        if analysis_period == 'custom':
            custom_start = request.form.get('start_date')
            custom_end = request.form.get('end_date')
            if custom_start:
                start_date_calc = pd.to_datetime(custom_start)
            if custom_end:
                end_date_calc = pd.to_datetime(custom_end)
        elif analysis_period == 'last_3_months':
            start_date_calc = end_date_calc - pd.DateOffset(months=3)
        elif analysis_period == 'last_6_months':
            start_date_calc = end_date_calc - pd.DateOffset(months=6)
        elif analysis_period == 'last_12_months':
            start_date_calc = end_date_calc - pd.DateOffset(months=12)
        elif analysis_period == 'last_24_months':
            start_date_calc = end_date_calc - pd.DateOffset(months=24)
        
        print(f"📅 Analysis Date Range: {start_date_calc} to {end_date_calc}")
        
        # Filter data by date range
        filtered_data = demand_data.copy()
        filtered_data['date'] = pd.to_datetime(filtered_data['date'])
        
        if start_date_calc:
            filtered_data = filtered_data[filtered_data['date'] >= start_date_calc]
        if end_date_calc:
            filtered_data = filtered_data[filtered_data['date'] <= end_date_calc]
        
        print(f"🔽 After date filtering: {len(filtered_data)} rows")
        
        # Apply other filters similar to insights_apply_standard_filters
        if location_filter and location_filter.strip():
            locations = [loc.strip() for loc in location_filter.split(',') if loc.strip()]
            if locations and 'location_id' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['location_id'].isin(locations)]
                print(f"🔽 After location filter: {len(filtered_data)} rows")
        
        if category_filter and category_filter.strip():
            categories = [cat.strip() for cat in category_filter.split(',') if cat.strip()]
            if categories:
                if 'product_category' in filtered_data.columns:
                    filtered_data = filtered_data[filtered_data['product_category'].isin(categories)]
                    print(f"🔽 After category filter (from demand data): {len(filtered_data)} rows")
                elif not product_master.empty and 'product_category' in product_master.columns:
                    category_products = product_master[product_master['product_category'].isin(categories)]['product_id'].unique()
                    filtered_data = filtered_data[filtered_data['product_id'].isin(category_products)]
                    print(f"🔽 After category filter (from master): {len(filtered_data)} rows")
        
        if product_id_filter:
            filtered_data = filtered_data[filtered_data['product_id'].str.contains(product_id_filter, case=False, na=False)]
            print(f"🔽 After product ID filter: {len(filtered_data)} rows")
        
        if not include_zero_demand:
            filtered_data = filtered_data[filtered_data['demand'] > 0]
            print(f"🔽 After excluding zero demand: {len(filtered_data)} rows")
        
        if filtered_data.empty:
            return jsonify({
                "success": False,
                "error": "No data matches the selected filters"
            })
        
        # Calculate revenue if needed
        revenue_column = None
        if 'unit_price' in filtered_data.columns:
            filtered_data['revenue'] = filtered_data['demand'] * filtered_data['unit_price']
            revenue_column = 'revenue'
        elif 'price' in filtered_data.columns:
            filtered_data['revenue'] = filtered_data['demand'] * filtered_data['price'] 
            revenue_column = 'revenue'
        elif 'revenue' in filtered_data.columns:
            revenue_column = 'revenue'
        
        # Perform demand classification
        group_variables = [col.strip() for col in group_by.split(',')]
        print(f"📊 Running demand classification with group_variables: {group_variables}")
        
        try:
            classification_result = demand_classification(
                filtered_data,
                group_variables,
                time_unit,
                revenue_column,
                start_date_calc.strftime('%Y-%m-%d') if start_date_calc else None,
                end_date_calc.strftime('%Y-%m-%d') if end_date_calc else None
            )
            print(f"📊 Classification completed: {len(classification_result)} products classified")
        except Exception as class_error:
            print(f"❌ Error in demand classification: {class_error}")
            return jsonify({
                "success": False,
                "error": f"Classification error: {str(class_error)}"
            })
        
        # Calculate metrics and generate charts (simplified for now)
        total_unique_products = filtered_data['product_id'].nunique()
        total_skus = len(filtered_data.groupby(['product_id', 'location_id']))
        total_demand = filtered_data['demand'].sum()
        total_revenue = filtered_data[revenue_column].sum() if revenue_column and revenue_column in filtered_data.columns else 0
        
        # Get classification counts
        if 'type' in classification_result.columns:
            classification_counts = classification_result['type'].value_counts()
        else:
            classification_counts = pd.Series()
        
        # Generate classification plot as grouped bar chart (like customer-projects)
        if 'type' in classification_result.columns:
            classification_counts = classification_result['type'].value_counts()
        else:
            classification_counts = pd.Series()
        
        # Calculate additional metrics for the 4-bar chart
        revenue_by_type = None
        inventory_value_by_type = None
        avg_inventory_by_type = None
        
        if not classification_result.empty and not classification_counts.empty:
            # Calculate revenue by classification type
            if revenue_column and revenue_column in filtered_data.columns:
                revenue_data = filtered_data.merge(
                    classification_result[['product_id', 'type']], 
                    on='product_id', 
                    how='inner'
                )
                revenue_by_type = revenue_data.groupby('type')[revenue_column].sum()
                total_revenue_calc = revenue_by_type.sum()
                if total_revenue_calc > 0:
                    revenue_by_type = (revenue_by_type / total_revenue_calc * 100).round(1)
            
            # Generate the proper 4-bar classification chart
            classification_plotly = build_plotly_classification_chart(
                classification_counts, 
                len(classification_result),
                revenue_by_type,
                show_values_with_percent
            )
        else:
            classification_plotly = None
        
        return jsonify({
            "success": True,
            "metrics": {
                "total_unique_products": int(total_unique_products),
                "total_skus": int(total_skus),
                "total_demand": float(total_demand),
                "total_revenue": float(total_revenue)
            },
            "classification_plotly": classification_plotly,
            "classification_summary": {k: int(v) for k, v in classification_counts.to_dict().items()},
        })
        
    except Exception as e:
        import traceback
        print(f"Error in get_demand_classification_plot: {e}")
        print(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": str(e)
        })


@app.route("/insights_filter_options")
def insights_filter_options():
    """Get filter options for insights dashboard"""
    try:
        # Load data using existing loader but with different variable names
        insights_outflow_data = loader.load_outflow()
        
        if insights_outflow_data.empty:
            return jsonify({
                "locations": [],
                "categories": [], 
                "products": []
            })
        
        # Get unique values for insights filters
        insights_locations = sorted(insights_outflow_data["location_id"].dropna().unique().tolist())
        
        # Always get categories from product master table, not from demand data
        try:
            insights_product_master_for_filters = loader.load_product_master()
            if not insights_product_master_for_filters.empty and "product_category" in insights_product_master_for_filters.columns:
                insights_categories = sorted(insights_product_master_for_filters["product_category"].dropna().unique().tolist())
            else:
                insights_categories = []
            print(f"📊 Insights categories loaded from master table: {len(insights_categories)} categories")
        except Exception as insights_cat_error:
            print(f"⚠️ Error loading categories from master table: {insights_cat_error}")
            insights_categories = []
        
        insights_products = sorted(insights_outflow_data["product_id"].dropna().unique().tolist())
        
        return jsonify({
            "locations": insights_locations,
            "categories": insights_categories,
            "products": insights_products
        })
        
    except Exception as insights_error:
        return jsonify({
            "locations": [],
            "categories": [],
            "products": [],
            "error": str(insights_error)
        })


@app.route("/insights_analysis_data", methods=["POST"])
def insights_analysis_data():
    """Generate analysis data for insights dashboard with enhanced filtering"""
    try:
        # Get form data with insights prefix
        insights_analysis_period = request.form.get('analysis_period', 'last_6_months')
        insights_start_date = request.form.get('start_date')
        insights_end_date = request.form.get('end_date')
        insights_location_filter = request.form.get('location_filter', '')
        insights_category_filter = request.form.get('category_filter', '')
        insights_product_id_filter = request.form.get('product_id_filter', '').strip()
        insights_include_zero_demand = request.form.get('include_zero_demand', 'true').lower() == 'true'
        insights_master_table_type = request.form.get('master_table_type', 'original')
        
        print(f"🔍 Insights Analysis Parameters:")
        print(f"  Analysis Period: {insights_analysis_period}")
        print(f"  Location Filter: {insights_location_filter}")
        print(f"  Category Filter: {insights_category_filter}")
        print(f"  Product ID Filter: {insights_product_id_filter}")
        print(f"  Master Table Type: {insights_master_table_type}")
        
        # Load data using existing loader
        print(f"📊 Loading outflow data with loader: {type(loader)}")
        insights_demand_data = loader.load_outflow()
        
        print(f"📊 Insights data loaded: {len(insights_demand_data)} rows")
        if not insights_demand_data.empty:
            print(f"📊 Insights data columns: {insights_demand_data.columns.tolist()}")
            print(f"📊 Insights data date range: {insights_demand_data['date'].min()} to {insights_demand_data['date'].max()}")
            print(f"📊 Insights non-zero demand rows: {len(insights_demand_data[insights_demand_data['demand'] > 0])}")
        else:
            print("❌ Insights data is completely empty after loading!")
        
        # Load master table based on selection for insights
        insights_product_master = insights_load_master_table_by_type(insights_master_table_type)
        
        if insights_demand_data.empty:
            print("❌ Insights demand data is empty!")
            return jsonify({"error": "No demand data available"})
        
        # Calculate analysis period dates for insights
        insights_analysis_start_date, insights_analysis_end_date = insights_calculate_analysis_period_dates(insights_analysis_period, insights_start_date, insights_end_date)
        
        print(f"📅 Insights Analysis Date Range: {insights_analysis_start_date} to {insights_analysis_end_date}")
        
        # Filter data based on insights analysis period
        insights_filtered_data = insights_demand_data.copy()
        insights_filtered_data['date'] = pd.to_datetime(insights_filtered_data['date'])
        
        if insights_analysis_start_date:
            insights_filtered_data = insights_filtered_data[insights_filtered_data['date'] >= insights_analysis_start_date]
            print(f"🔽 After start date filter: {len(insights_filtered_data)} rows")
        
        if insights_analysis_end_date:
            insights_filtered_data = insights_filtered_data[insights_filtered_data['date'] <= insights_analysis_end_date]
            print(f"🔽 After end date filter: {len(insights_filtered_data)} rows")
        
        # Apply advanced filtering for insights
        insights_filtered_data = insights_apply_standard_filters(
            insights_filtered_data, 
            insights_location_filter, 
            insights_category_filter, 
            insights_product_id_filter,
            insights_product_master
        )
        
        print(f"🔽 After all filters: {len(insights_filtered_data)} rows")
        
        # Apply zero demand filter for insights
        if not insights_include_zero_demand:
            insights_filtered_data = insights_filtered_data[insights_filtered_data['demand'] > 0]
            print(f"🔽 After excluding zero demand: {len(insights_filtered_data)} rows")
        
        print(f"🔽 After all filters: {len(insights_filtered_data)} rows")
        if insights_filtered_data.empty:
            print("❌ Insights filtered data is empty after filtering!")
            return jsonify({"error": "No data matches the selected filters"})
        
        # Always calculate OTIF metrics for insights (show all the time)
        insights_include_otif = request.form.get('include_otif', 'true').lower() == 'true'  # Default to true
        insights_otif_x_days = int(request.form.get('otif_x_days', 3))
        
        # Always calculate OTIF metrics regardless of checkbox state
        insights_otif_metrics = insights_calculate_otif_metrics(insights_filtered_data, insights_otif_x_days)
        print(f"📊 OTIF metrics calculated for insights: {insights_otif_metrics}")
        
        # Parse group variables for insights
        insights_group_by = request.form.get('group_by', 'product_id')
        insights_group_variables = [col.strip() for col in insights_group_by.split(',')]
        
        # Get other chart parameters for insights
        insights_time_unit = request.form.get('time_unit', 'day')
        insights_pareto_lines = request.form.get('pareto_lines', '50,80,90')
        insights_show_values = request.form.get('show_values_with_percent', 'true').lower() == 'true'
        
        # Calculate insights metrics
        insights_metrics = insights_calculate_metrics(insights_filtered_data, insights_product_master)
        print(f"📊 Insights calculated metrics: {insights_metrics}")
        
        # Always add OTIF metrics to insights response (even if "Data Not Available")
        insights_metrics.update(insights_otif_metrics)
        print(f"📊 Insights metrics after OTIF update: {insights_metrics}")
        
        # Convert numpy types to Python native types for JSON serialization
        insights_metrics_json = {}
        for insights_key, insights_value in insights_metrics.items():
            if isinstance(insights_value, (np.integer, np.int64, np.int32)):
                insights_metrics_json[insights_key] = int(insights_value)
            elif isinstance(insights_value, (np.floating, np.float64, np.float32)):
                # Handle NaN values properly
                if np.isnan(insights_value):
                    insights_metrics_json[insights_key] = None
                else:
                    insights_metrics_json[insights_key] = float(insights_value)
            elif pd.isna(insights_value) or insights_value != insights_value:  # Check for NaN
                insights_metrics_json[insights_key] = None
            elif isinstance(insights_value, str) and insights_value.lower() == 'nan':
                insights_metrics_json[insights_key] = None
            else:
                insights_metrics_json[insights_key] = insights_value
        
        # Perform demand classification to calculate proper smooth_percentage
        try:
            from forecaster.insights.demand_classification import demand_classification
            
            # Calculate revenue column if needed
            insights_revenue_column = None
            if 'unit_price' in insights_filtered_data.columns:
                insights_filtered_data['revenue'] = insights_filtered_data['demand'] * insights_filtered_data['unit_price']
                insights_revenue_column = 'revenue'
            elif 'price' in insights_filtered_data.columns:
                insights_filtered_data['revenue'] = insights_filtered_data['demand'] * insights_filtered_data['price']
                insights_revenue_column = 'revenue'
            elif 'revenue' in insights_filtered_data.columns:
                insights_revenue_column = 'revenue'
            
            # Perform demand classification
            print(f"📊 Running demand classification for insights...")
            insights_classification_result = demand_classification(
                insights_filtered_data,
                insights_group_variables,
                insights_time_unit,
                insights_revenue_column,
                insights_analysis_start_date.strftime('%Y-%m-%d') if insights_analysis_start_date else None,
                insights_analysis_end_date.strftime('%Y-%m-%d') if insights_analysis_end_date else None
            )
            print(f"📊 Classification completed for insights: {len(insights_classification_result)} products classified")
            
            # Calculate proper smooth percentage based on classification results
            if 'type' in insights_classification_result.columns:
                insights_smooth_count = len(insights_classification_result[insights_classification_result['type'] == 'Smooth'])
                insights_metrics_json['smooth_percentage'] = round((insights_smooth_count / len(insights_classification_result)) * 100, 1)
                print(f"📊 Smooth percentage calculated: {insights_metrics_json['smooth_percentage']}%")
            
            # Generate the 4-bar classification chart
            if not insights_classification_result.empty:
                insights_classification_counts = insights_classification_result['type'].value_counts()
                
                # Calculate revenue by classification type for chart
                insights_revenue_by_type = None
                if insights_revenue_column and insights_revenue_column in insights_filtered_data.columns:
                    insights_revenue_data = insights_filtered_data.merge(
                        insights_classification_result[['product_id', 'type']], 
                        on='product_id', 
                        how='inner'
                    )
                    insights_revenue_by_type = insights_revenue_data.groupby('type', observed=True)[insights_revenue_column].sum()
                    insights_total_revenue_calc = insights_revenue_by_type.sum()
                    if insights_total_revenue_calc > 0:
                        insights_revenue_by_type = (insights_revenue_by_type / insights_total_revenue_calc * 100).round(1)
                
                # Build classification chart
                insights_classification_chart = build_plotly_classification_chart(
                    insights_classification_counts, 
                    len(insights_classification_result),
                    insights_revenue_by_type,
                    insights_show_values
                )
            else:
                insights_classification_chart = None
                
        except Exception as insights_class_error:
            print(f"❌ Error in insights demand classification: {insights_class_error}")
            insights_classification_chart = None
            
        # Generate charts matching customer-projects implementation
        print(f"📊 Generating insights charts...")
        insights_charts = {}
        
        # Generate each chart individually with error handling
        try:
            print(f"📊 Building revenue by category chart...")
            insights_charts['revenue_by_category'] = insights_build_revenue_by_category_chart(insights_filtered_data, insights_product_master)
        except Exception as cat_error:
            print(f"❌ Error building category chart: {cat_error}")
            insights_charts['revenue_by_category'] = None
            
        try:
            print(f"📊 Building revenue by location chart...")
            print(f"📊 Input data shape: {insights_filtered_data.shape}")
            print(f"📊 Input data columns: {insights_filtered_data.columns.tolist()}")
            insights_charts['revenue_by_location'] = insights_build_revenue_by_location_chart(insights_filtered_data)
            if insights_charts['revenue_by_location']:
                print(f"✅ Revenue by location chart generated successfully")
            else:
                print(f"⚠️ Revenue by location chart returned None")
        except Exception as loc_error:
            print(f"❌ Error building location chart: {loc_error}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            insights_charts['revenue_by_location'] = None
            
        try:
            print(f"📊 Building COGS chart...")
            insights_charts['cogs_stock_value'] = insights_build_cogs_stock_value_chart(insights_filtered_data, insights_time_unit)
        except Exception as cogs_error:
            print(f"❌ Error building COGS chart: {cogs_error}")
            insights_charts['cogs_stock_value'] = None
            
        try:
            print(f"📊 Building Pareto revenue chart...")
            insights_charts['pareto_revenue'] = insights_build_pareto_revenue_chart(insights_filtered_data, insights_pareto_lines)
        except Exception as pareto_rev_error:
            print(f"❌ Error building Pareto revenue chart: {pareto_rev_error}")
            insights_charts['pareto_revenue'] = None
            
        try:
            print(f"📊 Building Pareto demand chart...")
            insights_charts['pareto_demand'] = insights_build_pareto_demand_chart(insights_filtered_data, insights_pareto_lines)
        except Exception as pareto_dem_error:
            print(f"❌ Error building Pareto demand chart: {pareto_dem_error}")
            insights_charts['pareto_demand'] = None
            
        insights_charts['classification'] = insights_classification_chart
        
        print(f"📊 Charts generated: {[k for k, v in insights_charts.items() if v is not None]}")
        
        # Generate missing charts from customer-projects version
        insights_leadtime_histogram = insights_build_plotly_leadtime_histogram(insights_product_master)
        insights_ordering_analysis = insights_build_plotly_ordering_analysis(insights_filtered_data)
        
        insights_response_data = {
            "metrics": insights_metrics_json,
            "revenue_by_category_plotly": insights_charts.get('revenue_by_category'),
            "revenue_by_location_plotly": insights_charts.get('revenue_by_location'),
            "cogs_stock_value_plotly": insights_charts.get('cogs_stock_value'),
            "pareto_revenue_plotly": insights_charts.get('pareto_revenue'),
            "pareto_demand_plotly": insights_charts.get('pareto_demand'),
            "classification_plotly": insights_charts.get('classification'),
            "leadtime_histogram_plotly": insights_leadtime_histogram,
            "ordering_analysis_plotly": insights_ordering_analysis
        }
        
        print(f"📊 Final insights response (metrics only): {insights_response_data['metrics']}")
        
        # Double-check for any remaining NaN values before sending
        insights_response_str = str(insights_response_data['metrics'])
        if 'nan' in insights_response_str.lower() or 'NaN' in insights_response_str:
            print(f"⚠️ Warning: Still found NaN in metrics after conversion: {insights_response_str}")
            
        print(f"📊 Insights response data keys: {list(insights_response_data.keys())}")
        
        return jsonify(insights_response_data)
        
    except Exception as insights_error:
        import traceback
        print(f"Error in insights analysis: {insights_error}")
        print(traceback.format_exc())
        return jsonify({"error": str(insights_error)})


@app.route("/get_product_analysis", methods=["POST"])
def get_product_analysis():
    """Get detailed product analysis data for the modal"""
    try:
        print("=== Product Analysis Debug ===")
        
        # Get same form data as main insights
        analysis_period = request.form.get('analysis_period', 'last_6_months')
        group_by = request.form.get('group_by', 'product_id')
        time_unit = request.form.get('time_unit', 'month')
        location_filter = request.form.get('location_filter', '')
        category_filter = request.form.get('category_filter', '')
        product_id_filter = request.form.get('product_id_filter', '').strip()
        include_zero_demand = request.form.get('include_zero_demand', 'true').lower() == 'true'
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        master_table_type = request.form.get('master_table_type', 'original')
        
        print(f"Product Analysis - period={analysis_period}, location={location_filter}, category={category_filter}, master_table={master_table_type}")
        
        # Load data using existing loader
        demand_data = loader.load_outflow()
        
        # Load master table based on selection
        if master_table_type == "selected":
            try:
                import os
                selected_path = os.path.join(os.path.dirname(__file__), 'selected_master_table.csv')
                if os.path.exists(selected_path):
                    product_master = pd.read_csv(selected_path)
                    print(f"✅ Loaded selected master table: {len(product_master)} products")
                else:
                    print("⚠️ Selected master table not found, using original")
                    product_master = loader.load_product_master()
            except Exception as e:
                print(f"⚠️ Error loading selected master table: {e}")
                product_master = loader.load_product_master()
        else:
            product_master = loader.load_product_master()
        
        if demand_data.empty:
            return jsonify({
                "success": False,
                "error": "No demand data available"
            })
        
        # Calculate analysis period dates
        end_date_calc = pd.Timestamp.now().normalize()
        start_date_calc = None
        
        if analysis_period == 'custom':
            if start_date:
                start_date_calc = pd.to_datetime(start_date)
            if end_date:
                end_date_calc = pd.to_datetime(end_date)
        elif analysis_period == 'last_3_months':
            start_date_calc = end_date_calc - pd.DateOffset(months=3)
        elif analysis_period == 'last_6_months':
            start_date_calc = end_date_calc - pd.DateOffset(months=6)
        elif analysis_period == 'last_12_months':
            start_date_calc = end_date_calc - pd.DateOffset(months=12)
        elif analysis_period == 'last_24_months':
            start_date_calc = end_date_calc - pd.DateOffset(months=24)
        
        # Filter data based on analysis period
        filtered_data = demand_data.copy()
        filtered_data['date'] = pd.to_datetime(filtered_data['date'])
        
        if start_date_calc:
            filtered_data = filtered_data[filtered_data['date'] >= start_date_calc]
        
        if end_date_calc:
            filtered_data = filtered_data[filtered_data['date'] <= end_date_calc]
        
        # Apply advanced filtering
        if location_filter and location_filter.strip():
            locations = [loc.strip() for loc in location_filter.split(',') if loc.strip()]
            if locations and 'location_id' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['location_id'].isin(locations)]
        
        if category_filter and category_filter.strip():
            categories = [cat.strip() for cat in category_filter.split(',') if cat.strip()]
            if categories:
                if 'product_category' in filtered_data.columns:
                    filtered_data = filtered_data[filtered_data['product_category'].isin(categories)]
                elif not product_master.empty and 'product_category' in product_master.columns:
                    category_products = product_master[product_master['product_category'].isin(categories)]['product_id'].unique()
                    filtered_data = filtered_data[filtered_data['product_id'].isin(category_products)]
        
        if product_id_filter:
            filtered_data = filtered_data[filtered_data['product_id'].str.contains(product_id_filter, case=False, na=False)]
        
        # Apply zero demand filter
        if not include_zero_demand:
            filtered_data = filtered_data[filtered_data['demand'] > 0]
        
        if filtered_data.empty:
            return jsonify({
                "success": False,
                "error": "No data matches the selected filters"
            })
        
        # Calculate product-level summary
        agg_dict = {'demand': 'sum'}
        if 'revenue' in filtered_data.columns:
            agg_dict['revenue'] = 'sum'
        elif 'unit_price' in filtered_data.columns:
            # Calculate revenue from demand * unit_price
            filtered_data['revenue'] = filtered_data['demand'] * filtered_data['unit_price']
            agg_dict['revenue'] = 'sum'
        
        product_summary = filtered_data.groupby('product_id').agg(agg_dict).reset_index()
        
        # Calculate total revenue
        total_revenue = product_summary['revenue'].sum() if 'revenue' in product_summary.columns else 0
        
        # Convert to list of products
        products_list = []
        for _, product in product_summary.iterrows():
            products_list.append({
                'product_id': product['product_id'],
                'total_demand': float(product['demand']) if pd.notna(product['demand']) else 0,
                'total_revenue': float(product['revenue']) if 'revenue' in product and pd.notna(product['revenue']) else 0
            })
        
        return jsonify({
            "success": True,
            "total_revenue": float(total_revenue),
            "products": products_list,
            "product_count": len(products_list)
        })
        
    except Exception as error:
        import traceback
        print(f"Error in product analysis: {error}")
        print(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": str(error)
        })


@app.route("/insights_product_analysis", methods=["POST"])
def insights_product_analysis():
    """Get detailed product analysis data for the insights modal"""
    try:
        print("=== Insights Product Analysis Debug ===")
        
        # Get form data with insights prefix
        insights_analysis_period = request.form.get('analysis_period', 'last_6_months')
        insights_group_by = request.form.get('group_by', 'product_id')
        insights_time_unit = request.form.get('time_unit', 'month')
        insights_location_filter = request.form.get('location_filter', '')
        insights_category_filter = request.form.get('category_filter', '')
        insights_product_id_filter = request.form.get('product_id_filter', '').strip()
        insights_include_zero_demand = request.form.get('include_zero_demand', 'true').lower() == 'true'
        insights_start_date = request.form.get('start_date')
        insights_end_date = request.form.get('end_date')
        insights_master_table_type = request.form.get('master_table_type', 'original')
        
        print(f"Insights Product Analysis - period={insights_analysis_period}, location={insights_location_filter}, category={insights_category_filter}, master_table={insights_master_table_type}")
        
        # Load data using existing loader
        insights_demand_data = loader.load_outflow()
        
        # Load master table based on selection for insights
        insights_product_master = insights_load_master_table_by_type(insights_master_table_type)
        
        if insights_demand_data.empty:
            return jsonify({
                "success": False,
                "error": "No demand data available"
            })
        
        # Calculate analysis period dates for insights
        insights_analysis_start_date, insights_analysis_end_date = insights_calculate_analysis_period_dates(insights_analysis_period, insights_start_date, insights_end_date)
        
        # Filter data based on insights analysis period
        insights_filtered_data = insights_demand_data.copy()
        insights_filtered_data['date'] = pd.to_datetime(insights_filtered_data['date'])
        
        if insights_analysis_start_date:
            insights_filtered_data = insights_filtered_data[insights_filtered_data['date'] >= insights_analysis_start_date]
        
        if insights_analysis_end_date:
            insights_filtered_data = insights_filtered_data[insights_filtered_data['date'] <= insights_analysis_end_date]
        
        # Apply advanced filtering for insights
        insights_filtered_data = insights_apply_standard_filters(
            insights_filtered_data, 
            insights_location_filter, 
            insights_category_filter, 
            insights_product_id_filter,
            insights_product_master
        )
        
        # Apply zero demand filter for insights
        if not insights_include_zero_demand:
            insights_filtered_data = insights_filtered_data[insights_filtered_data['demand'] > 0]
        
        if insights_filtered_data.empty:
            return jsonify({
                "success": False,
                "error": "No data matches the selected filters"
            })
        
        # Calculate product-level summary for insights
        insights_agg_dict = {'demand': 'sum'}
        if 'revenue' in insights_filtered_data.columns:
            insights_agg_dict['revenue'] = 'sum'
        elif 'unit_price' in insights_filtered_data.columns:
            # Calculate revenue from demand * unit_price
            insights_filtered_data['revenue'] = insights_filtered_data['demand'] * insights_filtered_data['unit_price']
            insights_agg_dict['revenue'] = 'sum'
        
        insights_product_summary = insights_filtered_data.groupby('product_id').agg(insights_agg_dict).reset_index()
        
        # Calculate total revenue for insights
        insights_total_revenue = insights_product_summary['revenue'].sum() if 'revenue' in insights_product_summary.columns else 0
        
        # Convert to list of products for insights
        insights_products_list = []
        for _, insights_product in insights_product_summary.iterrows():
            insights_products_list.append({
                'product_id': insights_product['product_id'],
                'total_demand': float(insights_product['demand']) if pd.notna(insights_product['demand']) else 0,
                'total_revenue': float(insights_product['revenue']) if 'revenue' in insights_product and pd.notna(insights_product['revenue']) else 0
            })
        
        return jsonify({
            "success": True,
            "total_revenue": float(insights_total_revenue),
            "products": insights_products_list,
            "product_count": len(insights_products_list)
        })
        
    except Exception as insights_error:
        import traceback
        print(f"Error in insights product analysis: {insights_error}")
        print(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": str(insights_error)
        })


def insights_build_plotly_leadtime_histogram(insights_product_master_data):
    """Build simple bar chart showing product distribution by leadtime from product master."""
    import plotly.graph_objs as go
    
    try:
        if insights_product_master_data is None or insights_product_master_data.empty or 'leadtime' not in insights_product_master_data.columns:
            return None
        
        # Get leadtime data from product master
        insights_leadtimes = insights_product_master_data['leadtime'].dropna()
        insights_leadtimes = pd.to_numeric(insights_leadtimes, errors='coerce').dropna()
        insights_leadtimes = insights_leadtimes[insights_leadtimes > 0]
        
        if len(insights_leadtimes) == 0:
            return None
        
        # Round to integers and count products by leadtime
        insights_leadtimes_rounded = insights_leadtimes.round().astype(int)
        insights_leadtime_counts = insights_leadtimes_rounded.value_counts().sort_index()
        
        # Create bar chart
        insights_fig = go.Figure()
        
        insights_fig.add_trace(go.Bar(
            x=[f"{int(x)} days" for x in insights_leadtime_counts.index.tolist()],
            y=insights_leadtime_counts.values.tolist(),
            marker_color='rgba(54, 162, 235, 0.8)',
            hovertemplate='<b>Lead Time: %{x}</b><br>Products: %{y}<extra></extra>'
        ))
        
        # Update layout
        insights_fig.update_layout(
            xaxis_title='Lead Time (days)',
            yaxis_title='Number of Products',
            height=380,
            showlegend=False,
            template='plotly_white',
            margin=dict(t=20, b=50, l=50, r=20),
            xaxis=dict(type='category'),
            bargap=0.2
        )
        
        return insights_fig.to_json()
    except Exception as insights_error:
        print(f"Error in insights leadtime histogram: {insights_error}")
        return None


def insights_analyze_ordering_patterns(insights_demand_data):
    """Analyze ordering patterns using actual incoming_inventory data with ARI and CV calculations like customer-projects"""
    try:
        if insights_demand_data.empty or 'incoming_inventory' not in insights_demand_data.columns:
            return pd.DataFrame()
        
        # Filter to only receipts (incoming_inventory > 0) and create receipt data
        insights_receipts = insights_demand_data[insights_demand_data['incoming_inventory'] > 0].copy()
        
        if insights_receipts.empty:
            return pd.DataFrame()
        
        # Ensure date column is datetime
        insights_receipts['date'] = pd.to_datetime(insights_receipts['date'])
        
        # Create weekly buckets for ARI calculation 
        INSIGHTS_BUCKET = "W"  # Weekly buckets
        insights_receipts["bucket"] = insights_receipts["date"].dt.to_period(INSIGHTS_BUCKET).dt.start_time
        
        def insights_supply_metrics(insights_group):
            """Calculate supply metrics for each product like customer-projects"""
            # Number of buckets in history (T)
            insights_T = insights_group["bucket"].nunique()
            # Quantities of each receipt
            insights_qtys = insights_group["incoming_inventory"].values
            insights_m = len(insights_qtys)

            # Average Receipt Interval (ARI) – use inf if m==0 to avoid div-zero
            insights_ARI = insights_T / insights_m if insights_m else np.inf

            # CV and CV² of receipt quantities
            insights_mu_q = np.mean(insights_qtys) if insights_m else 0
            insights_sigma_q = np.std(insights_qtys, ddof=1) if insights_m > 1 else 0
            insights_CV_q = insights_sigma_q / insights_mu_q if insights_mu_q else 0
            insights_CV2_q = insights_CV_q ** 2

            # Lead-time reliability (simulated)
            insights_CV_lt = np.random.uniform(0, 0.2)  # Simulate CV of lead time

            return pd.Series(
                dict(T=insights_T, m=insights_m, ARI=insights_ARI, CV2_qty=insights_CV2_q, CV_LT=insights_CV_lt,
                     total_inbound=insights_qtys.sum())
            )

        # Group by product and calculate metrics
        insights_sku_stats = insights_receipts.groupby("product_id").apply(insights_supply_metrics, include_groups=False)

        def insights_classify(insights_row):
            """Classify supply patterns like customer-projects"""
            if insights_row.m == 0:
                return "No Receipts"
            if insights_row.m == 1:
                return "One-Off Receipt"

            insights_timing_regular = insights_row.ARI <= 1.32
            insights_qty_stable = insights_row.CV2_qty <= 0.49
            insights_lt_reliable = (insights_row.CV_LT <= 0.10) if pd.notna(insights_row.CV_LT) else True

            if insights_timing_regular and insights_qty_stable and insights_lt_reliable:
                return "Regular"
            if insights_timing_regular and not insights_qty_stable:
                return "Regular Variable"
            if not insights_timing_regular and insights_qty_stable:
                return "Intermittent Supply"
            return "Sporadic & Variable"

        insights_sku_stats["supply_class"] = insights_sku_stats.apply(insights_classify, axis=1)

        # Build insight table like customer-projects
        insights_summary = (
            insights_sku_stats
            .groupby("supply_class")
            .agg(
                skus=("supply_class", "size"),
                inbound=("total_inbound", "sum")
            )
            .sort_values("inbound", ascending=False)
        )

        insights_summary["percent_skus"] = insights_summary.skus / insights_summary.skus.sum() * 100
        insights_summary["percent_inbound"] = insights_summary.inbound / insights_summary.inbound.sum() * 100
        insights_summary = insights_summary.reset_index()
        
        return insights_summary
        
    except Exception as insights_error:
        print(f"Error analyzing ordering patterns for insights (ARI method): {insights_error}")
        return pd.DataFrame()


def insights_build_plotly_ordering_analysis(insights_demand_data):
    """Generate Plotly ordering analysis chart for insights."""
    import plotly.graph_objs as go
    import json
    
    try:
        insights_summary = insights_analyze_ordering_patterns(insights_demand_data)
        
        if insights_summary.empty or len(insights_summary) == 0:
            # Create a placeholder chart with "No Data" message
            insights_fig = go.Figure()
            insights_fig.add_annotation(
                text="No supply pattern data available<br>Try different filters or date range",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False,
                font=dict(size=16, color="gray")
            )
            insights_fig.update_layout(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                plot_bgcolor='white',
                paper_bgcolor='white',
                height=380,
                margin=dict(t=20, b=20, l=20, r=20)
            )
            return insights_fig.to_json()
        
        insights_fig = go.Figure()
        
        # Ensure we have proper data
        insights_categories = insights_summary["supply_class"].tolist()
        insights_inbound_pct = insights_summary["percent_inbound"].tolist()
        insights_skus_pct = insights_summary["percent_skus"].tolist()
        
        # Add bars for percent of inbound volume
        insights_fig.add_trace(go.Bar(
            x=insights_categories,
            y=insights_inbound_pct,
            name="% of Inbound Volume",
            hovertemplate='<b>%{x}</b><br>% of Inbound Volume: %{y:.1f}%<extra></extra>',
            text=[f"{p:.2f}%" for p in insights_inbound_pct],
            textposition='outside',
            marker_color='#3D315A',
            offsetgroup=1
        ))
        
        # Add bars for percent of SKUs  
        insights_fig.add_trace(go.Bar(
            x=insights_categories,
            y=insights_skus_pct,
            name="% of SKUs",
            hovertemplate='<b>%{x}</b><br>% of SKUs: %{y:.1f}%<extra></extra>',
            text=[f"{p:.2f}%" for p in insights_skus_pct],
            textposition='outside',
            marker_color='#807368',
            offsetgroup=2
        ))
        
        insights_max_value = max(max(insights_inbound_pct), max(insights_skus_pct)) if insights_inbound_pct and insights_skus_pct else 100
        
        insights_fig.update_layout(
            template='plotly_white',
            # xaxis_title='Supply Pattern Classification',
            yaxis_title='Percentage (%)',
            height=380,
            barmode='group',
            legend=dict(orientation='h', yanchor='bottom', y=-0.15, xanchor='center', x=0.5),
            xaxis=dict(tickangle=0),
            yaxis=dict(
                title='Percentage (%)',
                ticksuffix='%',
                range=[0, insights_max_value * 1.1]
            ),
            margin=dict(l=40, r=20, t=20, b=60)
        )
        
        # Add grid
        insights_fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        return insights_fig.to_json()
        
    except Exception as insights_error:
        print(f"Error in insights ordering analysis: {insights_error}")
        return json.dumps({})


def format_number_with_suffix(value):
    """Format number with K/M/B suffix"""
    if value >= 1_000_000_000:
        return f"{value/1_000_000_000:.1f}B"
    elif value >= 1_000_000:
        return f"{value/1_000_000:.1f}M"
    elif value >= 1_000:
        return f"{value/1_000:.1f}K"
    else:
        return f"{value:,.0f}"


def insights_build_revenue_by_category_chart(insights_data, insights_product_master):
    """Build revenue by category pie chart for insights"""
    import plotly.graph_objs as go
    
    try:
        if insights_data.empty:
            return None
        
        # Calculate revenue if needed
        if 'revenue' not in insights_data.columns:
            if 'unit_price' in insights_data.columns:
                insights_data['revenue'] = insights_data['demand'] * insights_data['unit_price']
            else:
                return None
        
        # Try different ways to get product categories
        insights_category_revenue = None
        
        # First try: Use category from demand data directly if available
        if 'product_category' in insights_data.columns:
            print(f"📊 Using product_category from demand data")
            print(f"📊 Unique categories in data: {insights_data['product_category'].nunique()}")
            print(f"📊 Sample categories: {insights_data['product_category'].unique()[:10]}")
            
            # Remove rows with NaN or empty categories
            insights_data_clean = insights_data.dropna(subset=['product_category'])
            print(f"📊 After removing NaN categories: {len(insights_data_clean)} rows")
            
            # Check for empty strings
            insights_data_clean = insights_data_clean[insights_data_clean['product_category'].str.strip() != '']
            print(f"📊 After removing empty categories: {len(insights_data_clean)} rows")
            
            if not insights_data_clean.empty:
                try:
                    insights_category_revenue = insights_data_clean.groupby('product_category', observed=True)['revenue'].sum().sort_values(ascending=False)
                    print(f"📊 Category revenue calculated: {len(insights_category_revenue)} categories with revenue")
                    print(f"📊 Top categories: {dict(insights_category_revenue.head())}")
                except Exception as cat_error:
                    print(f"📊 Error calculating category revenue: {cat_error}")
                    insights_category_revenue = None
            else:
                print(f"📊 No valid categories found after cleaning")
                insights_category_revenue = None
        
        # Second try: Merge with product master to get categories
        elif not insights_product_master.empty and 'product_category' in insights_product_master.columns:
            print(f"📊 Merging with product master for categories")
            try:
                # Merge on both product_id and location_id if available
                if 'location_id' in insights_data.columns and 'location_id' in insights_product_master.columns:
                    insights_merged = insights_data.merge(
                        insights_product_master[['product_id', 'location_id', 'product_category']], 
                        on=['product_id', 'location_id'], 
                        how='left'
                    )
                else:
                    # Fallback to just product_id
                    insights_merged = insights_data.merge(
                        insights_product_master[['product_id', 'product_category']], 
                        on='product_id', 
                        how='left'
                    )
                
                # Check if merge was successful
                if 'product_category' in insights_merged.columns:
                    # Remove rows where category is null
                    insights_merged = insights_merged.dropna(subset=['product_category'])
                    if not insights_merged.empty:
                        insights_category_revenue = insights_merged.groupby('product_category', observed=True)['revenue'].sum().sort_values(ascending=False)
                        print(f"📊 Successfully merged categories: {len(insights_category_revenue)} categories found")
                    else:
                        print(f"📊 No valid categories found after merge")
                else:
                    print(f"📊 Merge failed - no product_category column in result")
            except Exception as merge_error:
                print(f"📊 Error merging with product master: {merge_error}")
        else:
            print(f"📊 No product_category available in data or master")
        
        # If we still don't have category revenue, return None
        if insights_category_revenue is None or insights_category_revenue.empty:
            return None
        
        # Create pie chart
        insights_fig = go.Figure(data=[go.Pie(
            labels=insights_category_revenue.index.tolist(),
            values=insights_category_revenue.values.tolist(),
            textinfo='label+percent',
            textposition='auto',
            hovertemplate='<b>%{label}</b><br>Revenue: $%{value:,.0f}<br>Percentage: %{percent}<extra></extra>'
        )])
        
        insights_fig.update_layout(
            template='plotly_white',
            height=380,
            margin=dict(t=20, b=20, l=20, r=20),
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.01
            )
        )
        
        return insights_fig.to_json()
        
    except Exception as insights_error:
        print(f"Error building revenue by category chart for insights: {insights_error}")
        return None


def insights_build_revenue_by_location_chart(insights_data):
    """Build revenue by location pie chart for insights"""
    import plotly.graph_objs as go
    
    try:
        if insights_data.empty or 'location_id' not in insights_data.columns:
            print(f"📊 Location chart: Data is empty or no location_id column")
            return None
        
        print(f"📊 Location chart: Processing {len(insights_data)} rows")
        print(f"📊 Available columns: {insights_data.columns.tolist()}")
        print(f"📊 Unique locations: {insights_data['location_id'].nunique()}")
        
        # Calculate revenue if needed
        if 'revenue' not in insights_data.columns:
            if 'unit_price' in insights_data.columns:
                insights_data['revenue'] = insights_data['demand'] * insights_data['unit_price']
                print(f"📊 Location chart: Calculated revenue from demand * unit_price")
            elif 'price' in insights_data.columns:
                insights_data['revenue'] = insights_data['demand'] * insights_data['price']
                print(f"📊 Location chart: Calculated revenue from demand * price")
            else:
                print(f"📊 Location chart: No revenue or price columns available")
                # Try to proceed with demand instead of revenue
                insights_location_revenue = insights_data.groupby('location_id', observed=True)['demand'].sum().sort_values(ascending=False)
                chart_title = 'Demand by Location'
                value_label = 'Demand'
        else:
            print(f"📊 Location chart: Using existing revenue column")
            insights_location_revenue = insights_data.groupby('location_id', observed=True)['revenue'].sum().sort_values(ascending=False)
            chart_title = 'Revenue by Location'
            value_label = 'Revenue'
        
        # If we haven't set the location revenue yet, calculate it
        if 'insights_location_revenue' not in locals():
            if 'revenue' in insights_data.columns:
                insights_location_revenue = insights_data.groupby('location_id', observed=True)['revenue'].sum().sort_values(ascending=False)
                chart_title = 'Revenue by Location'
                value_label = 'Revenue'
            else:
                print(f"📊 Location chart: Falling back to demand")
                insights_location_revenue = insights_data.groupby('location_id', observed=True)['demand'].sum().sort_values(ascending=False)
                chart_title = 'Demand by Location'
                value_label = 'Demand'
        
        print(f"📊 Location chart: Calculated {chart_title} with {len(insights_location_revenue)} locations")
        print(f"📊 Top locations: {dict(insights_location_revenue.head())}")
        
        if insights_location_revenue.empty:
            print(f"📊 Location chart: No data after grouping")
            return None
        
        # Create pie chart
        insights_fig = go.Figure(data=[go.Pie(
            labels=insights_location_revenue.index.tolist(),
            values=insights_location_revenue.values.tolist(),
            textinfo='label+percent',
            textposition='auto',
            hovertemplate=f'<b>%{{label}}</b><br>{value_label}: %{{value:,.0f}}<br>Percentage: %{{percent}}<extra></extra>'
        )])
        
        insights_fig.update_layout(
            template='plotly_white',
            height=380,
            margin=dict(t=20, b=20, l=20, r=20),
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.01
            )
        )
        
        print(f"📊 Location chart: Successfully created")
        return insights_fig.to_json()
        
    except Exception as insights_error:
        print(f"❌ Error building revenue by location chart for insights: {insights_error}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return None


def insights_build_cogs_stock_value_chart(insights_data, insights_time_unit):
    """Build COGS and stock value trends chart for insights"""
    import plotly.graph_objs as go
    
    try:
        if insights_data.empty or 'date' not in insights_data.columns:
            return None
        
        # Ensure date column
        insights_data['date'] = pd.to_datetime(insights_data['date'])
        
        # Calculate revenue if needed
        if 'revenue' not in insights_data.columns:
            if 'unit_price' in insights_data.columns:
                insights_data['revenue'] = insights_data['demand'] * insights_data['unit_price']
            else:
                return None
        
        # Group by date and calculate trends
        insights_daily_trends = insights_data.groupby('date', observed=True).agg({
            'revenue': 'sum',
            'stock_level': 'sum' if 'stock_level' in insights_data.columns else lambda x: 0
        }).reset_index()
        
        if insights_daily_trends.empty:
            return None
        
        # Create line chart
        insights_fig = go.Figure()
        
        # Revenue trend
        insights_fig.add_trace(go.Scatter(
            x=insights_daily_trends['date'],
            y=insights_daily_trends['revenue'],
            mode='lines',
            name='Revenue',
            line=dict(color='#de6a45', width=2),
            yaxis='y'
        ))
        
        # Stock level trend (if available)
        if 'stock_level' in insights_data.columns:
            insights_fig.add_trace(go.Scatter(
                x=insights_daily_trends['date'],
                y=insights_daily_trends['stock_level'],
                mode='lines',
                name='Stock Level',
                line=dict(color='#3d315a', width=2),
                yaxis='y2'
            ))
        
        insights_fig.update_layout(
            template='plotly_white',
            height=380,
            margin=dict(t=20, b=40, l=40, r=40),
            xaxis_title='Date',
            yaxis=dict(
                title='Revenue ($)',
                side='left'
            ),
            yaxis2=dict(
                title='Stock Level',
                side='right',
                overlaying='y'
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )
        
        return insights_fig.to_json()
        
    except Exception as insights_error:
        print(f"Error building COGS stock value chart for insights: {insights_error}")
        return None


def insights_build_pareto_revenue_chart(insights_data, insights_pareto_lines):
    """Build Pareto revenue analysis chart for insights"""
    import plotly.graph_objs as go
    
    try:
        if insights_data.empty:
            return None
        
        # Calculate revenue if needed
        if 'revenue' not in insights_data.columns:
            if 'unit_price' in insights_data.columns:
                insights_data['revenue'] = insights_data['demand'] * insights_data['unit_price']
            else:
                return None
        
        # Group by product and sum revenue
        insights_product_revenue = insights_data.groupby('product_id', observed=True)['revenue'].sum().sort_values(ascending=False)
        
        if insights_product_revenue.empty:
            return None
        
        # Calculate cumulative percentage
        insights_cumulative_pct = (insights_product_revenue.cumsum() / insights_product_revenue.sum() * 100)
        
        # Create Pareto chart
        insights_fig = go.Figure()
        
        # Bar chart for revenue
        insights_fig.add_trace(go.Bar(
            x=list(range(len(insights_product_revenue))),
            y=insights_product_revenue.values,
            name='Revenue',
            yaxis='y',
            marker_color='#de6a45'
        ))
        
        # Line chart for cumulative percentage
        insights_fig.add_trace(go.Scatter(
            x=list(range(len(insights_cumulative_pct))),
            y=insights_cumulative_pct.values,
            mode='lines',
            name='Cumulative %',
            yaxis='y2',
            line=dict(color='#3d315a', width=2)
        ))
        
        # Add Pareto lines
        insights_pareto_values = [int(x.strip()) for x in insights_pareto_lines.split(',') if x.strip().isdigit()]
        for insights_line in insights_pareto_values:
            insights_fig.add_hline(y=insights_line, line_dash="dash", line_color="red", 
                                 annotation_text=f"{insights_line}%", yref='y2')
        
        insights_fig.update_layout(
            template='plotly_white',
            height=380,
            margin=dict(t=20, b=40, l=40, r=40),
            xaxis_title='Products (ranked by revenue)',
            yaxis=dict(
                title='Revenue ($)',
                side='left'
            ),
            yaxis2=dict(
                title='Cumulative Percentage (%)',
                side='right',
                overlaying='y',
                range=[0, 100]
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )
        
        return insights_fig.to_json()
        
    except Exception as insights_error:
        print(f"Error building Pareto revenue chart for insights: {insights_error}")
        return None


def insights_build_pareto_demand_chart(insights_data, insights_pareto_lines):
    """Build Pareto demand analysis chart for insights"""
    import plotly.graph_objs as go
    
    try:
        if insights_data.empty:
            return None
        
        # Group by product and sum demand
        insights_product_demand = insights_data.groupby('product_id', observed=True)['demand'].sum().sort_values(ascending=False)
        
        if insights_product_demand.empty:
            return None
        
        # Calculate cumulative percentage
        insights_cumulative_pct = (insights_product_demand.cumsum() / insights_product_demand.sum() * 100)
        
        # Create Pareto chart
        insights_fig = go.Figure()
        
        # Bar chart for demand
        insights_fig.add_trace(go.Bar(
            x=list(range(len(insights_product_demand))),
            y=insights_product_demand.values,
            name='Demand',
            yaxis='y',
            marker_color='#2a44d4'
        ))
        
        # Line chart for cumulative percentage
        insights_fig.add_trace(go.Scatter(
            x=list(range(len(insights_cumulative_pct))),
            y=insights_cumulative_pct.values,
            mode='lines',
            name='Cumulative %',
            yaxis='y2',
            line=dict(color='#3d315a', width=2)
        ))
        
        # Add Pareto lines
        insights_pareto_values = [int(x.strip()) for x in insights_pareto_lines.split(',') if x.strip().isdigit()]
        for insights_line in insights_pareto_values:
            insights_fig.add_hline(y=insights_line, line_dash="dash", line_color="red", 
                                 annotation_text=f"{insights_line}%", yref='y2')
        
        insights_fig.update_layout(
            template='plotly_white',
            height=380,
            margin=dict(t=20, b=40, l=40, r=40),
            xaxis_title='Products (ranked by demand)',
            yaxis=dict(
                title='Demand',
                side='left'
            ),
            yaxis2=dict(
                title='Cumulative Percentage (%)',
                side='right',
                overlaying='y',
                range=[0, 100]
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )
        
        return insights_fig.to_json()
        
    except Exception as insights_error:
        print(f"Error building Pareto demand chart for insights: {insights_error}")
        return None


def build_plotly_classification_chart(class_counts, total_products, revenue_by_type=None, show_values_with_percent=True):
    """Build Plotly JSON grouped bar chart with 4 bars like customer-projects version"""
    import plotly.graph_objs as go
    
    if class_counts is None or len(class_counts) == 0:
        return None
    
    categories = list(class_counts.index)
    values = list(class_counts.values)
    product_percentages = [round((v/total_products)*100, 1) if total_products else 0 for v in values]
    
    fig = go.Figure()
    
    # Bar 1: % of SKUs
    fig.add_trace(go.Bar(
        name='% of SKUs',
        x=categories,
        y=product_percentages,
        text=[f"{p:.1f}%<br>({format_number_with_suffix(v)})" if show_values_with_percent else f"{p:.1f}%" for p, v in zip(product_percentages, values)],
        textposition='outside',
        marker_color='#3D315A',
        width=0.15,
        hovertemplate='<b>%{x}</b><br>' +
                        'SKU Percentage: %{y:.1f}%<br>' +
                        'Number of SKUs: %{customdata:,}<br>' +
                        '<extra></extra>',
        customdata=values,
        offsetgroup=1
    ))
    
    # Bar 2: Revenue Percentage (if available)
    if revenue_by_type is not None and not revenue_by_type.empty:
        revenue_percentages = []
        for cat in categories:
            if cat in revenue_by_type.index:
                revenue_percentages.append(revenue_by_type[cat])
            else:
                revenue_percentages.append(0)
        
        fig.add_trace(go.Bar(
            name='% of Revenue',
            x=categories,
            y=revenue_percentages,
            text=[f"{p:.1f}%" for p in revenue_percentages] if show_values_with_percent else '',
            textposition='outside',
            marker_color='#DE6A45',
            width=0.15,
            hovertemplate='<b>%{x}</b><br>' +
                            'Revenue Percentage: %{y:.1f}%<br>' +
                            '<extra></extra>',
            offsetgroup=2
        ))
    
    # Bar 3: Current Inventory Value % (placeholder - using same as SKUs for now)
    fig.add_trace(go.Bar(
        name='% of Current Inventory Value',
        x=categories,
        y=product_percentages,  # Using same values as placeholder
        text=[f"{p:.1f}%" for p in product_percentages] if show_values_with_percent else '',
        textposition='outside',
        marker_color='#807368',
        width=0.15,
        hovertemplate='<b>%{x}</b><br>' +
                        'Current Inventory Value %: %{y:.1f}%<br>' +
                        '<extra></extra>',
        offsetgroup=3
    ))
    
    # Bar 4: Avg Inventory Value % (placeholder - using same as SKUs for now)
    fig.add_trace(go.Bar(
        name='% of Avg Inventory Value',
        x=categories,
        y=product_percentages,  # Using same values as placeholder
        text=[f"{p:.1f}%" for p in product_percentages] if show_values_with_percent else '',
        textposition='outside',
        marker_color='#CFC8C2',
        width=0.15,
        hovertemplate='<b>%{x}</b><br>' +
                        'Avg Inventory Value %: %{y:.1f}%<br>' +
                        '<extra></extra>',
        offsetgroup=4
    ))

    # Update layout
    fig.update_layout(
        template='plotly_white',
        # xaxis_title='Demand Classification Type',
        yaxis_title='Percentage (%)',
        height=900,  # Match the CSS height expectation
        barmode='group',
        bargap=0.2,
        bargroupgap=0.1,
        legend=dict(orientation='h', yanchor='bottom', y=-0.05, xanchor='center', x=0.5),  # Position legend below chart
        xaxis=dict(tickangle=0),
        yaxis=dict(
            title='Percentage (%)',
            ticksuffix='%',
            range=[0, 100]
        ),
        margin=dict(t=10, b=100, l=60, r=20),  # Consistent margins with frontend
        autosize=True
    )
    
    return fig.to_json()


def insights_calculate_metrics(insights_data, insights_product_master):
    """Calculate KPI metrics for insights dashboard with comprehensive analysis"""
    try:
        insights_metrics = {}
        
        # Basic business metrics for insights
        insights_metrics['total_unique_products'] = insights_data['product_id'].nunique()
        insights_metrics['total_skus'] = len(insights_data.groupby(['product_id', 'location_id']))
        insights_metrics['total_demand'] = insights_data['demand'].sum()
        
        # Calculate revenue for insights
        if 'unit_price' in insights_data.columns:
            insights_data['revenue'] = insights_data['demand'] * insights_data['unit_price']
            insights_metrics['total_revenue'] = insights_data['revenue'].sum()
        elif 'price' in insights_data.columns:
            insights_data['revenue'] = insights_data['demand'] * insights_data['price'] 
            insights_metrics['total_revenue'] = insights_data['revenue'].sum()
        elif 'revenue' in insights_data.columns:
            insights_metrics['total_revenue'] = insights_data['revenue'].sum()
        else:
            insights_metrics['total_revenue'] = 0
        
        # Calculate mean margin percentage for insights (weighted by revenue like customer-projects)
        if 'margin_pct' in insights_data.columns and 'revenue' in insights_data.columns:
            # Calculate weighted average margin by product (weighted by revenue)
            insights_product_margins = []
            for insights_product_id in insights_data['product_id'].unique():
                insights_product_data = insights_data[insights_data['product_id'] == insights_product_id]
                if len(insights_product_data) > 0 and insights_product_data['revenue'].sum() > 0:
                    # Weighted margin = sum(margin * revenue) / sum(revenue)
                    insights_weighted_margin = (insights_product_data['margin_pct'] * insights_product_data['revenue']).sum() / insights_product_data['revenue'].sum()
                    insights_product_margins.append(insights_weighted_margin)
            
            if insights_product_margins:
                insights_metrics['mean_margin_pct'] = round(np.mean(insights_product_margins), 1)
            else:
                insights_metrics['mean_margin_pct'] = 0.0
        elif 'margin_pct' in insights_data.columns:
            # Fallback to simple mean if no revenue data
            insights_metrics['mean_margin_pct'] = insights_data['margin_pct'].mean()
        else:
            # Cannot calculate margin without required data
            insights_metrics['mean_margin_pct'] = 'N/A - Missing cost/price data'
        
        # Calculate smooth percentage for insights (from classification results - NOT implemented here)
        # This will be calculated in insights_analysis_data using actual classification results
        insights_metrics['smooth_percentage'] = 0  # Placeholder, calculated elsewhere
        
        # Enhanced inventory metrics for insights
        if 'stock_value' in insights_data.columns:
            insights_current_holding = insights_data['stock_value'].iloc[-1] if not insights_data.empty else 0
            insights_avg_holding = insights_data['stock_value'].mean()
            # Handle NaN values
            insights_metrics['current_inventory_holding'] = float(insights_current_holding) if not pd.isna(insights_current_holding) else 0
            insights_metrics['avg_inventory_holding'] = float(insights_avg_holding) if not pd.isna(insights_avg_holding) else 0
        else:
            # Calculate from product master if available
            if not insights_product_master.empty and 'inventory_cost' in insights_product_master.columns:
                insights_merged = insights_data.merge(insights_product_master[['product_id', 'location_id', 'inventory_cost']], 
                                                    on=['product_id', 'location_id'], how='left')
                if 'stock_level' in insights_data.columns:
                    insights_merged['stock_value'] = insights_merged['stock_level'] * insights_merged['inventory_cost']
                    insights_current_holding = insights_merged['stock_value'].iloc[-1] if not insights_merged.empty else 0
                    insights_avg_holding = insights_merged['stock_value'].mean()
                    # Handle NaN values
                    insights_metrics['current_inventory_holding'] = float(insights_current_holding) if not pd.isna(insights_current_holding) else 0
                    insights_metrics['avg_inventory_holding'] = float(insights_avg_holding) if not pd.isna(insights_avg_holding) else 0
                else:
                    insights_metrics['current_inventory_holding'] = 0
                    insights_metrics['avg_inventory_holding'] = 0
            else:
                insights_metrics['current_inventory_holding'] = 0
                insights_metrics['avg_inventory_holding'] = 0
        
        # Calculate inventory coverage for insights
        try:
            insights_coverage = insights_calculate_inventory_coverage(insights_data)
            insights_metrics['avg_inventory_coverage'] = insights_coverage
        except:
            insights_metrics['avg_inventory_coverage'] = 0
        
        # Calculate turnover ratio for insights
        if insights_metrics['avg_inventory_holding'] > 0 and not pd.isna(insights_metrics['avg_inventory_holding']):
            insights_turnover = insights_metrics['total_revenue'] / insights_metrics['avg_inventory_holding']
            insights_metrics['inventory_turnover_ratio'] = float(insights_turnover) if not pd.isna(insights_turnover) else 0
        else:
            insights_metrics['inventory_turnover_ratio'] = 0
        
        # Enhanced service level and stockout metrics for insights
        if 'stock_level' in insights_data.columns:
            insights_stockout_periods = (insights_data['stock_level'] <= 0).sum()
            insights_total_periods = len(insights_data)
            
            # Service level calculation - avoid division by zero
            if insights_total_periods > 0:
                insights_metrics['service_level'] = ((insights_total_periods - insights_stockout_periods) / insights_total_periods) * 100
                insights_metrics['stockout_frequency'] = (insights_stockout_periods / insights_total_periods) * 100
                
                # Availability percentage
                insights_available_periods = (insights_data['stock_level'] > 0).sum()
                insights_metrics['availability_percentage'] = (insights_available_periods / insights_total_periods) * 100
            else:
                insights_metrics['service_level'] = 95.0
                insights_metrics['stockout_frequency'] = 0
                insights_metrics['availability_percentage'] = 95.0
            
            # SKUs with stock
            if 'location_id' in insights_data.columns:
                insights_current_stock = insights_data.groupby(['product_id', 'location_id'])['stock_level'].last()
                insights_metrics['skus_with_stock'] = (insights_current_stock > 0).sum()
            else:
                insights_current_stock = insights_data.groupby('product_id')['stock_level'].last()
                insights_metrics['skus_with_stock'] = (insights_current_stock > 0).sum()
        else:
            insights_metrics['service_level'] = 95.0  # Default assumption
            insights_metrics['stockout_frequency'] = 5.0  # Default assumption
            insights_metrics['availability_percentage'] = 95.0
            insights_metrics['skus_with_stock'] = insights_metrics['total_skus']
        
        # Missed demand and revenue calculations for insights
        if 'missed_demand' in insights_data.columns:
            insights_total_missed_demand = insights_data['missed_demand'].sum()
            insights_total_periods = insights_data['date'].nunique() if 'date' in insights_data.columns else len(insights_data)
            insights_metrics['avg_missed_demand_per_period'] = insights_total_missed_demand / insights_total_periods if insights_total_periods > 0 else 0
            
            # Calculate missed revenue
            if 'unit_price' in insights_data.columns:
                insights_metrics['total_missed_revenue'] = (insights_data['missed_demand'] * insights_data['unit_price']).sum()
            elif 'price' in insights_data.columns:
                insights_metrics['total_missed_revenue'] = (insights_data['missed_demand'] * insights_data['price']).sum()
            else:
                insights_metrics['total_missed_revenue'] = 0
        else:
            insights_metrics['avg_missed_demand_per_period'] = 0
            insights_metrics['total_missed_revenue'] = 0
        
        # Decapable inventory calculation for insights (simplified)
        try:
            if 'stock_level' in insights_data.columns and not insights_product_master.empty:
                insights_merged_master = insights_data.merge(insights_product_master, on=['product_id', 'location_id'], how='left')
                if 'inventory_cost' in insights_merged_master.columns:
                    # Assume decapable if stock > 0 but demand = 0 for extended period
                    insights_no_demand_products = insights_data.groupby(['product_id', 'location_id']).agg({
                        'demand': 'sum',
                        'stock_level': 'last'
                    }).reset_index()
                    insights_no_demand_products = insights_no_demand_products[
                        (insights_no_demand_products['demand'] == 0) & (insights_no_demand_products['stock_level'] > 0)
                    ]
                    
                    if len(insights_no_demand_products) > 0:
                        insights_decapable_merged = insights_no_demand_products.merge(
                            insights_product_master[['product_id', 'location_id', 'inventory_cost']], 
                            on=['product_id', 'location_id'], how='left'
                        )
                        insights_decapable_merged['decapable_value'] = insights_decapable_merged['stock_level'] * insights_decapable_merged['inventory_cost']
                        insights_metrics['decapable_inventory_holding'] = insights_decapable_merged['decapable_value'].sum()
                    else:
                        insights_metrics['decapable_inventory_holding'] = 0
                else:
                    insights_metrics['decapable_inventory_holding'] = 0
            else:
                insights_metrics['decapable_inventory_holding'] = 0
        except:
            insights_metrics['decapable_inventory_holding'] = 0
        
        return insights_metrics
        
    except Exception as insights_error:
        print(f"Error calculating insights metrics: {insights_error}")
        import traceback
        print(f"Insights metrics error traceback: {traceback.format_exc()}")
        return {}


def insights_calculate_inventory_coverage(insights_data):
    """Calculate inventory coverage percentage for insights"""
    try:
        if 'stock_level' in insights_data.columns and 'demand' in insights_data.columns:
            # Calculate coverage as days of stock based on average demand
            insights_product_stats = insights_data.groupby('product_id').agg({
                'stock_level': 'last',  # Current stock
                'demand': 'mean'  # Average daily demand
            }).reset_index()
            
            # Calculate days of coverage
            insights_product_stats['days_coverage'] = insights_product_stats['stock_level'] / insights_product_stats['demand']
            insights_product_stats['days_coverage'] = insights_product_stats['days_coverage'].replace([np.inf, -np.inf], 0).fillna(0)
            
            # Calculate percentage of products with adequate coverage (e.g., >= 30 days)
            insights_adequate_coverage = (insights_product_stats['days_coverage'] >= 30).sum()
            insights_coverage_percentage = (insights_adequate_coverage / len(insights_product_stats)) * 100
            
            return insights_coverage_percentage
        else:
            return 0
    except:
        return 0


# ============================================================================
# INSIGHTS DEMAND CLASSIFICATION (Independent from other tabs)
# ============================================================================

# Insights-specific classification parameters
INSIGHTS_INTERDEMAND_INTERVAL = 1.32
INSIGHTS_CV2_INTERVAL = 0.49

def insights_demand_classification(insights_table, insights_group_variables, insights_unit_to_round_over='day', insights_revenue_column=None, insights_start_date=None, insights_end_date=None):
    """
    Returns demand classifications by product from insights table containing daily demand and inventory data
    (Independent version for insights tab with insights_ prefixes)
    """
    
    # Set frequency based on rounding unit for insights
    insights_freq_map = {
        'year': 365,
        'month': 30,
        'week': 7,
        'day': 1
    }
    
    if insights_unit_to_round_over in insights_freq_map:
        insights_freq = insights_freq_map[insights_unit_to_round_over]
    else:
        insights_freq = 1
    
    # Ensure date column is datetime for insights
    insights_table = insights_table.copy()
    insights_table['date'] = pd.to_datetime(insights_table['date'])
    
    # Filter by date range if provided for insights
    if insights_start_date is not None:
        insights_start_date = pd.to_datetime(insights_start_date)
        insights_table = insights_table[insights_table['date'] >= insights_start_date]
    
    if insights_end_date is not None:
        insights_end_date = pd.to_datetime(insights_end_date)
        insights_table = insights_table[insights_table['date'] <= insights_end_date]
    
    # Get demand for insights - filter for non-zero demand
    insights_demand_data = insights_table[insights_table['demand'] != 0].copy()
    
    # Round dates based on unit for insights
    if insights_unit_to_round_over == 'day':
        insights_demand_data['date'] = insights_demand_data['date'].dt.date
    elif insights_unit_to_round_over == 'week':
        insights_demand_data['date'] = insights_demand_data['date'].dt.to_period('W').dt.start_time.dt.date
    elif insights_unit_to_round_over == 'month':
        insights_demand_data['date'] = insights_demand_data['date'].dt.to_period('M').dt.start_time.dt.date
    elif insights_unit_to_round_over == 'year':
        insights_demand_data['date'] = insights_demand_data['date'].dt.to_period('Y').dt.start_time.dt.date
    
    # Group by specified variables and date, sum demand for insights
    insights_agg_dict = {'demand': 'sum'}
    if insights_revenue_column:
        insights_agg_dict[insights_revenue_column] = 'sum'
    
    insights_grouped_demand = (insights_demand_data
                     .groupby(insights_group_variables + ['date'])
                     .agg(insights_agg_dict)
                     .reset_index())
    
    # Calculate interdemand intervals and statistics for each group in insights
    def insights_calculate_stats(insights_group):
        """Calculate demand statistics for insights classification"""
        
        # Calculate total demand and revenue for insights
        insights_total_demand = insights_group['demand'].sum()
        insights_total_revenue = insights_group[insights_revenue_column].sum() if insights_revenue_column else 0
        
        # Calculate yearly orders for insights
        insights_nyearlyorders = len(insights_group) * (365.25 / insights_freq)
        
        # Calculate interdemand intervals for insights
        insights_group_sorted = insights_group.sort_values('date')
        if len(insights_group_sorted) > 1:
            insights_dates = pd.to_datetime(insights_group_sorted['date'])
            insights_intervals = insights_dates.diff().dt.days.dropna()
            insights_mean_interdemand_interval = insights_intervals.mean() if len(insights_intervals) > 0 else 0
        else:
            insights_mean_interdemand_interval = 0
        
        # Calculate CV2 for insights
        if len(insights_group) > 1 and insights_group['demand'].std() > 0:
            insights_cv = insights_group['demand'].std() / insights_group['demand'].mean()
            insights_cv2 = insights_cv ** 2
        else:
            insights_cv2 = 0
        
        insights_stats = {
            'total_demand': insights_total_demand,
            'total_revenue': insights_total_revenue,
            'nyearlyorders': insights_nyearlyorders,
            'mean_interdemand_interval': insights_mean_interdemand_interval,
            'CV2': insights_cv2
        }
        
        return pd.Series(insights_stats)
    
    try:
        # Use a safer approach for groupby/apply in insights
        insights_result_list = []
        for insights_name, insights_group in insights_grouped_demand.groupby(insights_group_variables):
            # Handle both single and multiple group variables for insights
            if isinstance(insights_name, tuple):
                insights_group_dict = {insights_var: insights_val for insights_var, insights_val in zip(insights_group_variables, insights_name)}
            else:
                insights_group_dict = {insights_group_variables[0]: insights_name}
            
            # Calculate stats for this group in insights
            insights_stats = insights_calculate_stats(insights_group)
            
            # Combine group info with stats for insights
            insights_result = {**insights_group_dict, **insights_stats.to_dict()}
            insights_result_list.append(insights_result)
        
        # Convert results to DataFrame for insights
        insights_x_classified = pd.DataFrame(insights_result_list)
        
    except Exception as insights_e:
        print(f"ERROR in insights demand classification groupby/apply: {str(insights_e)}")
        raise
    
    # Classify based on purchase patterns for insights
    insights_only_one = insights_x_classified[insights_x_classified['nyearlyorders'] == 1].copy()
    insights_only_one['type'] = "One Purchase"
    
    insights_more_than_one = insights_x_classified[insights_x_classified['nyearlyorders'] > 1].copy()
    
    # Lumpy: high interdemand interval AND high CV2 for insights
    insights_lumpy = insights_more_than_one[
        (insights_more_than_one['mean_interdemand_interval'] > INSIGHTS_INTERDEMAND_INTERVAL) & 
        (insights_more_than_one['CV2'] >= INSIGHTS_CV2_INTERVAL)
    ].copy()
    insights_lumpy = insights_lumpy.sort_values(['mean_interdemand_interval', 'CV2'], ascending=[False, False])
    insights_lumpy['type'] = "Lumpy"
    
    # Intermittent: high interdemand interval BUT low CV2 for insights
    insights_intermittent = insights_more_than_one[
        (insights_more_than_one['mean_interdemand_interval'] > INSIGHTS_INTERDEMAND_INTERVAL) & 
        (insights_more_than_one['CV2'] < INSIGHTS_CV2_INTERVAL)
    ].copy()
    insights_intermittent = insights_intermittent.sort_values(['mean_interdemand_interval'], ascending=[False])
    insights_intermittent['type'] = "Intermittent"
    
    # Erratic: low interdemand interval BUT high CV2 for insights  
    insights_erratic = insights_more_than_one[
        (insights_more_than_one['mean_interdemand_interval'] <= INSIGHTS_INTERDEMAND_INTERVAL) & 
        (insights_more_than_one['CV2'] >= INSIGHTS_CV2_INTERVAL)
    ].copy()
    insights_erratic = insights_erratic.sort_values(['CV2'], ascending=[False])
    insights_erratic['type'] = "Erratic"
    
    # Smooth: low interdemand interval AND low CV2 for insights
    insights_smooth = insights_more_than_one[
        (insights_more_than_one['mean_interdemand_interval'] <= INSIGHTS_INTERDEMAND_INTERVAL) & 
        (insights_more_than_one['CV2'] < INSIGHTS_CV2_INTERVAL)
    ].copy()
    insights_smooth = insights_smooth.sort_values(['CV2'], ascending=[True])
    insights_smooth['type'] = "Smooth"
    
    # Combine all classifications for insights
    insights_all_classified = pd.concat([
        insights_only_one, insights_lumpy, insights_intermittent, 
        insights_erratic, insights_smooth
    ], ignore_index=True)
    
    return insights_all_classified


def insights_generate_charts(insights_data, insights_group_variables=['product_id'], insights_time_unit='day', insights_pareto_lines="50,80,90", insights_show_values=True):
    """Generate enhanced chart data for insights dashboard"""
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        
        insights_charts = {}
        
        # Enhanced Revenue by Category chart for insights
        if 'product_category' in insights_data.columns and 'revenue' in insights_data.columns:
            insights_category_revenue = insights_data.groupby('product_category')['revenue'].sum().sort_values(ascending=False)
            insights_total_revenue_cat = insights_category_revenue.sum()
            
            if insights_show_values:
                insights_custom_text_cat = [f'{label}<br>${insights_format_number_with_suffix(value)}<br>({value/insights_total_revenue_cat*100:.1f}%)' 
                                          for label, value in zip(insights_category_revenue.index, insights_category_revenue.values)]
            else:
                insights_custom_text_cat = [f'{label}<br>({value/insights_total_revenue_cat*100:.1f}%)' 
                                          for label, value in zip(insights_category_revenue.index, insights_category_revenue.values)]
            
            insights_fig_cat = go.Figure(data=[go.Pie(
                labels=list(insights_category_revenue.index),
                values=list(insights_category_revenue.values),
                hole=0.3,
                text=insights_custom_text_cat
            )])
            insights_fig_cat.update_traces(
                textposition='outside', 
                textinfo='text',
                marker=dict(colors=['#de6a45','#2a44d4','#3d315a','#cfc8c2','#5f554c','#807368','#737574'])
            )
            insights_fig_cat.update_layout(
                title="Revenue by Category",
                height=420,
                showlegend=False,
                template='plotly_white'
            )
            insights_charts['revenue_by_category'] = insights_fig_cat.to_json()
        
        # Enhanced Revenue by Location chart for insights
        if 'location_id' in insights_data.columns and 'revenue' in insights_data.columns:
            insights_location_revenue = insights_data.groupby('location_id')['revenue'].sum().sort_values(ascending=False)
            insights_total_revenue_loc = insights_location_revenue.sum()
            
            if insights_show_values:
                insights_custom_text_loc = [f'{label}<br>${insights_format_number_with_suffix(value)}<br>({value/insights_total_revenue_loc*100:.1f}%)' 
                                          for label, value in zip(insights_location_revenue.index, insights_location_revenue.values)]
            else:
                insights_custom_text_loc = [f'{label}<br>({value/insights_total_revenue_loc*100:.1f}%)' 
                                          for label, value in zip(insights_location_revenue.index, insights_location_revenue.values)]
            
            insights_fig_loc = go.Figure(data=[go.Pie(
                labels=list(insights_location_revenue.index),
                values=list(insights_location_revenue.values),
                hole=0.3,
                text=insights_custom_text_loc
            )])
            insights_fig_loc.update_traces(
                textposition='outside',
                textinfo='text',
                marker=dict(colors=['#de6a45','#2a44d4','#3d315a','#cfc8c2','#5f554c','#807368','#737574'])
            )
            insights_fig_loc.update_layout(
                title="Revenue by Location",
                height=420,
                showlegend=False,
                template='plotly_white'
            )
            insights_charts['revenue_by_location'] = insights_fig_loc.to_json()
        
        # Enhanced COGS and Stock Value trends for insights
        if 'date' in insights_data.columns:
            insights_daily_data = insights_data.groupby('date').agg({
                'demand': 'sum',
                'revenue': 'sum' if 'revenue' in insights_data.columns else lambda x: 0
            }).reset_index()
            
            insights_fig_trends = go.Figure()
            insights_fig_trends.add_trace(go.Scatter(
                x=insights_daily_data['date'],
                y=insights_daily_data['demand'],
                mode='lines',
                name='Daily Demand',
                line=dict(color='#2a44d4', width=2)
            ))
            if 'revenue' in insights_daily_data.columns and insights_daily_data['revenue'].sum() > 0:
                insights_fig_trends.add_trace(go.Scatter(
                    x=insights_daily_data['date'],
                    y=insights_daily_data['revenue'],
                    mode='lines',
                    name='Daily Revenue',
                    yaxis='y2',
                    line=dict(color='#de6a45', width=2)
                ))
            
            insights_fig_trends.update_layout(
                title="Demand and Revenue Trends Over Time",
                height=420,
                xaxis_title="Date",
                yaxis_title="Daily Demand",
                yaxis2=dict(
                    title="Daily Revenue ($)",
                    overlaying='y',
                    side='right'
                ),
                template='plotly_white'
            )
            insights_charts['cogs_stock_value'] = insights_fig_trends.to_json()
        
        # Enhanced Pareto analysis for insights with configurable reference lines
        if 'revenue' in insights_data.columns:
            insights_product_revenue = insights_data.groupby('product_id')['revenue'].sum().sort_values(ascending=False)
        else:
            insights_product_revenue = insights_data.groupby('product_id')['demand'].sum().sort_values(ascending=False)
        
        insights_cumulative_pct = (insights_product_revenue.cumsum() / insights_product_revenue.sum() * 100)
        insights_product_pct = [(i+1)/len(insights_product_revenue)*100 for i in range(len(insights_product_revenue))]
        
        insights_fig_pareto = go.Figure()
        insights_fig_pareto.add_trace(go.Scatter(
            x=insights_product_pct,
            y=insights_cumulative_pct.values,
            mode='lines+markers',
            name='Revenue Curve' if 'revenue' in insights_data.columns else 'Demand Curve',
            line=dict(color='#A23B72', width=3),
            marker=dict(size=6, color='#A23B72')
        ))
        
        # Add configurable reference lines for insights
        if insights_pareto_lines:
            insights_pareto_vals = [float(x.strip()) for x in insights_pareto_lines.split(',') if x.strip().replace('.','').isdigit()]
            insights_colors = ['green', 'orange', 'red', 'purple', 'brown']
            for insights_i, insights_val in enumerate(insights_pareto_vals):
                if 0 <= insights_val <= 100:
                    insights_color = insights_colors[insights_i % len(insights_colors)]
                    insights_fig_pareto.add_hline(
                        y=insights_val,
                        line=dict(color=insights_color, width=2, dash='dash'),
                        annotation_text=f"{insights_val}%"
                    )
        
        insights_fig_pareto.update_layout(
            title="Revenue Pareto Analysis (80/20 Rule)" if 'revenue' in insights_data.columns else "Demand Pareto Analysis",
            height=420,
            xaxis_title="% of Products (Ranked by Revenue)" if 'revenue' in insights_data.columns else "% of Products (Ranked by Demand)", 
            yaxis_title="Cumulative % of Total Revenue" if 'revenue' in insights_data.columns else "Cumulative % of Total Demand",
            template='plotly_white'
        )
        insights_charts['pareto_revenue'] = insights_fig_pareto.to_json()
        insights_charts['pareto_demand'] = insights_fig_pareto.to_json()  # Use same enhanced chart
        
        # Enhanced demand classification plot for insights
        try:
            insights_classification_data = insights_demand_classification(
                insights_data, 
                insights_group_variables, 
                insights_time_unit,
                'revenue' if 'revenue' in insights_data.columns else None
            )
            
            if not insights_classification_data.empty:
                insights_fig_class = go.Figure()
                
                # Color mapping for insights classification
                insights_color_map = {
                    'Smooth': '#28a745',
                    'Erratic': '#ffc107', 
                    'Intermittent': '#17a2b8',
                    'Lumpy': '#dc3545',
                    'One Purchase': '#6c757d'
                }
                
                # Add data points by classification type for insights
                for insights_class_type in insights_color_map.keys():
                    insights_class_data = insights_classification_data[insights_classification_data['type'] == insights_class_type]
                    
                    if len(insights_class_data) > 0:
                        insights_fig_class.add_trace(go.Scatter(
                            x=insights_class_data['mean_interdemand_interval'],
                            y=insights_class_data['CV2'],
                            mode='markers',
                            name=f'{insights_class_type} ({len(insights_class_data)})',
                            marker=dict(
                                color=insights_color_map[insights_class_type],
                                size=8,
                                opacity=0.7
                            ),
                            hovertemplate=f'<b>{insights_class_type}</b><br>' +
                                        'Product: %{customdata[0]}<br>' +
                                        'Mean Interdemand Interval: %{x:.2f} days<br>' +
                                        'CV²: %{y:.2f}<br>' +
                                        'Total Demand: %{customdata[1]:,.0f}<br>' +
                                        'Total Revenue: $%{customdata[2]:,.0f}<br>' +
                                        '<extra></extra>',
                            customdata=list(zip(
                                insights_class_data['product_id'],
                                insights_class_data['total_demand'],
                                insights_class_data['total_revenue']
                            ))
                        ))
                
                # Add reference lines for insights classification
                insights_fig_class.add_vline(
                    x=INSIGHTS_INTERDEMAND_INTERVAL,
                    line=dict(color='black', width=2, dash='dash'),
                    annotation_text=f"Interdemand Interval = {INSIGHTS_INTERDEMAND_INTERVAL}"
                )
                insights_fig_class.add_hline(
                    y=INSIGHTS_CV2_INTERVAL,
                    line=dict(color='black', width=2, dash='dash'),
                    annotation_text=f"CV² = {INSIGHTS_CV2_INTERVAL}"
                )
                
                insights_fig_class.update_layout(
                    title="Demand Classification Analysis<br><sub>Products classified by demand patterns</sub>",
                    height=900,  # Match the CSS height expectation
                    xaxis_title="Mean Interdemand Interval (days)",
                    yaxis_title="CV² (Coefficient of Variation Squared)",
                    template='plotly_white',
                    showlegend=True,
                    margin=dict(t=40, b=60, l=60, r=20),
                    autosize=True
                )
                
                insights_charts['classification'] = insights_fig_class.to_json()
            else:
                # Fallback simple chart for insights
                insights_fig_class = go.Figure()
                insights_fig_class.add_annotation(
                    text="Insufficient data for demand classification analysis",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, xanchor='center', yanchor='middle',
                    font=dict(size=16, color="red")
                )
                insights_fig_class.update_layout(
                    title="Demand Classification Analysis",
                    height=900,  # Match the CSS height expectation
                    template='plotly_white',
                    margin=dict(t=40, b=60, l=60, r=20),
                    autosize=True
                )
                insights_charts['classification'] = insights_fig_class.to_json()
        
        except Exception as insights_class_error:
            print(f"Error in insights classification plot: {insights_class_error}")
            # Create empty placeholder for insights
            insights_fig_class = go.Figure()
            insights_fig_class.add_annotation(
                text=f"Error generating classification: {str(insights_class_error)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                font=dict(size=14, color="red")
            )
            insights_fig_class.update_layout(
                title="Demand Classification Analysis", 
                height=900,  # Match the CSS height expectation
                template='plotly_white',
                margin=dict(t=40, b=60, l=60, r=20),
                autosize=True
            )
            insights_charts['classification'] = insights_fig_class.to_json()
        
        return insights_charts
        
    except Exception as insights_error:
        print(f"Error generating insights charts: {insights_error}")
        import traceback
        print(f"Insights charts error traceback: {traceback.format_exc()}")
        return {}


def insights_format_number_with_suffix(insights_value):
    """Format numbers with K/M/B suffixes for insights readability"""
    if insights_value >= 1000000000:
        return f"{insights_value / 1000000000:.1f}B"
    elif insights_value >= 1000000:
        return f"{insights_value / 1000000:.1f}M"
    elif insights_value >= 1000:
        return f"{insights_value / 1000:.1f}K"
    else:
        return f"{insights_value:.0f}"


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
