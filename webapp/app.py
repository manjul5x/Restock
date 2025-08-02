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
        products = sorted(safety_stock_data["product_id"].unique().tolist())
        locations = sorted(safety_stock_data["location_id"].unique().tolist())
        review_dates = sorted(
            safety_stock_data["review_date"].dt.strftime("%Y-%m-%d").unique().tolist()
        )
        forecast_methods = sorted(
            safety_stock_data["forecast_method"].unique().tolist()
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
        products = sorted(forecast_data["product_id"].unique().tolist())
        locations = sorted(forecast_data["location_id"].unique().tolist())
        analysis_dates = sorted(
            forecast_data["analysis_date"].dt.strftime("%Y-%m-%d").unique().tolist()
        )
        forecast_methods = sorted(forecast_data["forecast_method"].unique().tolist())

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
        products = sorted(simulation_data["product_id"].unique().tolist())
        locations = sorted(simulation_data["location_id"].unique().tolist())
        forecast_methods = sorted(simulation_data["forecast_method"].unique().tolist())

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
        products = sorted(summary_data["product_id"].unique().tolist())
        locations = sorted(summary_data["location_id"].unique().tolist())
        forecast_methods = sorted(summary_data["forecast_method"].unique().tolist())

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


@app.route("/get_comparison_data", methods=["POST"])
def get_comparison_data():
    """Generate comprehensive inventory comparison data comparing actual vs simulated inventory levels"""
    try:
        # Get form data
        products = request.form.getlist("products")
        locations = request.form.getlist("locations")
        forecast_methods = request.form.getlist("forecast_methods")

        # Load detailed simulation data for comparison
        detailed_dir = loader.get_output_path("simulation", "detailed_results")
        detailed_data = []

        if not detailed_dir.exists():
            return jsonify({"error": "No detailed simulation data available"})

        for file_path in detailed_dir.glob("*_simulation.csv"):
            filename = file_path.stem
            parts = filename.split("_")
            if len(parts) >= 4:
                product_id = parts[0]
                location_id = parts[1]

                # Extract forecast method
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

                try:
                    data = pd.read_csv(file_path)
                    data["date"] = pd.to_datetime(data["date"])
                    detailed_data.append(data)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    continue

        if not detailed_data:
            return jsonify({"error": "No detailed simulation data available"})

        combined_detailed = pd.concat(detailed_data, ignore_index=True)

        # Calculate comprehensive comparison metrics
        comparison_results = []

        # Group by product-location-method
        for (
            product_id,
            location_id,
            forecast_method,
        ), group in combined_detailed.groupby(
            ["product_id", "location_id", "forecast_method"]
        ):
            if len(group) == 0:
                continue

            # Calculate metrics comparing actual vs simulated inventory
            total_days = int(len(group))

            # Actual inventory metrics
            actual_inventory = group["actual_inventory"].values
            actual_avg_inventory = float(actual_inventory.mean())
            actual_min_inventory = float(actual_inventory.min())
            actual_max_inventory = float(actual_inventory.max())
            actual_stockout_days = int((actual_inventory <= 0).sum())
            actual_stockout_rate = float((actual_stockout_days / total_days) * 100)

            # Simulated inventory metrics
            simulated_inventory = group["inventory_on_hand"].values
            simulated_avg_inventory = float(simulated_inventory.mean())
            simulated_min_inventory = float(simulated_inventory.min())
            simulated_max_inventory = float(simulated_inventory.max())
            simulated_stockout_days = int((simulated_inventory <= 0).sum())
            simulated_stockout_rate = float(
                (simulated_stockout_days / total_days) * 100
            )

            # Comparison metrics
            inventory_difference = float(simulated_avg_inventory - actual_avg_inventory)
            inventory_difference_percentage = float(
                (inventory_difference / actual_avg_inventory) * 100
                if actual_avg_inventory > 0
                else 0
            )

            # Stockout rate difference
            stockout_rate_difference = float(
                simulated_stockout_rate - actual_stockout_rate
            )

            # Overstocking/Understocking analysis
            # Overstocking: days when inventory > max_level
            actual_overstock_days = int((actual_inventory > group["max_level"]).sum())
            simulated_overstock_days = int(
                (simulated_inventory > group["max_level"]).sum()
            )
            actual_overstock_percentage = float(
                (actual_overstock_days / total_days) * 100
            )
            simulated_overstock_percentage = float(
                (simulated_overstock_days / total_days) * 100
            )

            # Understocking: days when inventory < min_level
            actual_understock_days = int((actual_inventory < group["min_level"]).sum())
            simulated_understock_days = int(
                (simulated_inventory < group["min_level"]).sum()
            )
            actual_understock_percentage = float(
                (actual_understock_days / total_days) * 100
            )
            simulated_understock_percentage = float(
                (simulated_understock_days / total_days) * 100
            )

            # Service level calculation
            actual_demand = group["actual_demand"].values
            actual_service_level = 0.0
            simulated_service_level = 0.0

            if actual_demand.sum() > 0:
                # Actual service level: demand met from actual inventory
                actual_demand_met = int(
                    ((actual_demand > 0) & (actual_inventory >= actual_demand)).sum()
                )
                actual_service_level = float(
                    (actual_demand_met / (actual_demand > 0).sum()) * 100
                )

                # Simulated service level: demand met from simulated inventory
                simulated_demand_met = int(
                    ((actual_demand > 0) & (simulated_inventory >= actual_demand)).sum()
                )
                simulated_service_level = float(
                    (simulated_demand_met / (actual_demand > 0).sum()) * 100
                )

            # Inventory days calculation (instead of turns)
            avg_daily_demand = float(actual_demand.mean())
            actual_inventory_days = float(
                actual_avg_inventory / avg_daily_demand if avg_daily_demand > 0 else 0
            )
            simulated_inventory_days = float(
                simulated_avg_inventory / avg_daily_demand
                if avg_daily_demand > 0
                else 0
            )

            # Stock days calculation
            actual_stock_days = float(
                actual_avg_inventory / avg_daily_demand if avg_daily_demand > 0 else 0
            )
            simulated_stock_days = float(
                simulated_avg_inventory / avg_daily_demand
                if avg_daily_demand > 0
                else 0
            )

            # Total demand calculation
            total_demand = float(actual_demand.sum())

            # Total inventory holding calculations
            # Get product master data for cost information
            product_master = loader.load_product_master()
            inventory_cost = 0.0
            if not product_master.empty:
                product_record = product_master[
                    (product_master["product_id"] == product_id)
                    & (product_master["location_id"] == location_id)
                ]
                if len(product_record) > 0:
                    inventory_cost = float(
                        product_record.iloc[0].get("inventory_cost", 0.0)
                    )

            # Calculate total inventory holding
            actual_total_inventory_units = float(
                actual_avg_inventory + group["inventory_on_order"].mean()
            )
            simulated_total_inventory_units = float(
                simulated_avg_inventory + group["inventory_on_order"].mean()
            )

            actual_total_inventory_cost = float(
                actual_total_inventory_units * inventory_cost
            )
            simulated_total_inventory_cost = float(
                simulated_total_inventory_units * inventory_cost
            )

            # Missed demand calculation
            actual_missed_demand = float(
                actual_demand[actual_inventory < actual_demand].sum()
            )
            simulated_missed_demand = float(
                actual_demand[simulated_inventory < actual_demand].sum()
            )

            # Create comparison result
            comparison_result = {
                "product_id": str(product_id),
                "location_id": str(location_id),
                "forecast_method": str(forecast_method),
                "total_days": total_days,
                # Actual inventory metrics
                "actual_avg_inventory": round(actual_avg_inventory, 0),
                "actual_min_inventory": round(actual_min_inventory, 0),
                "actual_max_inventory": round(actual_max_inventory, 0),
                "actual_stockout_days": actual_stockout_days,
                "actual_stockout_rate": round(actual_stockout_rate, 2),
                "actual_service_level": round(actual_service_level, 2),
                "actual_inventory_days": round(actual_inventory_days, 2),
                "actual_stock_days": round(actual_stock_days, 2),
                "actual_overstock_percentage": round(actual_overstock_percentage, 2),
                "actual_understock_percentage": round(actual_understock_percentage, 2),
                "actual_missed_demand": round(actual_missed_demand, 0),
                "actual_total_inventory_units": round(actual_total_inventory_units, 0),
                "actual_total_inventory_cost": round(actual_total_inventory_cost, 2),
                # Simulated inventory metrics
                "simulated_avg_inventory": round(simulated_avg_inventory, 0),
                "simulated_min_inventory": round(simulated_min_inventory, 0),
                "simulated_max_inventory": round(simulated_max_inventory, 0),
                "simulated_stockout_days": simulated_stockout_days,
                "simulated_stockout_rate": round(simulated_stockout_rate, 2),
                "simulated_service_level": round(simulated_service_level, 2),
                "simulated_inventory_days": round(simulated_inventory_days, 2),
                "simulated_stock_days": round(simulated_stock_days, 2),
                "simulated_overstock_percentage": round(
                    simulated_overstock_percentage, 2
                ),
                "simulated_understock_percentage": round(
                    simulated_understock_percentage, 2
                ),
                "simulated_missed_demand": round(simulated_missed_demand, 0),
                "simulated_total_inventory_units": round(
                    simulated_total_inventory_units, 0
                ),
                "simulated_total_inventory_cost": round(
                    simulated_total_inventory_cost, 2
                ),
                # Comparison metrics
                "inventory_difference": round(inventory_difference, 0),
                "inventory_difference_percentage": round(
                    inventory_difference_percentage, 2
                ),
                "stockout_rate_difference": round(stockout_rate_difference, 2),
                "service_level_difference": round(
                    simulated_service_level - actual_service_level, 2
                ),
                "inventory_days_difference": round(
                    simulated_inventory_days - actual_inventory_days, 2
                ),
                "stock_days_difference": round(
                    simulated_stock_days - actual_stock_days, 2
                ),
                "overstock_difference": round(
                    simulated_overstock_percentage - actual_overstock_percentage, 2
                ),
                "understock_difference": round(
                    simulated_understock_percentage - actual_understock_percentage, 2
                ),
                "missed_demand_difference": round(
                    simulated_missed_demand - actual_missed_demand, 0
                ),
                "total_inventory_units_difference": round(
                    simulated_total_inventory_units - actual_total_inventory_units, 0
                ),
                "total_inventory_cost_difference": round(
                    simulated_total_inventory_cost - actual_total_inventory_cost, 2
                ),
                # Total demand for context
                "total_demand": round(total_demand, 0),
            }

            comparison_results.append(comparison_result)

        # Calculate overall averages
        if comparison_results:
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
                "avg_actual_inventory_days": round(
                    sum(r["actual_inventory_days"] for r in comparison_results)
                    / len(comparison_results),
                    2,
                ),
                "avg_simulated_inventory_days": round(
                    sum(r["simulated_inventory_days"] for r in comparison_results)
                    / len(comparison_results),
                    2,
                ),
                "avg_inventory_difference_percentage": round(
                    sum(
                        r["inventory_difference_percentage"] for r in comparison_results
                    )
                    / len(comparison_results),
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
                "avg_inventory_days_difference": round(
                    sum(r["inventory_days_difference"] for r in comparison_results)
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
                    sum(
                        r["simulated_total_inventory_units"] for r in comparison_results
                    ),
                    0,
                ),
                "total_actual_inventory_cost": round(
                    sum(r["actual_total_inventory_cost"] for r in comparison_results), 2
                ),
                "total_simulated_inventory_cost": round(
                    sum(
                        r["simulated_total_inventory_cost"] for r in comparison_results
                    ),
                    2,
                ),
                "total_products": len(comparison_results),
            }
        else:
            overall_metrics = {}

        return jsonify(
            {
                "comparison_results": comparison_results,
                "overall_metrics": overall_metrics,
            }
        )

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
        for file_path in detailed_dir.glob("*_simulation.csv"):
            # Extract product, location, and method from filename
            filename = file_path.stem  # e.g., "RSWQ_WB_moving_average_simulation"
            parts = filename.split("_")
            if len(parts) >= 4:
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

                # Apply filters
                if products and product_id not in products:
                    continue
                if locations and location_id not in locations:
                    continue
                if forecast_methods and forecast_method not in forecast_methods:
                    continue

                # Load the data
                try:
                    data = pd.read_csv(file_path)
                    data["date"] = pd.to_datetime(data["date"])
                    all_data.append(data)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    continue

        if not all_data:
            return jsonify({"error": "No data matches the selected filters"})

        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)

        # Create the simulation chart
        fig, ax = plt.subplots(figsize=(20, 8))

        # Plot for each product-location-method combination
        for (product_id, location_id, forecast_method), group in combined_data.groupby(
            ["product_id", "location_id", "forecast_method"]
        ):
            # Sort by date
            group = group.sort_values("date")

            # Plot simulated stock on hand (blue line)
            ax.plot(
                group["date"],
                group["inventory_on_hand"],
                label="Stock on Hand",
                linewidth=3,
                alpha=0.9,
                color="#1f77b4",
            )

            # Plot actual demand (orange line)
            ax.plot(
                group["date"],
                group["actual_demand"],
                label="Actual Demand",
                linewidth=2,
                alpha=0.8,
                linestyle="--",
                color="#ff7f0e",
            )

            # Plot safety stock (red dashed line)
            ax.plot(
                group["date"],
                group["safety_stock"],
                label="Safety Stock",
                linewidth=2,
                alpha=0.7,
                linestyle=":",
                color="#d62728",
            )

            # Plot actual inventory as shaded area (light blue)
            ax.fill_between(
                group["date"],
                group["actual_inventory"],
                alpha=0.2,
                label="Actual Inventory",
                color="#aec7e8",
            )

            # Plot orders placed as vertical bars (cyan)
            order_dates = group[group["order_placed"] > 0]["date"]
            order_values = group[group["order_placed"] > 0]["order_placed"]
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
        ax.set_ylabel("Quantity", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.4, linestyle="-", linewidth=0.5)
        ax.legend(loc="upper right", fontsize=10)

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
                forecast_group = first_risk_forecasts[forecast_filter]

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
        ax.set_title(
            "Demand Analysis: Actual vs Forecasted vs Orders",
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
