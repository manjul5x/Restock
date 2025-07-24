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

# Add the forecaster package to the path
sys.path.append(str(Path(__file__).parent.parent))

from forecaster.data import DemandDataLoader
from forecaster.utils import DemandVisualizer
from forecaster.safety_stocks import SafetyStockCalculator, SafetyStockModels
from forecaster.backtesting.config import BacktestConfig

app = Flask(__name__)

# Global variables to store data and visualizer
data_loader = None
visualizer = None


def initialize_data():
    """Initialize data loader and visualizer"""
    global data_loader, visualizer
    if data_loader is None:
        try:
            data_loader = DemandDataLoader()
            # Load customer demand data instead of dummy data
            daily_data = data_loader.load_customer_demand()
            visualizer = DemandVisualizer(daily_data)
        except Exception as e:
            print(f"Warning: Could not initialize data loader: {e}")
            # Create empty visualizer with dummy data
            data_loader = None
            visualizer = None


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

        # Generate plot based on type
        if plot_type == "demand_trend":
            fig = visualizer.plot_demand_trend(
                locations=locations,
                categories=categories,
                products=products,
                start_date=start_date,
                end_date=end_date,
                time_aggregation=time_aggregation,
                title=f"Demand Trend ({time_aggregation.title()})",
            )
        elif plot_type == "demand_decomposition":
            fig = visualizer.plot_demand_decomposition(
                locations=locations,
                categories=categories,
                products=products,
                start_date=start_date,
                end_date=end_date,
                time_aggregation=time_aggregation,
                title=f"Demand Decomposition ({time_aggregation.title()})",
            )
        elif plot_type == "category_comparison":
            fig = visualizer.plot_category_comparison(
                locations=locations,
                start_date=start_date,
                end_date=end_date,
                time_aggregation=time_aggregation,
                title=f"Category Comparison ({time_aggregation.title()})",
            )
        elif plot_type == "location_comparison":
            fig = visualizer.plot_location_comparison(
                categories=categories,
                start_date=start_date,
                end_date=end_date,
                time_aggregation=time_aggregation,
                title=f"Location Comparison ({time_aggregation.title()})",
            )
        else:
            return jsonify({"error": "Invalid plot type"}), 400

        # Convert plot to base64 string
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format="png", dpi=300, bbox_inches="tight")
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close(fig)  # Close the figure to free memory

        return jsonify(
            {
                "success": True,
                "image": img_str,
                "plot_type": plot_type,
                "time_aggregation": time_aggregation,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/get_data_summary")
def get_data_summary():
    """Get summary statistics about the data"""
    initialize_data()

    try:
        # Calculate some basic statistics
        total_records = len(visualizer.data)
        date_range = f"{visualizer.data['date'].min().strftime('%Y-%m-%d')} to {visualizer.data['date'].max().strftime('%Y-%m-%d')}"
        total_demand = visualizer.data["demand"].sum()
        avg_demand = visualizer.data["demand"].mean()

        return jsonify(
            {
                "total_records": total_records,
                "date_range": date_range,
                "total_demand": f"{total_demand:,.0f}",
                "avg_demand": f"{avg_demand:.2f}",
                "unique_products": visualizer.data["product_id"].nunique(),
                "unique_locations": visualizer.data["location_id"].nunique(),
                "unique_categories": visualizer.data["product_category"].nunique(),
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/safety_stocks")
def safety_stocks():
    """Safety stock visualization page"""
    try:
        # Check for both regular and enhanced safety stock results
        regular_safety_stock_file = (
            Path(__file__).parent.parent
            / "output/safety_stocks/safety_stock_results.csv"
        )
        enhanced_safety_stock_file = (
            Path(__file__).parent.parent
            / "output/enhanced_safety_stocks/safety_stock_results.csv"
        )

        # Load data from both sources if available
        all_safety_stock_data = []

        if regular_safety_stock_file.exists():
            regular_data = pd.read_csv(regular_safety_stock_file)
            regular_data["data_source"] = "Regular Backtest"
            all_safety_stock_data.append(regular_data)

        if enhanced_safety_stock_file.exists():
            enhanced_data = pd.read_csv(enhanced_safety_stock_file)
            enhanced_data["data_source"] = "Enhanced Backtest"
            all_safety_stock_data.append(enhanced_data)

        if not all_safety_stock_data:
            return render_template(
                "safety_stocks.html",
                error="Safety stock results not found. Please run safety stock calculation first.",
            )

        # Combine all data
        safety_stock_data = pd.concat(all_safety_stock_data, ignore_index=True)

        # Convert errors string back to list
        safety_stock_data["errors"] = safety_stock_data["errors"].apply(
            lambda x: (
                [float(e) for e in x.split(",")]
                if pd.notna(x) and x and x != ""
                else []
            )
        )

        # Get available filter options
        products = sorted(safety_stock_data["product_id"].unique().tolist())
        locations = sorted(safety_stock_data["location_id"].unique().tolist())
        review_dates = sorted(safety_stock_data["review_date"].unique().tolist())
        data_sources = sorted(safety_stock_data["data_source"].unique().tolist())

        return render_template(
            "safety_stocks.html",
            products=products,
            locations=locations,
            review_dates=review_dates,
            data_sources=data_sources,
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
        # Check for both regular and enhanced backtest data
        regular_forecast_file = (
            Path(__file__).parent.parent
            / "output/customer_backtest/forecast_visualization_data.csv"
        )
        enhanced_forecast_file = (
            Path(__file__).parent.parent
            / "output/enhanced_backtest/enhanced_forecast_visualization_data.csv"
        )

        # Load data from both sources if available
        all_forecast_data = []

        if regular_forecast_file.exists():
            regular_data = pd.read_csv(regular_forecast_file)
            regular_data["data_source"] = "Regular Backtest"
            all_forecast_data.append(regular_data)

        if enhanced_forecast_file.exists():
            enhanced_data = pd.read_csv(enhanced_forecast_file)
            enhanced_data["data_source"] = "Enhanced Backtest"
            all_forecast_data.append(enhanced_data)

        if not all_forecast_data:
            return render_template(
                "forecast_visualization.html",
                error="Forecast visualization data not found. Please run backtest first.",
            )

        # Combine all data for dropdown options
        forecast_data = pd.concat(all_forecast_data, ignore_index=True)

        # Convert string lists back to actual lists with proper NaN handling
        def safe_eval_list(x):
            if not x or x == "[]":
                return []
            try:
                # Replace 'nan' with 'float("nan")' for proper evaluation
                x_clean = x.replace("nan", 'float("nan")')
                return eval(x_clean)
            except:
                return []

        # Process columns that exist in the combined data
        # Enhanced backtest columns
        if "forecast_values" in forecast_data.columns:
            forecast_data["forecast_values"] = forecast_data["forecast_values"].apply(
                safe_eval_list
            )
        if "actual_values" in forecast_data.columns:
            forecast_data["actual_values"] = forecast_data["actual_values"].apply(
                safe_eval_list
            )
        if "best_parameters" in forecast_data.columns:
            forecast_data["best_parameters"] = forecast_data["best_parameters"].apply(
                safe_eval_list
            )

        # Regular backtest columns
        if "historical_bucket_start_dates" in forecast_data.columns:
            forecast_data["historical_bucket_start_dates"] = forecast_data[
                "historical_bucket_start_dates"
            ].apply(safe_eval_list)
        if "historical_demands" in forecast_data.columns:
            forecast_data["historical_demands"] = forecast_data[
                "historical_demands"
            ].apply(safe_eval_list)
        if "forecast_horizon_start_dates" in forecast_data.columns:
            forecast_data["forecast_horizon_start_dates"] = forecast_data[
                "forecast_horizon_start_dates"
            ].apply(safe_eval_list)
        if "forecast_horizon_actual_demands" in forecast_data.columns:
            forecast_data["forecast_horizon_actual_demands"] = forecast_data[
                "forecast_horizon_actual_demands"
            ].apply(safe_eval_list)
        if "forecast_horizon_forecast_demands" in forecast_data.columns:
            forecast_data["forecast_horizon_forecast_demands"] = forecast_data[
                "forecast_horizon_forecast_demands"
            ].apply(safe_eval_list)
        if "forecast_horizon_errors" in forecast_data.columns:
            forecast_data["forecast_horizon_errors"] = forecast_data[
                "forecast_horizon_errors"
            ].apply(safe_eval_list)

        # Get available filter options
        products = sorted(forecast_data["product_id"].unique().tolist())
        locations = sorted(forecast_data["location_id"].unique().tolist())
        analysis_dates = sorted(forecast_data["analysis_date"].unique().tolist())
        data_sources = sorted(forecast_data["data_source"].unique().tolist())

        return render_template(
            "forecast_visualization.html",
            products=products,
            locations=locations,
            analysis_dates=analysis_dates,
            data_sources=data_sources,
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
        # Load simulation summary data from both sources
        regular_simulation_file = (
            Path(__file__).parent.parent / "output/simulation/simulation_summary.csv"
        )
        enhanced_simulation_file = (
            Path(__file__).parent.parent
            / "output/enhanced_simulation/simulation_summary.csv"
        )

        # Load data from both sources if available
        all_simulation_data = []

        if regular_simulation_file.exists():
            regular_data = pd.read_csv(regular_simulation_file)
            regular_data["data_source"] = "Regular Backtest"
            all_simulation_data.append(regular_data)

        if enhanced_simulation_file.exists():
            enhanced_data = pd.read_csv(enhanced_simulation_file)
            enhanced_data["data_source"] = "Enhanced Backtest"
            all_simulation_data.append(enhanced_data)

        if not all_simulation_data:
            return render_template(
                "simulation_visualization.html",
                error="Simulation data not found. Please run simulation first.",
            )

        # Combine all data for dropdown options
        simulation_data = pd.concat(all_simulation_data, ignore_index=True)

        # Get unique products and locations for filter dropdowns
        products = sorted(simulation_data["product_id"].unique().tolist())
        locations = sorted(simulation_data["location_id"].unique().tolist())
        data_sources = sorted(simulation_data["data_source"].unique().tolist())

        return render_template(
            "simulation_visualization.html",
            products=products,
            locations=locations,
            data_sources=data_sources,
            error=None,
        )
    except Exception as e:
        return render_template(
            "simulation_visualization.html",
            products=[],
            locations=[],
            data_sources=[],
            error=f"Error loading data: {str(e)}",
        )


@app.route("/inventory_comparison")
def inventory_comparison():
    """Inventory comparison page showing actual vs simulated metrics"""
    try:
        # Load simulation summary data
        summary_file = (
            Path(__file__).parent.parent / "output/simulation/simulation_summary.csv"
        )
        if summary_file.exists():
            simulation_summary = pd.read_csv(summary_file)

            # Get unique products and locations for filtering
            products = sorted(simulation_summary["product_id"].unique().tolist())
            locations = sorted(simulation_summary["location_id"].unique().tolist())

            # Calculate aggregated metrics
            aggregated_metrics = {
                "total_products": len(products),
                "total_locations": len(locations),
                "avg_stockout_rate": float(simulation_summary["stockout_rate"].mean()),
                "avg_service_level": float(simulation_summary["service_level"].mean()),
                "avg_inventory_turns": float(
                    simulation_summary["inventory_turns"].mean()
                ),
                "avg_on_hand": float(simulation_summary["avg_on_hand"].mean()),
                "total_orders": float(simulation_summary["total_orders"].sum()),
            }

            return render_template(
                "inventory_comparison.html",
                products=products,
                locations=locations,
                aggregated_metrics=aggregated_metrics,
            )
        else:
            return render_template(
                "inventory_comparison.html",
                products=[],
                locations=[],
                aggregated_metrics={},
                error="No simulation data found",
            )
    except Exception as e:
        return render_template(
            "inventory_comparison.html",
            products=[],
            locations=[],
            aggregated_metrics={},
            error=f"Error loading data: {str(e)}",
        )


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    import numpy as np

    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def calculate_actual_metrics(actual_data, product_master=None):
    """Calculate inventory metrics from actual historical data"""
    import numpy as np

    # Calculate stockout periods (when stock_level = 0)
    stockout_periods = np.sum(actual_data["stock_level"] == 0)
    total_periods = len(actual_data)
    stockout_rate = stockout_periods / total_periods if total_periods > 0 else 0

    # Calculate average inventory levels
    avg_on_hand = np.mean(actual_data["stock_level"])
    avg_on_order = np.mean(actual_data["incoming_inventory"])
    avg_net_stock = avg_on_hand + avg_on_order

    # Calculate inventory turns
    total_demand = np.sum(actual_data["demand"])
    avg_inventory = avg_on_hand + avg_on_order
    inventory_turns = total_demand / avg_inventory if avg_inventory > 0 else 0

    # Calculate service level (fill rate)
    total_demand_periods = np.sum(actual_data["demand"] > 0)
    demand_met_periods = np.sum(
        (actual_data["demand"] > 0)
        & (actual_data["stock_level"] >= actual_data["demand"])
    )
    service_level = (
        demand_met_periods / total_demand_periods if total_demand_periods > 0 else 1.0
    )

    # For actual data, we don't have order information, so we'll estimate
    # based on inventory changes and demand
    actual_data_sorted = actual_data.sort_values("date").copy()
    actual_data_sorted["inventory_change"] = actual_data_sorted["stock_level"].diff()
    actual_data_sorted["estimated_orders"] = np.where(
        actual_data_sorted["inventory_change"] > 0,
        actual_data_sorted["inventory_change"],
        0,
    )

    total_orders = np.sum(actual_data_sorted["estimated_orders"])
    non_zero_orders = actual_data_sorted["estimated_orders"][
        actual_data_sorted["estimated_orders"] > 0
    ]
    avg_order_size = np.mean(non_zero_orders) if len(non_zero_orders) > 0 else 0

    # Calculate working capital tied up in inventory
    working_capital = 0.0
    if product_master is not None:
        # Get unique products in the actual data
        unique_products = actual_data["product_id"].unique()
        for product in unique_products:
            # Get the inventory cost for this product
            product_cost = (
                product_master[product_master["product_id"] == product][
                    "inventory_cost"
                ].iloc[0]
                if len(product_master[product_master["product_id"] == product]) > 0
                else 0.0
            )

            # Calculate average on-hand for this product
            product_data = actual_data[actual_data["product_id"] == product]
            product_avg_on_hand = np.mean(product_data["stock_level"])

            # Add to total working capital
            working_capital += product_avg_on_hand * product_cost

    # Calculate forecast accuracy (we'll use a simple comparison)
    # For actual data, we'll assume perfect forecasting for comparison
    forecast_mae = 0.0  # Since we're comparing actual vs actual

    return {
        "stockout_rate": float(stockout_rate),
        "avg_on_hand": float(avg_on_hand),
        "avg_on_order": float(avg_on_order),
        "avg_net_stock": float(avg_net_stock),
        "inventory_turns": float(inventory_turns),
        "service_level": float(service_level),
        "total_orders": float(total_orders),
        "avg_order_size": float(avg_order_size),
        "forecast_mae": float(forecast_mae),
        "total_periods": int(total_periods),
        "stockout_periods": int(stockout_periods),
        "working_capital": float(working_capital),
    }


@app.route("/get_comparison_data", methods=["POST"])
def get_comparison_data():
    """Get comparison data for selected filters"""
    try:
        # Get filter parameters
        selected_products = request.form.getlist("products[]")
        selected_locations = request.form.getlist("locations[]")

        # Load simulation summary
        summary_file = (
            Path(__file__).parent.parent / "output/simulation/simulation_summary.csv"
        )
        if not summary_file.exists():
            return jsonify({"error": "Simulation summary not found"}), 404

        simulation_summary = pd.read_csv(summary_file)

        # Load actual inventory data
        actual_data_file = (
            Path(__file__).parent.parent / "forecaster/data/customer_demand.csv"
        )
        if not actual_data_file.exists():
            return jsonify({"error": "Actual inventory data not found"}), 404

        actual_data = pd.read_csv(actual_data_file)
        actual_data["date"] = pd.to_datetime(actual_data["date"])

        # Load product master data for inventory costs
        product_master_file = (
            Path(__file__).parent.parent / "forecaster/data/customer_product_master.csv"
        )
        if not product_master_file.exists():
            return jsonify({"error": "Product master data not found"}), 404

        product_master = pd.read_csv(product_master_file)

        # Apply filters
        if selected_products:
            simulation_summary = simulation_summary[
                simulation_summary["product_id"].isin(selected_products)
            ]
            actual_data = actual_data[actual_data["product_id"].isin(selected_products)]
        if selected_locations:
            simulation_summary = simulation_summary[
                simulation_summary["location_id"].isin(selected_locations)
            ]
            actual_data = actual_data[
                actual_data["location_id"].isin(selected_locations)
            ]

        if simulation_summary.empty:
            return jsonify({"error": "No data found for selected filters"}), 404

        # Calculate SIMULATED metrics
        simulated_metrics = {
            "stockout_rate": float(simulation_summary["stockout_rate"].mean()),
            "avg_on_hand": float(simulation_summary["avg_on_hand"].mean()),
            "avg_on_order": float(simulation_summary["avg_on_order"].mean()),
            "avg_net_stock": float(simulation_summary["avg_net_stock"].mean()),
            "inventory_turns": float(simulation_summary["inventory_turns"].mean()),
            "service_level": float(simulation_summary["service_level"].mean()),
            "total_orders": float(simulation_summary["total_orders"].sum()),
            "avg_order_size": float(simulation_summary["avg_order_size"].mean()),
            "forecast_mae": float(simulation_summary["forecast_mae"].mean()),
            "total_periods": int(simulation_summary["total_periods"].sum()),
            "stockout_periods": int(simulation_summary["stockout_periods"].sum()),
        }

        # Calculate working capital for simulated metrics
        simulated_working_capital = 0.0
        for _, row in simulation_summary.iterrows():
            product_id = row["product_id"]
            avg_on_hand = row["avg_on_hand"]

            # Get the inventory cost for this product
            product_cost = (
                product_master[product_master["product_id"] == product_id][
                    "inventory_cost"
                ].iloc[0]
                if len(product_master[product_master["product_id"] == product_id]) > 0
                else 0.0
            )

            # Add to total working capital
            simulated_working_capital += avg_on_hand * product_cost

        simulated_metrics["working_capital"] = float(simulated_working_capital)

        # Calculate ACTUAL metrics from historical data
        actual_metrics = calculate_actual_metrics(actual_data, product_master)

        comparison_data = {
            "simulated_metrics": simulated_metrics,
            "actual_metrics": actual_metrics,
            "product_count": len(simulation_summary),
            "date_range": f"{actual_data['date'].min().strftime('%Y-%m-%d')} to {actual_data['date'].max().strftime('%Y-%m-%d')}",
            "filtered_products": selected_products if selected_products else "All",
            "filtered_locations": selected_locations if selected_locations else "All",
        }

        # Convert any remaining numpy types
        comparison_data = convert_numpy_types(comparison_data)

        return jsonify(comparison_data)

    except Exception as e:
        import traceback

        print(f"Error in comparison data: {e}")
        print(traceback.format_exc())
        return jsonify({"error": f"Error generating comparison data: {str(e)}"}), 500


@app.route("/get_safety_stock_plot", methods=["POST"])
def get_safety_stock_plot():
    """Generate safety stock distribution plot"""
    try:
        print("🔍 Safety Stock Plot Request Received")
        print("=" * 40)

        # Get form data
        product = request.form.get("product")
        location = request.form.get("location")
        review_date = request.form.get("review_date")
        data_source = request.form.get("data_source", "All")

        print(f"Product: {product}")
        print(f"Location: {location}")
        print(f"Review Date: {review_date}")
        print(f"Data Source: {data_source}")

        if not all([product, location, review_date]):
            return jsonify({"error": "All fields are required"}), 400

        # Load safety stock data from both sources
        all_safety_stock_data = []

        # Load regular safety stock data
        regular_safety_stock_file = (
            Path(__file__).parent.parent
            / "output/safety_stocks/safety_stock_results.csv"
        )
        if regular_safety_stock_file.exists():
            regular_data = pd.read_csv(regular_safety_stock_file)
            regular_data["data_source"] = "Regular Backtest"
            all_safety_stock_data.append(regular_data)

        # Load enhanced safety stock data
        enhanced_safety_stock_file = (
            Path(__file__).parent.parent
            / "output/enhanced_safety_stocks/safety_stock_results.csv"
        )
        if enhanced_safety_stock_file.exists():
            enhanced_data = pd.read_csv(enhanced_safety_stock_file)
            enhanced_data["data_source"] = "Enhanced Backtest"
            all_safety_stock_data.append(enhanced_data)

        if not all_safety_stock_data:
            return jsonify({"error": "Safety stock results not found"}), 404

        # Combine all data
        safety_stock_data = pd.concat(all_safety_stock_data, ignore_index=True)

        # Convert errors string back to list
        safety_stock_data["errors"] = safety_stock_data["errors"].apply(
            lambda x: (
                [float(e) for e in x.split(",")]
                if pd.notna(x) and x and x != ""
                else []
            )
        )

        # Filter data
        filtered_data = safety_stock_data[
            (safety_stock_data["product_id"] == product)
            & (safety_stock_data["location_id"] == location)
            & (safety_stock_data["review_date"] == review_date)
        ]

        # Filter by data source if specified
        if data_source and data_source != "All":
            filtered_data = filtered_data[filtered_data["data_source"] == data_source]

        if filtered_data.empty:
            return jsonify({"error": "No data found for the selected criteria"}), 404

        # Check if we have multiple data sources for comparison
        if data_source == "All" and len(filtered_data) > 1:
            # Create comparison plot with one-below-the-other layout
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

            # Process each data source
            for idx, (_, row) in enumerate(filtered_data.iterrows()):
                ax = ax1 if idx == 0 else ax2
                source_name = row.get("data_source", "Unknown")

                errors = row["errors"]
                safety_stock = row["safety_stock"]
                distribution_type = row["distribution_type"]
                service_level = row["service_level"]

                if not errors:
                    ax.text(
                        0.5,
                        0.5,
                        "No error data available",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                        fontsize=14,
                        bbox=dict(boxstyle="round", facecolor="lightgray"),
                    )
                    ax.set_title(f"{source_name} - {product} at {location}")
                    continue

                # Generate plot data
                models = SafetyStockModels()
                plot_data = models.get_distribution_plot_data(errors, distribution_type)

                # Plot histogram first
                if plot_data["histogram"]["x"] and plot_data["histogram"]["y"]:
                    hist_x = plot_data["histogram"]["x"]
                    hist_y = plot_data["histogram"]["y"]

                    # Use bin edges for gap-free plotting if available
                    if "bin_edges" in plot_data["histogram"]:
                        bin_edges = plot_data["histogram"]["bin_edges"]
                        # Plot bars using bin edges for exact positioning
                        ax.bar(
                            bin_edges[:-1],
                            hist_y,
                            alpha=0.7,
                            color="lightblue",
                            label="Error Count",
                            width=np.diff(bin_edges),
                            align="edge",
                        )
                    else:
                        # Fallback to center-based plotting
                        if len(hist_x) > 1:
                            bar_width = (max(hist_x) - min(hist_x)) / len(hist_x) * 0.8
                        else:
                            bar_width = 20  # Default width
                        ax.bar(
                            hist_x,
                            hist_y,
                            alpha=0.7,
                            color="lightblue",
                            label="Error Count",
                            width=bar_width,
                        )

                # Create a secondary y-axis for the KDE density
                ax2_density = ax.twinx()

                # Plot KDE line on secondary axis
                if plot_data["x"] and plot_data["y"]:
                    ax2_density.plot(
                        plot_data["x"],
                        plot_data["y"],
                        "b-",
                        linewidth=2,
                        label=f"{distribution_type.upper()} Density",
                    )

                # Plot service level line on primary axis
                ax.axvline(
                    x=safety_stock,
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    label=f"Safety Stock ({safety_stock:.2f})",
                )

                # Add labels and title
                ax.set_xlabel("Forecast Error")
                ax.set_ylabel("Error Count", color="blue")
                ax2_density.set_ylabel("Density", color="red")
                ax.set_title(
                    f"{source_name} - Safety Stock Distribution\n{product} at {location} - {review_date}"
                )

                # Add legends
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2_density.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

                ax.grid(True, alpha=0.3)

            plt.tight_layout()

            # Convert plot to base64 string
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format="png", dpi=300, bbox_inches="tight")
            img_buffer.seek(0)
            img_str = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close(fig)

            # Prepare response data for comparison
            response_data = {
                "success": True,
                "image": img_str,
                "data_source": data_source,
            }

            return jsonify(response_data)

        else:
            # Single plot
            row = filtered_data.iloc[0]
            errors = row["errors"]
            safety_stock = row["safety_stock"]
            distribution_type = row["distribution_type"]
            service_level = row["service_level"]
            data_source_used = row.get("data_source", "Unknown")

            if not errors:
                return (
                    jsonify({"error": "No error data available for this selection"}),
                    404,
                )

            # Generate plot
            models = SafetyStockModels()
            plot_data = models.get_distribution_plot_data(errors, distribution_type)

            # Create the plot
            fig, ax = plt.subplots(figsize=(10, 6))

            # Plot histogram first
            if plot_data["histogram"]["x"] and plot_data["histogram"]["y"]:
                hist_x = plot_data["histogram"]["x"]
                hist_y = plot_data["histogram"]["y"]

                # Use bin edges for gap-free plotting if available
                if "bin_edges" in plot_data["histogram"]:
                    bin_edges = plot_data["histogram"]["bin_edges"]
                    # Plot bars using bin edges for exact positioning
                    ax.bar(
                        bin_edges[:-1],
                        hist_y,
                        alpha=0.7,
                        color="lightblue",
                        label="Error Count",
                        width=np.diff(bin_edges),
                        align="edge",
                    )
                else:
                    # Fallback to center-based plotting
                    if len(hist_x) > 1:
                        bar_width = (max(hist_x) - min(hist_x)) / len(hist_x) * 0.8
                    else:
                        bar_width = 20  # Default width
                    ax.bar(
                        hist_x,
                        hist_y,
                        alpha=0.7,
                        color="lightblue",
                        label="Error Count",
                        width=bar_width,
                    )

            # Create a secondary y-axis for the KDE density
            ax2 = ax.twinx()

            # Plot KDE line on secondary axis
            if plot_data["x"] and plot_data["y"]:
                ax2.plot(
                    plot_data["x"],
                    plot_data["y"],
                    "b-",
                    linewidth=2,
                    label=f"{distribution_type.upper()} Density",
                )

            # Plot service level line on primary axis
            ax.axvline(
                x=safety_stock,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Safety Stock ({safety_stock:.2f})",
            )

            # Add labels and title
            ax.set_xlabel("Forecast Error")
            ax.set_ylabel("Error Count", color="blue")
            ax2.set_ylabel("Density", color="red")
            ax.set_title(
                f"Safety Stock Distribution\n{product} at {location} - {review_date}"
            )

            # Add legends
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

            ax.grid(True, alpha=0.3)

            # Convert plot to base64 string
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format="png", dpi=300, bbox_inches="tight")
            img_buffer.seek(0)
            img_str = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close(fig)

            # Prepare response data
            response_data = {
                "success": True,
                "image": img_str,
                "data_source": data_source,
            }

            # Add single plot data if not comparison
            if data_source != "All" or len(filtered_data) == 1:
                response_data.update(
                    {
                        "safety_stock": safety_stock,
                        "error_count": len(errors),
                        "distribution_type": distribution_type,
                        "service_level": service_level,
                        "data_source": data_source_used,
                    }
                )

            return jsonify(response_data)

    except Exception as e:
        import traceback

        error_details = traceback.format_exc()
        print(f"Error in safety stock plot generation: {e}")
        print(error_details)
        return (
            jsonify(
                {"error": f"An error occurred while generating the plot: {str(e)}"}
            ),
            500,
        )


@app.route("/get_simulation_plot", methods=["POST"])
def get_simulation_plot():
    """Generate inventory simulation plot as PNG image with support for data source selection and comparison"""
    try:
        # Get form data
        product = request.form.get("product")
        location = request.form.get("location")
        data_source = request.form.get("data_source", "All")

        if not all([product, location]):
            return jsonify({"error": "Product and location are required"}), 400

        # Load detailed simulation data from both sources
        all_simulation_data = []

        # Load regular simulation data
        regular_detailed_dir = (
            Path(__file__).parent.parent / "output/simulation/detailed_results"
        )
        regular_simulation_file = (
            regular_detailed_dir / f"{product}_{location}_simulation.csv"
        )
        if regular_simulation_file.exists():
            regular_data = pd.read_csv(regular_simulation_file)
            regular_data["date"] = pd.to_datetime(regular_data["date"])
            regular_data["data_source"] = "Regular Backtest"
            all_simulation_data.append(regular_data)

        # Load enhanced simulation data
        enhanced_detailed_dir = (
            Path(__file__).parent.parent / "output/enhanced_simulation/detailed_results"
        )
        enhanced_simulation_file = (
            enhanced_detailed_dir / f"{product}_{location}_simulation.csv"
        )
        if enhanced_simulation_file.exists():
            enhanced_data = pd.read_csv(enhanced_simulation_file)
            enhanced_data["date"] = pd.to_datetime(enhanced_data["date"])
            enhanced_data["data_source"] = "Enhanced Backtest"
            all_simulation_data.append(enhanced_data)

        if not all_simulation_data:
            return (
                jsonify(
                    {
                        "error": "Simulation data not found for the selected product-location combination"
                    }
                ),
                404,
            )

        # Combine all data
        simulation_data = pd.concat(all_simulation_data, ignore_index=True)

        # Filter by data source if specified
        if data_source and data_source != "All":
            simulation_data = simulation_data[
                simulation_data["data_source"] == data_source
            ]

        if simulation_data.empty:
            return jsonify({"error": "No data found for the selected criteria"}), 404

        # Ensure numeric columns are properly formatted
        numeric_columns = [
            "actual_inventory",
            "inventory_on_hand",
            "actual_demand",
            "safety_stock",
            "order_placed",
            "min_level",
            "max_level",
        ]
        for col in numeric_columns:
            if col in simulation_data.columns:
                simulation_data[col] = pd.to_numeric(
                    simulation_data[col], errors="coerce"
                ).fillna(0)

        # Debug: Print data info
        print(f"Simulation data shape: {simulation_data.shape}")
        print(f"Data sources: {simulation_data['data_source'].unique()}")
        print(f"All simulation data sources: {len(all_simulation_data)}")
        print(
            f"Date range: {simulation_data['date'].min()} to {simulation_data['date'].max()}"
        )

        # Create the plot using matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        # Check if we have multiple data sources for comparison
        if data_source == "All" and len(all_simulation_data) > 1:
            # Create comparison plot with one-below-the-other layout
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

            # Get the original unfiltered data for comparison
            original_simulation_data = pd.concat(all_simulation_data, ignore_index=True)

            # Process each data source separately
            sources = sorted(original_simulation_data["data_source"].unique())
            for idx, source in enumerate(sources):
                ax = ax1 if idx == 0 else ax2
                source_data = original_simulation_data[
                    original_simulation_data["data_source"] == source
                ]

                # Ensure numeric columns are properly formatted
                numeric_columns = [
                    "actual_inventory",
                    "inventory_on_hand",
                    "actual_demand",
                    "safety_stock",
                    "order_placed",
                    "min_level",
                    "max_level",
                ]
                for col in numeric_columns:
                    if col in source_data.columns:
                        source_data[col] = pd.to_numeric(
                            source_data[col], errors="coerce"
                        ).fillna(0)

                # Plot inventory levels
                ax.bar(
                    source_data["date"],
                    source_data["actual_inventory"],
                    alpha=0.4,
                    color="lightblue",
                    label="Actual Inventory",
                    width=1,
                )

                ax.plot(
                    source_data["date"],
                    source_data["inventory_on_hand"],
                    color="blue",
                    linewidth=2,
                    label="Simulated Stock on Hand",
                )

                ax.plot(
                    source_data["date"],
                    source_data["actual_demand"],
                    color="orange",
                    linewidth=2,
                    label="Actual Demand",
                )

                ax.plot(
                    source_data["date"],
                    source_data["safety_stock"],
                    color="red",
                    linewidth=2,
                    linestyle="--",
                    label="Safety Stock",
                )

                ax.plot(
                    source_data["date"],
                    source_data["min_level"],
                    color="green",
                    linewidth=2,
                    linestyle=":",
                    label="Min Level",
                )

                ax.plot(
                    source_data["date"],
                    source_data["max_level"],
                    color="purple",
                    linewidth=2,
                    linestyle=":",
                    label="Max Level",
                )

                # Plot orders placed
                ax.bar(
                    source_data["date"],
                    source_data["order_placed"],
                    color="cyan",
                    alpha=0.7,
                    label="Orders Placed",
                    width=1,
                )

                # Set labels and titles
                ax.set_xlabel("Date")
                ax.set_ylabel("Quantity")
                ax.set_title(
                    f"{source} - Inventory Simulation: {product} at {location}"
                )

                # Format x-axis dates
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
                ax.xaxis.set_major_locator(mdates.MonthLocator())
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

                # Add legend
                ax.legend(loc="upper left")

                # Add grid
                ax.grid(True, alpha=0.3)

            plt.tight_layout()

        else:
            # Single plot
            # Ensure numeric columns are properly formatted
            numeric_columns = [
                "actual_inventory",
                "inventory_on_hand",
                "actual_demand",
                "safety_stock",
                "order_placed",
                "min_level",
                "max_level",
            ]
            for col in numeric_columns:
                if col in simulation_data.columns:
                    simulation_data[col] = pd.to_numeric(
                        simulation_data[col], errors="coerce"
                    ).fillna(0)

            # Create the plot with single y-axis
            fig, ax = plt.subplots(figsize=(15, 8))

            # Plot 1: Inventory Levels
            # Actual inventory as shaded bars
            ax.bar(
                simulation_data["date"],
                simulation_data["actual_inventory"],
                alpha=0.4,
                color="lightblue",
                label="Actual Inventory",
                width=1,
            )

            # Simulated stock on hand as line
            ax.plot(
                simulation_data["date"],
                simulation_data["inventory_on_hand"],
                color="blue",
                linewidth=2,
                label="Simulated Stock on Hand",
            )

            # Actual demand as line
            ax.plot(
                simulation_data["date"],
                simulation_data["actual_demand"],
                color="orange",
                linewidth=2,
                label="Actual Demand",
            )

            # Safety stock as dashed line
            ax.plot(
                simulation_data["date"],
                simulation_data["safety_stock"],
                color="red",
                linewidth=2,
                linestyle="--",
                label="Safety Stock",
            )

            # Min level as dotted line
            ax.plot(
                simulation_data["date"],
                simulation_data["min_level"],
                color="green",
                linewidth=2,
                linestyle=":",
                label="Min Level",
            )

            # Max level as dotted line
            ax.plot(
                simulation_data["date"],
                simulation_data["max_level"],
                color="purple",
                linewidth=2,
                linestyle=":",
                label="Max Level",
            )

            # Plot 2: Orders Placed (same y-axis)
            ax.bar(
                simulation_data["date"],
                simulation_data["order_placed"],
                color="cyan",
                alpha=0.7,
                label="Orders Placed",
                width=1,
            )

            # Set labels and titles
            ax.set_xlabel("Date")
            ax.set_ylabel("Quantity")
            ax.set_title(f"Inventory Simulation: {product} at {location}")

            # Format x-axis dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

            # Add legend
            ax.legend(loc="upper left")

            # Add grid
            ax.grid(True, alpha=0.3)

            # Adjust layout
            plt.tight_layout()

        # Convert plot to base64 string
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format="png", dpi=300, bbox_inches="tight")
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close(fig)

        # Prepare response data
        response_data = {
            "success": True,
            "image": img_str,
            "product": product,
            "location": location,
            "data_source": data_source,
        }

        return jsonify(response_data)

    except Exception as e:
        import traceback

        print(f"Error in simulation plot: {e}")
        print(traceback.format_exc())
        return jsonify({"error": f"Error generating simulation plot: {str(e)}"}), 500


@app.route("/get_forecast_visualization_plot", methods=["POST"])
def get_forecast_visualization_plot():
    """Generate forecast visualization plot with support for data source selection and comparison"""
    try:
        # Get form data
        product = request.form.get("product")
        location = request.form.get("location")
        analysis_date = request.form.get("analysis_date")
        data_source = request.form.get("data_source", "All")

        if not all([product, location, analysis_date]):
            return jsonify({"error": "All fields are required"}), 400

        # Load forecast visualization data from both sources
        all_forecast_data = []

        # Load regular forecast data
        regular_forecast_file = (
            Path(__file__).parent.parent
            / "output/customer_backtest/forecast_visualization_data.csv"
        )
        if regular_forecast_file.exists():
            regular_data = pd.read_csv(regular_forecast_file)
            regular_data["data_source"] = "Regular Backtest"
            all_forecast_data.append(regular_data)

        # Load enhanced forecast data
        enhanced_forecast_file = (
            Path(__file__).parent.parent
            / "output/enhanced_backtest/enhanced_forecast_visualization_data.csv"
        )
        if enhanced_forecast_file.exists():
            enhanced_data = pd.read_csv(enhanced_forecast_file)
            enhanced_data["data_source"] = "Enhanced Backtest"
            all_forecast_data.append(enhanced_data)

        if not all_forecast_data:
            return jsonify({"error": "Forecast visualization data not found"}), 404

        # Combine all data
        forecast_data = pd.concat(all_forecast_data, ignore_index=True)

        # Convert string lists back to actual lists with proper NaN handling
        def safe_eval_list(x):
            if not x or x == "[]":
                return []
            try:
                # Replace 'nan' with 'float("nan")' for proper evaluation
                x_clean = x.replace("nan", 'float("nan")')
                return eval(x_clean)
            except:
                return []

        # Process columns that exist in the combined data
        # Enhanced backtest columns
        if "forecast_values" in forecast_data.columns:
            forecast_data["forecast_values"] = forecast_data["forecast_values"].apply(
                safe_eval_list
            )
        if "actual_values" in forecast_data.columns:
            forecast_data["actual_values"] = forecast_data["actual_values"].apply(
                safe_eval_list
            )
        if "best_parameters" in forecast_data.columns:
            forecast_data["best_parameters"] = forecast_data["best_parameters"].apply(
                safe_eval_list
            )

        # Regular backtest columns
        if "historical_bucket_start_dates" in forecast_data.columns:
            forecast_data["historical_bucket_start_dates"] = forecast_data[
                "historical_bucket_start_dates"
            ].apply(safe_eval_list)
        if "historical_demands" in forecast_data.columns:
            forecast_data["historical_demands"] = forecast_data[
                "historical_demands"
            ].apply(safe_eval_list)
        if "forecast_horizon_start_dates" in forecast_data.columns:
            forecast_data["forecast_horizon_start_dates"] = forecast_data[
                "forecast_horizon_start_dates"
            ].apply(safe_eval_list)
        if "forecast_horizon_actual_demands" in forecast_data.columns:
            forecast_data["forecast_horizon_actual_demands"] = forecast_data[
                "forecast_horizon_actual_demands"
            ].apply(safe_eval_list)
        if "forecast_horizon_forecast_demands" in forecast_data.columns:
            forecast_data["forecast_horizon_forecast_demands"] = forecast_data[
                "forecast_horizon_forecast_demands"
            ].apply(safe_eval_list)
        if "forecast_horizon_errors" in forecast_data.columns:
            forecast_data["forecast_horizon_errors"] = forecast_data[
                "forecast_horizon_errors"
            ].apply(safe_eval_list)

        # Filter data
        filtered_data = forecast_data[
            (forecast_data["product_id"] == product)
            & (forecast_data["location_id"] == location)
            & (forecast_data["analysis_date"] == analysis_date)
        ]

        # Filter by data source if specified
        if data_source and data_source != "All":
            filtered_data = filtered_data[filtered_data["data_source"] == data_source]

        if filtered_data.empty:
            return jsonify({"error": "No data found for the selected criteria"}), 404

        # Create the plot
        if data_source == "All" and len(filtered_data) > 1:
            # One-below-the-other comparison for better visual comparison
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

            # Process each data source
            for idx, (_, row) in enumerate(filtered_data.iterrows()):
                ax = ax1 if idx == 0 else ax2
                source_name = row.get("data_source", "Unknown")

                # Determine data format and extract data
                if "forecast_values" in row and row["forecast_values"]:
                    # Enhanced format
                    forecast_values = row["forecast_values"]
                    actual_values = row.get("actual_values", [])
                    best_parameters = row.get("best_parameters", {})

                    # Create x-axis indices
                    x_indices = list(range(len(forecast_values)))

                    # Plot forecast demand
                    ax.plot(
                        x_indices,
                        forecast_values,
                        "r-s",
                        linewidth=2,
                        markersize=6,
                        label="Forecast Demand",
                    )

                    # Plot actual demand if available
                    if actual_values and any(actual_values):
                        ax.plot(
                            x_indices,
                            actual_values,
                            "b-o",
                            linewidth=2,
                            markersize=6,
                            label="Actual Demand",
                        )

                    # Add best parameters info
                    if best_parameters:
                        param_text = f"Best Parameters: {best_parameters}"
                        ax.text(
                            0.02,
                            0.98,
                            param_text,
                            transform=ax.transAxes,
                            fontsize=10,
                            verticalalignment="top",
                            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
                        )

                    ax.set_xlabel("Time Period")
                    ax.set_ylabel("Demand")
                    ax.set_title(f"{source_name} - {product} at {location}")
                    ax.legend()
                    ax.grid(True, alpha=0.3)

                elif "historical_demands" in row and row["historical_demands"]:
                    # Regular format
                    historical_dates = row.get("historical_bucket_start_dates", [])
                    historical_demands = row["historical_demands"]
                    forecast_dates = row.get("forecast_horizon_start_dates", [])
                    actual_demands = row.get("forecast_horizon_actual_demands", [])
                    forecast_demands = row.get("forecast_horizon_forecast_demands", [])

                    # Convert dates for plotting
                    hist_dates = []
                    for d in historical_dates:
                        if isinstance(d, str):
                            if " " in d:  # datetime format
                                hist_dates.append(date.fromisoformat(d.split(" ")[0]))
                            else:  # date format
                                hist_dates.append(date.fromisoformat(d))
                        else:
                            hist_dates.append(d)

                    # Plot historical demand as bars
                    if hist_dates and historical_demands:
                        ax.bar(
                            hist_dates,
                            historical_demands,
                            alpha=0.7,
                            color="lightblue",
                            label="Historical Demand",
                        )

                    # Plot forecast and actual demand
                    if forecast_dates and forecast_demands:
                        forecast_dates_conv = []
                        for d in forecast_dates:
                            if isinstance(d, str):
                                if " " in d:
                                    forecast_dates_conv.append(
                                        date.fromisoformat(d.split(" ")[0])
                                    )
                                else:
                                    forecast_dates_conv.append(date.fromisoformat(d))
                            else:
                                forecast_dates_conv.append(d)

                        ax.plot(
                            forecast_dates_conv,
                            forecast_demands,
                            "r-s",
                            linewidth=2,
                            markersize=6,
                            label="Forecast Demand",
                        )

                    if forecast_dates and actual_demands:
                        forecast_dates_conv = []
                        for d in forecast_dates:
                            if isinstance(d, str):
                                if " " in d:
                                    forecast_dates_conv.append(
                                        date.fromisoformat(d.split(" ")[0])
                                    )
                                else:
                                    forecast_dates_conv.append(date.fromisoformat(d))
                            else:
                                forecast_dates_conv.append(d)

                        ax.plot(
                            forecast_dates_conv,
                            actual_demands,
                            "b-o",
                            linewidth=2,
                            markersize=6,
                            label="Actual Demand",
                        )

                    ax.set_xlabel("Date")
                    ax.set_ylabel("Demand")
                    ax.set_title(f"{source_name} - {product} at {location}")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    ax.tick_params(axis="x", rotation=45)

                else:
                    ax.text(
                        0.5,
                        0.5,
                        "No data available",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                        fontsize=14,
                        bbox=dict(boxstyle="round", facecolor="lightgray"),
                    )
                    ax.set_title(f"{source_name} - {product} at {location}")

            plt.tight_layout()

        else:
            # Single plot
            fig, ax = plt.subplots(figsize=(12, 8))
            row = filtered_data.iloc[0]
            source_name = row.get("data_source", "Unknown")

            # Determine data format and extract data
            if "forecast_values" in row and row["forecast_values"]:
                # Enhanced format
                forecast_values = row["forecast_values"]
                actual_values = row.get("actual_values", [])
                best_parameters = row.get("best_parameters", {})

                # Create x-axis indices
                x_indices = list(range(len(forecast_values)))

                # Plot forecast demand
                ax.plot(
                    x_indices,
                    forecast_values,
                    "r-s",
                    linewidth=2,
                    markersize=6,
                    label="Forecast Demand",
                )

                # Plot actual demand if available
                if actual_values and any(actual_values):
                    ax.plot(
                        x_indices,
                        actual_values,
                        "b-o",
                        linewidth=2,
                        markersize=6,
                        label="Actual Demand",
                    )

                # Add best parameters info
                if best_parameters:
                    param_text = f"Best Parameters: {best_parameters}"
                    ax.text(
                        0.02,
                        0.98,
                        param_text,
                        transform=ax.transAxes,
                        fontsize=10,
                        verticalalignment="top",
                        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
                    )

                ax.set_xlabel("Time Period")
                ax.set_ylabel("Demand")
                ax.set_title(f"{source_name} - {product} at {location}")
                ax.legend()
                ax.grid(True, alpha=0.3)

            elif "historical_demands" in row and row["historical_demands"]:
                # Regular format
                historical_dates = row.get("historical_bucket_start_dates", [])
                historical_demands = row["historical_demands"]
                forecast_dates = row.get("forecast_horizon_start_dates", [])
                actual_demands = row.get("forecast_horizon_actual_demands", [])
                forecast_demands = row.get("forecast_horizon_forecast_demands", [])

                # Convert dates for plotting
                hist_dates = []
                for d in historical_dates:
                    if isinstance(d, str):
                        if " " in d:  # datetime format
                            hist_dates.append(date.fromisoformat(d.split(" ")[0]))
                        else:  # date format
                            hist_dates.append(date.fromisoformat(d))
                    else:
                        hist_dates.append(d)

                # Plot historical demand as bars
                if hist_dates and historical_demands:
                    ax.bar(
                        hist_dates,
                        historical_demands,
                        alpha=0.7,
                        color="lightblue",
                        label="Historical Demand",
                    )

                # Plot forecast and actual demand
                if forecast_dates and forecast_demands:
                    forecast_dates_conv = []
                    for d in forecast_dates:
                        if isinstance(d, str):
                            if " " in d:
                                forecast_dates_conv.append(
                                    date.fromisoformat(d.split(" ")[0])
                                )
                            else:
                                forecast_dates_conv.append(date.fromisoformat(d))
                        else:
                            forecast_dates_conv.append(d)

                    ax.plot(
                        forecast_dates_conv,
                        forecast_demands,
                        "r-s",
                        linewidth=2,
                        markersize=6,
                        label="Forecast Demand",
                    )

                if forecast_dates and actual_demands:
                    forecast_dates_conv = []
                    for d in forecast_dates:
                        if isinstance(d, str):
                            if " " in d:
                                forecast_dates_conv.append(
                                    date.fromisoformat(d.split(" ")[0])
                                )
                            else:
                                forecast_dates_conv.append(date.fromisoformat(d))
                        else:
                            forecast_dates_conv.append(d)

                    ax.plot(
                        forecast_dates_conv,
                        actual_demands,
                        "b-o",
                        linewidth=2,
                        markersize=6,
                        label="Actual Demand",
                    )

                ax.set_xlabel("Date")
                ax.set_ylabel("Demand")
                ax.set_title(f"{source_name} - {product} at {location}")
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis="x", rotation=45)

            else:
                ax.text(
                    0.5,
                    0.5,
                    "No data available",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=14,
                    bbox=dict(boxstyle="round", facecolor="lightgray"),
                )
                ax.set_title(f"{source_name} - {product} at {location}")

        # Convert plot to base64 string
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format="png", dpi=300, bbox_inches="tight")
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()

        # Calculate statistics
        total_historical_demand = 0
        total_actual_demand = 0
        total_forecast_demand = 0
        historical_periods = 0
        forecast_periods = 0

        for _, row in filtered_data.iterrows():
            if "historical_demands" in row and row["historical_demands"]:
                total_historical_demand += sum(row["historical_demands"])
                historical_periods += len(row["historical_demands"])

            if (
                "forecast_horizon_actual_demands" in row
                and row["forecast_horizon_actual_demands"]
            ):
                total_actual_demand += sum(row["forecast_horizon_actual_demands"])

            if (
                "forecast_horizon_forecast_demands" in row
                and row["forecast_horizon_forecast_demands"]
            ):
                total_forecast_demand += sum(row["forecast_horizon_forecast_demands"])
                forecast_periods += len(row["forecast_horizon_forecast_demands"])

            # Enhanced format
            if "actual_values" in row and row["actual_values"]:
                total_actual_demand += sum(row["actual_values"])

            if "forecast_values" in row and row["forecast_values"]:
                total_forecast_demand += sum(row["forecast_values"])
                forecast_periods += len(row["forecast_values"])

        return jsonify(
            {
                "success": True,
                "image": img_base64,
                "total_historical_demand": total_historical_demand,
                "total_actual_demand": total_actual_demand,
                "total_forecast_demand": total_forecast_demand,
                "historical_periods": historical_periods,
                "forecast_periods": forecast_periods,
                "data_source": data_source,
            }
        )

    except Exception as e:
        import traceback

        print(f"Error in get_forecast_visualization_plot: {e}")
        print(traceback.format_exc())
        return (
            jsonify(
                {"error": f"Error generating forecast visualization plot: {str(e)}"}
            ),
            500,
        )


def generate_forecast_with_best_parameters(forecaster, data):
    """Generate forecast data using best parameters for visualization"""
    try:
        from datetime import datetime, timedelta
        import pandas as pd

        # Get the best parameters from the forecaster
        best_parameters = forecaster.get_parameters()

        # Create a new forecaster with best parameters
        from forecaster.forecasting.prophet import ProphetForecaster

        # Extract key parameters
        changepoint_prior_scale = best_parameters.get("changepoint_prior_scale", 0.05)
        seasonality_prior_scale = best_parameters.get("seasonality_prior_scale", 10.0)
        holidays_prior_scale = best_parameters.get("holidays_prior_scale", 10.0)
        seasonality_mode = best_parameters.get("seasonality_mode", "additive")

        # Create optimized forecaster
        optimized_forecaster = ProphetForecaster(
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
            holidays_prior_scale=holidays_prior_scale,
            seasonality_mode=seasonality_mode,
            include_quarterly_effects=best_parameters.get(
                "include_quarterly_effects", True
            ),
            include_monthly_effects=best_parameters.get(
                "include_monthly_effects", True
            ),
            include_indian_holidays=best_parameters.get(
                "include_indian_holidays", True
            ),
            include_festival_seasons=best_parameters.get(
                "include_festival_seasons", True
            ),
            include_monsoon_effect=best_parameters.get("include_monsoon_effect", True),
        )

        # Fit the optimized model
        optimized_forecaster.fit(data)

        # Generate forecast for next 4 months (120 days)
        forecast_horizon = 120
        forecast_series = optimized_forecaster.forecast(forecast_horizon)

        # Prepare historical data
        historical_dates = data["date"].dt.strftime("%Y-%m-%d").tolist()
        historical_demands = data["demand"].tolist()

        # Prepare forecast data
        forecast_dates = forecast_series.index.strftime("%Y-%m-%d").tolist()
        forecast_values = forecast_series.values.tolist()

        return {
            "historical_dates": historical_dates,
            "historical_demands": historical_demands,
            "forecast_dates": forecast_dates,
            "forecast_values": forecast_values,
        }

    except Exception as e:
        print(f"Error generating forecast: {e}")
        return None


@app.route("/seasonality_analysis")
def seasonality_analysis():
    """Seasonality analysis page for Prophet model components"""
    initialize_data()

    # Get available filter options
    if visualizer is not None:
        filters = visualizer.get_available_filters()
    else:
        filters = {"locations": [], "categories": [], "products": []}

    return render_template(
        "seasonality_analysis.html",
        locations=filters["locations"],
        categories=filters["categories"],
        products=filters["products"],
    )


@app.route("/run_seasonality_analysis", methods=["POST"])
def run_seasonality_analysis_route():
    """Run seasonality analysis and return results"""
    try:
        # Get form data
        locations = request.form.getlist("locations")
        categories = request.form.getlist("categories")
        products = request.form.getlist("products")

        # Convert empty lists to None for the visualizer
        locations = locations if locations else None
        categories = categories if categories else None
        products = products if products else None

        # Get filtered data
        if visualizer is None:
            raise ValueError("Data visualizer not initialized")

        filtered_data = visualizer.get_filtered_data(
            locations=locations, categories=categories, products=products
        )

        if filtered_data.empty:
            return jsonify(
                {"success": False, "error": "No data found for the selected filters"}
            )

        # Import Prophet forecaster
        from forecaster.forecasting.prophet import ProphetForecaster

        # Create and fit Prophet model with comprehensive analysis
        forecaster = ProphetForecaster(
            include_quarterly_effects=True,
            include_monthly_effects=True,
            include_indian_holidays=True,
            include_festival_seasons=True,
            include_monsoon_effect=True,
        )

        # Fit the model (this will trigger seasonality analysis)
        forecaster.fit(filtered_data)

        # Get seasonality analysis results
        seasonality_analysis = forecaster.get_seasonality_analysis()

        if seasonality_analysis is None:
            return jsonify({"success": False, "error": "Seasonality analysis failed"})

        # Generate forecast data using best parameters
        forecast_data = generate_forecast_with_best_parameters(
            forecaster, filtered_data
        )

        # Prepare results for frontend
        results = {
            "seasonalities": seasonality_analysis.get("seasonalities", {}),
            "summary": seasonality_analysis.get("summary", {}),
            "recommendations": seasonality_analysis.get("recommendations", {}),
            "model_parameters": forecaster.get_parameters(),
            "forecast_data": forecast_data,
        }

        return jsonify({"success": True, "results": results})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
