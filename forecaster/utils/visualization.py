"""
Visualization module for demand forecasting data.
Provides interactive line graphs with filtering and aggregation capabilities.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Dict, Tuple
from datetime import date, timedelta
import warnings
from statsmodels.tsa.seasonal import seasonal_decompose

# Suppress matplotlib warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


class DemandVisualizer:
    """
    Visualization class for demand data with filtering and aggregation capabilities.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize the visualizer with demand data.

        Args:
            data: DataFrame with demand data (must include columns:
                  product_id, product_category, location_id, date, demand)
        """
        self.data = data.copy()
        self.data["date"] = pd.to_datetime(self.data["date"])

        # Set style
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

    def get_available_filters(self) -> Dict[str, List]:
        """
        Get available filter options from the data.

        Returns:
            Dictionary with available locations, categories, and products
        """
        return {
            "locations": sorted(self.data["location_id"].unique().tolist()),
            "categories": sorted(self.data["product_category"].unique().tolist()),
            "products": sorted(self.data["product_id"].unique().tolist()),
        }

    def get_filtered_data(
        self,
        locations: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        products: Optional[List[str]] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """
        Get filtered data based on specified criteria.
        This is a convenience method that calls filter_data.

        Args:
            locations: List of location IDs to include
            categories: List of product categories to include
            products: List of product IDs to include
            start_date: Start date for filtering
            end_date: End date for filtering

        Returns:
            Filtered DataFrame
        """
        return self.filter_data(
            locations=locations,
            categories=categories,
            products=products,
            start_date=start_date,
            end_date=end_date,
        )

    def filter_data(
        self,
        locations: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        products: Optional[List[str]] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """
        Filter data based on specified criteria.

        Args:
            locations: List of location IDs to include
            categories: List of product categories to include
            products: List of product IDs to include
            start_date: Start date for filtering
            end_date: End date for filtering

        Returns:
            Filtered DataFrame
        """
        filtered_data = self.data.copy()

        if locations:
            # Case-insensitive matching for locations
            location_matches = []
            for loc in locations:
                matches = filtered_data[
                    filtered_data["location_id"].str.lower() == loc.lower()
                ]
                if not matches.empty:
                    location_matches.append(matches)
            if location_matches:
                filtered_data = pd.concat(location_matches).drop_duplicates()

        if categories:
            # Case-insensitive matching for categories
            category_matches = []
            for cat in categories:
                matches = filtered_data[
                    filtered_data["product_category"].str.lower() == cat.lower()
                ]
                if not matches.empty:
                    category_matches.append(matches)
            if category_matches:
                filtered_data = pd.concat(category_matches).drop_duplicates()

        if products:
            # Case-insensitive matching for products
            product_matches = []
            for prod in products:
                matches = filtered_data[
                    filtered_data["product_id"].str.lower() == prod.lower()
                ]
                if not matches.empty:
                    product_matches.append(matches)
            if product_matches:
                filtered_data = pd.concat(product_matches).drop_duplicates()

        if start_date:
            # Convert start_date to datetime for comparison
            start_datetime = pd.to_datetime(start_date)
            filtered_data = filtered_data[filtered_data["date"] >= start_datetime]

        if end_date:
            # Convert end_date to datetime for comparison
            end_datetime = pd.to_datetime(end_date)
            filtered_data = filtered_data[filtered_data["date"] <= end_datetime]

        return filtered_data

    def aggregate_data(
        self,
        data: pd.DataFrame,
        group_by: List[str],
        time_aggregation: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Aggregate data by specified grouping columns and time period.

        Args:
            data: DataFrame to aggregate
            group_by: List of columns to group by (e.g., ['date', 'location_id'])
            time_aggregation: Time aggregation level ('daily', 'weekly', 'monthly', 'risk_period', None)

        Returns:
            Aggregated DataFrame
        """
        # Ensure date is in the group_by for time series
        if "date" not in group_by:
            group_by = ["date"] + group_by

        # Apply time aggregation if specified
        if time_aggregation:
            data = data.copy()
            # Convert date column to datetime for time aggregation operations
            data["date"] = pd.to_datetime(data["date"])

            if time_aggregation == "weekly":
                # Convert to weekly periods and get start of week
                data["date"] = data["date"].dt.to_period("W").dt.start_time
            elif time_aggregation == "monthly":
                # Convert to monthly periods and get start of month
                data["date"] = data["date"].dt.to_period("M").dt.start_time
            elif time_aggregation == "risk_period":
                # Use risk period aggregation based on product master data
                data = self._aggregate_by_risk_period(data)
            # For 'daily', no change needed

        # Aggregate demand by summing
        aggregated = data.groupby(group_by)["demand"].sum().reset_index()

        return aggregated.sort_values("date")

    def _aggregate_by_risk_period(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate data by risk periods based on product master configuration.

        Args:
            data: DataFrame with demand data

        Returns:
            DataFrame with risk period aggregated data
        """
        try:
            # Import here to avoid circular imports
            from forecaster.data import DemandDataLoader
            from forecaster.data.product_master_schema import ProductMasterSchema

            # Load product master data
            loader = DemandDataLoader()
            try:
                # Try to load customer product master first
                product_master = loader.load_csv("customer_product_master.csv")
            except:
                # Fallback to daily product master
                product_master = loader.load_product_master_daily()

            # Create aggregator for risk period bucketing
            from forecaster.data.aggregator import DemandAggregator

            aggregator = DemandAggregator(loader)

            # Get the latest date in the data as cutoff
            latest_date = data["date"].max().date()

            # Create risk period buckets
            risk_period_data = aggregator.create_risk_period_buckets_with_data(
                data, product_master, latest_date
            )

            if risk_period_data.empty:
                # Fallback to daily aggregation if no risk period data
                return data

            # Convert bucket data to the expected format
            aggregated_data = []
            for _, row in risk_period_data.iterrows():
                aggregated_data.append(
                    {
                        "date": row["bucket_start_date"],
                        "product_id": row["product_id"],
                        "location_id": row["location_id"],
                        "product_category": row["product_category"],
                        "demand": row["total_demand"],
                    }
                )

            return pd.DataFrame(aggregated_data)

        except Exception as e:
            print(f"Warning: Risk period aggregation failed: {e}")
            # Fallback to daily aggregation
            return data

    def plot_demand_trend(
        self,
        locations: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        products: Optional[List[str]] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        group_by: Optional[List[str]] = None,
        time_aggregation: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6),
        title: Optional[str] = None,
        show_legend: bool = True,
    ) -> plt.Figure:
        """
        Create a line plot of demand trends with filtering and aggregation.

        Args:
            locations: List of location IDs to include
            categories: List of product categories to include
            products: List of product IDs to include
            start_date: Start date for filtering
            end_date: End date for filtering
            group_by: Columns to group by for aggregation (e.g., ['location_id', 'product_category'])
            figsize: Figure size (width, height)
            title: Plot title
            show_legend: Whether to show legend

        Returns:
            Matplotlib figure object
        """
        # Filter data
        filtered_data = self.filter_data(
            locations=locations,
            categories=categories,
            products=products,
            start_date=start_date,
            end_date=end_date,
        )

        if filtered_data.empty:
            raise ValueError("No data matches the specified filters")

        # Determine grouping
        if group_by is None:
            # Default: group by date only (total demand over time)
            group_by = ["date"]

        # Aggregate data
        aggregated_data = self.aggregate_data(filtered_data, group_by, time_aggregation)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot based on grouping
        if len(group_by) == 1 and group_by[0] == "date":
            # Simple time series
            ax.plot(
                aggregated_data["date"],
                aggregated_data["demand"],
                linewidth=2,
                marker="o",
                markersize=4,
            )
            ax.set_ylabel("Total Demand")

        elif "date" in group_by and len(group_by) == 2:
            # Group by date and one other dimension
            other_dim = [col for col in group_by if col != "date"][0]

            for value in aggregated_data[other_dim].unique():
                subset = aggregated_data[aggregated_data[other_dim] == value]
                ax.plot(
                    subset["date"],
                    subset["demand"],
                    linewidth=2,
                    marker="o",
                    markersize=4,
                    label=str(value),
                )

            ax.set_ylabel("Demand")
            if show_legend:
                ax.legend(title=other_dim.replace("_", " ").title())

        elif "date" in group_by and len(group_by) == 3:
            # Group by date and two other dimensions
            other_dims = [col for col in group_by if col != "date"]

            # Create a unique identifier for each combination
            aggregated_data["combination"] = (
                aggregated_data[other_dims[0]] + " - " + aggregated_data[other_dims[1]]
            )

            for combo in aggregated_data["combination"].unique():
                subset = aggregated_data[aggregated_data["combination"] == combo]
                ax.plot(
                    subset["date"],
                    subset["demand"],
                    linewidth=2,
                    marker="o",
                    markersize=4,
                    label=combo,
                )

            ax.set_ylabel("Demand")
            if show_legend:
                ax.legend(
                    title=f"{other_dims[0].replace('_', ' ').title()} - {other_dims[1].replace('_', ' ').title()}"
                )

        # Customize plot
        ax.set_xlabel("Date")
        ax.grid(True, alpha=0.3)

        if title:
            ax.set_title(title)
        else:
            # Generate title based on filters
            title_parts = []
            if locations:
                title_parts.append(f"Locations: {', '.join(locations)}")
            if categories:
                title_parts.append(f"Categories: {', '.join(categories)}")
            if products:
                title_parts.append(f"Products: {', '.join(products)}")

            if time_aggregation:
                title_parts.append(f"Aggregation: {time_aggregation.title()}")

            if title_parts:
                ax.set_title(" | ".join(title_parts))
            else:
                ax.set_title("Demand Trend")

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.tight_layout()

        return fig

    def plot_sku_location_demand(
        self,
        sku_location_pairs: List[Tuple[str, str]],
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        figsize: Tuple[int, int] = (12, 6),
        title: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot demand for specific SKU-location combinations.

        Args:
            sku_location_pairs: List of (product_id, location_id) tuples
            start_date: Start date for filtering
            end_date: End date for filtering
            figsize: Figure size

        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        for product_id, location_id in sku_location_pairs:
            # Filter for specific SKU-location combination
            subset = self.filter_data(
                products=[product_id],
                locations=[location_id],
                start_date=start_date,
                end_date=end_date,
            )

            if not subset.empty:
                # Sort by date
                subset = subset.sort_values("date")
                ax.plot(
                    subset["date"],
                    subset["demand"],
                    linewidth=2,
                    marker="o",
                    markersize=4,
                    label=f"{product_id} - {location_id}",
                )

        ax.set_xlabel("Date")
        ax.set_ylabel("Demand")
        if title:
            ax.set_title(title)
        else:
            ax.set_title("SKU-Location Demand Trends")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        return fig

    def plot_category_comparison(
        self,
        locations: Optional[List[str]] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        time_aggregation: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6),
        title: Optional[str] = None,
    ) -> plt.Figure:
        """
        Compare demand trends across different product categories.

        Args:
            locations: List of location IDs to include
            start_date: Start date for filtering
            end_date: End date for filtering
            figsize: Figure size

        Returns:
            Matplotlib figure object
        """
        # Filter data
        filtered_data = self.filter_data(
            locations=locations, start_date=start_date, end_date=end_date
        )

        # Aggregate by date and category
        aggregated_data = self.aggregate_data(
            filtered_data, ["date", "product_category"], time_aggregation
        )

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

        for category in aggregated_data["product_category"].unique():
            subset = aggregated_data[aggregated_data["product_category"] == category]
            ax.plot(
                subset["date"],
                subset["demand"],
                linewidth=2,
                marker="o",
                markersize=4,
                label=category,
            )

        ax.set_xlabel("Date")
        ax.set_ylabel("Total Demand")
        if title:
            ax.set_title(title)
        else:
            ax.set_title("Demand by Product Category")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        return fig

    def plot_location_comparison(
        self,
        categories: Optional[List[str]] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        time_aggregation: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6),
        title: Optional[str] = None,
    ) -> plt.Figure:
        """
        Compare demand trends across different locations.

        Args:
            categories: List of product categories to include
            start_date: Start date for filtering
            end_date: End date for filtering
            figsize: Figure size

        Returns:
            Matplotlib figure object
        """
        # Filter data
        filtered_data = self.filter_data(
            categories=categories, start_date=start_date, end_date=end_date
        )

        # Aggregate by date and location
        aggregated_data = self.aggregate_data(
            filtered_data, ["date", "location_id"], time_aggregation
        )

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

        for location in aggregated_data["location_id"].unique():
            subset = aggregated_data[aggregated_data["location_id"] == location]
            ax.plot(
                subset["date"],
                subset["demand"],
                linewidth=2,
                marker="o",
                markersize=4,
                label=location,
            )

        ax.set_xlabel("Date")
        ax.set_ylabel("Total Demand")
        if title:
            ax.set_title(title)
        else:
            ax.set_title("Demand by Location")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        return fig

    def plot_demand_decomposition(
        self,
        locations: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        products: Optional[List[str]] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        time_aggregation: Optional[str] = None,
        period: Optional[int] = None,
        figsize: Tuple[int, int] = (15, 10),
        title: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot demand decomposition into trend, seasonal, and residual components.

        Args:
            locations: List of location IDs to include
            categories: List of product categories to include
            products: List of product IDs to include
            start_date: Start date for filtering
            end_date: End date for filtering
            time_aggregation: Time aggregation level ('daily', 'weekly', 'monthly')
            period: Period for seasonal decomposition (e.g., 7 for weekly, 30 for monthly)
            figsize: Figure size
            title: Plot title

        Returns:
            Matplotlib figure with decomposition plots
        """
        # Filter data
        filtered_data = self.filter_data(
            locations=locations,
            categories=categories,
            products=products,
            start_date=start_date,
            end_date=end_date,
        )

        if filtered_data.empty:
            raise ValueError("No data available for the specified filters")

        # Aggregate data by date
        if time_aggregation:
            group_cols = ["date"]
            if locations:
                group_cols.append("location_id")
            if categories:
                group_cols.append("product_category")
            if products:
                group_cols.append("product_id")

            aggregated_data = self.aggregate_data(
                filtered_data, group_cols, time_aggregation
            )
        else:
            aggregated_data = (
                filtered_data.groupby("date")["demand"].sum().reset_index()
            )

        # Sort by date and set as index for decomposition
        aggregated_data = aggregated_data.sort_values("date")
        aggregated_data = aggregated_data.set_index("date")

        # Determine period for seasonal decomposition
        if period is None:
            if time_aggregation == "weekly":
                period = 4  # 4 weeks per month
            elif time_aggregation == "monthly":
                period = 12  # 12 months per year
            else:  # daily
                period = 7  # 7 days per week

        # Ensure we have enough data for decomposition
        if len(aggregated_data) < period * 2:
            raise ValueError(
                f"Insufficient data for decomposition. Need at least {period * 2} periods, got {len(aggregated_data)}"
            )

        # Perform seasonal decomposition
        try:
            decomposition = seasonal_decompose(
                aggregated_data["demand"],
                model="additive",
                period=period,
                extrapolate_trend="freq",
            )
        except Exception as e:
            # Fallback to multiplicative model if additive fails
            try:
                decomposition = seasonal_decompose(
                    aggregated_data["demand"],
                    model="multiplicative",
                    period=period,
                    extrapolate_trend="freq",
                )
            except Exception as e2:
                raise ValueError(f"Failed to perform decomposition: {e2}")

        # Create subplots
        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
        fig.suptitle(
            title or "Demand Decomposition Analysis", fontsize=16, fontweight="bold"
        )

        # Original data
        axes[0].plot(
            aggregated_data.index,
            aggregated_data["demand"],
            color="#667eea",
            linewidth=2,
            label="Original Demand",
        )
        axes[0].set_ylabel("Demand")
        axes[0].set_title("Original Time Series")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # Trend component
        axes[1].plot(
            aggregated_data.index,
            decomposition.trend,
            color="#764ba2",
            linewidth=2,
            label="Trend",
        )
        axes[1].set_ylabel("Demand")
        axes[1].set_title("Trend Component")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        # Seasonal component
        axes[2].plot(
            aggregated_data.index,
            decomposition.seasonal,
            color="#f093fb",
            linewidth=2,
            label="Seasonal",
        )
        axes[2].set_ylabel("Demand")
        axes[2].set_title("Seasonal Component")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()

        # Residual component
        axes[3].plot(
            aggregated_data.index,
            decomposition.resid,
            color="#f5576c",
            linewidth=1,
            label="Residual",
        )
        axes[3].set_ylabel("Demand")
        axes[3].set_title("Residual Component")
        axes[3].set_xlabel("Date")
        axes[3].grid(True, alpha=0.3)
        axes[3].legend()

        # Format x-axis
        plt.xticks(rotation=45)
        plt.tight_layout()

        return fig

    def save_plot(self, fig: plt.Figure, filename: str, dpi: int = 300):
        """
        Save a plot to file.

        Args:
            fig: Matplotlib figure object
            filename: Output filename
            dpi: Resolution for saving
        """
        fig.savefig(filename, dpi=dpi, bbox_inches="tight")
        print(f"Plot saved as {filename}")

    def show_plot(self, fig: plt.Figure):
        """
        Display a plot.

        Args:
            fig: Matplotlib figure object
        """
        plt.show()
