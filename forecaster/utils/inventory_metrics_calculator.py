"""
Standardized inventory metrics calculation engine.
Handles calculation of all inventory comparison metrics in a consistent, performant way.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class InventoryMetricsCalculator:
    """Standardized inventory metrics calculation engine"""
    
    def __init__(self, data_loader=None):
        """
        Initialize the calculator.
        
        Args:
            data_loader: Optional DataLoader instance. If not provided, will create a new one.
        """
        from data.loader import DataLoader
        self.data_loader = data_loader if data_loader is not None else DataLoader()
    
    def calculate_all_metrics_vectorized(self, group: pd.DataFrame, inventory_cost: float) -> Dict[str, Any]:
        """
        Calculate all metrics using vectorized operations for performance
        
        Args:
            group: DataFrame containing simulation data for a single product-location-method
            inventory_cost: Unit cost of inventory from product master
            
        Returns:
            Dictionary containing all calculated metrics
        """
        # Validate input data
        self.validate_simulation_data(group)
        
        # Check if we should calculate metrics after leadtime
        calc_after_lt = self.data_loader.config.get('simulation', {}).get('calc_metrics_after_lt', False)
        leadtime = group['leadtime'].iloc[0]  # Assuming leadtime is constant for the group
        
        if calc_after_lt:
            # Skip the first leadtime
            start_idx = leadtime
            group = group.iloc[start_idx:]
            if group.empty:
                raise ValueError("No data available after risk period")
        
        # Extract arrays for vectorized operations
        actual_inventory = group["actual_inventory"].values
        simulated_inventory = group["inventory_on_hand"].values
        actual_demand = group["actual_demand"].values
        max_level = group["max_level"].values
        min_level = group["min_level"].values
        on_order = group["inventory_on_order"].values
        total_days = len(group)
        
        # Vectorized calculations for basic metrics
        # Inventory levels
        actual_metrics = {
            "avg": np.mean(actual_inventory),
            "min": np.min(actual_inventory),
            "max": np.max(actual_inventory)
        }
        simulated_metrics = {
            "avg": np.mean(simulated_inventory),
            "min": np.min(simulated_inventory),
            "max": np.max(simulated_inventory)
        }
        
        # Stockout calculations (vectorized)
        stockout_mask_actual = actual_inventory <= 0
        stockout_mask_sim = simulated_inventory <= 0
        actual_stockout_days = np.sum(stockout_mask_actual)
        simulated_stockout_days = np.sum(stockout_mask_sim)
        actual_stockout_rate = (actual_stockout_days / total_days) * 100
        simulated_stockout_rate = (simulated_stockout_days / total_days) * 100
        
        # Service level calculations (vectorized)
        demand_mask = actual_demand > 0
        demand_periods = np.sum(demand_mask)
        if demand_periods > 0:
            actual_demand_met = np.sum((actual_demand > 0) & (actual_inventory >= actual_demand))
            simulated_demand_met = np.sum((actual_demand > 0) & (simulated_inventory >= actual_demand))
            actual_service_level = (actual_demand_met / demand_periods) * 100
            simulated_service_level = (simulated_demand_met / demand_periods) * 100
        else:
            actual_service_level = simulated_service_level = 100.0
        
        # Inventory days calculation (vectorized)
        avg_daily_demand = np.mean(actual_demand)
        if avg_daily_demand > 0:
            actual_inventory_days = actual_metrics["avg"] / avg_daily_demand
            simulated_inventory_days = simulated_metrics["avg"] / avg_daily_demand
        else:
            actual_inventory_days = simulated_inventory_days = 0.0
        
        # Overstock/understock calculations (vectorized)
        overstock_mask_actual = actual_inventory > max_level
        overstock_mask_sim = simulated_inventory > max_level
        understock_mask_actual = actual_inventory < min_level
        understock_mask_sim = simulated_inventory < min_level
        
        overstock_metrics = {
            "actual_overstock_percentage": (np.sum(overstock_mask_actual) / total_days) * 100,
            "simulated_overstock_percentage": (np.sum(overstock_mask_sim) / total_days) * 100,
            "actual_understock_percentage": (np.sum(understock_mask_actual) / total_days) * 100,
            "simulated_understock_percentage": (np.sum(understock_mask_sim) / total_days) * 100
        }
        
        # Cost metrics calculation (vectorized) - only considering on-hand inventory
        actual_total_inventory_units = actual_metrics["avg"]
        simulated_total_inventory_units = simulated_metrics["avg"]
        
        # Calculate Cost of Goods Sold (COGS) using actual demand
        total_actual_demand = np.sum(actual_demand)
        cogs = total_actual_demand * inventory_cost if inventory_cost > 0 else total_actual_demand
        
        # Calculate average inventory value
        avg_actual_inventory_value = actual_total_inventory_units * inventory_cost if inventory_cost > 0 else actual_total_inventory_units
        avg_simulated_inventory_value = simulated_total_inventory_units * inventory_cost if inventory_cost > 0 else simulated_total_inventory_units
        
        # Calculate inventory turnover ratios
        actual_turnover_ratio = round(cogs / avg_actual_inventory_value, 2) if avg_actual_inventory_value > 0 else 0.0
        simulated_turnover_ratio = round(cogs / avg_simulated_inventory_value, 2) if avg_simulated_inventory_value > 0 else 0.0
        
        # Calculate surplus stock percentage (vectorized)
        # Handle division by zero by masking where inventory is zero
        actual_mask = actual_inventory > 0
        rolling_max = group['rolling_max_inventory'].values
        
        # Calculate surplus percentage where inventory > 0
        actual_surplus_pct = np.where(
            actual_mask,
            ((actual_inventory - rolling_max) / actual_inventory) * 100,
            -100
        )
        
        # Calculate averages, excluding any invalid values
        actual_surplus_avg = max(0, np.mean(actual_surplus_pct[actual_mask])) if np.any(actual_mask) else 0
        
        cost_metrics = {
            "actual_total_inventory_units": actual_total_inventory_units,
            "simulated_total_inventory_units": simulated_total_inventory_units,
            "actual_total_inventory_cost": actual_total_inventory_units * inventory_cost,
            "simulated_total_inventory_cost": simulated_total_inventory_units * inventory_cost,
            "actual_turnover_ratio": actual_turnover_ratio,
            "simulated_turnover_ratio": simulated_turnover_ratio
        }
        
        # Missed demand calculation (vectorized)
        actual_missed_demand = np.sum(actual_demand[actual_inventory < actual_demand])
        simulated_missed_demand = np.sum(actual_demand[simulated_inventory < actual_demand])
        
        # Combine all metrics with proper rounding
        return {
            # Basic inventory metrics
            "actual_avg_inventory": round(actual_metrics["avg"], 0),
            "actual_min_inventory": round(actual_metrics["min"], 0),
            "actual_max_inventory": round(actual_metrics["max"], 0),
            "simulated_avg_inventory": round(simulated_metrics["avg"], 0),
            "simulated_min_inventory": round(simulated_metrics["min"], 0),
            "simulated_max_inventory": round(simulated_metrics["max"], 0),
            
            # Service level metrics
            "actual_service_level": round(actual_service_level, 2),
            "simulated_service_level": round(simulated_service_level, 2),
            
            # Stockout metrics
            "actual_stockout_days": int(actual_stockout_days),
            "simulated_stockout_days": int(simulated_stockout_days),
            "actual_stockout_rate": round(actual_stockout_rate, 2),
            "simulated_stockout_rate": round(simulated_stockout_rate, 2),
            
            # Inventory days metrics
            "actual_inventory_days": round(actual_inventory_days, 2),
            "simulated_inventory_days": round(simulated_inventory_days, 2),
            
            # Overstock/understock metrics
            "actual_overstock_percentage": round(overstock_metrics["actual_overstock_percentage"], 2),
            "simulated_overstock_percentage": round(overstock_metrics["simulated_overstock_percentage"], 2),
            "actual_understock_percentage": round(overstock_metrics["actual_understock_percentage"], 2),
            "simulated_understock_percentage": round(overstock_metrics["simulated_understock_percentage"], 2),
            
            # Cost metrics
            "actual_total_inventory_units": round(cost_metrics["actual_total_inventory_units"], 0),
            "simulated_total_inventory_units": round(cost_metrics["simulated_total_inventory_units"], 0),
            "actual_total_inventory_cost": round(cost_metrics["actual_total_inventory_cost"], 2),
            "simulated_total_inventory_cost": round(cost_metrics["simulated_total_inventory_cost"], 2),
            
            # Missed demand metrics
            "actual_missed_demand": round(actual_missed_demand, 0),
            "simulated_missed_demand": round(simulated_missed_demand, 0),
            
            # Difference metrics
            "total_days": total_days,
            "inventory_difference": round(cost_metrics["actual_total_inventory_cost"] - cost_metrics["simulated_total_inventory_cost"], 2),
            "inventory_difference_percentage": round(
                ((cost_metrics["actual_total_inventory_cost"] - cost_metrics["simulated_total_inventory_cost"]) / cost_metrics["actual_total_inventory_cost"] * 100)
                if cost_metrics["actual_total_inventory_cost"] > 0 else 0, 2
            ),
            "stockout_rate_difference": round(simulated_stockout_rate - actual_stockout_rate, 2),
            "service_level_difference": round(simulated_service_level - actual_service_level, 2),
            "inventory_days_difference": round(simulated_inventory_days - actual_inventory_days, 2),
            "overstock_difference": round(
                overstock_metrics["simulated_overstock_percentage"] - overstock_metrics["actual_overstock_percentage"], 2
            ),
            "understock_difference": round(
                overstock_metrics["simulated_understock_percentage"] - overstock_metrics["actual_understock_percentage"], 2
            ),
            "missed_demand_difference": round(simulated_missed_demand - actual_missed_demand, 0),
            "total_inventory_units_difference": round(
                cost_metrics["simulated_total_inventory_units"] - cost_metrics["actual_total_inventory_units"], 0
            ),
            "total_inventory_cost_difference": round(
                cost_metrics["simulated_total_inventory_cost"] - cost_metrics["actual_total_inventory_cost"], 2
            ),
            
            # Inventory turnover metrics
            "actual_turnover_ratio": cost_metrics["actual_turnover_ratio"],
            "simulated_turnover_ratio": cost_metrics["simulated_turnover_ratio"],
            "turnover_ratio_difference": round(
                cost_metrics["simulated_turnover_ratio"] - cost_metrics["actual_turnover_ratio"], 2
            ),
            
            # Surplus stock metrics
            "actual_surplus_stock_percentage": round(actual_surplus_avg, 2),
            
            # Availability metrics
            "actual_availability_percentage": round((1 - np.mean(group["understock_flag"].values)) * 100, 2),
            "simulated_availability_percentage": round((1 - np.mean(group["simulated_understock_flag"].values)) * 100, 2),
            "availability_percentage_difference": round(
                (1 - np.mean(group["simulated_understock_flag"].values)) * 100 - 
                (1 - np.mean(group["understock_flag"].values)) * 100, 
                2
            )
        }
        
    def calculate_all_metrics(self, group: pd.DataFrame, inventory_cost: float) -> Dict[str, Any]:
        """
        Calculate all metrics for a product-location-method group.
        This is now a wrapper around the vectorized implementation.
        
        Args:
            group: DataFrame containing simulation data for a single product-location-method
            inventory_cost: Unit cost of inventory from product master
            
        Returns:
            Dictionary containing all calculated metrics
        """
        return self.calculate_all_metrics_vectorized(group, inventory_cost)
    
    def calculate_service_level(self, demand: np.ndarray, inventory: np.ndarray) -> float:
        """
        Calculate service level (fill rate) as percentage of demand met from inventory
        
        Args:
            demand: Array of actual demand values
            inventory: Array of inventory values
            
        Returns:
            Service level as percentage
        """
        demand_mask = demand > 0
        if not demand_mask.any():
            return 100.0
            
        demand_met = np.sum((demand > 0) & (inventory >= demand))
        total_demand_periods = np.sum(demand > 0)
        
        return (demand_met / total_demand_periods * 100) if total_demand_periods > 0 else 100.0
    
    def calculate_stockout_rate(self, inventory: np.ndarray) -> float:
        """
        Calculate stockout rate as percentage of days with zero inventory
        
        Args:
            inventory: Array of inventory values
            
        Returns:
            Stockout rate as percentage
        """
        stockout_days = np.sum(inventory <= 0)
        total_days = len(inventory)
        
        return (stockout_days / total_days * 100) if total_days > 0 else 0.0
    
    def calculate_inventory_days(self, avg_inventory: float, avg_demand: float) -> float:
        """
        Calculate inventory coverage in days
        
        Args:
            avg_inventory: Average inventory level
            avg_demand: Average daily demand
            
        Returns:
            Number of days of inventory coverage
        """
        return avg_inventory / avg_demand if avg_demand > 0 else 0.0
    
    def calculate_overstock_metrics(self, actual_inventory: np.ndarray, simulated_inventory: np.ndarray, 
                                  max_level: np.ndarray, min_level: np.ndarray) -> Dict[str, float]:
        """
        Calculate overstocking and understocking metrics
        
        Args:
            actual_inventory: Array of actual inventory values
            simulated_inventory: Array of simulated inventory values
            max_level: Array of maximum inventory levels
            min_level: Array of minimum inventory levels
            
        Returns:
            Dictionary of overstock/understock metrics
        """
        total_days = len(actual_inventory)
        if total_days == 0:
            return {
                "actual_overstock_percentage": 0.0,
                "simulated_overstock_percentage": 0.0,
                "actual_understock_percentage": 0.0,
                "simulated_understock_percentage": 0.0
            }
            
        # Calculate overstock days (inventory > max_level)
        actual_overstock_days = np.sum(actual_inventory > max_level)
        simulated_overstock_days = np.sum(simulated_inventory > max_level)
        
        # Calculate understock days (inventory < min_level)
        actual_understock_days = np.sum(actual_inventory < min_level)
        simulated_understock_days = np.sum(simulated_inventory < min_level)
        
        return {
            "actual_overstock_percentage": round(actual_overstock_days / total_days * 100, 2),
            "simulated_overstock_percentage": round(simulated_overstock_days / total_days * 100, 2),
            "actual_understock_percentage": round(actual_understock_days / total_days * 100, 2),
            "simulated_understock_percentage": round(simulated_understock_days / total_days * 100, 2),
            "overstock_difference": round(
                (simulated_overstock_days - actual_overstock_days) / total_days * 100, 2
            ),
            "understock_difference": round(
                (simulated_understock_days - actual_understock_days) / total_days * 100, 2
            )
        }
    
    def calculate_cost_metrics(self, actual_inventory: float, simulated_inventory: float,
                             avg_on_order: float, unit_cost: float) -> Dict[str, float]:
        """
        Calculate cost-related metrics
        
        Args:
            actual_inventory: Average actual inventory level
            simulated_inventory: Average simulated inventory level
            avg_on_order: Average inventory on order
            unit_cost: Cost per unit of inventory
            
        Returns:
            Dictionary of cost metrics
        """
        # Calculate total inventory units (inventory + on order)
        actual_total_inventory_units = actual_inventory + avg_on_order
        simulated_total_inventory_units = simulated_inventory + avg_on_order
        
        # Calculate total inventory costs
        actual_total_inventory_cost = actual_total_inventory_units * unit_cost
        simulated_total_inventory_cost = simulated_total_inventory_units * unit_cost
        
        return {
            "actual_total_inventory_units": round(actual_total_inventory_units, 0),
            "simulated_total_inventory_units": round(simulated_total_inventory_units, 0),
            "actual_total_inventory_cost": round(actual_total_inventory_cost, 2),
            "simulated_total_inventory_cost": round(simulated_total_inventory_cost, 2),
            "total_inventory_units_difference": round(
                simulated_total_inventory_units - actual_total_inventory_units, 0
            ),
            "total_inventory_cost_difference": round(
                simulated_total_inventory_cost - actual_total_inventory_cost, 2
            )
        }
        
    def validate_simulation_data(self, data: pd.DataFrame) -> bool:
        """
        Validate simulation data quality and completeness.
        Performs comprehensive validation of simulation data including:
        - Required columns
        - Data types
        - Value ranges
        - Logical relationships
        - Data quality checks
        
        Args:
            data: DataFrame containing simulation data
            
        Returns:
            True if validation passes
            
        Raises:
            ValueError: If validation fails with details about the failure
            TypeError: If data types are incorrect
        """
        # 1. Basic DataFrame Validation
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
            
        if data.empty:
            raise ValueError("Input DataFrame is empty")
            
        # 2. Required Columns Check
        required_columns = {
            "actual_inventory": np.number,
            "inventory_on_hand": np.number,
            "actual_demand": np.number,
            "min_level": np.number,
            "max_level": np.number,
            "inventory_on_order": np.number,
            "date": "datetime64[ns]"
        }
        
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        # 3. Data Type Validation
        for col, expected_type in required_columns.items():
            if col == "date":
                if not pd.api.types.is_datetime64_any_dtype(data[col]):
                    raise TypeError(f"Column {col} must be datetime type")
            else:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    raise TypeError(f"Column {col} must be numeric type")
        
        # 4. Value Range Validation
        # Check for negative values where they shouldn't exist
        negative_checks = {
            "inventory_on_hand": "Negative inventory values found in simulation data",
            "actual_demand": "Negative demand values found",
            "min_level": "Negative minimum level values found",
            "max_level": "Negative maximum level values found",
            "inventory_on_order": "Negative inventory on order values found"
        }
        
        for col, message in negative_checks.items():
            negative_mask = data[col] < 0
            if negative_mask.any():
                if col in ["inventory_on_hand", "actual_demand"]:  # Critical columns
                    raise ValueError(message)
                else:  # Non-critical columns
                    logger.warning(message)
                    
        # 5. Logical Relationship Validation
        # Check min_level <= max_level
        if (data["min_level"] > data["max_level"]).any():
            raise ValueError("Found min_level values greater than max_level")
            
        # Check date sequence
        if not data["date"].is_monotonic_increasing:
            raise ValueError("Dates must be in ascending order")
            
        # 6. Data Quality Checks
        # Check for unreasonable values (outliers)
        for col in ["actual_inventory", "inventory_on_hand", "actual_demand"]:
            mean = data[col].mean()
            std = data[col].std()
            if std > 0:  # Only check if we have variation
                extreme_values = data[col] > (mean + 5 * std)  # 5 sigma rule
                if extreme_values.any():
                    logger.warning(f"Found potential outliers in {col}")
                    
        # 7. Missing Value Checks
        null_counts = data[list(required_columns.keys())].isnull().sum()
        if null_counts.any():
            null_columns = null_counts[null_counts > 0].index.tolist()
            raise ValueError(f"Found missing values in columns: {null_columns}")
            
        # 8. Consistency Checks
        # Check if inventory changes are consistent with demand and orders
        inventory_changes = data["inventory_on_hand"].diff()
        demand_effect = -data["actual_demand"]
        incoming_effect = data["inventory_on_order"].shift(1).fillna(0)
        
        # Allow for small numerical differences due to floating point arithmetic
        tolerance = 1e-10
        inconsistent_changes = np.abs(inventory_changes - (demand_effect + incoming_effect)) > tolerance
        if inconsistent_changes.any():
            logger.warning("Found potentially inconsistent inventory changes")
            
        # 9. Time Series Completeness
        date_diff = data["date"].diff().dropna()
        if not (date_diff == pd.Timedelta(days=1)).all():
            logger.warning("Found gaps in daily data sequence")
            
        # 10. Business Logic Validation
        # Service level should be achievable with given inventory levels
        if data["actual_demand"].sum() > 0:
            current_service_level = (
                (data["actual_demand"] > 0) & 
                (data["inventory_on_hand"] >= data["actual_demand"])
            ).mean() * 100
            
            if current_service_level < 50:  # Arbitrary threshold for example
                logger.warning(f"Low service level detected: {current_service_level:.2f}%")
        
        return True
