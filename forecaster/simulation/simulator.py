"""
Inventory simulation engine.

Executes inventory simulations for product-location combinations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path
import sys
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from .data_loader import SimulationDataLoader
from .order_policies import OrderPolicy, OrderPolicyFactory
from forecaster.utils.logger import get_logger


class InventorySimulator:
    """
    Main inventory simulation engine.
    
    Executes simulations for product-location combinations using specified order policies.
    Designed for efficient parallel processing of thousands of product lines.
    """
    
    def __init__(self, default_policy: str = "review_ordering",
                 safety_stock_file: str = None,
                 forecast_comparison_file: str = None,
                 log_level: str = "INFO"):
        """
        Initialize the inventory simulator.
        
        Args:
            default_policy: Default order policy to use
            safety_stock_file: Path to safety stock results file (optional)
            forecast_comparison_file: Path to forecast comparison file (optional)
            log_level: Logging level for the simulator
        """
        
        self.default_policy = default_policy
        self.log_level = log_level
        self.data_loader = SimulationDataLoader(
            safety_stock_file=safety_stock_file,
            forecast_comparison_file=forecast_comparison_file,
            log_level=log_level
        )
        
        # Get MOQ configuration from data loader config
        self.enable_moq = self.data_loader.loader.config.get('simulation', {}).get('enable_moq', False)
        
        # Log MOQ status
        logger = get_logger(__name__, level=self.log_level)
        if self.enable_moq:
            logger.info("🔒 MOQ Constraints: Enabled")
        else:
            logger.info("🔓 MOQ Constraints: Disabled")
        
        self.results = {}
        
    def run_single_simulation(self, product_location_key: str, 
                            order_policy: Optional[OrderPolicy] = None) -> Dict[str, Any]:
        """
        Run simulation for a single product-location-method combination.
        
        Args:
            product_location_key: Key in format "product_id_location_id_forecast_method"
            order_policy: Order policy to use (uses default if None)
            
        Returns:
            Dictionary containing simulation results
        """
        logger = get_logger(__name__, level=self.log_level)
        try:
            # Get simulation data
            simulation_data = self.data_loader.get_all_simulation_data()
            
            if product_location_key not in simulation_data:
                raise ValueError(f"No simulation data found for {product_location_key}")
            
            data = simulation_data[product_location_key]
            arrays = data['arrays'].copy()  # Make a copy to avoid modifying original
            period_info = data['period_info']
            
            # Log sunset date and stop ordering date information if present
            sunset_date = period_info.get('sunset_date')
            stop_ordering_date = period_info.get('stop_ordering_date')
            
            if sunset_date is not None:
                logger.info(f"Simulating {product_location_key} with sunset date {sunset_date}, stop ordering date {stop_ordering_date} - simulation will run for {period_info['num_steps']} steps")
                logger.info(f"  Orders will be stopped when date >= {stop_ordering_date}")
            else:
                logger.debug(f"Simulating {product_location_key} with no sunset date - running full simulation")
            
            # Create order policy if not provided
            if order_policy is None:
                order_policy = OrderPolicyFactory.create_policy_with_moq(
                    self.default_policy, 
                    enable_moq=self.enable_moq
                )
            
            # Run simulation
            arrays = self._execute_simulation(arrays, period_info, order_policy)
            
            # Calculate performance metrics
            metrics = self._calculate_performance_metrics(arrays, period_info)
            
            result = {
                'product_location_key': product_location_key,
                'period_info': period_info,
                'arrays': arrays,
                'metrics': metrics,
                'order_policy': order_policy.__class__.__name__,
                'forecast_method': period_info.get('forecast_method', 'unknown')
            }
            
            logger.debug(f"Completed simulation for {product_location_key}")
            return result
            
        except Exception as e:
            logger.error(f"Error in simulation for {product_location_key}: {e}")
            raise
    
    def run_batch_simulation(self, product_location_keys: Optional[List[str]] = None,
                           order_policy: Optional[OrderPolicy] = None,
                           max_workers: Optional[int] = None) -> Dict[str, Dict]:
        """
        Run simulations for multiple product-location-method combinations in parallel.
        
        Args:
            product_location_keys: List of product-location-method keys to simulate
                                 (runs all if None)
            order_policy: Order policy to use (uses default if None)
            max_workers: Maximum number of parallel workers (uses CPU count if None)
            
        Returns:
            Dictionary mapping product-location-method keys to simulation results
        """
        logger = get_logger(__name__, level=self.log_level)
        
        # Get all simulation data first
        simulation_data = self.data_loader.get_all_simulation_data()
        
        if product_location_keys is None:
            product_location_keys = list(simulation_data.keys())
        
        # Filter to only keys that have data
        available_keys = [key for key in product_location_keys if key in simulation_data]
        
        if len(available_keys) == 0:
            logger.warning("No valid product-location-method combinations found")
            return {}
        
        logger.info(f"Starting batch simulation for {len(available_keys)} product-location-method combinations")
        
        # Set up parallel processing
        if max_workers is None:
            max_workers = min(mp.cpu_count(), len(available_keys))
        
        results = {}
        
        # For small batches, run sequentially to avoid overhead
        if len(available_keys) <= 4:
            for key in available_keys:
                try:
                    results[key] = self.run_single_simulation(key, order_policy)
                except Exception as e:
                    logger.error(f"Failed to simulate {key}: {e}")
                    continue
        else:
            # Run in parallel
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_key = {
                    executor.submit(self.run_single_simulation, key, order_policy): key
                    for key in available_keys
                }
                
                # Collect results
                for future in as_completed(future_to_key):
                    key = future_to_key[future]
                    try:
                        results[key] = future.result()
                    except Exception as e:
                        logger.error(f"Failed to simulate {key}: {e}")
                        continue
        
        self.results = results
        logger.info(f"Completed batch simulation. Successfully simulated {len(results)} combinations")
        return results
    

    def _execute_simulation(self, arrays: Dict[str, np.ndarray], 
                          period_info: Dict[str, Any], 
                          order_policy: OrderPolicy) -> Dict[str, np.ndarray]:
        """
        Execute the simulation logic for a single product-location-method.
        
        Args:
            arrays: Simulation arrays
            period_info: Period information
            order_policy: Order policy to use
            
        Returns:
            Updated arrays after simulation
        """
        num_steps = period_info['num_steps']
        leadtime = period_info['leadtime']
        
        # Get stop ordering date information
        stop_ordering_date = period_info.get('stop_ordering_date')
        sunset_date = period_info.get('sunset_date')
        
        # Step through simulation
        for step in range(num_steps):
            # Step 1: Calculate order placed
            # Check if we should stop ordering due to sunset date
            current_date = arrays['date'][step]
            should_stop_ordering = False
            
            if stop_ordering_date is not None and current_date >= stop_ordering_date:
                should_stop_ordering = True
                logger = get_logger(__name__, level=self.log_level)
                logger.debug(f"Step {step}: Stop ordering enforced - current date {current_date} >= stop ordering date {stop_ordering_date}")
            
            if should_stop_ordering:
                order_quantity = 0  # No orders after stop ordering date
            else:
                order_quantity = order_policy.calculate_order(step, arrays, period_info)
            
            arrays['order_placed'][step] = order_quantity
            
            # Update on-order and incoming inventory arrays
            if order_quantity > 0:
                # Add to on-order array for steps step+1 to step+leadtime-1
                for j in range(1, leadtime):
                    if step + j < num_steps:
                        arrays['inventory_on_order'][step + j] += order_quantity
                
                # Add to incoming inventory array at step+leadtime
                if step + leadtime < num_steps:
                    arrays['incoming_inventory'][step + leadtime] += order_quantity
            
            # Step 2: Calculate min and max levels
            safety_stock = arrays['safety_stock'][step]
            frsp = arrays['FRSP'][step]
            arrays['min_level'][step] = safety_stock + frsp
            arrays['max_level'][step] = arrays['min_level'][step] + order_quantity
            
            # Step 3: Update inventory for next step (if not the last step)
            if step < num_steps - 1:
                # Calculate inventory on hand for next step
                current_on_hand = arrays['inventory_on_hand'][step]
                current_demand = arrays['actual_demand'][step]
                next_incoming = arrays['incoming_inventory'][step + 1]
                
                arrays['inventory_on_hand'][step + 1] = max(0, current_on_hand - current_demand + next_incoming)
                
                # Calculate net stock for next step
                next_on_hand = arrays['inventory_on_hand'][step + 1]
                next_on_order = arrays['inventory_on_order'][step + 1]
                arrays['net_stock'][step + 1] = next_on_hand + next_on_order
        
        return arrays
    
    def _calculate_performance_metrics(self, arrays: Dict[str, np.ndarray], 
                                     period_info: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate performance metrics for the simulation.
        
        Args:
            arrays: Simulation arrays
            period_info: Period information
            
        Returns:
            Dictionary of performance metrics
        """
        # Calculate stockout periods
        stockout_periods = np.sum(arrays['inventory_on_hand'] == 0)
        total_periods = len(arrays['inventory_on_hand'])
        stockout_rate = stockout_periods / total_periods if total_periods > 0 else 0
        
        # Calculate average inventory levels
        avg_on_hand = np.mean(arrays['inventory_on_hand'])
        avg_on_order = np.mean(arrays['inventory_on_order'])
        avg_net_stock = np.mean(arrays['net_stock'])
        
        # Calculate inventory turns
        total_demand = np.sum(arrays['actual_demand'])
        avg_inventory = avg_on_hand + avg_on_order
        inventory_turns = total_demand / avg_inventory if avg_inventory > 0 else 0
        
        # Calculate service level (fill rate)
        total_demand_periods = np.sum(arrays['actual_demand'] > 0)
        demand_met_periods = np.sum(
            (arrays['actual_demand'] > 0) & (arrays['inventory_on_hand'] >= arrays['actual_demand'])
        )
        service_level = demand_met_periods / total_demand_periods if total_demand_periods > 0 else 1.0
        
        # Calculate total orders placed
        total_orders = np.sum(arrays['order_placed'])
        
        # Calculate average order size
        non_zero_orders = arrays['order_placed'][arrays['order_placed'] > 0]
        avg_order_size = np.mean(non_zero_orders) if len(non_zero_orders) > 0 else 0
        
        # Calculate forecast accuracy (if we have actual vs forecast)
        forecast_errors = arrays['FRSP'] - arrays['actual_demand']
        mae = np.mean(np.abs(forecast_errors))
        mape = np.mean(np.abs(forecast_errors / (arrays['actual_demand'] + 1e-8))) * 100
        
        metrics = {
            'stockout_rate': stockout_rate,
            'avg_on_hand': avg_on_hand,
            'avg_on_order': avg_on_order,
            'avg_net_stock': avg_net_stock,
            'inventory_turns': inventory_turns,
            'service_level': service_level,
            'total_orders': total_orders,
            'avg_order_size': avg_order_size,
            'forecast_mae': mae,
            'forecast_mape': mape,
            'total_periods': total_periods,
            'stockout_periods': stockout_periods
        }
        
        return metrics
    
    def save_results(self):
        """
        Save simulation results to files using DataLoader.
        """
        logger = get_logger(__name__, level=self.log_level)
        if not self.results:
            logger.warning("No results to save")
            return
        
        # Save summary metrics
        summary_data = []
        for key, result in self.results.items():
            metrics = result['metrics']
            period_info = result['period_info']
            
            summary_row = {
                'product_location_key': key,
                'product_id': period_info['product_id'],
                'location_id': period_info['location_id'],
                'forecast_method': result.get('forecast_method', 'unknown'),
                'order_policy': result['order_policy'],
                'num_steps': period_info['num_steps'],
                'leadtime': period_info['leadtime'],
                **metrics
            }
            summary_data.append(summary_row)
        
        summary_df = pd.DataFrame(summary_data)
        self.data_loader.loader.save_simulation_results(summary_df)
        
        # Save detailed results for each product-location
        for key, result in self.results.items():
            # Convert arrays to DataFrame
            arrays_df = pd.DataFrame(result['arrays'])
            
            # Add metadata
            arrays_df['product_location_key'] = key
            arrays_df['product_id'] = result['period_info']['product_id']
            arrays_df['location_id'] = result['period_info']['location_id']
            arrays_df['forecast_method'] = result.get('forecast_method', 'unknown')
            arrays_df['order_policy'] = result['order_policy']
            arrays_df['leadtime'] = result['period_info']['leadtime']
            
            # Save to file
            safe_key = key.replace('/', '_').replace('\\', '_')
            self.data_loader.loader.save_results(
                arrays_df,
                "simulation/detailed_results",
                f"{safe_key}_simulation.csv"
            )
        
        # Return paths for backward compatibility
        filename = self.data_loader.loader.config['paths']['output_files']['simulation_results']
        summary_file = self.data_loader.loader.get_output_path("simulation", filename)
        detailed_dir = self.data_loader.loader.get_output_path("simulation/detailed_results", "")
        
        return {
            'summary_file': summary_file,
            'detailed_dir': detailed_dir
        } 