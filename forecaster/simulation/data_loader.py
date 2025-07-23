"""
Data loader for inventory simulation.

Handles loading and preparing data for simulation runs.
"""

import pandas as pd
import numpy as np
from datetime import date, timedelta
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


from forecaster.utils.logger import get_logger

logger = get_logger(__name__)


class SimulationDataLoader:
    """
    Loads and prepares data for inventory simulation.
    
    Handles:
    - Loading customer demand, safety stocks, forecasts, and product master data
    - Creating simulation arrays for each product-location combination
    - Determining simulation periods and frequencies
    """
    
    def __init__(self, data_dir: str = "forecaster/data"):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = Path(data_dir)
        
        # Load all required data
        self._load_data()
        
    def _load_data(self):
        """Load all required data files."""
        try:
            # Load customer demand data
            customer_demand_file = self.data_dir / "customer_demand.csv"
            if customer_demand_file.exists():
                self.customer_demand = pd.read_csv(customer_demand_file)
            else:
                raise FileNotFoundError(f"Customer demand file not found: {customer_demand_file}")
            
            # Load product master
            product_master_file = self.data_dir / "customer_product_master.csv"
            if product_master_file.exists():
                self.product_master = pd.read_csv(product_master_file)
            else:
                raise FileNotFoundError(f"Product master file not found: {product_master_file}")
            
            # Load safety stock results
            safety_stock_file = Path("output/safety_stocks/safety_stock_results.csv")
            if safety_stock_file.exists():
                self.safety_stocks = pd.read_csv(safety_stock_file)
                # Convert review_date to datetime
                self.safety_stocks['review_date'] = pd.to_datetime(self.safety_stocks['review_date'])
            else:
                raise FileNotFoundError("Safety stock results not found. Please run safety stock calculation first.")
            
            # Load forecast comparison data
            forecast_file = Path("output/customer_backtest/forecast_comparison.csv")
            if forecast_file.exists():
                self.forecast_comparison = pd.read_csv(forecast_file)
                # Convert analysis_date to datetime
                self.forecast_comparison['analysis_date'] = pd.to_datetime(self.forecast_comparison['analysis_date'])
            else:
                raise FileNotFoundError("Forecast comparison data not found. Please run backtest first.")
            
            logger.info("Successfully loaded all simulation data")
            
        except Exception as e:
            logger.error(f"Error loading simulation data: {e}")
            raise
    
    def get_simulation_periods(self) -> Dict[str, Dict]:
        """
        Get simulation periods for all product-location combinations.
        
        Returns:
            Dictionary mapping product-location keys to simulation period info
        """
        periods = {}
        
        for _, product_record in self.product_master.iterrows():
            product_id = product_record['product_id']
            location_id = product_record['location_id']
            key = f"{product_id}_{location_id}"
            
            # Get safety stocks for this product-location
            product_safety_stocks = self.safety_stocks[
                (self.safety_stocks['product_id'] == product_id) &
                (self.safety_stocks['location_id'] == location_id)
            ]
            
            if len(product_safety_stocks) == 0:
                logger.warning(f"No safety stock data found for {key}")
                continue
            
            # Get first and last review dates
            first_review_date = product_safety_stocks['review_date'].min()
            last_review_date = product_safety_stocks['review_date'].max()
            
            # Determine frequency from customer demand data
            product_demand = self.customer_demand[
                (self.customer_demand['product_id'] == product_id) &
                (self.customer_demand['location_id'] == location_id)
            ]
            
            if len(product_demand) == 0:
                logger.warning(f"No demand data found for {key}")
                continue
            
            # Sort by date and calculate frequency
            # TODO: calculate / set frequency in the beginning of this whole thing
            product_demand = product_demand.sort_values('date')
            product_demand['date'] = pd.to_datetime(product_demand['date'])
            
            if len(product_demand) > 1:
                date_diffs = product_demand['date'].diff().dropna()
                most_common_diff = date_diffs.mode().iloc[0] if len(date_diffs.mode()) > 0 else pd.Timedelta(days=1)
                frequency_days = most_common_diff.days
            else:
                frequency_days = 1
            
            
            # Create date range
            # TODO: we need to do something about making sure frequency and review period and lead time and risk period match up in the way they need to.
            date_range = pd.date_range(
                start=first_review_date,
                end=last_review_date,
                freq=f'{frequency_days}D'
            )
            
            periods[key] = {
                'product_id': product_id,
                'location_id': location_id,
                'first_review_date': first_review_date,
                'last_review_date': last_review_date,
                'frequency_days': frequency_days,
                'date_range': date_range,
                'num_steps': len(date_range),
                'leadtime': product_record['leadtime']
            }
        
        logger.info(f"Created simulation periods for {len(periods)} product-location combinations")
        return periods
    
    def create_simulation_arrays(self, product_location_key: str, period_info: Dict) -> Dict[str, np.ndarray]:
        """
        Create simulation arrays for a specific product-location combination.
        
        Args:
            product_location_key: Key in format "product_id_location_id"
            period_info: Period information from get_simulation_periods()
            
        Returns:
            Dictionary containing all simulation arrays
        """
        product_id = period_info['product_id']
        location_id = period_info['location_id']
        date_range = period_info['date_range']
        num_steps = period_info['num_steps']
        leadtime = period_info['leadtime']
        
        # Initialize arrays
        arrays = {
            'date': date_range,
            'step': np.arange(num_steps),
            'decision_day': np.zeros(num_steps, dtype=int),
            'safety_stock': np.zeros(num_steps),
            'FRSP': np.zeros(num_steps),  # Forecast over first risk period
            'actual_demand': np.zeros(num_steps),
            'inventory_on_hand': np.zeros(num_steps),
            'inventory_on_order': np.zeros(num_steps),
            'net_stock': np.zeros(num_steps),
            'order_placed': np.zeros(num_steps),
            'incoming_inventory': np.zeros(num_steps),
            'actual_inventory': np.zeros(num_steps),
            'min_level': np.zeros(num_steps),
            'max_level': np.zeros(num_steps)
        }
        
        # Populate decision_day array (1 for review dates, 0 otherwise)
        product_safety_stocks = self.safety_stocks[
            (self.safety_stocks['product_id'] == product_id) &
            (self.safety_stocks['location_id'] == location_id)
        ]
        review_dates = set(product_safety_stocks['review_date'].dt.date)
        
        for i, sim_date in enumerate(date_range):
            if sim_date.date() in review_dates:
                arrays['decision_day'][i] = 1
        
        # Populate safety_stock array (forward fill from review dates)
        safety_stock_dict = dict(zip(
            product_safety_stocks['review_date'].dt.date,
            product_safety_stocks['safety_stock']
        ))
        
        current_safety_stock = None
        for i, sim_date in enumerate(date_range):
            if sim_date.date() in safety_stock_dict:
                current_safety_stock = safety_stock_dict[sim_date.date()]
            if current_safety_stock is not None:
                arrays['safety_stock'][i] = current_safety_stock
        
        # Populate FRSP array (forecast for step=1)
        product_forecasts = self.forecast_comparison[
            (self.forecast_comparison['product_id'] == product_id) &
            (self.forecast_comparison['location_id'] == location_id) &
            (self.forecast_comparison['step'] == 1)
        ]
        
        forecast_dict = dict(zip(
            product_forecasts['analysis_date'].dt.date,
            product_forecasts['forecast_demand']
        ))
        
        for i, sim_date in enumerate(date_range):
            if sim_date.date() in forecast_dict:
                arrays['FRSP'][i] = forecast_dict[sim_date.date()]
        
        # Populate actual_demand array
        product_demand = self.customer_demand[
            (self.customer_demand['product_id'] == product_id) &
            (self.customer_demand['location_id'] == location_id)
        ]
        product_demand['date'] = pd.to_datetime(product_demand['date'])
        
        demand_dict = dict(zip(
            product_demand['date'].dt.date,
            product_demand['demand']
        ))
        
        for i, sim_date in enumerate(date_range):
            if sim_date.date() in demand_dict:
                arrays['actual_demand'][i] = demand_dict[sim_date.date()]
        
        # Initialize inventory_on_hand (first entry from stock_level)
        product_demand_sorted = product_demand.sort_values('date')
        starting_date = date_range[0].date()
        
        # Find the closest stock level to starting date
        starting_stock_data = product_demand_sorted[
            product_demand_sorted['date'].dt.date >= starting_date
        ]
        
        if len(starting_stock_data) > 0:
            arrays['inventory_on_hand'][0] = starting_stock_data.iloc[0]['stock_level']
        else:
            # Fallback: use the last available stock level
            arrays['inventory_on_hand'][0] = product_demand_sorted.iloc[-1]['stock_level'] if len(product_demand_sorted) > 0 else 0
        
        # Initialize inventory_on_order (sum of incoming inventory from start to start + leadtime)
        incoming_dict = dict(zip(
            product_demand['date'].dt.date,
            product_demand['incoming_inventory']
        ))
        
        for i, sim_date in enumerate(date_range):
            if i == 0:  # Only for first step
                total_incoming = 0
                for j in range(leadtime):
                    if i + j < len(date_range):
                        check_date = date_range[i + j].date()
                        total_incoming += incoming_dict.get(check_date, 0)
                arrays['inventory_on_order'][i] = total_incoming
        
        # Initialize incoming_inventory array
        for i, sim_date in enumerate(date_range):
            if i < leadtime:
                arrays['incoming_inventory'][i] = incoming_dict.get(sim_date.date(), 0)
        
        # Populate actual_inventory array
        stock_level_dict = dict(zip(
            product_demand['date'].dt.date,
            product_demand['stock_level']
        ))
        
        for i, sim_date in enumerate(date_range):
            if sim_date.date() in stock_level_dict:
                arrays['actual_inventory'][i] = stock_level_dict[sim_date.date()]
        
        # Calculate initial net_stock
        arrays['net_stock'][0] = arrays['inventory_on_hand'][0] + arrays['inventory_on_order'][0]
        
        logger.debug(f"Created simulation arrays for {product_location_key} with {num_steps} steps")
        return arrays
    
    def get_all_simulation_data(self) -> Dict[str, Dict]:
        """
        Get all simulation data for all product-location combinations.
        
        Returns:
            Dictionary mapping product-location keys to simulation arrays
        """
        periods = self.get_simulation_periods()
        simulation_data = {}
        
        for key, period_info in periods.items():
            try:
                arrays = self.create_simulation_arrays(key, period_info)
                simulation_data[key] = {
                    'period_info': period_info,
                    'arrays': arrays
                }
            except Exception as e:
                logger.error(f"Error creating simulation arrays for {key}: {e}")
                continue
        
        logger.info(f"Created simulation data for {len(simulation_data)} product-location combinations")
        return simulation_data 