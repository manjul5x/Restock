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


from data.loader import DataLoader
from ..validation.product_master_schema import ProductMasterSchema
from ..utils.logger import get_logger


class SimulationDataLoader:
    """
    Loads and prepares data for inventory simulation using the standardized DataLoader.
    """

    def __init__(self,
                 safety_stock_file: Optional[str] = None,
                 forecast_comparison_file: Optional[str] = None,
                 log_level: str = "INFO"):
        """
        Initialize the data loader.

        Args:
            safety_stock_file: Path to safety stock results file.
            forecast_comparison_file: Path to forecast comparison file.
            log_level: Logging level for the data loader.
        """
        self.loader = DataLoader()
        self.safety_stock_file = safety_stock_file
        self.forecast_comparison_file = forecast_comparison_file
        self.log_level = log_level

        # Data attributes to be loaded
        self.product_master: Optional[pd.DataFrame] = None
        self.customer_demand: Optional[pd.DataFrame] = None
        self.safety_stocks: Optional[pd.DataFrame] = None
        self.forecast_comparison: Optional[pd.DataFrame] = None

        # Load all required data
        self._load_required_data()

    def _expand_product_master_by_methods(self) -> pd.DataFrame:
        """
        Expand product master data to create separate entries for each forecast method.
        Uses the standardized schema helper for consistency.

        Returns:
            Expanded product master DataFrame with a 'forecast_method' column.
        """
        if self.product_master is None:
            self._load_required_data()

        expanded_df = ProductMasterSchema.expand_product_master_for_methods(
            self.product_master
        )
        logger = get_logger(__name__, level=self.log_level)
        logger.info(f"Expanded product master from {len(self.product_master)} to {len(expanded_df)} entries")
        return expanded_df

    def _load_required_data(self):
        """Load all required data files using the DataLoader and direct reads for output files."""
        logger = get_logger(__name__, level=self.log_level)
        try:
            logger.info("Loading core data using DataLoader...")
            # Load product master and filtered outflow (demand) data
            self.product_master = self.loader.load_product_master()
            self.customer_demand = self.loader.load_outflow(product_master=self.product_master)

            logger.info("Loading simulation-specific output files...")
            # Load safety stock results
            filename = self.loader.config['paths']['output_files']['safety_stocks']
            ss_path = Path(self.safety_stock_file or str(self.loader.get_output_path("safety_stocks", filename)))
            if ss_path.exists():
                self.safety_stocks = pd.read_csv(ss_path)
                self.safety_stocks['review_date'] = pd.to_datetime(self.safety_stocks['review_date'])
            else:
                raise FileNotFoundError(f"Safety stock results not found: {ss_path}. Please run safety stock calculation first.")

            # Load forecast comparison data
            filename = self.loader.config['paths']['output_files']['forecast_comparison']
            fc_path = Path(self.forecast_comparison_file or str(self.loader.get_output_path("backtesting", filename)))
            if fc_path.exists():
                self.forecast_comparison = pd.read_csv(fc_path)
                self.forecast_comparison['analysis_date'] = pd.to_datetime(self.forecast_comparison['analysis_date'])
            else:
                raise FileNotFoundError(f"Forecast comparison data not found: {fc_path}. Please run backtest first.")

            logger.info("Successfully loaded all simulation data")

        except Exception as e:
            logger.error(f"Error loading simulation data: {e}")
            raise
    
    def get_simulation_periods(self) -> Dict[str, Dict]:
        """
        Get simulation periods for all product-location-method combinations.
        
        Returns:
            Dictionary mapping product-location-method keys to simulation period info
        """
        periods = {}
        
        # Expand product master by forecast methods
        expanded_product_master = self._expand_product_master_by_methods()
        
        for _, product_record in expanded_product_master.iterrows():
            product_id = product_record['product_id']
            location_id = product_record['location_id']
            forecast_method = product_record['forecast_method']
            key = f"{product_id}_{location_id}_{forecast_method}"
            
            # Get safety stocks for this product-location-method
            product_safety_stocks = self.safety_stocks[
                (self.safety_stocks['product_id'] == product_id) &
                (self.safety_stocks['location_id'] == location_id) &
                (self.safety_stocks['forecast_method'] == forecast_method)
            ]
            
            if len(product_safety_stocks) == 0:
                logger = get_logger(__name__, level=self.log_level)
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
                logger = get_logger(__name__, level=self.log_level)
                logger.warning(f"No demand data found for {key}")
                continue
            
            # Sort by date and calculate frequency
            # TODO: calculate / set frequency in the beginning of this whole thing
            product_demand = product_demand.sort_values('date')
            product_demand.loc[:, 'date'] = pd.to_datetime(product_demand['date'])
            
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
                'forecast_method': forecast_method,
                'first_review_date': first_review_date,
                'last_review_date': last_review_date,
                'frequency_days': frequency_days,
                'date_range': date_range,
                'num_steps': len(date_range),
                'leadtime': product_record['leadtime'],
                'moq': product_record['moq'] 
            }
        
        logger = get_logger(__name__, level=self.log_level)
        logger.info(f"Created simulation periods for {len(periods)} product-location-method combinations")
        return periods
    
    def create_simulation_arrays(self, product_location_key: str, period_info: Dict) -> Dict[str, np.ndarray]:
        """
        Create simulation arrays for a specific product-location-method combination.
        
        Args:
            product_location_key: Key in format "product_id_location_id_forecast_method"
            period_info: Period information from get_simulation_periods()
            
        Returns:
            Dictionary containing all simulation arrays
        """
        product_id = period_info['product_id']
        location_id = period_info['location_id']
        forecast_method = period_info['forecast_method']
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
            (self.safety_stocks['location_id'] == location_id) &
            (self.safety_stocks['forecast_method'] == forecast_method)
        ]
        # Convert review_date to date if it's datetime
        if pd.api.types.is_datetime64_any_dtype(product_safety_stocks['review_date']):
            review_dates = set(product_safety_stocks['review_date'].dt.date)
        else:
            review_dates = set(product_safety_stocks['review_date'])
        
        for i, sim_date in enumerate(date_range):
            if sim_date.date() in review_dates:
                arrays['decision_day'][i] = 1
        
        # Populate safety_stock array (forward fill from review dates)
        # Convert review_date to date if it's datetime
        if pd.api.types.is_datetime64_any_dtype(product_safety_stocks['review_date']):
            safety_stock_dict = dict(zip(
                product_safety_stocks['review_date'].dt.date,
                product_safety_stocks['safety_stock']
            ))
        else:
            safety_stock_dict = dict(zip(
                product_safety_stocks['review_date'],
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
            (self.forecast_comparison['forecast_method'] == forecast_method) &
            (self.forecast_comparison['step'] == 1)
        ]
        
        # Convert analysis_date to date if it's datetime
        if pd.api.types.is_datetime64_any_dtype(product_forecasts['analysis_date']):
            forecast_dict = dict(zip(
                product_forecasts['analysis_date'].dt.date,
                product_forecasts['forecast_demand']
            ))
        else:
            forecast_dict = dict(zip(
                product_forecasts['analysis_date'],
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
        # Convert date to datetime for processing, then back to date
        product_demand.loc[:, 'date'] = pd.to_datetime(product_demand['date']).dt.date
        
        demand_dict = dict(zip(
            product_demand['date'],
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
            product_demand_sorted['date'] >= starting_date
        ]
        
        if len(starting_stock_data) > 0:
            arrays['inventory_on_hand'][0] = starting_stock_data.iloc[0]['stock_level']
        else:
            # Fallback: use the last available stock level
            arrays['inventory_on_hand'][0] = product_demand_sorted.iloc[-1]['stock_level'] if len(product_demand_sorted) > 0 else 0
        
        # Initialize inventory_on_order (sum of incoming inventory from start to start + leadtime)
        incoming_dict = dict(zip(
            product_demand['date'],
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
            product_demand['date'],
            product_demand['stock_level']
        ))
        
        for i, sim_date in enumerate(date_range):
            if sim_date.date() in stock_level_dict:
                arrays['actual_inventory'][i] = stock_level_dict[sim_date.date()]
        
        # Calculate initial net_stock
        arrays['net_stock'][0] = arrays['inventory_on_hand'][0] + arrays['inventory_on_order'][0]
        
        logger = get_logger(__name__, level=self.log_level)
        logger.debug(f"Created simulation arrays for {product_location_key} with {num_steps} steps")
        return arrays
    
    def get_all_simulation_data(self) -> Dict[str, Dict]:
        """
        Get all simulation data for all product-location-method combinations.
        
        Returns:
            Dictionary mapping product-location-method keys to simulation arrays
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
                logger = get_logger(__name__, level=self.log_level)
                logger.error(f"Error creating simulation arrays for {key}: {e}")
                continue
        
        logger = get_logger(__name__, level=self.log_level)
        logger.info(f"Created simulation data for {len(simulation_data)} product-location-method combinations")
        return simulation_data 