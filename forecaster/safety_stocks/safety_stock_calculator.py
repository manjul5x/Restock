"""
Safety Stock Calculator

This module handles the calculation of safety stocks based on forecast errors.
"""

import pandas as pd
import numpy as np
from datetime import date, timedelta
from typing import List, Dict, Any
from ..utils.logger import ForecasterLogger
from .safety_stock_models import SafetyStockModels


class SafetyStockCalculator:
    """
    Calculates safety stocks based on forecast errors.
    """
    
    def __init__(self, product_master_data: pd.DataFrame):
        """
        Initialize the safety stock calculator.
        
        Args:
            product_master_data: Product master data with safety stock parameters
        """
        self.product_master_data = product_master_data
        self.logger = ForecasterLogger(__name__)
        self.models = SafetyStockModels()
    
    def _expand_product_master_by_methods(self) -> pd.DataFrame:
        """
        Expand product master data to create separate entries for each forecast method.
        
        Returns:
            Expanded product master DataFrame with forecast_method column
        """
        expanded_data = []
        
        for _, row in self.product_master_data.iterrows():
            # Get forecast methods from the row
            forecast_methods_str = row.get('forecast_methods', '')
            if not forecast_methods_str:
                self.logger.warning(f"No forecast_methods found for {row['product_id']} at {row['location_id']}")
                continue
                
            # Split methods and create separate rows for each
            methods = [m.strip() for m in forecast_methods_str.split(',')]
            
            for method in methods:
                expanded_row = row.copy()
                expanded_row['forecast_method'] = method
                expanded_data.append(expanded_row)
        
        expanded_df = pd.DataFrame(expanded_data)
        self.logger.info(f"Expanded product master from {len(self.product_master_data)} to {len(expanded_df)} entries")
        return expanded_df
    
    def calculate_safety_stocks(
        self, 
        forecast_comparison_data: pd.DataFrame,
        review_dates: List[date]
    ) -> pd.DataFrame:
        """
        Calculate safety stocks for all product-location-method combinations.
        
        Args:
            forecast_comparison_data: Forecast comparison data with errors
            review_dates: List of review dates to calculate safety stocks for
            
        Returns:
            DataFrame with safety stock calculations
        """
        self.logger.info("Starting safety stock calculations")
        self.logger.info(f"Processing {len(review_dates)} review dates")
        
        # Expand product master data by forecast methods
        expanded_product_master = self._expand_product_master_by_methods()
        
        # Initialize results list
        safety_stock_results = []
        
        # Process each product-location-method combination
        for _, product_row in expanded_product_master.iterrows():
            product = product_row['product_id']
            location = product_row['location_id']
            forecast_method = product_row['forecast_method']
            distribution_type = product_row.get('distribution', 'kde')
            service_level = product_row.get('service_level', 0.95)
            ss_window_length = product_row.get('ss_window_length', 180)
            
            self.logger.info(f"Processing {product} at {location} with method {forecast_method}")
            
            # Calculate safety stocks for this product-location-method
            product_safety_stocks = self._calculate_product_safety_stocks(
                forecast_comparison_data=forecast_comparison_data,
                product=product,
                location=location,
                forecast_method=forecast_method,
                review_dates=review_dates,
                distribution_type=distribution_type,
                service_level=service_level,
                ss_window_length=ss_window_length,
                product_row=product_row # Pass the product_row to _calculate_product_safety_stocks
            )
            
            safety_stock_results.extend(product_safety_stocks)
        
        # Create results DataFrame
        results_df = pd.DataFrame(safety_stock_results)
        
        self.logger.info(f"Completed safety stock calculations for {len(results_df)} combinations")
        return results_df
    
    def _calculate_product_safety_stocks(
        self,
        forecast_comparison_data: pd.DataFrame,
        product: str,
        location: str,
        forecast_method: str,
        review_dates: List[date],
        distribution_type: str,
        service_level: float,
        ss_window_length: int,
        product_row: pd.Series # Added product_row parameter
    ) -> List[Dict[str, Any]]:
        """
        Calculate safety stocks for a specific product-location-method combination.
        
        Args:
            forecast_comparison_data: Forecast comparison data
            product: Product name
            location: Location name
            forecast_method: Forecast method used
            review_dates: List of review dates
            distribution_type: Type of distribution to use
            service_level: Service level percentage
            ss_window_length: Rolling window length for safety stock calculation in demand frequency units
            product_row: The row from the expanded product master data for the current product-location-method
            
        Returns:
            List of safety stock results
        """
        results = []
        
        # Filter data for this product-location-method
        product_data = forecast_comparison_data[
            (forecast_comparison_data['product_id'] == product) &
            (forecast_comparison_data['location_id'] == location) &
            (forecast_comparison_data['forecast_method'] == forecast_method)
        ].copy()
        
        if product_data.empty:
            self.logger.warning(f"No forecast comparison data found for {product} at {location} with method {forecast_method}")
            return results
        
        # Get minimum and maximum safety stock from product master
        min_safety_stock = product_row.get('min_safety_stock', 0.0)
        max_safety_stock = product_row.get('max_safety_stock', None)
        
        # Ensure min_safety_stock is numeric and handle any edge cases
        try:
            if pd.isna(min_safety_stock) or min_safety_stock == "":
                min_safety_stock = 0.0
            else:
                min_safety_stock = float(min_safety_stock)
                if min_safety_stock < 0:
                    self.logger.warning(f"Negative min_safety_stock found for {product} at {location}: {min_safety_stock}, setting to 0.0")
                    min_safety_stock = 0.0
        except (ValueError, TypeError) as e:
            self.logger.warning(f"Invalid min_safety_stock value for {product} at {location}: {min_safety_stock}, setting to 0.0. Error: {e}")
            min_safety_stock = 0.0
        
        # Ensure max_safety_stock is numeric and handle any edge cases
        try:
            if pd.isna(max_safety_stock) or max_safety_stock == "":
                max_safety_stock = None
            else:
                max_safety_stock = float(max_safety_stock)
                if max_safety_stock < 0:
                    self.logger.warning(f"Negative max_safety_stock found for {product} at {location}: {max_safety_stock}, setting to None")
                    max_safety_stock = None
        except (ValueError, TypeError) as e:
            self.logger.warning(f"Invalid max_safety_stock value for {product} at {location}: {max_safety_stock}, setting to None. Error: {e}")
            max_safety_stock = None
        
        # Convert ss_window_length to timedelta (assuming daily frequency for now)
        # This should be made more flexible based on demand frequency
        window_timedelta = timedelta(days=ss_window_length)
        
        for review_date in review_dates:
            # Filter errors for this review date
            errors = self._get_relevant_errors(
                product_data=product_data,
                review_date=review_date,
                window_timedelta=window_timedelta
            )
            
            if len(errors) < 10:  # Need minimum data points for reliable calculation
                self.logger.warning(
                    f"Insufficient error data for {product} at {location} with method {forecast_method} on {review_date}: {len(errors)} points"
                )
                safety_stock = 0.0
            else:
                # Calculate safety stock using the specified distribution
                safety_stock = self.models.calculate_safety_stock(
                    errors=errors,
                    distribution_type=distribution_type,
                    service_level=service_level
                )
            
            # Apply minimum and maximum safety stock constraints
            safety_stock = max(safety_stock, min_safety_stock)
            if max_safety_stock is not None:
                safety_stock = min(safety_stock, max_safety_stock)
            
            results.append({
                'product_id': product,
                'location_id': location,
                'forecast_method': forecast_method,
                'review_date': review_date,
                'errors': errors,
                'safety_stock': safety_stock,
                'distribution_type': distribution_type,
                'service_level': service_level,
                'ss_window_length': ss_window_length,
                'error_count': len(errors),
                'min_safety_stock': min_safety_stock,
                'max_safety_stock': max_safety_stock
            })
        
        return results
    
    def _get_relevant_errors(
        self,
        product_data: pd.DataFrame,
        review_date: date,
        window_timedelta: timedelta
    ) -> List[float]:
        """
        Get relevant errors for safety stock calculation at a specific review date.
        
        Args:
            product_data: Filtered data for a product-location
            review_date: Review date
            window_timedelta: Rolling window length as timedelta
            
        Returns:
            List of relevant errors
        """
        # Filter for step 1 forecasts (next period forecasts)
        step1_data = product_data[product_data['step'] == 1].copy()
        
        if step1_data.empty:
            return []
        
        # Convert analysis_date to date if it's not already
        if not pd.api.types.is_datetime64_any_dtype(step1_data['analysis_date']):
            step1_data['analysis_date'] = pd.to_datetime(step1_data['analysis_date']).dt.date
        
        # Convert risk_period_end_date to date if it's not already
        if not pd.api.types.is_datetime64_any_dtype(step1_data['risk_period_end']):
            step1_data['risk_period_end'] = pd.to_datetime(step1_data['risk_period_end']).dt.date
        
        # Convert review_date to date if it's datetime
        if hasattr(review_date, 'date'):  # Check if it's a datetime object
            review_date = review_date.date()
        
        # Filter for relevant errors:
        # 1. Analysis date within rolling window length
        # 2. Risk period end date less than review date
        window_start = review_date - window_timedelta
        
        relevant_data = step1_data[
            (step1_data['analysis_date'] >= window_start) &
            (step1_data['analysis_date'] <= review_date) &
            (step1_data['risk_period_end'] < review_date)
        ]
        
        # Extract errors
        errors = relevant_data['forecast_error'].tolist()
        
        return errors 