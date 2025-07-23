"""
Outlier handling utilities for demand data.
Processes demand data, handles outliers, and creates cleaned and outlier tables.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import date
from pathlib import Path
from ..data import DemandDataLoader, ProductMasterSchema
from .detection import OutlierDetector

class OutlierHandler:
    """Utilities for handling outliers in demand data"""
    
    def __init__(self, data_loader: DemandDataLoader = None):
        self.data_loader = data_loader or DemandDataLoader()
        self.detector = OutlierDetector(data_loader)
    
    def process_demand_outliers_with_data(self, 
                                         demand_df: pd.DataFrame,
                                         product_master_df: pd.DataFrame,
                                         default_method: str = "iqr",
                                         default_threshold: float = 1.5) -> Dict[str, Any]:
        """
        Process outliers using provided demand data and product master data
        
        Args:
            demand_df: Demand DataFrame
            product_master_df: Product master DataFrame
            default_method: Default outlier detection method ('iqr', 'zscore', 'mad', 'rolling')
            default_threshold: Default threshold for outlier detection
            
        Returns:
            Dictionary with 'cleaned_data' and 'summary'
        """
        # Initialize output dataframes
        cleaned_demand = []
        outlier_records = []
        
        # Process each product-location combination
        for _, master_record in product_master_df.iterrows():
            product_id = master_record['product_id']
            location_id = master_record['location_id']
            
            # Get demand data for this product-location
            product_demand = demand_df[
                (demand_df['product_id'] == product_id) &
                (demand_df['location_id'] == location_id)
            ].copy()
            
            if len(product_demand) == 0:
                continue
            
            # Get outlier parameters from product master (or use defaults)
            outlier_method = getattr(master_record, 'outlier_method', default_method)
            outlier_threshold = getattr(master_record, 'outlier_threshold', default_threshold)
            
            # Detect only high outliers
            outliers_mask = self._detect_high_outliers_only(
                product_demand, outlier_method, outlier_threshold
            )
            
            # Create cleaned demand records
            cleaned_records = product_demand[~outliers_mask].copy()
            cleaned_records['outlier_handled'] = False
            cleaned_demand.extend(cleaned_records.to_dict('records'))
            
            # Create outlier records
            outlier_records_data = product_demand[outliers_mask].copy()
            outlier_records_data['outlier_handled'] = True
            outlier_records_data['outlier_method'] = outlier_method
            outlier_records_data['outlier_threshold'] = outlier_threshold
            
            # Add outlier statistics
            if len(outlier_records_data) > 0:
                outlier_records_data['original_demand'] = outlier_records_data['demand']
                outlier_records_data['demand'] = self._handle_outlier_demand(
                    outlier_records_data, cleaned_records, outlier_method, outlier_threshold
                )
            
            outlier_records.extend(outlier_records_data.to_dict('records'))
        
        # Create final dataframes
        cleaned_df = pd.DataFrame(cleaned_demand) if cleaned_demand else pd.DataFrame()
        outlier_df = pd.DataFrame(outlier_records) if outlier_records else pd.DataFrame()
        
        # Sort and reset index
        if len(cleaned_df) > 0:
            cleaned_df = cleaned_df.sort_values(['date', 'product_id', 'location_id']).reset_index(drop=True)
        
        if len(outlier_df) > 0:
            outlier_df = outlier_df.sort_values(['date', 'product_id', 'location_id']).reset_index(drop=True)
        
        # Generate summary
        summary = self.get_outlier_summary(outlier_df)
        
        return {
            'cleaned_data': cleaned_df,
            'outlier_data': outlier_df,
            'summary': summary
        }

    def process_demand_outliers(self, 
                               frequency: str = "daily",
                               default_method: str = "iqr",
                               default_threshold: float = 1.5) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process demand data to handle outliers and create cleaned and outlier tables
        
        Args:
            frequency: 'daily' or 'weekly'
            default_method: Default outlier detection method ('iqr', 'zscore', 'mad', 'rolling')
            default_threshold: Default threshold for outlier detection
            
        Returns:
            Tuple of (cleaned_demand_df, outlier_df)
        """
        # Load demand data
        demand_df = self.data_loader.load_dummy_data(frequency=frequency)
        
        # Load product master to get outlier parameters
        if frequency == "daily":
            master_df = self.data_loader.load_product_master_daily()
        else:
            master_df = self.data_loader.load_product_master_weekly()
        
        # Initialize output dataframes
        cleaned_demand = []
        outlier_records = []
        
        # Process each product-location combination
        for _, master_record in master_df.iterrows():
            product_id = master_record['product_id']
            location_id = master_record['location_id']
            
            # Get demand data for this product-location
            product_demand = demand_df[
                (demand_df['product_id'] == product_id) &
                (demand_df['location_id'] == location_id)
            ].copy()
            
            if len(product_demand) == 0:
                continue
            
            # Get outlier parameters from product master (or use defaults)
            outlier_method = getattr(master_record, 'outlier_method', default_method)
            outlier_threshold = getattr(master_record, 'outlier_threshold', default_threshold)
            
            # Detect only high outliers
            outliers_mask = self._detect_high_outliers_only(
                product_demand, outlier_method, outlier_threshold
            )
            
            # Create cleaned demand records
            cleaned_records = product_demand[~outliers_mask].copy()
            cleaned_records['outlier_handled'] = False
            cleaned_demand.extend(cleaned_records.to_dict('records'))
            
            # Create outlier records
            outlier_records_data = product_demand[outliers_mask].copy()
            outlier_records_data['outlier_handled'] = True
            outlier_records_data['outlier_method'] = outlier_method
            outlier_records_data['outlier_threshold'] = outlier_threshold
            
            # Add outlier statistics
            if len(outlier_records_data) > 0:
                outlier_records_data['original_demand'] = outlier_records_data['demand']
                outlier_records_data['demand'] = self._handle_outlier_demand(
                    outlier_records_data, cleaned_records, outlier_method, outlier_threshold
                )
            
            outlier_records.extend(outlier_records_data.to_dict('records'))
        
        # Create final dataframes
        cleaned_df = pd.DataFrame(cleaned_demand) if cleaned_demand else pd.DataFrame()
        outlier_df = pd.DataFrame(outlier_records) if outlier_records else pd.DataFrame()
        
        # Sort and reset index
        if len(cleaned_df) > 0:
            cleaned_df = cleaned_df.sort_values(['date', 'product_id', 'location_id']).reset_index(drop=True)
        
        if len(outlier_df) > 0:
            outlier_df = outlier_df.sort_values(['date', 'product_id', 'location_id']).reset_index(drop=True)
        
        return cleaned_df, outlier_df
    
    def _detect_outliers_for_product_location(self, 
                                             product_demand: pd.DataFrame,
                                             method: str,
                                             threshold: float) -> pd.Series:
        """Detect outliers for a specific product-location combination"""
        demand_series = product_demand['demand']
        
        if method == 'iqr':
            return self.detector.detect_outliers_iqr(demand_series, threshold)
        elif method == 'zscore':
            return self.detector.detect_outliers_zscore(demand_series, threshold)
        elif method == 'mad':
            return self.detector.detect_outliers_mad(demand_series, threshold)
        elif method == 'rolling':
            return self.detector.detect_outliers_rolling(demand_series, window=int(threshold))
        else:
            # Default to IQR
            return self.detector.detect_outliers_iqr(demand_series, 1.5)
    
    def _detect_high_outliers_only(self, 
                                  product_demand: pd.DataFrame,
                                  method: str,
                                  threshold: float) -> pd.Series:
        """Detect only high outliers (above upper threshold)"""
        demand_series = product_demand['demand']
        
        if method == 'iqr':
            Q1 = demand_series.quantile(0.25)
            Q3 = demand_series.quantile(0.75)
            IQR = Q3 - Q1
            upper_bound = Q3 + threshold * IQR
            return demand_series > upper_bound
        elif method == 'zscore':
            z_scores = (demand_series - demand_series.mean()) / demand_series.std()
            return z_scores > threshold
        elif method == 'mad':
            median = demand_series.median()
            mad = np.median(np.abs(demand_series - median))
            mad_std = mad * 1.4826
            z_scores = (demand_series - median) / mad_std
            return z_scores > threshold
        elif method == 'rolling':
            rolling_mean = demand_series.rolling(window=int(threshold), center=True).mean()
            rolling_std = demand_series.rolling(window=int(threshold), center=True).std()
            upper_bound = rolling_mean + 2.0 * rolling_std  # Use 2.0 as default multiplier
            return demand_series > upper_bound
        else:
            # Default to IQR
            Q1 = demand_series.quantile(0.25)
            Q3 = demand_series.quantile(0.75)
            IQR = Q3 - Q1
            upper_bound = Q3 + 1.5 * IQR
            return demand_series > upper_bound
    
    def _get_upper_threshold(self, 
                            product_demand: pd.DataFrame,
                            method: str,
                            threshold: float) -> float:
        """Get the upper threshold value for capping outliers"""
        demand_series = product_demand['demand']
        
        if method == 'iqr':
            Q1 = demand_series.quantile(0.25)
            Q3 = demand_series.quantile(0.75)
            IQR = Q3 - Q1
            return Q3 + threshold * IQR
        elif method == 'zscore':
            mean_val = demand_series.mean()
            std_val = demand_series.std()
            return mean_val + threshold * std_val
        elif method == 'mad':
            median = demand_series.median()
            mad = np.median(np.abs(demand_series - median))
            mad_std = mad * 1.4826
            return median + threshold * mad_std
        elif method == 'rolling':
            # For rolling, we'll use a simple approach with the overall statistics
            mean_val = demand_series.mean()
            std_val = demand_series.std()
            return mean_val + 2.0 * std_val  # Use 2.0 as default multiplier
        else:
            # Default to IQR
            Q1 = demand_series.quantile(0.25)
            Q3 = demand_series.quantile(0.75)
            IQR = Q3 - Q1
            return Q3 + 1.5 * IQR
    
    def _handle_outlier_demand(self, 
                              outlier_data: pd.DataFrame,
                              cleaned_data: pd.DataFrame,
                              method: str,
                              threshold: float) -> pd.Series:
        """Handle outlier demand values by capping at the threshold"""
        # Get the upper threshold for capping
        all_data = pd.concat([cleaned_data, outlier_data]).reset_index(drop=True)
        upper_bound = self._get_upper_threshold(all_data, method, threshold)
        
        # Cap outliers at the upper threshold
        capped_demand = outlier_data['demand'].copy()
        capped_demand = capped_demand.clip(upper=upper_bound)
        
        return capped_demand
    
    def save_processed_data(self, 
                           cleaned_df: pd.DataFrame,
                           outlier_df: pd.DataFrame,
                           frequency: str = "daily",
                           output_dir: Optional[Path] = None) -> Dict[str, str]:
        """
        Save processed demand data to CSV files
        
        Args:
            cleaned_df: Cleaned demand dataframe
            outlier_df: Outlier dataframe
            frequency: 'daily' or 'weekly'
            output_dir: Output directory (defaults to data directory)
            
        Returns:
            Dictionary with file paths
        """
        if output_dir is None:
            output_dir = self.data_loader.data_dir
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filenames
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cleaned_filename = f"demand_cleaned_{frequency}_{timestamp}.csv"
        outlier_filename = f"demand_outliers_{frequency}_{timestamp}.csv"
        
        # Save files
        cleaned_path = output_dir / cleaned_filename
        outlier_path = output_dir / outlier_filename
        
        cleaned_df.to_csv(cleaned_path, index=False)
        outlier_df.to_csv(outlier_path, index=False)
        
        return {
            'cleaned_file': str(cleaned_path),
            'outlier_file': str(outlier_path)
        }
    
    def get_outlier_summary(self, outlier_df: pd.DataFrame) -> Dict:
        """
        Get summary statistics for outlier data
        
        Args:
            outlier_df: Outlier dataframe
            
        Returns:
            Dictionary with outlier summary statistics
        """
        if len(outlier_df) == 0:
            return {
                'total_outliers': 0,
                'products_with_outliers': 0,
                'locations_with_outliers': 0,
                'total_original_demand': 0,
                'total_replaced_demand': 0,
                'demand_reduction': 0
            }
        
        summary = {
            'total_outliers': len(outlier_df),
            'products_with_outliers': outlier_df['product_id'].nunique(),
            'locations_with_outliers': outlier_df['location_id'].nunique(),
            'total_original_demand': outlier_df['original_demand'].sum(),
            'total_replaced_demand': outlier_df['demand'].sum(),
            'demand_reduction': outlier_df['original_demand'].sum() - outlier_df['demand'].sum()
        }
        
        # Add method breakdown
        method_counts = outlier_df['outlier_method'].value_counts().to_dict()
        summary['outlier_methods'] = method_counts
        
        # Add date range
        summary['date_range'] = {
            'earliest': outlier_df['date'].min(),
            'latest': outlier_df['date'].max()
        }
        
        return summary

def process_demand_outliers(frequency: str = "daily", 
                           default_method: str = "iqr",
                           default_threshold: float = 1.5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Quick function to process demand outliers"""
    handler = OutlierHandler()
    return handler.process_demand_outliers(frequency, default_method, default_threshold)

def save_processed_data(cleaned_df: pd.DataFrame,
                       outlier_df: pd.DataFrame,
                       frequency: str = "daily") -> Dict[str, str]:
    """Quick function to save processed data"""
    handler = OutlierHandler()
    return handler.save_processed_data(cleaned_df, outlier_df, frequency) 