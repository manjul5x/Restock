"""
Demand aggregation utilities.
Aggregates demand data into risk period sized buckets for backtesting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import date, timedelta
from pathlib import Path
from data.loader import DataLoader
try:
    from ..validation.product_master_schema import ProductMasterSchema
except ImportError:
    # Schema module requires pydantic which may not be installed
    ProductMasterSchema = None

class DemandAggregator:
    """Utilities for aggregating demand data into risk period buckets"""
    
    def __init__(self, data_loader: DataLoader = None):
        self.data_loader = data_loader or DataLoader()
    
    def create_risk_period_buckets_with_data(self, 
                                            demand_df: pd.DataFrame,
                                            product_master_df: pd.DataFrame,
                                            cutoff_date) -> pd.DataFrame:
        """
        Create risk period buckets using provided data
        
        Args:
            demand_df: Demand DataFrame
            product_master_df: Product master DataFrame
            cutoff_date: Date to cut off the data (exclusive)
            
        Returns:
            DataFrame with aggregated demand in risk period buckets
        """
        # Filter demand data to only include data before cutoff date
        cutoff_timestamp = pd.Timestamp(cutoff_date)
        demand_filtered = demand_df[pd.to_datetime(demand_df['date']) < cutoff_timestamp].copy()
        
        aggregated_buckets = []
        
        # Process each product-location combination
        for _, master_record in product_master_df.iterrows():
            product_id = master_record['product_id']
            location_id = master_record['location_id']
            risk_period = master_record['risk_period']
            demand_freq = master_record['demand_frequency']
            
            # Get demand data for this product-location
            product_demand = demand_filtered[
                (demand_filtered['product_id'] == product_id) &
                (demand_filtered['location_id'] == location_id)
            ].copy()
            
            if len(product_demand) == 0:
                continue
            
            # Create buckets for this product-location
            buckets = self._create_buckets_for_product_location(
                product_demand, 
                cutoff_date, 
                risk_period, 
                demand_freq,
                product_id,
                location_id
            )
            
            aggregated_buckets.extend(buckets)
        
        if not aggregated_buckets:
            return pd.DataFrame()
        
        # Create final DataFrame
        result_df = pd.DataFrame(aggregated_buckets)
        
        # Sort by product, location, and bucket start date
        result_df = result_df.sort_values(['product_id', 'location_id', 'bucket_start_date']).reset_index(drop=True)
        
        return result_df

    def create_risk_period_buckets(self, 
                                  cutoff_date: date,
                                  frequency: str = "daily") -> pd.DataFrame:
        """
        Create risk period buckets for all product-location combinations
        
        Args:
            cutoff_date: Date to cut off the data (exclusive)
            frequency: 'daily' or 'weekly'
            
        Returns:
            DataFrame with aggregated demand in risk period buckets
        """
        # Load demand data
        demand_df = self.data_loader.load_dummy_data(frequency=frequency)
        
        # Load product master
        if frequency == "daily":
            master_df = self.data_loader.load_product_master_daily()
        else:
            master_df = self.data_loader.load_product_master_weekly()
        
        # Filter demand data to only include data before cutoff date
        demand_filtered = demand_df[demand_df['date'] < cutoff_date].copy()
        
        aggregated_buckets = []
        
        # Process each product-location combination
        for _, master_record in master_df.iterrows():
            product_id = master_record['product_id']
            location_id = master_record['location_id']
            risk_period = master_record['risk_period']
            demand_freq = master_record['demand_frequency']
            
            # Get demand data for this product-location
            product_demand = demand_filtered[
                (demand_filtered['product_id'] == product_id) &
                (demand_filtered['location_id'] == location_id)
            ].copy()
            
            if len(product_demand) == 0:
                continue
            
            # Create buckets for this product-location
            buckets = self._create_buckets_for_product_location(
                product_demand, 
                cutoff_date, 
                risk_period, 
                demand_freq,
                product_id,
                location_id
            )
            
            aggregated_buckets.extend(buckets)
        
        if not aggregated_buckets:
            return pd.DataFrame()
        
        # Create final DataFrame
        result_df = pd.DataFrame(aggregated_buckets)
        
        # Sort by product, location, and bucket start date
        result_df = result_df.sort_values(['product_id', 'location_id', 'bucket_start_date']).reset_index(drop=True)
        
        return result_df
    
    def _create_buckets_for_product_location(self,
                                           product_demand: pd.DataFrame,
                                           cutoff_date,
                                           risk_period: int,
                                           demand_freq: str,
                                           product_id: str,
                                           location_id: str) -> List[Dict]:
        """
        Create risk period buckets for a specific product-location combination
        
        Args:
            product_demand: Demand data for this product-location
            cutoff_date: Cutoff date (exclusive)
            risk_period: Risk period size
            demand_freq: Demand frequency ('d', 'w', 'm')
            product_id: Product identifier
            location_id: Location identifier
            
        Returns:
            List of bucket dictionaries
        """
        buckets = []
        
        # Sort demand data by date
        product_demand = product_demand.sort_values('date')
        
        # Calculate bucket size in days
        bucket_size_days = ProductMasterSchema.get_risk_period_days(demand_freq, risk_period)
        
        # Find the earliest date in the data
        earliest_date = pd.to_datetime(product_demand['date'].min()).date()
        
        # Calculate the latest possible bucket start date
        # We want buckets that end before the cutoff date
        # For June 8th cutoff, the latest bucket should end on June 7th
        latest_bucket_end = cutoff_date - timedelta(days=1)
        latest_bucket_start = latest_bucket_end - timedelta(days=bucket_size_days - 1)
        
        # Generate bucket start dates working backwards from the latest possible bucket
        # This ensures we get the most recent complete buckets first
        current_bucket_start = latest_bucket_start
        
        while current_bucket_start >= earliest_date:
            # Calculate bucket end date
            bucket_end_date = current_bucket_start + timedelta(days=bucket_size_days - 1)
            
            # Filter demand data for this bucket
            bucket_mask = (
                (pd.to_datetime(product_demand['date']).dt.date >= current_bucket_start) &
                (pd.to_datetime(product_demand['date']).dt.date <= bucket_end_date)
            )
            
            bucket_data = product_demand[bucket_mask]
            
            # Only create bucket if we have complete data (all days in the bucket)
            expected_days = bucket_size_days
            actual_days = len(bucket_data)
            
            if actual_days == expected_days:
                # Aggregate demand for this bucket
                total_demand = bucket_data['demand'].sum()
                avg_demand = bucket_data['demand'].mean()
                min_demand = bucket_data['demand'].min()
                max_demand = bucket_data['demand'].max()
                std_demand = bucket_data['demand'].std()
                
                # Get stock levels (use the last available stock level in the bucket)
                last_stock = bucket_data.iloc[-1]['stock_level'] if len(bucket_data) > 0 else 0
                
                bucket_info = {
                    'product_id': product_id,
                    'location_id': location_id,
                    'product_category': bucket_data.iloc[0]['product_category'],
                    'bucket_start_date': current_bucket_start,
                    'bucket_end_date': bucket_end_date,
                    'bucket_size_days': bucket_size_days,
                    'demand_frequency': demand_freq,
                    'risk_period': risk_period,
                    'total_demand': total_demand,
                    'avg_demand': avg_demand,
                    'min_demand': min_demand,
                    'max_demand': max_demand,
                    'std_demand': std_demand,
                    'demand_records': actual_days,
                    'last_stock_level': last_stock,
                    'bucket_completeness': actual_days / expected_days
                }
                
                buckets.append(bucket_info)
            
            # Move to previous bucket
            current_bucket_start -= timedelta(days=bucket_size_days)
        
        return buckets
    
    def get_aggregation_summary(self, 
                               cutoff_date: date,
                               frequency: str = "daily") -> Dict:
        """
        Get summary statistics for aggregated demand buckets
        
        Args:
            cutoff_date: Date to cut off the data (exclusive)
            frequency: 'daily' or 'weekly'
            
        Returns:
            Dictionary with aggregation summary
        """
        aggregated_df = self.create_risk_period_buckets(cutoff_date, frequency)
        
        if len(aggregated_df) == 0:
            return {
                'cutoff_date': cutoff_date,
                'frequency': frequency,
                'total_buckets': 0,
                'products': 0,
                'locations': 0,
                'total_demand': 0,
                'avg_buckets_per_product_location': 0
            }
        
        summary = {
            'cutoff_date': cutoff_date,
            'frequency': frequency,
            'total_buckets': len(aggregated_df),
            'products': aggregated_df['product_id'].nunique(),
            'locations': aggregated_df['location_id'].nunique(),
            'total_demand': aggregated_df['total_demand'].sum(),
            'avg_buckets_per_product_location': len(aggregated_df) / (aggregated_df['product_id'].nunique() * aggregated_df['location_id'].nunique()),
            'bucket_size_days': aggregated_df['bucket_size_days'].iloc[0],
            'date_range': {
                'earliest_bucket': aggregated_df['bucket_start_date'].min(),
                'latest_bucket': aggregated_df['bucket_end_date'].max()
            },
            'demand_stats': {
                'avg_total_demand': aggregated_df['total_demand'].mean(),
                'min_total_demand': aggregated_df['total_demand'].min(),
                'max_total_demand': aggregated_df['total_demand'].max(),
                'std_total_demand': aggregated_df['total_demand'].std()
            }
        }
        
        return summary

def create_risk_period_buckets(cutoff_date: date, frequency: str = "daily") -> pd.DataFrame:
    """Quick function to create risk period buckets"""
    aggregator = DemandAggregator()
    return aggregator.create_risk_period_buckets(cutoff_date, frequency)

def get_aggregation_summary(cutoff_date: date, frequency: str = "daily") -> Dict:
    """Quick function to get aggregation summary"""
    aggregator = DemandAggregator()
    return aggregator.get_aggregation_summary(cutoff_date, frequency) 