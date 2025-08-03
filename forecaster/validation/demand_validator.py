"""
Demand data validation utilities.
Checks for missing dates, frequency consistency, and data completeness.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import date, timedelta
from pathlib import Path
from data.loader import DataLoader
from .product_master_schema import ProductMasterSchema

class DemandValidator:
    """Utilities for validating demand data completeness and frequency"""
    
    def __init__(self, data_loader: DataLoader = None):
        self.data_loader = data_loader or DataLoader()
    
    def detect_demand_frequency(self, dates: pd.Series) -> str:
        """
        Detect the frequency of demand data from date series
        
        Args:
            dates: Series of date objects
            
        Returns:
            Frequency string: 'd' (daily), 'w' (weekly), 'm' (monthly)
        """
        if len(dates) < 2:
            return 'd'  # Default to daily if insufficient data
        
        # Ensure dates are date type (not datetime)
        if not pd.api.types.is_object_dtype(dates) and not pd.api.types.is_datetime64_any_dtype(dates):
            dates = pd.to_datetime(dates).dt.date
        
        # Sort dates
        sorted_dates = dates.sort_values().reset_index(drop=True)
        
        # Calculate time differences using date arithmetic
        time_diffs = []
        for i in range(1, len(sorted_dates)):
            diff = (sorted_dates.iloc[i] - sorted_dates.iloc[i-1]).days
            time_diffs.append(diff)
        
        if not time_diffs:
            return 'd'  # Default to daily if no differences
        
        # Analyze the most common difference
        time_diffs_series = pd.Series(time_diffs)
        mode_diff = time_diffs_series.mode().iloc[0]
        
        # Determine frequency based on mode difference
        if mode_diff <= 1.5:  # Allow some tolerance for daily
            return 'd'
        elif mode_diff <= 8:  # Allow tolerance for weekly
            return 'w'
        elif mode_diff <= 35:  # Allow tolerance for monthly
            return 'm'
        else:
            # If unclear, return the most common pattern
            return 'd'
    
    def find_missing_dates(self, 
                          product_id: str, 
                          location_id: str, 
                          dates: pd.Series,
                          frequency: str = 'd') -> List[date]:
        """
        Find missing dates in a demand series for a product-location combination
        
        Args:
            product_id: Product identifier
            location_id: Location identifier
            dates: Series of dates for this product-location
            frequency: Expected frequency ('d', 'w', 'm')
            
        Returns:
            List of missing dates
        """
        if len(dates) < 2:
            return []
        
        # Sort dates
        sorted_dates = dates.sort_values().reset_index(drop=True)
        start_date = sorted_dates.iloc[0]
        end_date = sorted_dates.iloc[-1]
        
        # Generate expected date range
        expected_dates = self._generate_date_range(start_date, end_date, frequency)
        
        # Find missing dates
        actual_dates_set = set(sorted_dates)
        missing_dates = [date for date in expected_dates if date not in actual_dates_set]
        
        return missing_dates
    
    def _generate_date_range(self, start_date: date, end_date: date, frequency: str) -> List[date]:
        """Generate expected date range based on frequency"""
        dates = []
        current_date = start_date
        
        if frequency == 'd':
            while current_date <= end_date:
                dates.append(current_date)
                current_date += timedelta(days=1)
        elif frequency == 'w':
            while current_date <= end_date:
                dates.append(current_date)
                current_date += timedelta(weeks=1)
        elif frequency == 'm':
            while current_date <= end_date:
                dates.append(current_date)
                # Approximate monthly increment
                if current_date.month == 12:
                    current_date = current_date.replace(year=current_date.year + 1, month=1)
                else:
                    current_date = current_date.replace(month=current_date.month + 1)
        
        return dates
    
    def validate_demand_completeness(self, frequency: str = "daily") -> Dict:
        """
        Validate demand data completeness for all product-location combinations
        
        Args:
            frequency: 'daily' or 'weekly'
            
        Returns:
            Dictionary with validation results
        """
        # Load demand data
        demand_df = self.data_loader.load_dummy_data(frequency=frequency)
        
        # Load product master for frequency validation
        if frequency == "daily":
            master_df = self.data_loader.load_product_master_daily()
        else:
            master_df = self.data_loader.load_product_master_weekly()
        
        return self._validate_demand_completeness_with_data(demand_df, master_df, frequency)
    
    def _validate_demand_completeness_with_data(self, demand_df: pd.DataFrame, master_df: pd.DataFrame, frequency: str = "daily") -> Dict:
        """
        Validate demand data completeness using provided data
        
        Args:
            demand_df: Demand DataFrame
            master_df: Product master DataFrame
            frequency: 'daily' or 'weekly'
            
        Returns:
            Dictionary with validation results
        """
        
        validation_results = {
            'frequency': frequency,
            'total_combinations': 0,
            'valid_combinations': 0,
            'invalid_combinations': 0,
            'missing_dates_total': 0,
            'frequency_mismatches': 0,
            'issues': [],
            'frequency_validation': {}
        }
        
        # Group by product-location
        grouped = demand_df.groupby(['product_id', 'location_id'])
        
        for (product_id, location_id), group in grouped:
            validation_results['total_combinations'] += 1
            
            # Get dates for this combination
            dates = group['date'].sort_values()
            
            # Detect actual frequency from data
            detected_frequency = self.detect_demand_frequency(dates)
            
            # Get expected frequency from product master
            master_record = master_df[
                (master_df['product_id'] == product_id) & 
                (master_df['location_id'] == location_id)
            ]
            
            if len(master_record) == 0:
                validation_results['issues'].append({
                    'type': 'missing_master_record',
                    'product_id': product_id,
                    'location_id': location_id,
                    'message': f"No product master record found for {product_id}-{location_id}"
                })
                validation_results['invalid_combinations'] += 1
                continue
            
            expected_frequency = master_record['demand_frequency'].iloc[0]
            
            # Check frequency consistency
            if detected_frequency != expected_frequency:
                validation_results['frequency_mismatches'] += 1
                validation_results['frequency_validation'][f"{product_id}-{location_id}"] = {
                    'expected': expected_frequency,
                    'detected': detected_frequency,
                    'dates_count': len(dates),
                    'date_range': f"{dates.iloc[0]} to {dates.iloc[-1]}"
                }
            
            # Find missing dates
            missing_dates = self.find_missing_dates(product_id, location_id, dates, expected_frequency)
            
            if missing_dates:
                validation_results['missing_dates_total'] += len(missing_dates)
                validation_results['issues'].append({
                    'type': 'missing_dates',
                    'product_id': product_id,
                    'location_id': location_id,
                    'expected_frequency': expected_frequency,
                    'detected_frequency': detected_frequency,
                    'missing_dates': [d for d in missing_dates],
                    'missing_count': len(missing_dates),
                    'date_range': f"{dates.iloc[0]} to {dates.iloc[-1]}",
                    'total_expected': len(self._generate_date_range(dates.iloc[0], dates.iloc[-1], expected_frequency)),
                    'total_actual': len(dates)
                })
                validation_results['invalid_combinations'] += 1
            else:
                validation_results['valid_combinations'] += 1
        
        # Calculate percentages
        total = validation_results['total_combinations']
        if total > 0:
            validation_results['valid_percentage'] = (validation_results['valid_combinations'] / total) * 100
            validation_results['invalid_percentage'] = (validation_results['invalid_combinations'] / total) * 100
        else:
            validation_results['valid_percentage'] = 0
            validation_results['invalid_percentage'] = 0
        
        return validation_results
    
    def validate_demand_completeness_with_data(self, demand_df: pd.DataFrame, master_df: pd.DataFrame, frequency: str = "daily") -> Dict:
        """
        Validate demand data completeness using provided data
        
        Args:
            demand_df: Demand DataFrame
            master_df: Product master DataFrame
            frequency: 'daily' or 'weekly'
            
        Returns:
            Dictionary with validation results
        """
        return self._validate_demand_completeness_with_data(demand_df, master_df, frequency)
    
    def generate_completeness_report(self, frequency: str = "daily") -> str:
        """
        Generate a human-readable completeness report
        
        Args:
            frequency: 'daily' or 'weekly'
            
        Returns:
            Formatted report string
        """
        results = self.validate_demand_completeness(frequency)
        
        report = f"""
Demand Data Completeness Report - {frequency.upper()}
{'='*60}

Summary:
- Total product-location combinations: {results['total_combinations']}
- Valid combinations: {results['valid_combinations']} ({results['valid_percentage']:.1f}%)
- Invalid combinations: {results['invalid_combinations']} ({results['invalid_percentage']:.1f}%)
- Total missing dates: {results['missing_dates_total']}
- Frequency mismatches: {results['frequency_mismatches']}

"""
        
        if results['issues']:
            report += "Issues Found:\n"
            report += "-" * 40 + "\n"
            
            for i, issue in enumerate(results['issues'], 1):
                if issue['type'] == 'missing_dates':
                    report += f"{i}. Missing Dates: {issue['product_id']}-{issue['location_id']}\n"
                    report += f"   Expected frequency: {issue['expected_frequency']}\n"
                    report += f"   Detected frequency: {issue['detected_frequency']}\n"
                    report += f"   Date range: {issue['date_range']}\n"
                    report += f"   Missing dates: {len(issue['missing_dates'])} dates\n"
                    if len(issue['missing_dates']) <= 10:
                        report += f"   Missing: {issue['missing_dates']}\n"
                    else:
                        report += f"   Missing: {issue['missing_dates'][:5]} ... {issue['missing_dates'][-5:]}\n"
                    report += "\n"
                
                elif issue['type'] == 'missing_master_record':
                    report += f"{i}. Missing Master Record: {issue['product_id']}-{issue['location_id']}\n"
                    report += f"   {issue['message']}\n\n"
        
        if results['frequency_validation']:
            report += "Frequency Mismatches:\n"
            report += "-" * 40 + "\n"
            for combo, details in results['frequency_validation'].items():
                report += f"- {combo}: Expected {details['expected']}, Detected {details['detected']}\n"
                report += f"  Date range: {details['date_range']} ({details['dates_count']} records)\n"
        
        return report

def validate_demand_completeness(frequency: str = "daily") -> Dict:
    """Quick demand completeness validation"""
    validator = DemandValidator()
    return validator.validate_demand_completeness(frequency)

def generate_completeness_report(frequency: str = "daily") -> str:
    """Quick completeness report generation"""
    validator = DemandValidator()
    return validator.generate_completeness_report(frequency) 