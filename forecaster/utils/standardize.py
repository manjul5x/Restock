"""
Data standardization and quality check utilities for the forecaster package.
Provides functions to check, convert, and clean dataframes to standard schema.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import date, timedelta
import warnings
from .logger import get_logger

class DataStandardizer:
    """Utilities for standardizing and validating data"""
    
    def __init__(self):
        self.logger = get_logger()
    
    def check_dataframe_schema(self, df: pd.DataFrame, 
                              required_columns: List[str] = None) -> Dict[str, any]:
        """
        Comprehensive check of dataframe against schema
        
        Args:
            df: DataFrame to check
            required_columns: List of required columns (uses default if None)
            
        Returns:
            Dictionary with check results and issues found
        """
        if required_columns is None:
            required_columns = ['product_id', 'product_category', 'location_id', 
                              'date', 'demand', 'stock_level']
        
        results = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'column_info': {},
            'data_quality': {}
        }
        
        # Check for required columns
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            results['is_valid'] = False
            results['issues'].append(f"Missing required columns: {missing_cols}")
        
        # Check data types and quality for each column
        for col in df.columns:
            col_info = self._analyze_column(df[col], col)
            results['column_info'][col] = col_info
            
            if col_info['issues']:
                results['issues'].extend(col_info['issues'])
                results['is_valid'] = False
            
            if col_info['warnings']:
                results['warnings'].extend(col_info['warnings'])
        
        # Overall data quality metrics
        results['data_quality'] = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'date_range': self._get_date_range(df) if 'date' in df.columns else None
        }
        
        return results
    
    def _analyze_column(self, series: pd.Series, column_name: str) -> Dict[str, any]:
        """Analyze a single column for data quality issues"""
        info = {
            'dtype': str(series.dtype),
            'null_count': series.isnull().sum(),
            'null_percentage': (series.isnull().sum() / len(series)) * 100,
            'unique_count': series.nunique(),
            'issues': [],
            'warnings': []
        }
        
        # Check for null values
        if info['null_count'] > 0:
            info['warnings'].append(f"Column {column_name} has {info['null_count']} null values")
        
        # Column-specific checks
        if column_name == 'date':
            info.update(self._check_date_column(series))
        elif column_name in ['demand', 'stock_level']:
            info.update(self._check_numeric_column(series, column_name))
        elif column_name in ['product_id', 'location_id', 'product_category']:
            info.update(self._check_string_column(series, column_name))
        
        return info
    
    def _check_date_column(self, series: pd.Series) -> Dict[str, any]:
        """Check date column for issues"""
        info = {'issues': [], 'warnings': []}
        
        # Check if it's date (pandas stores dates as datetime64 internally)
        if not pd.api.types.is_datetime64_any_dtype(series):
            info['issues'].append("Date column is not datetime64 type (required for date handling)")
        
        # Check for future dates
        if pd.api.types.is_datetime64_any_dtype(series):
            from datetime import datetime
            future_dates = series > datetime.now()
            if future_dates.any():
                info['warnings'].append(f"Found {future_dates.sum()} future dates")
        
        return info
    
    def _check_numeric_column(self, series: pd.Series, column_name: str) -> Dict[str, any]:
        """Check numeric column for issues"""
        info = {'issues': [], 'warnings': []}
        
        # Check if numeric
        if not pd.api.types.is_numeric_dtype(series):
            info['issues'].append(f"{column_name} column is not numeric")
            return info
        
        # Check for negative values
        negative_count = (series < 0).sum()
        if negative_count > 0:
            info['issues'].append(f"{column_name} has {negative_count} negative values")
        
        # Check for outliers (using IQR method)
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        outlier_count = ((series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR))).sum()
        
        if outlier_count > 0:
            info['warnings'].append(f"{column_name} has {outlier_count} potential outliers")
        
        # Add statistics
        info['stats'] = {
            'min': series.min(),
            'max': series.max(),
            'mean': series.mean(),
            'std': series.std(),
            'median': series.median()
        }
        
        return info
    
    def _check_string_column(self, series: pd.Series, column_name: str) -> Dict[str, any]:
        """Check string column for issues"""
        info = {'issues': [], 'warnings': []}
        
        # Check for empty strings
        empty_strings = (series == '').sum()
        if empty_strings > 0:
            info['warnings'].append(f"{column_name} has {empty_strings} empty strings")
        
        # Check for whitespace-only strings
        whitespace_only = series.str.strip().eq('').sum() if series.dtype == 'object' else 0
        if whitespace_only > 0:
            info['warnings'].append(f"{column_name} has {whitespace_only} whitespace-only strings")
        
        return info
    
    def _get_date_range(self, df: pd.DataFrame) -> Dict[str, str]:
        """Get date range information"""
        if 'date' not in df.columns or not pd.api.types.is_datetime64_any_dtype(df['date']):
            return None
        
        return {
            'start': df['date'].min().strftime('%Y-%m-%d'),
            'end': df['date'].max().strftime('%Y-%m-%d'),
            'days': (df['date'].max() - df['date'].min()).days
        }
    
    def convert_to_standard_format(self, df: pd.DataFrame, 
                                 target_schema: Dict[str, str] = None) -> pd.DataFrame:
        """
        Convert dataframe to standard format
        
        Args:
            df: DataFrame to convert
            target_schema: Dictionary mapping column names to target types
            
        Returns:
            Standardized DataFrame
        """
        if target_schema is None:
            target_schema = {
                'product_id': 'string',
                'product_category': 'string', 
                'location_id': 'string',
                'date': 'date',
                'demand': 'float64',
                'stock_level': 'float64'
            }
        
        df_std = df.copy()
        
        # Convert columns to target types
        for col, target_type in target_schema.items():
            if col in df_std.columns:
                try:
                    if target_type == 'date':
                        df_std[col] = pd.to_datetime(df_std[col]).dt.date
                    elif target_type == 'string':
                        df_std[col] = df_std[col].astype(str)
                    else:
                        df_std[col] = df_std[col].astype(target_type)
                except Exception as e:
                    self.logger.warning(f"Could not convert column {col} to {target_type}: {e}")
        
        # Handle missing values
        df_std = self._handle_missing_values(df_std)
        
        # Sort for consistency
        if 'date' in df_std.columns:
            sort_cols = ['date']
            for col in ['product_id', 'product_category', 'location_id']:
                if col in df_std.columns:
                    sort_cols.append(col)
            df_std = df_std.sort_values(sort_cols).reset_index(drop=True)
        
        return df_std
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataframe"""
        df_clean = df.copy()
        
        # For numeric columns, fill with 0
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in ['demand', 'stock_level']:
                df_clean[col] = df_clean[col].fillna(0)
        
        # For string columns, fill with 'UNKNOWN'
        string_cols = df_clean.select_dtypes(include=['object']).columns
        for col in string_cols:
            if col in ['product_id', 'location_id', 'product_category']:
                df_clean[col] = df_clean[col].fillna('UNKNOWN')
        
        return df_clean
    
    def validate_and_clean_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, any]]:
        """
        Validate and clean data in one step
        
        Args:
            df: DataFrame to validate and clean
            
        Returns:
            Tuple of (cleaned_dataframe, validation_results)
        """
        # Check current state
        validation_results = self.check_dataframe_schema(df)
        
        # Convert to standard format
        df_cleaned = self.convert_to_standard_format(df)
        
        # Check final state
        final_validation = self.check_dataframe_schema(df_cleaned)
        
        # Log results
        if final_validation['is_valid']:
            self.logger.info("Data validation and cleaning completed successfully")
        else:
            self.logger.warning(f"Data cleaning completed with issues: {final_validation['issues']}")
        
        return df_cleaned, final_validation

# Convenience functions
def check_data_quality(df: pd.DataFrame) -> Dict[str, any]:
    """Quick data quality check"""
    standardizer = DataStandardizer()
    
    # Determine schema based on columns present
    if 'demand_frequency' in df.columns and 'risk_period' in df.columns:
        # Product master schema
        required_columns = ['product_id', 'product_category', 'location_id', 'demand_frequency', 'risk_period']
    else:
        # Demand data schema
        required_columns = ['product_id', 'product_category', 'location_id', 'date', 'demand', 'stock_level']
    
    return standardizer.check_dataframe_schema(df, required_columns)

def standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Quick dataframe standardization"""
    standardizer = DataStandardizer()
    
    # Determine target schema based on columns present
    if 'demand_frequency' in df.columns and 'risk_period' in df.columns:
        # Product master schema
        target_schema = {
            'product_id': 'object',
            'product_category': 'object', 
            'location_id': 'object',
            'demand_frequency': 'object',
            'risk_period': 'int64'
        }
    else:
        # Demand data schema
        target_schema = {
            'product_id': 'object',
            'product_category': 'object',
            'location_id': 'object', 
            'date': 'date',
            'demand': 'float64',
            'stock_level': 'float64'
        }
    
    return standardizer.convert_to_standard_format(df, target_schema)

def validate_and_clean(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """Quick validation and cleaning"""
    standardizer = DataStandardizer()
    return standardizer.validate_and_clean_data(df)
