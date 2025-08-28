"""
Schema validation component for demand and product master data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, date
import time

from .schema import DemandSchema
from .product_master_schema import ProductMasterSchema
from .types import ValidationResult, ValidationIssue, ValidationSeverity


class SchemaValidator:
    """
    Validates data schema and data types.
    Ensures data meets required format and type specifications.
    """
    
    def __init__(self):
        pass
    
    def _standardize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize data types before validation.
        Creates a working copy and ensures consistent types.
        """
        if df is None or len(df) == 0:
            return df
        
        working_df = df.copy()
        
        # Standardize date column
        if 'date' in working_df.columns:
            try:
                working_df['date'] = pd.to_datetime(working_df['date'], errors='coerce').dt.date
                # Remove rows where date conversion failed
                working_df = working_df.dropna(subset=['date'])
            except Exception as e:
                print(f"Warning: Date conversion failed: {e}")
                return pd.DataFrame()
        
        # Ensure numeric columns are numeric
        numeric_cols = ['demand', 'stock_level']
        for col in numeric_cols:
            if col in working_df.columns:
                try:
                    working_df[col] = pd.to_numeric(working_df[col], errors='coerce')
                except Exception:
                    # If conversion fails, keep original but will be flagged in validation
                    pass
        
        # Ensure string columns are strings
        string_cols = ['product_id', 'location_id', 'product_category']
        for col in string_cols:
            if col in working_df.columns:
                working_df[col] = working_df[col].astype(str)
        
        return working_df
    
    def validate_demand_schema(self, demand_data: pd.DataFrame) -> ValidationResult:
        """
        Validate demand data schema and data types.
        
        Args:
            demand_data: Demand DataFrame to validate
            
        Returns:
            Validation result with issues and summary
        """
        start_time = time.time()
        issues = []
        summary = {}
        
        try:
            # Standardize data types BEFORE validation
            standardized_data = self._standardize_data_types(demand_data)
            
            if len(standardized_data) == 0:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    category="schema",
                    message="Data is empty or cannot be standardized",
                    details={"original_length": len(demand_data) if demand_data is not None else 0},
                    affected_records=0
                ))
                return ValidationResult(
                    is_valid=False,
                    issues=issues,
                    summary={"error": "Data standardization failed"},
                    execution_time=time.time() - start_time
                )
            
            # Check required columns
            required_cols = DemandSchema.REQUIRED_COLUMNS
            missing_cols = set(required_cols) - set(standardized_data.columns)
            
            if missing_cols:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    category="schema",
                    message=f"Missing required columns: {missing_cols}",
                    details={"missing_columns": list(missing_cols)},
                    affected_records=len(standardized_data)
                ))
            
            # Check data types using standardized data
            if len(standardized_data) > 0:
                # Date column validation - already standardized
                if 'date' in standardized_data.columns:
                    # Check if any dates are still NaT after standardization
                    nat_count = standardized_data['date'].isna().sum()
                    if nat_count > 0:
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            category="schema",
                            message=f"Found {nat_count} invalid dates that could not be converted",
                            details={"nat_count": int(nat_count)},
                            affected_records=int(nat_count)
                        ))
                
                # Numeric columns validation
                numeric_cols = ['demand', 'stock_level']
                for col in numeric_cols:
                    if col in standardized_data.columns:
                        # Check for non-numeric values (NaN from conversion)
                        non_numeric_count = standardized_data[col].isna().sum()
                        if non_numeric_count > 0:
                            issues.append(ValidationIssue(
                                severity=ValidationSeverity.ERROR,
                                category="schema",
                                message=f"Column '{col}' contains {non_numeric_count} non-numeric values",
                                details={"non_numeric_count": int(non_numeric_count)},
                                affected_records=int(non_numeric_count)
                            ))
                
                # String columns validation
                string_cols = ['product_id', 'location_id', 'product_category']
                for col in string_cols:
                    if col in standardized_data.columns:
                        # Check for empty strings
                        empty_string_count = (standardized_data[col] == '').sum()
                        if empty_string_count > 0:
                            issues.append(ValidationIssue(
                                severity=ValidationSeverity.WARNING,
                                category="schema",
                                message=f"Column '{col}' contains empty strings",
                                details={"empty_string_count": int(empty_string_count)},
                                affected_records=int(empty_string_count)
                            ))
                
                # Check for duplicate product-location-date combinations
                if all(col in standardized_data.columns for col in ['product_id', 'location_id', 'date']):
                    duplicates = standardized_data.duplicated(subset=['product_id', 'location_id', 'date']).sum()
                    if duplicates > 0:
                        # Get details about the duplicates
                        duplicate_records = standardized_data[standardized_data.duplicated(subset=['product_id', 'location_id', 'date'], keep=False)]
                        duplicate_groups = duplicate_records.groupby(['product_id', 'location_id', 'date']).size()
                        
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            category="schema",
                            message=f"Found {duplicates} duplicate product-location-date combinations",
                            details={
                                "duplicate_count": int(duplicates),
                                "duplicate_combinations": len(duplicate_groups),
                                "max_duplicates_per_combination": int(duplicate_groups.max()),
                                "sample_duplicates": duplicate_groups.head(5).to_dict()
                            },
                            affected_records=int(duplicates)
                        ))
            
            # Create summary
            summary = {
                "total_records": len(standardized_data),
                "total_columns": len(standardized_data.columns),
                "required_columns_present": len(missing_cols) == 0,
                "missing_columns": list(missing_cols) if missing_cols else [],
                "has_date_column": 'date' in standardized_data.columns,
                "has_numeric_columns": all(col in standardized_data.columns for col in ['demand', 'stock_level']),
                "standardized_records": len(standardized_data),
                "original_records": len(demand_data) if demand_data is not None else 0
            }
            
        except Exception as e:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                category="schema",
                message=f"Schema validation failed: {str(e)}",
                details={"error": str(e)},
                affected_records=len(demand_data) if demand_data is not None else 0
            ))
            summary = {"error": str(e)}
        
        execution_time = time.time() - start_time
        is_valid = len([i for i in issues if i.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR]]) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            summary=summary,
            execution_time=execution_time
        )
    
    def validate_product_master_schema(self, product_master_data: pd.DataFrame) -> ValidationResult:
        """
        Validate product master data schema.
        
        Args:
            product_master_data: Product master DataFrame
            
        Returns:
            Validation result with issues and summary
        """
        start_time = time.time()
        issues = []
        summary = {}
        
        try:
            # Check required columns
            required_cols = ProductMasterSchema.REQUIRED_COLUMNS
            missing_cols = set(required_cols) - set(product_master_data.columns)
            
            if missing_cols:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    category="schema",
                    message=f"Missing required columns: {missing_cols}",
                    details={"missing_columns": list(missing_cols)},
                    affected_records=len(product_master_data)
                ))
            
            # Check data types and values
            if len(product_master_data) > 0:
                # Demand frequency validation
                if 'demand_frequency' in product_master_data.columns:
                    valid_frequencies = ['d', 'w', 'm']
                    invalid_freq = set(product_master_data['demand_frequency'].unique()) - set(valid_frequencies)
                    if invalid_freq:
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            category="schema",
                            message=f"Invalid demand frequencies found: {invalid_freq}",
                            details={"invalid_frequencies": list(invalid_freq)},
                            affected_records=len(product_master_data[product_master_data['demand_frequency'].isin(invalid_freq)])
                        ))
                
                # Risk period validation
                if 'risk_period' in product_master_data.columns:
                    try:
                        risk_periods = pd.to_numeric(product_master_data['risk_period'], errors='coerce')
                        invalid_risk = (risk_periods <= 0).sum()
                        if invalid_risk > 0:
                            issues.append(ValidationIssue(
                                severity=ValidationSeverity.ERROR,
                                category="schema",
                                message=f"Found {invalid_risk} non-positive risk periods",
                                details={"invalid_risk_count": int(invalid_risk)},
                                affected_records=int(invalid_risk)
                            ))
                        
                        # Check for reasonable risk period limits
                        if 'demand_frequency' in product_master_data.columns:
                            for freq in ['d', 'w', 'm']:
                                freq_data = product_master_data[product_master_data['demand_frequency'] == freq]
                                if len(freq_data) > 0:
                                    freq_risk = pd.to_numeric(freq_data['risk_period'], errors='coerce')
                                    max_allowed = {'d': 365, 'w': 52, 'm': 12}[freq]
                                    excessive_risk = (freq_risk > max_allowed).sum()
                                    if excessive_risk > 0:
                                        issues.append(ValidationIssue(
                                            severity=ValidationSeverity.WARNING,
                                            category="schema",
                                            message=f"Found {excessive_risk} excessive risk periods for {freq} frequency",
                                            details={"frequency": freq, "max_allowed": max_allowed, "excessive_count": int(excessive_risk)},
                                            affected_records=int(excessive_risk)
                                        ))
                    except Exception as e:
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.CRITICAL,
                            category="schema",
                            message=f"Risk period conversion failed: {str(e)}",
                            details={"error": str(e)},
                            affected_records=len(product_master_data)
                        ))
                
                # Lead time validation
                if 'leadtime' in product_master_data.columns:
                    try:
                        leadtimes = pd.to_numeric(product_master_data['leadtime'], errors='coerce')
                        invalid_leadtime = (leadtimes <= 0).sum()
                        if invalid_leadtime > 0:
                            issues.append(ValidationIssue(
                                severity=ValidationSeverity.ERROR,
                                category="schema",
                                message=f"Found {invalid_leadtime} non-positive lead times",
                                details={"invalid_leadtime_count": int(invalid_leadtime)},
                                affected_records=int(invalid_leadtime)
                            ))
                    except Exception as e:
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.CRITICAL,
                            category="schema",
                            message=f"Lead time conversion failed: {str(e)}",
                            details={"error": str(e)},
                            affected_records=len(product_master_data)
                        ))
                
                # MOQ validation
                if 'moq' in product_master_data.columns:
                    try:
                        moq_values = pd.to_numeric(product_master_data['moq'], errors='coerce')
                        invalid_moq = (moq_values < 0).sum()
                        if invalid_moq > 0:
                            issues.append(ValidationIssue(
                                severity=ValidationSeverity.ERROR,
                                category="schema",
                                message=f"Found {invalid_moq} negative MOQ values",
                                details={"invalid_moq_count": int(invalid_moq)},
                                affected_records=int(invalid_moq)
                            ))
                    except Exception as e:
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.CRITICAL,
                            category="schema",
                            message=f"MOQ conversion failed: {str(e)}",
                            details={"error": str(e)},
                            affected_records=len(product_master_data)
                        ))
                
                # Min safety stock validation
                if 'min_safety_stock' in product_master_data.columns:
                    try:
                        # Handle empty strings and whitespace by cleaning the data
                        def clean_min_safety_stock(val):
                            if pd.isna(val):
                                return 0.0
                            if isinstance(val, str):
                                if val.strip() == "":
                                    return 0.0
                                try:
                                    return float(val)
                                except ValueError:
                                    return 0.0
                            return val
                        
                        min_ss_clean = product_master_data['min_safety_stock'].apply(clean_min_safety_stock)
                        min_ss_values = pd.to_numeric(min_ss_clean, errors='coerce')
                        invalid_min_ss = (min_ss_values < 0).sum()
                        if invalid_min_ss > 0:
                            issues.append(ValidationIssue(
                                severity=ValidationSeverity.ERROR,
                                category="schema",
                                message=f"Found {invalid_min_ss} negative minimum safety stock values",
                                details={"invalid_min_ss_count": int(invalid_min_ss)},
                                affected_records=int(invalid_min_ss)
                            ))
                    except Exception as e:
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.CRITICAL,
                            category="schema",
                            message=f"Minimum safety stock conversion failed: {str(e)}",
                            details={"error": str(e)},
                            affected_records=len(product_master_data)
                        ))
                
                # Max safety stock validation
                if 'max_safety_stock' in product_master_data.columns:
                    try:
                        # Handle empty strings and whitespace by cleaning the data
                        def clean_max_safety_stock(val):
                            if pd.isna(val):
                                return None
                            if isinstance(val, str):
                                if val.strip() == "":
                                    return None
                                try:
                                    return float(val)
                                except ValueError:
                                    return None
                            return val
                        
                        max_ss_clean = product_master_data['max_safety_stock'].apply(clean_max_safety_stock)
                        
                        # Count negative values (excluding None values)
                        invalid_max_ss = 0
                        for val in max_ss_clean:
                            if val is not None and val < 0:
                                invalid_max_ss += 1
                        
                        if invalid_max_ss > 0:
                            issues.append(ValidationIssue(
                                severity=ValidationSeverity.ERROR,
                                category="schema",
                                message=f"Found {invalid_max_ss} negative maximum safety stock values",
                                details={"invalid_max_ss_count": int(invalid_max_ss)},
                                affected_records=int(invalid_max_ss)
                            ))
                            
                    except Exception as e:
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.CRITICAL,
                            category="schema",
                            message=f"Maximum safety stock conversion failed: {str(e)}",
                            details={"error": str(e)},
                            affected_records=len(product_master_data)
                        ))
                
                # Sunset date validation
                if 'sunset_date' in product_master_data.columns:
                    try:
                        # Handle empty strings and convert to dates or None
                        def clean_sunset_date(val):
                            if pd.isna(val):
                                return None
                            if isinstance(val, str):
                                if val.strip() == "":
                                    return None
                                try:
                                    from datetime import datetime
                                    return datetime.strptime(val.strip(), "%Y-%m-%d").date()
                                except ValueError:
                                    return None  # Invalid date format becomes None
                            return val
                        
                        sunset_dates_clean = product_master_data['sunset_date'].apply(clean_sunset_date)
                        
                        # Count invalid dates (those that couldn't be parsed and became None when they shouldn't have)
                        # Only count as invalid if the original value was a non-empty string that couldn't be parsed
                        invalid_dates = 0
                        for i, (original, cleaned) in enumerate(zip(product_master_data['sunset_date'], sunset_dates_clean)):
                            if isinstance(original, str) and original.strip() != "" and cleaned is None:
                                invalid_dates += 1
                        
                        if invalid_dates > 0:
                            issues.append(ValidationIssue(
                                severity=ValidationSeverity.ERROR,
                                category="schema",
                                message=f"Found {invalid_dates} invalid sunset date formats",
                                details={"invalid_date_count": int(invalid_dates)},
                                affected_records=int(invalid_dates)
                            ))
                            
                    except Exception as e:
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.CRITICAL,
                            category="schema",
                            message=f"Sunset date conversion failed: {str(e)}",
                            details={"error": str(e)},
                            affected_records=len(product_master_data)
                        ))
                
                # Check for duplicate product-location combinations
                if all(col in product_master_data.columns for col in ['product_id', 'location_id']):
                    duplicates = product_master_data.duplicated(subset=['product_id', 'location_id']).sum()
                    if duplicates > 0:
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            category="schema",
                            message=f"Found {duplicates} duplicate product-location combinations",
                            details={"duplicate_count": int(duplicates)},
                            affected_records=int(duplicates)
                        ))
            
            # Create summary
            summary = {
                "total_records": len(product_master_data),
                "total_columns": len(product_master_data.columns),
                "required_columns_present": len(missing_cols) == 0,
                "missing_columns": list(missing_cols) if missing_cols else [],
                "unique_products": product_master_data['product_id'].nunique() if 'product_id' in product_master_data.columns else 0,
                "unique_locations": product_master_data['location_id'].nunique() if 'location_id' in product_master_data.columns else 0,
                "unique_combinations": product_master_data.groupby(['product_id', 'location_id']).size().shape[0] if all(col in product_master_data.columns for col in ['product_id', 'location_id']) else 0
            }
            
        except Exception as e:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                category="schema",
                message=f"Schema validation failed: {str(e)}",
                details={"error": str(e)},
                affected_records=len(product_master_data) if product_master_data is not None else 0
            ))
            summary = {"error": str(e)}
        
        execution_time = time.time() - start_time
        is_valid = len([i for i in issues if i.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR]]) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            summary=summary,
            execution_time=execution_time
        ) 