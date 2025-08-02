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
    """Validates data schemas for demand and product master data"""
    
    def validate_demand_schema(self, demand_data: pd.DataFrame) -> ValidationResult:
        """
        Validate demand data schema.
        
        Args:
            demand_data: Demand DataFrame
            
        Returns:
            Validation result with issues and summary
        """
        start_time = time.time()
        issues = []
        summary = {}
        
        try:
            # Check required columns
            required_cols = DemandSchema.REQUIRED_COLUMNS
            missing_cols = set(required_cols) - set(demand_data.columns)
            
            if missing_cols:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    category="schema",
                    message=f"Missing required columns: {missing_cols}",
                    details={"missing_columns": list(missing_cols)},
                    affected_records=len(demand_data)
                ))
            
            # Check data types
            if len(demand_data) > 0:
                # Date column validation
                if 'date' in demand_data.columns:
                    try:
                        pd.to_datetime(demand_data['date'])
                    except Exception as e:
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.CRITICAL,
                            category="schema",
                            message="Date column cannot be converted to datetime",
                            details={"error": str(e)},
                            affected_records=len(demand_data)
                        ))
                
                # Numeric columns validation
                numeric_cols = ['demand', 'stock_level']
                for col in numeric_cols:
                    if col in demand_data.columns:
                        try:
                            pd.to_numeric(demand_data[col], errors='coerce')
                            # Check for non-numeric values
                            non_numeric_count = demand_data[col].apply(
                                lambda x: not pd.isna(x) and not isinstance(x, (int, float, np.number))
                            ).sum()
                            if non_numeric_count > 0:
                                issues.append(ValidationIssue(
                                    severity=ValidationSeverity.ERROR,
                                    category="schema",
                                    message=f"Column '{col}' contains non-numeric values",
                                    details={"non_numeric_count": int(non_numeric_count)},
                                    affected_records=int(non_numeric_count)
                                ))
                        except Exception as e:
                            issues.append(ValidationIssue(
                                severity=ValidationSeverity.CRITICAL,
                                category="schema",
                                message=f"Column '{col}' cannot be converted to numeric",
                                details={"error": str(e)},
                                affected_records=len(demand_data)
                            ))
                
                # String columns validation
                string_cols = ['product_id', 'location_id', 'product_category']
                for col in string_cols:
                    if col in demand_data.columns:
                        # Check for empty strings
                        empty_string_count = (demand_data[col] == '').sum()
                        if empty_string_count > 0:
                            issues.append(ValidationIssue(
                                severity=ValidationSeverity.WARNING,
                                category="schema",
                                message=f"Column '{col}' contains empty strings",
                                details={"empty_string_count": int(empty_string_count)},
                                affected_records=int(empty_string_count)
                            ))
            
            # Create summary
            summary = {
                "total_records": len(demand_data),
                "total_columns": len(demand_data.columns),
                "required_columns_present": len(missing_cols) == 0,
                "missing_columns": list(missing_cols) if missing_cols else [],
                "has_date_column": 'date' in demand_data.columns,
                "has_numeric_columns": all(col in demand_data.columns for col in ['demand', 'stock_level'])
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