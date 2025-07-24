"""
Data quality validation component.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import time

from .types import ValidationResult, ValidationIssue, ValidationSeverity


class QualityValidator:
    """Validates data quality including negative values, outliers, and statistical anomalies"""
    
    def validate_quality(
        self,
        demand_data: pd.DataFrame,
        product_master_data: pd.DataFrame
    ) -> ValidationResult:
        """
        Validate data quality.
        
        Args:
            demand_data: Demand DataFrame
            product_master_data: Product master DataFrame
            
        Returns:
            Validation result with issues and summary
        """
        start_time = time.time()
        issues = []
        summary = {}
        
        try:
            # Check for negative values
            self._check_negative_values(demand_data, issues)
            
            # Check for zero values
            self._check_zero_values(demand_data, issues)
            
            # Check for extreme values
            self._check_extreme_values(demand_data, issues)
            
            # Check for statistical anomalies
            self._check_statistical_anomalies(demand_data, product_master_data, issues)
            
            # Check for data consistency
            self._check_data_consistency(demand_data, product_master_data, issues)
            
            # Create summary
            summary = {
                "total_records": len(demand_data),
                "negative_demand_count": len(demand_data[demand_data['demand'] < 0]) if 'demand' in demand_data.columns else 0,
                "negative_stock_count": len(demand_data[demand_data['stock_level'] < 0]) if 'stock_level' in demand_data.columns else 0,
                "zero_demand_count": len(demand_data[demand_data['demand'] == 0]) if 'demand' in demand_data.columns else 0,
                "zero_stock_count": len(demand_data[demand_data['stock_level'] == 0]) if 'stock_level' in demand_data.columns else 0,
                "extreme_demand_count": 0,  # Will be calculated in _check_extreme_values
                "extreme_stock_count": 0,   # Will be calculated in _check_extreme_values
                "unique_products": demand_data['product_id'].nunique() if 'product_id' in demand_data.columns else 0,
                "unique_locations": demand_data['location_id'].nunique() if 'location_id' in demand_data.columns else 0
            }
            
        except Exception as e:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                category="quality",
                message=f"Quality validation failed: {str(e)}",
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
    
    def _check_negative_values(self, demand_data: pd.DataFrame, issues: List[ValidationIssue]):
        """Check for negative values in numeric columns"""
        if len(demand_data) == 0:
            return
        
        # Check demand column
        if 'demand' in demand_data.columns:
            negative_demand = demand_data[demand_data['demand'] < 0]
            if len(negative_demand) > 0:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="quality",
                    message=f"Found {len(negative_demand)} records with negative demand values",
                    details={
                        "negative_count": len(negative_demand),
                        "min_demand": float(negative_demand['demand'].min()),
                        "max_demand": float(negative_demand['demand'].max()),
                        "affected_products": negative_demand['product_id'].unique().tolist() if 'product_id' in negative_demand.columns else []
                    },
                    affected_records=len(negative_demand),
                    affected_products=negative_demand['product_id'].unique().tolist() if 'product_id' in negative_demand.columns else None
                ))
        
        # Check stock_level column
        if 'stock_level' in demand_data.columns:
            negative_stock = demand_data[demand_data['stock_level'] < 0]
            if len(negative_stock) > 0:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="quality",
                    message=f"Found {len(negative_stock)} records with negative stock levels",
                    details={
                        "negative_count": len(negative_stock),
                        "min_stock": float(negative_stock['stock_level'].min()),
                        "max_stock": float(negative_stock['stock_level'].max()),
                        "affected_products": negative_stock['product_id'].unique().tolist() if 'product_id' in negative_stock.columns else []
                    },
                    affected_records=len(negative_stock),
                    affected_products=negative_stock['product_id'].unique().tolist() if 'product_id' in negative_stock.columns else None
                ))
    
    def _check_zero_values(self, demand_data: pd.DataFrame, issues: List[ValidationIssue]):
        """Check for zero values that might indicate data issues"""
        if len(demand_data) == 0:
            return
        
        # Check for all-zero demand periods
        if 'demand' in demand_data.columns:
            zero_demand = demand_data[demand_data['demand'] == 0]
            if len(zero_demand) > 0:
                # Check for consecutive zero demand periods
                zero_demand = zero_demand.sort_values(['product_id', 'location_id', 'date'])
                consecutive_zeros = 0
                
                for _, group in zero_demand.groupby(['product_id', 'location_id']):
                    if len(group) > 1:
                        # Check for consecutive dates with zero demand
                        group = group.sort_values('date')
                        date_diffs = pd.to_datetime(group['date']).diff().dt.days
                        consecutive_periods = (date_diffs == 1).sum()
                        if consecutive_periods > 7:  # More than a week of consecutive zeros
                            consecutive_zeros += consecutive_periods
                
                if consecutive_zeros > 0:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category="quality",
                        message=f"Found {consecutive_zeros} consecutive zero demand periods",
                        details={
                            "consecutive_zero_periods": consecutive_zeros,
                            "total_zero_records": len(zero_demand)
                        },
                        affected_records=consecutive_zeros
                    ))
    
    def _check_extreme_values(self, demand_data: pd.DataFrame, issues: List[ValidationIssue]):
        """Check for extreme values that might be outliers"""
        if len(demand_data) == 0:
            return
        
        # Check demand column for extreme values
        if 'demand' in demand_data.columns:
            # Calculate statistics
            demand_stats = demand_data['demand'].describe()
            q1 = demand_stats['25%']
            q3 = demand_stats['75%']
            iqr = q3 - q1
            upper_bound = q3 + 3 * iqr  # 3x IQR for extreme outliers
            
            extreme_demand = demand_data[demand_data['demand'] > upper_bound]
            if len(extreme_demand) > 0:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="quality",
                    message=f"Found {len(extreme_demand)} records with extreme demand values",
                    details={
                        "extreme_count": len(extreme_demand),
                        "upper_bound": float(upper_bound),
                        "max_demand": float(extreme_demand['demand'].max()),
                        "iqr": float(iqr),
                        "q1": float(q1),
                        "q3": float(q3)
                    },
                    affected_records=len(extreme_demand),
                    affected_products=extreme_demand['product_id'].unique().tolist() if 'product_id' in extreme_demand.columns else None
                ))
        
        # Check stock_level column for extreme values
        if 'stock_level' in demand_data.columns:
            # Calculate statistics
            stock_stats = demand_data['stock_level'].describe()
            q1 = stock_stats['25%']
            q3 = stock_stats['75%']
            iqr = q3 - q1
            upper_bound = q3 + 3 * iqr  # 3x IQR for extreme outliers
            
            extreme_stock = demand_data[demand_data['stock_level'] > upper_bound]
            if len(extreme_stock) > 0:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="quality",
                    message=f"Found {len(extreme_stock)} records with extreme stock levels",
                    details={
                        "extreme_count": len(extreme_stock),
                        "upper_bound": float(upper_bound),
                        "max_stock": float(extreme_stock['stock_level'].max()),
                        "iqr": float(iqr),
                        "q1": float(q1),
                        "q3": float(q3)
                    },
                    affected_records=len(extreme_stock),
                    affected_products=extreme_stock['product_id'].unique().tolist() if 'product_id' in extreme_stock.columns else None
                ))
    
    def _check_statistical_anomalies(self, demand_data: pd.DataFrame, product_master_data: pd.DataFrame, issues: List[ValidationIssue]):
        """Check for statistical anomalies by product-location"""
        if len(demand_data) == 0 or len(product_master_data) == 0:
            return
        
        # Check for products with very low variance (might indicate data quality issues)
        for _, master_record in product_master_data.iterrows():
            product_id = master_record['product_id']
            location_id = master_record['location_id']
            
            # Get demand data for this combination
            product_demand = demand_data[
                (demand_data['product_id'] == product_id) &
                (demand_data['location_id'] == location_id)
            ].copy()
            
            if len(product_demand) < 10:  # Need sufficient data for statistical analysis
                continue
            
            # Check for zero variance
            demand_variance = product_demand['demand'].var()
            if demand_variance == 0:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="quality",
                    message=f"Zero variance in demand for {product_id}-{location_id}",
                    details={
                        "product_id": product_id,
                        "location_id": location_id,
                        "records_count": len(product_demand),
                        "constant_value": float(product_demand['demand'].iloc[0])
                    },
                    affected_records=len(product_demand),
                    affected_products=[product_id],
                    affected_locations=[location_id]
                ))
            
            # Check for very low variance (coefficient of variation < 0.01)
            demand_mean = product_demand['demand'].mean()
            if demand_mean > 0:
                coefficient_of_variation = np.sqrt(demand_variance) / demand_mean
                if coefficient_of_variation < 0.01:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        category="quality",
                        message=f"Very low demand variability for {product_id}-{location_id}",
                        details={
                            "product_id": product_id,
                            "location_id": location_id,
                            "coefficient_of_variation": float(coefficient_of_variation),
                            "mean_demand": float(demand_mean),
                            "std_demand": float(np.sqrt(demand_variance))
                        },
                        affected_records=len(product_demand),
                        affected_products=[product_id],
                        affected_locations=[location_id]
                    ))
    
    def _check_data_consistency(self, demand_data: pd.DataFrame, product_master_data: pd.DataFrame, issues: List[ValidationIssue]):
        """Check for data consistency issues"""
        if len(demand_data) == 0 or len(product_master_data) == 0:
            return
        
        # Check for demand > stock_level (might indicate data quality issues)
        if all(col in demand_data.columns for col in ['demand', 'stock_level']):
            inconsistent_records = demand_data[demand_data['demand'] > demand_data['stock_level']]
            if len(inconsistent_records) > 0:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="quality",
                    message=f"Found {len(inconsistent_records)} records where demand exceeds stock level",
                    details={
                        "inconsistent_count": len(inconsistent_records),
                        "max_demand_excess": float((inconsistent_records['demand'] - inconsistent_records['stock_level']).max()),
                        "affected_products": inconsistent_records['product_id'].unique().tolist()
                    },
                    affected_records=len(inconsistent_records),
                    affected_products=inconsistent_records['product_id'].unique().tolist()
                ))
        
        # Check for missing product categories
        if all(col in demand_data.columns for col in ['product_id', 'product_category']):
            missing_categories = demand_data[demand_data['product_category'].isna() | (demand_data['product_category'] == '')]
            if len(missing_categories) > 0:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="quality",
                    message=f"Found {len(missing_categories)} records with missing product categories",
                    details={
                        "missing_category_count": len(missing_categories),
                        "affected_products": missing_categories['product_id'].unique().tolist()
                    },
                    affected_records=len(missing_categories),
                    affected_products=missing_categories['product_id'].unique().tolist()
                )) 