"""
Data completeness validation component.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, date, timedelta
import time

from .demand_validator import DemandValidator
from .types import ValidationResult, ValidationIssue, ValidationSeverity


class CompletenessValidator:
    """Validates data completeness including missing dates and frequency consistency"""
    
    def __init__(self):
        self.demand_validator = DemandValidator()
    
    def validate_completeness(
        self,
        demand_data: pd.DataFrame,
        product_master_data: pd.DataFrame,
        demand_frequency: str = "d"
    ) -> ValidationResult:
        """
        Validate data completeness.
        
        Args:
            demand_data: Demand DataFrame
            product_master_data: Product master DataFrame
            demand_frequency: Expected demand frequency ('d', 'w', 'm')
            
        Returns:
            Validation result with issues and summary
        """
        start_time = time.time()
        issues = []
        summary = {}
        
        try:
            # Convert frequency format
            frequency_map = {'d': 'daily', 'w': 'weekly', 'm': 'monthly'}
            validation_frequency = frequency_map.get(demand_frequency, 'daily')
            
            # Use existing demand validator for completeness check
            completeness_result = self.demand_validator.validate_demand_completeness_with_data(
                demand_data, product_master_data, validation_frequency
            )
            
            # Process completeness issues
            if completeness_result.get('missing_dates_total', 0) > 0:
                for issue_detail in completeness_result.get('issues', []):
                    if issue_detail.get('type') == 'missing_dates':
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            category="completeness",
                            message=f"Missing dates for {issue_detail['product_id']}-{issue_detail['location_id']}",
                            details={
                                "product_id": issue_detail['product_id'],
                                "location_id": issue_detail['location_id'],
                                "missing_dates": [str(d) for d in issue_detail.get('missing_dates', [])],
                                "missing_count": issue_detail.get('missing_count', 0),
                                "expected_frequency": issue_detail.get('expected_frequency', ''),
                                "detected_frequency": issue_detail.get('detected_frequency', ''),
                                "date_range": issue_detail.get('date_range', ''),
                                "total_expected": issue_detail.get('total_expected', 0),
                                "total_actual": issue_detail.get('total_actual', 0)
                            },
                            affected_records=issue_detail.get('missing_count', 0),
                            affected_products=[issue_detail['product_id']],
                            affected_locations=[issue_detail['location_id']]
                        ))
                    elif issue_detail.get('type') == 'missing_master_record':
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            category="completeness",
                            message=f"No product master record found for {issue_detail['product_id']}-{issue_detail['location_id']}",
                            details={
                                "product_id": issue_detail['product_id'],
                                "location_id": issue_detail['location_id']
                            },
                            affected_products=[issue_detail['product_id']],
                            affected_locations=[issue_detail['location_id']]
                        ))
            
            # Check frequency consistency
            if completeness_result.get('frequency_mismatches', 0) > 0:
                for combo, freq_info in completeness_result.get('frequency_validation', {}).items():
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category="completeness",
                        message=f"Frequency mismatch for {combo}: expected {freq_info['expected']}, detected {freq_info['detected']}",
                        details={
                            "combination": combo,
                            "expected_frequency": freq_info['expected'],
                            "detected_frequency": freq_info['detected'],
                            "dates_count": freq_info['dates_count'],
                            "date_range": freq_info['date_range']
                        },
                        affected_records=freq_info['dates_count']
                    ))
            
            # Additional completeness checks
            self._check_data_gaps(demand_data, product_master_data, issues)
            self._check_date_continuity(demand_data, product_master_data, issues)
            
            # Create summary
            summary = {
                "total_combinations": completeness_result.get('total_combinations', 0),
                "valid_combinations": completeness_result.get('valid_combinations', 0),
                "invalid_combinations": completeness_result.get('invalid_combinations', 0),
                "missing_dates_total": completeness_result.get('missing_dates_total', 0),
                "frequency_mismatches": completeness_result.get('frequency_mismatches', 0),
                "valid_percentage": completeness_result.get('valid_percentage', 0),
                "invalid_percentage": completeness_result.get('invalid_percentage', 0),
                "expected_frequency": demand_frequency,
                "validation_frequency": validation_frequency
            }
            
        except Exception as e:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                category="completeness",
                message=f"Completeness validation failed: {str(e)}",
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
    
    def _check_data_gaps(
        self,
        demand_data: pd.DataFrame,
        product_master_data: pd.DataFrame,
        issues: List[ValidationIssue]
    ):
        """Check for large gaps in data"""
        if len(demand_data) == 0:
            return
        
        # Convert date column
        demand_data = demand_data.copy()
        demand_data['date'] = pd.to_datetime(demand_data['date'])
        
        # Check for gaps by product-location
        for _, master_record in product_master_data.iterrows():
            product_id = master_record['product_id']
            location_id = master_record['location_id']
            
            # Get demand data for this combination
            product_demand = demand_data[
                (demand_data['product_id'] == product_id) &
                (demand_data['location_id'] == location_id)
            ].copy()
            
            if len(product_demand) < 2:
                continue
            
            # Sort by date
            product_demand = product_demand.sort_values('date')
            
            # Calculate gaps
            date_diffs = product_demand['date'].diff().dt.days
            large_gaps = date_diffs[date_diffs > 30]  # Gaps larger than 30 days
            
            if len(large_gaps) > 0:
                max_gap = large_gaps.max()
                gap_count = len(large_gaps)
                
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="completeness",
                    message=f"Large data gaps detected for {product_id}-{location_id}",
                    details={
                        "product_id": product_id,
                        "location_id": location_id,
                        "max_gap_days": int(max_gap),
                        "gap_count": int(gap_count),
                        "total_records": len(product_demand)
                    },
                    affected_records=int(gap_count),
                    affected_products=[product_id],
                    affected_locations=[location_id]
                ))
    
    def _check_date_continuity(
        self,
        demand_data: pd.DataFrame,
        product_master_data: pd.DataFrame,
        issues: List[ValidationIssue]
    ):
        """Check for date continuity issues"""
        if len(demand_data) == 0:
            return
        
        # Convert date column
        demand_data = demand_data.copy()
        demand_data['date'] = pd.to_datetime(demand_data['date'])
        
        # Check for future dates
        today = pd.Timestamp.now().date()
        future_dates = demand_data[demand_data['date'].dt.date > today]
        
        if len(future_dates) > 0:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="completeness",
                message=f"Found {len(future_dates)} records with future dates",
                details={
                    "future_records_count": len(future_dates),
                    "future_date_range": f"{future_dates['date'].min()} to {future_dates['date'].max()}",
                    "today": str(today)
                },
                affected_records=len(future_dates)
            ))
        
        # Check for very old dates (more than 10 years ago)
        ten_years_ago = today - timedelta(days=3650)
        old_dates = demand_data[demand_data['date'].dt.date < ten_years_ago]
        
        if len(old_dates) > 0:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                category="completeness",
                message=f"Found {len(old_dates)} records with dates older than 10 years",
                details={
                    "old_records_count": len(old_dates),
                    "old_date_range": f"{old_dates['date'].min()} to {old_dates['date'].max()}",
                    "ten_years_ago": str(ten_years_ago)
                },
                affected_records=len(old_dates)
            )) 