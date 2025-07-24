"""
Data coverage validation component.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import time

from .types import ValidationResult, ValidationIssue, ValidationSeverity


class CoverageValidator:
    """Validates data coverage between demand and product master data"""
    
    def validate_coverage(
        self,
        demand_data: pd.DataFrame,
        product_master_data: pd.DataFrame
    ) -> ValidationResult:
        """
        Validate data coverage between demand and product master.
        
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
            # Get unique combinations from demand data
            demand_combinations = set(
                demand_data[["product_id", "location_id"]].apply(tuple, axis=1)
            )
            
            # Get unique combinations from product master
            master_combinations = set(
                product_master_data[["product_id", "location_id"]].apply(tuple, axis=1)
            )
            
            # Find missing combinations
            missing_combinations = demand_combinations - master_combinations
            extra_combinations = master_combinations - demand_combinations
            
            # Report missing combinations
            if missing_combinations:
                missing_list = list(missing_combinations)
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="coverage",
                    message=f"Found {len(missing_combinations)} product-location combinations in demand data missing from product master",
                    details={
                        "missing_combinations": [{"product_id": p, "location_id": l} for p, l in missing_list],
                        "missing_count": len(missing_combinations)
                    },
                    affected_records=len(demand_data[demand_data[["product_id", "location_id"]].apply(tuple, axis=1).isin(missing_combinations)]),
                    affected_products=[p for p, l in missing_list],
                    affected_locations=[l for p, l in missing_list]
                ))
            
            # Report extra combinations
            if extra_combinations:
                extra_list = list(extra_combinations)
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="coverage",
                    message=f"Found {len(extra_combinations)} product-location combinations in product master not present in demand data",
                    details={
                        "extra_combinations": [{"product_id": p, "location_id": l} for p, l in extra_list],
                        "extra_count": len(extra_combinations)
                    },
                    affected_products=[p for p, l in extra_list],
                    affected_locations=[l for p, l in extra_list]
                ))
            
            # Check for orphaned records
            self._check_orphaned_records(demand_data, product_master_data, issues)
            
            # Check for data volume coverage
            self._check_data_volume_coverage(demand_data, product_master_data, issues)
            
            # Create summary
            coverage_percentage = (
                len(demand_combinations - missing_combinations) / len(demand_combinations)
            ) * 100 if demand_combinations else 0
            
            summary = {
                "demand_combinations": len(demand_combinations),
                "master_combinations": len(master_combinations),
                "missing_combinations": len(missing_combinations),
                "extra_combinations": len(extra_combinations),
                "coverage_percentage": coverage_percentage,
                "is_valid": len(missing_combinations) == 0,
                "missing_details": [{"product_id": p, "location_id": l} for p, l in missing_combinations] if missing_combinations else [],
                "extra_details": [{"product_id": p, "location_id": l} for p, l in extra_combinations] if extra_combinations else []
            }
            
        except Exception as e:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                category="coverage",
                message=f"Coverage validation failed: {str(e)}",
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
    
    def _check_orphaned_records(self, demand_data: pd.DataFrame, product_master_data: pd.DataFrame, issues: List[ValidationIssue]):
        """Check for orphaned records that might indicate data quality issues"""
        if len(demand_data) == 0 or len(product_master_data) == 0:
            return
        
        # Check for products in demand data that don't exist in product master
        demand_products = set(demand_data['product_id'].unique())
        master_products = set(product_master_data['product_id'].unique())
        orphaned_products = demand_products - master_products
        
        if orphaned_products:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="coverage",
                message=f"Found {len(orphaned_products)} products in demand data not present in product master",
                details={
                    "orphaned_products": list(orphaned_products),
                    "orphaned_count": len(orphaned_products)
                },
                affected_records=len(demand_data[demand_data['product_id'].isin(orphaned_products)]),
                affected_products=list(orphaned_products)
            ))
        
        # Check for locations in demand data that don't exist in product master
        demand_locations = set(demand_data['location_id'].unique())
        master_locations = set(product_master_data['location_id'].unique())
        orphaned_locations = demand_locations - master_locations
        
        if orphaned_locations:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="coverage",
                message=f"Found {len(orphaned_locations)} locations in demand data not present in product master",
                details={
                    "orphaned_locations": list(orphaned_locations),
                    "orphaned_count": len(orphaned_locations)
                },
                affected_records=len(demand_data[demand_data['location_id'].isin(orphaned_locations)]),
                affected_locations=list(orphaned_locations)
            ))
    
    def _check_data_volume_coverage(self, demand_data: pd.DataFrame, product_master_data: pd.DataFrame, issues: List[ValidationIssue]):
        """Check for data volume coverage issues"""
        if len(demand_data) == 0 or len(product_master_data) == 0:
            return
        
        # Check for products with very few demand records
        product_record_counts = demand_data.groupby('product_id').size()
        low_volume_products = product_record_counts[product_record_counts < 10]  # Less than 10 records
        
        if len(low_volume_products) > 0:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="coverage",
                message=f"Found {len(low_volume_products)} products with very few demand records (< 10)",
                details={
                    "low_volume_products": low_volume_products.to_dict(),
                    "min_records": int(low_volume_products.min()),
                    "max_records": int(low_volume_products.max())
                },
                affected_records=int(low_volume_products.sum()),
                affected_products=low_volume_products.index.tolist()
            ))
        
        # Check for locations with very few demand records
        location_record_counts = demand_data.groupby('location_id').size()
        low_volume_locations = location_record_counts[location_record_counts < 10]  # Less than 10 records
        
        if len(low_volume_locations) > 0:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="coverage",
                message=f"Found {len(low_volume_locations)} locations with very few demand records (< 10)",
                details={
                    "low_volume_locations": low_volume_locations.to_dict(),
                    "min_records": int(low_volume_locations.min()),
                    "max_records": int(low_volume_locations.max())
                },
                affected_records=int(low_volume_locations.sum()),
                affected_locations=low_volume_locations.index.tolist()
            ))
        
        # Check for product-location combinations with very few records
        combo_record_counts = demand_data.groupby(['product_id', 'location_id']).size()
        low_volume_combos = combo_record_counts[combo_record_counts < 5]  # Less than 5 records
        
        if len(low_volume_combos) > 0:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="coverage",
                message=f"Found {len(low_volume_combos)} product-location combinations with very few demand records (< 5)",
                details={
                    "low_volume_combinations": low_volume_combos.to_dict(),
                    "min_records": int(low_volume_combos.min()),
                    "max_records": int(low_volume_combos.max())
                },
                affected_records=int(low_volume_combos.sum()),
                affected_products=[p for p, l in low_volume_combos.index],
                affected_locations=[l for p, l in low_volume_combos.index]
            )) 