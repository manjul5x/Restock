"""
Main data validator that orchestrates all validation components.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, date
from pathlib import Path
import json
import time

from ..utils.logger import get_logger
from .types import ValidationResult, ValidationIssue, ValidationReport, ValidationSeverity
from .schema_validator import SchemaValidator
from .completeness_validator import CompletenessValidator
from .quality_validator import QualityValidator
from .coverage_validator import CoverageValidator


class DataValidator:
    """
    Comprehensive data validator that orchestrates all validation components.
    """
    
    def __init__(self, log_level: str = "INFO"):
        self.logger = get_logger(__name__, log_level)
        
        # Initialize validation components
        self.schema_validator = SchemaValidator()
        self.completeness_validator = CompletenessValidator()
        self.quality_validator = QualityValidator()
        self.coverage_validator = CoverageValidator()
    
    def _standardize_data_types(self, df: pd.DataFrame, data_type: str = "demand") -> pd.DataFrame:
        """
        Utility method to standardize data types before validation.
        Creates a working copy and ensures consistent types.
        
        Args:
            df: DataFrame to standardize
            data_type: Type of data ('demand' or 'product_master')
            
        Returns:
            Standardized DataFrame
        """
        if df is None or len(df) == 0:
            return df
        
        working_df = df.copy()
        
        try:
            if data_type == "demand":
                # Standardize date column
                if 'date' in working_df.columns:
                    working_df['date'] = pd.to_datetime(working_df['date'], errors='coerce').dt.date
                    # Remove rows where date conversion failed
                    working_df = working_df.dropna(subset=['date'])
                
                # Ensure numeric columns are numeric
                numeric_cols = ['demand', 'stock_level']
                for col in numeric_cols:
                    if col in working_df.columns:
                        working_df[col] = pd.to_numeric(working_df[col], errors='coerce')
                
                # Ensure string columns are strings
                string_cols = ['product_id', 'location_id', 'product_category']
                for col in string_cols:
                    if col in working_df.columns:
                        working_df[col] = working_df[col].astype(str)
            
            elif data_type == "product_master":
                # Ensure string columns are strings
                string_cols = ['product_id', 'location_id', 'product_category', 'demand_frequency']
                for col in string_cols:
                    if col in working_df.columns:
                        working_df[col] = working_df[col].astype(str)
                
                # Ensure numeric columns are numeric
                numeric_cols = ['leadtime', 'service_level', 'moq', 'inventory_cost', 'outlier_threshold', 'ss_window_length']
                for col in numeric_cols:
                    if col in working_df.columns:
                        working_df[col] = pd.to_numeric(working_df[col], errors='coerce')
            
            self.logger.debug(f"Standardized {data_type} data: {len(working_df)} records")
            return working_df
            
        except Exception as e:
            self.logger.error(f"Data standardization failed for {data_type}: {e}")
            return pd.DataFrame()
    
    def validate_data(
        self,
        demand_data: pd.DataFrame,
        product_master_data: pd.DataFrame,
        demand_frequency: str = "d",
        validate_schema: bool = True,
        validate_completeness: bool = True,
        validate_quality: bool = True,
        validate_coverage: bool = True,
        strict_mode: bool = False
    ) -> ValidationReport:
        """
        Perform comprehensive data validation.
        
        Args:
            demand_data: Demand DataFrame
            product_master_data: Product master DataFrame
            demand_frequency: Expected demand frequency ('d', 'w', 'm')
            validate_schema: Whether to validate schema
            validate_completeness: Whether to validate completeness
            validate_quality: Whether to validate data quality
            validate_coverage: Whether to validate coverage
            strict_mode: If True, treats warnings as errors
            
        Returns:
            Comprehensive validation report
        """
        import time
        start_time = time.time()
        
        self.logger.info("Starting comprehensive data validation")
        self.logger.info(f"Demand records: {len(demand_data)}")
        self.logger.info(f"Product master records: {len(product_master_data)}")
        
        # Initialize results
        demand_validation = None
        product_master_validation = None
        coverage_validation = None
        quality_validation = None
        completeness_validation = None
        
        # Schema validation
        if validate_schema:
            self.logger.info("Validating schemas...")
            demand_validation = self.schema_validator.validate_demand_schema(demand_data)
            product_master_validation = self.schema_validator.validate_product_master_schema(product_master_data)
        
        # Coverage validation
        if validate_coverage:
            self.logger.info("Validating data coverage...")
            coverage_validation = self.coverage_validator.validate_coverage(
                demand_data, product_master_data
            )
        
        # Quality validation
        if validate_quality:
            self.logger.info("Validating data quality...")
            quality_validation = self.quality_validator.validate_quality(
                demand_data, product_master_data
            )
        
        # Completeness validation
        if validate_completeness:
            self.logger.info("Validating data completeness...")
            completeness_validation = self.completeness_validator.validate_completeness(
                demand_data, product_master_data, demand_frequency
            )
        
        # Calculate overall results
        execution_time = time.time() - start_time
        overall_valid, total_issues, critical_issues, warnings = self._calculate_overall_results(
            demand_validation, product_master_validation, coverage_validation,
            quality_validation, completeness_validation, strict_mode
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            demand_validation, product_master_validation, coverage_validation,
            quality_validation, completeness_validation
        )
        
        # Create validation report
        report = ValidationReport(
            overall_valid=overall_valid,
            demand_validation=demand_validation or ValidationResult(True, [], {}, 0),
            product_master_validation=product_master_validation or ValidationResult(True, [], {}, 0),
            coverage_validation=coverage_validation or ValidationResult(True, [], {}, 0),
            quality_validation=quality_validation or ValidationResult(True, [], {}, 0),
            completeness_validation=completeness_validation or ValidationResult(True, [], {}, 0),
            total_issues=total_issues,
            critical_issues=critical_issues,
            warnings=warnings,
            execution_time=execution_time,
            recommendations=recommendations
        )
        
        self.logger.info(f"Validation completed in {execution_time:.2f} seconds")
        self.logger.info(f"Overall valid: {overall_valid}")
        self.logger.info(f"Total issues: {total_issues}, Critical: {critical_issues}, Warnings: {warnings}")
        
        return report
    
    def _calculate_overall_results(
        self,
        demand_validation: Optional[ValidationResult],
        product_master_validation: Optional[ValidationResult],
        coverage_validation: Optional[ValidationResult],
        quality_validation: Optional[ValidationResult],
        completeness_validation: Optional[ValidationResult],
        strict_mode: bool
    ) -> tuple[bool, int, int, int]:
        """Calculate overall validation results"""
        all_results = [
            r for r in [demand_validation, product_master_validation, coverage_validation,
                       quality_validation, completeness_validation] if r is not None
        ]
        
        # Check if all validations passed
        overall_valid = all(r.is_valid for r in all_results)
        
        # Count issues
        total_issues = 0
        critical_issues = 0
        warnings = 0
        
        for result in all_results:
            for issue in result.issues:
                total_issues += 1
                if issue.severity == ValidationSeverity.CRITICAL:
                    critical_issues += 1
                    overall_valid = False
                elif issue.severity == ValidationSeverity.ERROR:
                    critical_issues += 1
                    overall_valid = False
                elif issue.severity == ValidationSeverity.WARNING:
                    warnings += 1
                    if strict_mode:
                        overall_valid = False
        
        return overall_valid, total_issues, critical_issues, warnings
    
    def _generate_recommendations(
        self,
        demand_validation: Optional[ValidationResult],
        product_master_validation: Optional[ValidationResult],
        coverage_validation: Optional[ValidationResult],
        quality_validation: Optional[ValidationResult],
        completeness_validation: Optional[ValidationResult]
    ) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        # Schema recommendations
        if demand_validation and not demand_validation.is_valid:
            recommendations.append("Fix demand data schema issues before proceeding")
        
        if product_master_validation and not product_master_validation.is_valid:
            recommendations.append("Fix product master schema issues before proceeding")
        
        # Coverage recommendations
        if coverage_validation and not coverage_validation.is_valid:
            recommendations.append("Add missing product-location combinations to product master")
        
        # Quality recommendations
        if quality_validation and not quality_validation.is_valid:
            recommendations.append("Review and clean data quality issues")
        
        # Completeness recommendations
        if completeness_validation and not completeness_validation.is_valid:
            recommendations.append("Address missing dates and data completeness issues")
        
        # General recommendations
        if not recommendations:
            recommendations.append("Data validation passed successfully")
        
        return recommendations
    
    def print_report(self, report: ValidationReport, detailed: bool = True):
        """Print validation report in a formatted way"""
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE DATA VALIDATION REPORT")
        print("="*80)
        
        # Overall status
        status_icon = "✅" if report.overall_valid else "❌"
        print(f"{status_icon} Overall Status: {'VALID' if report.overall_valid else 'INVALID'}")
        print(f"⏱️  Execution Time: {report.execution_time:.2f} seconds")
        print(f"📈 Total Issues: {report.total_issues}")
        print(f"🚨 Critical Issues: {report.critical_issues}")
        print(f"⚠️  Warnings: {report.warnings}")
        
        # Summary by category
        print("\n📋 VALIDATION SUMMARY:")
        print("-" * 40)
        
        categories = [
            ("Demand Schema", report.demand_validation),
            ("Product Master Schema", report.product_master_validation),
            ("Data Coverage", report.coverage_validation),
            ("Data Quality", report.quality_validation),
            ("Data Completeness", report.completeness_validation)
        ]
        
        for category_name, result in categories:
            if result is not None:
                status = "✅ PASS" if result.is_valid else "❌ FAIL"
                issues_count = len(result.issues)
                print(f"{status} {category_name}: {issues_count} issues")
        
        # Detailed issues
        if detailed and report.total_issues > 0:
            print("\n🔍 DETAILED ISSUES:")
            print("-" * 40)
            
            for category_name, result in categories:
                if result and result.issues:
                    print(f"\n📁 {category_name.upper()}:")
                    for i, issue in enumerate(result.issues, 1):
                        severity_icon = {
                            ValidationSeverity.CRITICAL: "🚨",
                            ValidationSeverity.ERROR: "❌",
                            ValidationSeverity.WARNING: "⚠️",
                            ValidationSeverity.INFO: "ℹ️"
                        }.get(issue.severity, "❓")
                        
                        print(f"  {i}. {severity_icon} {issue.message}")
                        if issue.details:
                            for key, value in issue.details.items():
                                print(f"     {key}: {value}")
        
        # Recommendations
        if report.recommendations:
            print("\n💡 RECOMMENDATIONS:")
            print("-" * 40)
            for i, rec in enumerate(report.recommendations, 1):
                print(f"  {i}. {rec}")
        
        print("\n" + "="*80) 