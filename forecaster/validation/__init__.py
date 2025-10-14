"""
Data validation module for the forecaster package.

This module provides comprehensive data validation capabilities including:
- Schema validation
- Data completeness validation
- Data quality validation
- Coverage validation
- Frequency validation
- Statistical validation
"""

from .types import ValidationResult, ValidationIssue, ValidationReport, ValidationSeverity
from .validator import DataValidator
from .schema_validator import SchemaValidator
from .completeness_validator import CompletenessValidator
from .quality_validator import QualityValidator
from .coverage_validator import CoverageValidator

# Make schema imports optional due to pydantic dependency
try:
    from .schema import DemandSchema, DemandRecord
    from .product_master_schema import ProductMasterSchema, ProductMasterRecord
    from .demand_validator import DemandValidator, validate_demand_completeness, generate_completeness_report
    _SCHEMA_AVAILABLE = True
except ImportError:
    # Schema modules require pydantic which may not be installed
    DemandSchema = None
    DemandRecord = None
    ProductMasterSchema = None
    ProductMasterRecord = None
    DemandValidator = None
    validate_demand_completeness = None
    generate_completeness_report = None
    _SCHEMA_AVAILABLE = False

__all__ = [
    'DataValidator',
    'ValidationResult', 
    'ValidationIssue',
    'ValidationReport',
    'ValidationSeverity',
    'SchemaValidator',
    'CompletenessValidator',
    'QualityValidator',
    'CoverageValidator'
]

# Add schema-related items only if available
if _SCHEMA_AVAILABLE:
    __all__.extend([
        'DemandSchema',
        'DemandRecord',
        'ProductMasterSchema',
        'ProductMasterRecord',
        'DemandValidator',
        'validate_demand_completeness',
        'generate_completeness_report'
    ]) 