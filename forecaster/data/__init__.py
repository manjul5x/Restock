"""
Data module for the forecaster package.

This module handles data loading, validation, and schema management.
"""

from .loader import DemandDataLoader
from .schema import DemandSchema, DemandRecord
from .product_master_schema import ProductMasterSchema, ProductMasterRecord
from .demand_validator import DemandValidator, validate_demand_completeness, generate_completeness_report
from .aggregator import DemandAggregator, create_risk_period_buckets, get_aggregation_summary

__all__ = [
    'DemandDataLoader', 
    'DemandSchema', 
    'DemandRecord',
    'ProductMasterSchema',
    'ProductMasterRecord',
    'DemandValidator',
    'validate_demand_completeness',
    'generate_completeness_report',
    'DemandAggregator',
    'create_risk_period_buckets',
    'get_aggregation_summary'
]
