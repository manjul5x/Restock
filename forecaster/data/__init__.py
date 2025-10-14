"""
Data module for the forecaster package.

This module handles data loading, validation, and schema management.
"""

from .aggregator import DemandAggregator, create_risk_period_buckets, get_aggregation_summary

__all__ = [
    'DemandAggregator',
    'create_risk_period_buckets',
    'get_aggregation_summary'
]
