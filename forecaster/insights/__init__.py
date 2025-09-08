"""
Insights module for demand classification and analysis.

This module contains functionality for analyzing demand patterns, classifying products,
and generating insights about inventory performance.
"""

from forecaster.insights.demand_classification import demand_classification, CV2_INTERVAL, INTERDEMAND_INTERVAL

__all__ = ["demand_classification", "CV2_INTERVAL", "INTERDEMAND_INTERVAL"]