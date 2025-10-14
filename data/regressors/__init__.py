"""
Regressor functions for the new backtesting pipeline.

This package provides modular, configurable regressor computation functions
that can be easily extended by users without modifying core pipeline code.
"""

from .lead_lag_aggregation import compute_lead_lag_aggregation
from .in_season import compute_in_season
from .week_of_month import compute_week_of_month
from .recency import compute_recency_weight

AVAILABLE_REGRESSORS = [
    'compute_lead_lag_aggregation',
    'compute_in_season',
    'compute_week_of_month',
    'compute_recency_weight',
]

# Keep __all__ for backward compatibility if needed
__all__ = AVAILABLE_REGRESSORS
