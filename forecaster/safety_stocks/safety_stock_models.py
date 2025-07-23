"""
Safety Stock Models

This module contains different models for calculating safety stocks based on error distributions.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import List


class SafetyStockModels:
    """
    Contains different models for calculating safety stocks.
    """
    
    def __init__(self):
        """Initialize the safety stock models."""
        pass
    
    def calculate_safety_stock(
        self,
        errors: List[float],
        distribution_type: str = 'kde',
        service_level: float = 0.95
    ) -> float:
        """
        Calculate safety stock based on error distribution.
        
        Args:
            errors: List of forecast errors
            distribution_type: Type of distribution ('kde', 'normal', etc.)
            service_level: Service level percentage (0.0 to 1.0)
            
        Returns:
            Safety stock value
        """
        if distribution_type == 'kde':
            return self._calculate_kde_safety_stock(errors, service_level)
        elif distribution_type == 'normal':
            return self._calculate_normal_safety_stock(errors, service_level)
        else:
            raise ValueError(f"Unsupported distribution type: {distribution_type}")
    
    def _calculate_kde_safety_stock(
        self,
        errors: List[float],
        service_level: float
    ) -> float:
        """
        Calculate safety stock using Kernel Density Estimation.
        
        Args:
            errors: List of forecast errors
            service_level: Service level percentage
            
        Returns:
            Safety stock value
        """
        if len(errors) < 2:
            return 0.0
        
        errors_array = np.array(errors)
        
        # Fit KDE using scipy.stats.gaussian_kde
        kde = stats.gaussian_kde(errors_array)
        
        # Create a range of values to evaluate
        min_error = min(errors)
        max_error = max(errors)
        
        x_range = np.linspace(
            min_error,
            max_error,
            1000
        )
        
        # Calculate density
        density = kde(x_range)
        
        # Calculate cumulative distribution function (CDF)
        cdf = np.cumsum(density) * (x_range[1] - x_range[0])
        cdf = cdf / cdf[-1]  # Normalize to 1
        
        # Use interpolation to find the exact value at the service level
        safety_stock = np.interp(service_level, cdf, x_range)
        
        # Safety stock should be positive (to cover shortages)
        # If the error distribution suggests negative safety stock, return 0
        return max(0.0, safety_stock)
    
    def _calculate_normal_safety_stock(
        self,
        errors: List[float],
        service_level: float
    ) -> float:
        """
        Calculate safety stock assuming normal distribution.
        
        Args:
            errors: List of forecast errors
            service_level: Service level percentage
            
        Returns:
            Safety stock value
        """
        if len(errors) < 2:
            return 0.0
        
        errors_array = np.array(errors)
        mean_error = np.mean(errors_array)
        std_error = np.std(errors_array, ddof=1)
        
        if std_error == 0:
            return max(0.0, mean_error)
        
        # Calculate z-score for service level
        z_score = stats.norm.ppf(service_level)
        
        # Calculate safety stock
        safety_stock = mean_error + z_score * std_error
        
        # Safety stock should be positive
        return max(0.0, safety_stock)
    
    def get_distribution_plot_data(
        self,
        errors: List[float],
        distribution_type: str = 'kde'
    ) -> dict:
        """
        Get data for plotting the error distribution.
        
        Args:
            errors: List of forecast errors
            distribution_type: Type of distribution
            
        Returns:
            Dictionary with plot data
        """
        if distribution_type == 'kde':
            return self._get_kde_plot_data(errors)
        elif distribution_type == 'normal':
            return self._get_normal_plot_data(errors)
        else:
            raise ValueError(f"Unsupported distribution type: {distribution_type}")
    
    def _get_kde_plot_data(self, errors: List[float]) -> dict:
        """Get KDE plot data."""
        if len(errors) < 2:
            return {'x': [], 'y': [], 'histogram': {'x': [], 'y': []}}
        
        errors_array = np.array(errors)
        
        # Fit KDE using scipy.stats.gaussian_kde
        kde = stats.gaussian_kde(errors_array)
        
        # Create range for plotting
        min_error = min(errors)
        max_error = max(errors)
        
        x_range = np.linspace(
            min_error,
            max_error,
            200
        )
        
        # Calculate density
        density = kde(x_range)
        
        # Create histogram data with actual counts (not density)
        # Use more bins for better resolution
        n_bins = int(np.ceil(1 + 3.322 * np.log10(len(errors))))
        n_bins = max(10, min(30, n_bins))  # Between 10 and 30 bins for more detail
        
        hist, bin_edges = np.histogram(errors, bins=n_bins, density=False)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        return {
            'x': x_range.tolist(),
            'y': density.tolist(),
            'histogram': {
                'x': bin_centers.tolist(),
                'y': hist.tolist(),
                'bin_edges': bin_edges.tolist()  # Add bin edges for gap-free plotting
            }
        }
    
    def _get_normal_plot_data(self, errors: List[float]) -> dict:
        """Get normal distribution plot data."""
        if len(errors) < 2:
            return {'x': [], 'y': [], 'histogram': {'x': [], 'y': []}}
        
        errors_array = np.array(errors)
        mean_error = np.mean(errors_array)
        std_error = np.std(errors_array, ddof=1)
        
        # Create range for plotting
        min_error = min(errors)
        max_error = max(errors)
        
        x_range = np.linspace(
            min_error,
            max_error,
            200
        )
        
        # Calculate normal density
        density = stats.norm.pdf(x_range, mean_error, std_error)
        
        # Create histogram data with actual counts (not density)
        # Use more bins for better resolution
        n_bins = int(np.ceil(1 + 3.322 * np.log10(len(errors))))
        n_bins = max(10, min(30, n_bins))  # Between 10 and 30 bins for more detail
        
        hist, bin_edges = np.histogram(errors, bins=n_bins, density=False)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        return {
            'x': x_range.tolist(),
            'y': density.tolist(),
            'histogram': {
                'x': bin_centers.tolist(),
                'y': hist.tolist(),
                'bin_edges': bin_edges.tolist()  # Add bin edges for gap-free plotting
            }
        } 