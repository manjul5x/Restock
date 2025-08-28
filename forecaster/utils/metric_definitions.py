"""
Standardized inventory metric definitions.
Provides consistent definitions and targets for inventory metrics across the system.
"""

from typing import Dict, Any

class InventoryMetrics:
    """Standardized inventory metric definitions"""
    
    METRICS: Dict[str, Dict[str, Any]] = {
        'service_level': {
            'description': 'Percentage of demand met from inventory',
            'calculation': 'demand_met / total_demand_periods * 100',
            'target': '>95%',
            'better_when': 'higher',
            'unit': '%',
            'typical_range': (80, 100)
        },
        'stockout_rate': {
            'description': 'Percentage of periods with zero inventory',
            'calculation': 'stockout_days / total_days * 100',
            'target': '<5%',
            'better_when': 'lower',
            'unit': '%',
            'typical_range': (0, 20)
        },
        'inventory_days': {
            'description': 'Number of days of inventory coverage based on average demand',
            'calculation': 'average_inventory / average_daily_demand',
            'target': 'varies by product',
            'better_when': 'optimal',
            'unit': 'days',
            'typical_range': (30, 120)
        },
        'overstock_percentage': {
            'description': 'Percentage of days where inventory exceeds maximum level',
            'calculation': 'overstock_days / total_days * 100',
            'target': '<10%',
            'better_when': 'lower',
            'unit': '%',
            'typical_range': (0, 30)
        },
        'understock_percentage': {
            'description': 'Percentage of days where inventory is below minimum level',
            'calculation': 'understock_days / total_days * 100',
            'target': '<15%',
            'better_when': 'lower',
            'unit': '%',
            'typical_range': (0, 30)
        },
        'total_inventory_cost': {
            'description': 'Total cost of on-hand inventory',
            'calculation': 'average_inventory * unit_cost',
            'target': 'minimize while maintaining service level',
            'better_when': 'lower',
            'unit': 'currency',
            'typical_range': None  # Varies too much by product to specify
        },
        'inventory_difference_percentage': {
            'description': 'Percentage difference in inventory cost (actual vs simulated)',
            'calculation': '(actual_total_inventory_cost - simulated_total_inventory_cost) / actual_total_inventory_cost * 100',
            'target': '±10%',
            'better_when': 'closer to zero',
            'unit': '%',
            'typical_range': (-20, 20)
        },
        'service_level_difference': {
            'description': 'Difference between simulated and actual service levels',
            'calculation': 'simulated_service_level - actual_service_level',
            'target': '≥0',
            'better_when': 'higher',
            'unit': 'percentage points',
            'typical_range': (-10, 10)
        },
        'inventory_turnover_ratio': {
            'description': 'Number of times inventory is sold and replaced over a period',
            'calculation': 'Cost of Goods Sold (actual demand * unit cost) / Cost of average inventory',
            'target': 'varies by industry',
            'better_when': 'higher',
            'unit': 'ratio',
            'typical_range': (4, 12)  # This can be adjusted based on industry standards
        },
        'surplus_stock_percentage': {
            'description': 'Average percentage of inventory that exceeds the rolling maximum level',
            'calculation': '((actual_inventory - rolling_max_inventory) / actual_inventory).mean()',
            'target': '<10%',
            'better_when': 'lower',
            'unit': '%',
            'typical_range': (0, 30)
        },
        'availability_percentage': {
            'description': 'Percentage of days where inventory is above safety stock level',
            'calculation': '(1 - days_with_understock_flag / total_days) * 100',
            'target': '>95%',
            'better_when': 'higher',
            'unit': '%',
            'typical_range': (80, 100)
        }
    }
    
    @classmethod
    def get_metric_info(cls, metric_name: str) -> Dict[str, Any]:
        """
        Get information about a specific metric
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            Dictionary containing metric information
            
        Raises:
            KeyError: If metric_name is not found
        """
        if metric_name not in cls.METRICS:
            raise KeyError(f"Unknown metric: {metric_name}")
        return cls.METRICS[metric_name]
    
    @classmethod
    def get_all_metrics(cls) -> Dict[str, Dict[str, Any]]:
        """Get information about all metrics"""
        return cls.METRICS
    
    @classmethod
    def get_metric_target(cls, metric_name: str) -> str:
        """Get target value for a specific metric"""
        return cls.get_metric_info(metric_name)['target']
    
    @classmethod
    def get_metric_unit(cls, metric_name: str) -> str:
        """Get unit for a specific metric"""
        return cls.get_metric_info(metric_name)['unit']
    
    @classmethod
    def is_better_when_higher(cls, metric_name: str) -> bool:
        """Check if a higher value is better for this metric"""
        return cls.get_metric_info(metric_name)['better_when'] == 'higher'
    
    @classmethod
    def get_typical_range(cls, metric_name: str) -> tuple:
        """Get typical range for a specific metric"""
        return cls.get_metric_info(metric_name)['typical_range']
    
    @classmethod
    def evaluate_metric(cls, metric_name: str, value: float) -> str:
        """
        Evaluate a metric value against its target
        
        Args:
            metric_name: Name of the metric
            value: Metric value to evaluate
            
        Returns:
            String indicating performance ('good', 'warning', 'poor')
        """
        metric_info = cls.get_metric_info(metric_name)
        typical_range = metric_info['typical_range']
        
        if typical_range is None:
            return 'unknown'
            
        min_val, max_val = typical_range
        better_when = metric_info['better_when']
        
        if better_when == 'higher':
            if value >= max_val * 0.9:  # Within 90% of max
                return 'good'
            elif value >= min_val:
                return 'warning'
            else:
                return 'poor'
        elif better_when == 'lower':
            if value <= min_val * 1.1:  # Within 110% of min
                return 'good'
            elif value <= max_val:
                return 'warning'
            else:
                return 'poor'
        elif better_when == 'closer to zero':
            abs_val = abs(value)
            if abs_val <= (max_val - min_val) * 0.2:  # Within 20% of range
                return 'good'
            elif abs_val <= max_val:
                return 'warning'
            else:
                return 'poor'
        else:  # optimal
            mid_point = (min_val + max_val) / 2
            range_width = max_val - min_val
            if abs(value - mid_point) <= range_width * 0.2:  # Within 20% of midpoint
                return 'good'
            elif min_val <= value <= max_val:
                return 'warning'
            else:
                return 'poor'
