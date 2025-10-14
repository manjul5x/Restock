"""
Order policies for inventory simulation.

Defines different strategies for calculating order quantities.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


class OrderPolicy(ABC):
    """
    Abstract base class for order policies.
    
    All order policies must implement the calculate_order method.
    """
    
    @abstractmethod
    def calculate_order(self, step: int, arrays: Dict[str, np.ndarray], 
                       period_info: Dict[str, Any]) -> float:
        """
        Calculate the order quantity for a given step.
        
        Args:
            step: Current simulation step
            arrays: Dictionary containing all simulation arrays
            period_info: Period information for the product-location
            
        Returns:
            Order quantity to place
        """
        pass


class MOQOrderPolicy(OrderPolicy):
    """
    Wrapper order policy that applies Minimum Order Quantity (MOQ) constraints.
    
    Takes the maximum of the recommended order quantity and the MOQ for the product.
    """
    
    def __init__(self, base_policy: OrderPolicy):
        """
        Initialize MOQ order policy.
        
        Args:
            base_policy: The base order policy to wrap
        """
        self.base_policy = base_policy
    
    def calculate_order(self, step: int, arrays: Dict[str, np.ndarray], 
                       period_info: Dict[str, Any]) -> float:
        """
        Calculate order quantity using base policy and apply MOQ constraint.
        
        Args:
            step: Current simulation step
            arrays: Dictionary containing all simulation arrays
            period_info: Period information for the product-location (contains MOQ)
            
        Returns:
            Order quantity to place (max of recommended order and MOQ)
        """
        # Get base order quantity
        base_order = self.base_policy.calculate_order(step, arrays, period_info)
        
        # If no order is recommended, return 0
        if base_order <= 0:
            return 0.0
        
        # Get MOQ from period_info
        moq = period_info.get('moq', 1.0)  # Default MOQ if not found
        
        # Return the maximum of base order and MOQ
        return max(base_order, moq)


class ReviewOrderingPolicy(OrderPolicy):
    """
    Review-based ordering policy.
    
    Calculates order as: max(0, safety_stock + FRSP - net_stock)
    Only places orders on decision days (review dates).
    """
    
    def calculate_order(self, step: int, arrays: Dict[str, np.ndarray], 
                       period_info: Dict[str, Any]) -> float:
        """
        Calculate order quantity using review-based policy.
        
        Args:
            step: Current simulation step
            arrays: Dictionary containing all simulation arrays
            period_info: Period information for the product-location
            
        Returns:
            Order quantity to place
        """
        # Only place orders on decision days
        if arrays['decision_day'][step] == 0:
            return 0.0
        
        # Calculate order quantity
        safety_stock = arrays['safety_stock'][step]
        frsp = arrays['FRSP'][step]
        net_stock = arrays['net_stock'][step]
        
        order_quantity = max(0.0, safety_stock + frsp - net_stock)
        
        return order_quantity


#### DO NOT USE BELOW POLICIES I HAVEN'T CHECKED THEM

class ContinuousReviewPolicy(OrderPolicy):
    """
    Continuous review ordering policy.
    
    Places orders whenever net stock falls below safety stock + FRSP.
    """
    
    def __init__(self, order_up_to_level: bool = True):
        """
        Initialize continuous review policy.
        
        Args:
            order_up_to_level: If True, order up to safety_stock + FRSP
                              If False, order exactly the deficit
        """
        self.order_up_to_level = order_up_to_level
    
    def calculate_order(self, step: int, arrays: Dict[str, np.ndarray], 
                       period_info: Dict[str, Any]) -> float:
        """
        Calculate order quantity using continuous review policy.
        
        Args:
            step: Current simulation step
            arrays: Dictionary containing all simulation arrays
            period_info: Period information for the product-location
            
        Returns:
            Order quantity to place
        """
        safety_stock = arrays['safety_stock'][step]
        frsp = arrays['FRSP'][step]
        net_stock = arrays['net_stock'][step]
        
        target_level = safety_stock + frsp
        
        if net_stock < target_level:
            if self.order_up_to_level:
                return target_level - net_stock
            else:
                return target_level - net_stock
        else:
            return 0.0


class MinMaxPolicy(OrderPolicy):
    """
    Min-Max ordering policy.
    
    Places orders when inventory falls below minimum level,
    ordering up to maximum level.
    """
    
    def calculate_order(self, step: int, arrays: Dict[str, np.ndarray], 
                       period_info: Dict[str, Any]) -> float:
        """
        Calculate order quantity using min-max policy.
        
        Args:
            step: Current simulation step
            arrays: Dictionary containing all simulation arrays
            period_info: Period information for the product-location
            
        Returns:
            Order quantity to place
        """
        min_level = arrays['min_level'][step]
        max_level = arrays['max_level'][step]
        net_stock = arrays['net_stock'][step]
        
        if net_stock <= min_level:
            return max_level - net_stock
        else:
            return 0.0


class OrderPolicyFactory:
    """
    Factory for creating order policies.
    """
    
    _policies = {
        'review_ordering': ReviewOrderingPolicy,
        'continuous_review': ContinuousReviewPolicy,
        'min_max': MinMaxPolicy
    }
    
    @classmethod
    def create_policy_with_moq(cls, policy_name: str, enable_moq: bool = False, **kwargs) -> OrderPolicy:
        """
        Create an order policy by name, optionally wrapped with MOQ constraints.
        
        Args:
            policy_name: Name of the policy to create
            enable_moq: If True, wrap the policy with MOQ constraints
            **kwargs: Additional arguments for the policy constructor
            
        Returns:
            OrderPolicy instance (wrapped with MOQ if enabled)
            
        Raises:
            ValueError: If policy name is not recognized
        """
        base_policy = cls.create_policy(policy_name, **kwargs)
        
        if enable_moq:
            return MOQOrderPolicy(base_policy)
        else:
            return base_policy
    
    @classmethod
    def create_policy(cls, policy_name: str, **kwargs) -> OrderPolicy:
        """
        Create an order policy by name.
        
        Args:
            policy_name: Name of the policy to create
            **kwargs: Additional arguments for the policy constructor
            
        Returns:
            OrderPolicy instance
            
        Raises:
            ValueError: If policy name is not recognized
        """
        if policy_name not in cls._policies:
            available_policies = list(cls._policies.keys())
            raise ValueError(f"Unknown policy '{policy_name}'. Available policies: {available_policies}")
        
        policy_class = cls._policies[policy_name]
        return policy_class(**kwargs)
    
    @classmethod
    def register_policy(cls, name: str, policy_class: type):
        """
        Register a new order policy.
        
        Args:
            name: Name for the policy
            policy_class: Policy class (must inherit from OrderPolicy)
        """
        if not issubclass(policy_class, OrderPolicy):
            raise ValueError("Policy class must inherit from OrderPolicy")
        
        cls._policies[name] = policy_class
    
    @classmethod
    def list_policies(cls) -> list:
        """
        Get list of available policy names.
        
        Returns:
            List of available policy names
        """
        return list(cls._policies.keys()) 