"""
Validation types and data structures.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, date
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from enum import Enum


class ValidationSeverity(Enum):
    """Validation severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Represents a single validation issue"""
    severity: ValidationSeverity
    category: str
    message: str
    details: Dict[str, Any]
    affected_records: Optional[int] = None
    affected_products: Optional[List[str]] = None
    affected_locations: Optional[List[str]] = None


@dataclass
class ValidationResult:
    """Result of a single validation check"""
    is_valid: bool
    issues: List[ValidationIssue]
    summary: Dict[str, Any]
    execution_time: float


@dataclass
class ValidationReport:
    """Comprehensive validation report"""
    overall_valid: bool
    demand_validation: ValidationResult
    product_master_validation: ValidationResult
    coverage_validation: ValidationResult
    quality_validation: ValidationResult
    completeness_validation: ValidationResult
    total_issues: int
    critical_issues: int
    warnings: int
    execution_time: float
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    def to_json(self, file_path: Optional[Union[str, Path]] = None) -> str:
        """Convert to JSON string or save to file"""
        # Convert enums to strings for JSON serialization
        def convert_enums(obj):
            if isinstance(obj, ValidationSeverity):
                return obj.value
            elif isinstance(obj, dict):
                return {k: convert_enums(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_enums(item) for item in obj]
            return obj
        
        json_data = convert_enums(self.to_dict())
        json_str = json.dumps(json_data, indent=2, default=str)
        
        if file_path:
            with open(file_path, 'w') as f:
                f.write(json_str)
        
        return json_str 