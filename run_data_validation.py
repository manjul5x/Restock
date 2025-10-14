#!/usr/bin/env python3
"""
Data Validation Runner

This script performs comprehensive data validation on demand and product master data
without running the full forecasting pipeline. It provides detailed reports on data
quality, completeness, coverage, and schema issues.

Usage:
    python run_data_validation.py [options]

Examples:
    # Run with default settings
    python run_data_validation.py
    
    # Run with custom data files
    python run_data_validation.py --demand-file my_demand.csv --product-master-file my_product_master.csv
    
    # Run with specific validation options
    python run_data_validation.py --no-quality --strict-mode
    
    # Save validation report to file
    python run_data_validation.py --output-report validation_report.json
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add the forecaster package to the path
sys.path.insert(0, str(Path(__file__).parent))

from data.loader import DataLoader
from forecaster.validation import DataValidator


def run_data_validation(
    demand_frequency: str = "d",
    validate_schema: bool = True,
    validate_completeness: bool = True,
    validate_quality: bool = True,
    validate_coverage: bool = True,
    strict_mode: bool = False,
    output_report: str = None,
    log_level: str = "INFO"
) -> bool:
    """
    Run comprehensive data validation.
    
    Args:
        data_dir: Directory containing data files
        demand_file: Demand data file name
        product_master_file: Product master file name
        demand_frequency: Expected demand frequency ('d', 'w', 'm')
        validate_schema: Whether to validate schema
        validate_completeness: Whether to validate completeness
        validate_quality: Whether to validate data quality
        validate_coverage: Whether to validate coverage
        strict_mode: If True, treats warnings as errors
        output_report: Path to save validation report JSON
        log_level: Logging level
        
    Returns:
        True if validation passed, False otherwise
    """
    print("🔍 Data Validation Runner")
    print("=" * 60)
    print("This script performs comprehensive data validation including:")
    print("• Schema validation (column structure and data types)")
    print("• Data completeness validation (missing dates, frequency consistency)")
    print("• Data quality validation (negative values, outliers, anomalies)")
    print("• Data coverage validation (product-location combinations)")
    print("=" * 60)
    
    # Configuration
    print(f"🔄 Demand Frequency: {demand_frequency}")
    print(f"⚙️  Strict Mode: {strict_mode}")
    print(f"📝 Output Report: {output_report or 'None'}")
    print("=" * 60)
    
    try:
        # Initialize DataLoader
        # Note: We are assuming the config YAML is correctly set up.
        # The script's --data-dir, --demand-file, etc., arguments are now for documentation
        # as the DataLoader gets its paths from data/config/data_config.yaml.
        print("📂 Initializing DataLoader...")
        loader = DataLoader()
        
        # Load data using the new loader
        print("📂 Loading data files via DataLoader...")
        product_master_data = loader.load_product_master()
        demand_data = loader.load_outflow(product_master=product_master_data)
        
        if demand_data is None or product_master_data is None:
            print("❌ Data loading failed. Please check the DataLoader configuration and file paths.")
            return False

        print(f"✅ Loaded {len(demand_data)} demand records (filtered by product master)")
        print(f"✅ Loaded {len(product_master_data)} product master records")
        
        # Initialize validator
        validator = DataValidator(log_level=log_level)
        
        # Run validation
        print("\n🔍 Running comprehensive validation...")
        report = validator.validate_data(
            demand_data=demand_data,
            product_master_data=product_master_data,
            demand_frequency=demand_frequency,
            validate_schema=validate_schema,
            validate_completeness=validate_completeness,
            validate_quality=validate_quality,
            validate_coverage=validate_coverage,
            strict_mode=strict_mode
        )
        
        # Print report
        validator.print_report(report, detailed=True)
        
        # Save report if requested
        if output_report:
            output_path = Path(output_report)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            report.to_json(output_path)
            print(f"\n📄 Validation report saved to: {output_path}")
        
        # Return validation result
        return report.overall_valid
        
    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        import traceback
        print(traceback.format_exc())
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive data validation (standalone)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python run_data_validation.py
  
  # Run with custom data directory and files
  python run_data_validation.py --data-dir forecaster/data --demand-file my_demand.csv --product-master-file my_product_master.csv
  
  # Run with specific demand frequency
  python run_data_validation.py --demand-frequency w
  
  # Run with specific validation options
  python run_data_validation.py --no-quality --no-completeness
  
  # Run in strict mode (warnings treated as errors)
  python run_data_validation.py --strict-mode
  
  # Save validation report to file
  python run_data_validation.py --output-report validation_report.json
  
  # Run with debug logging
  python run_data_validation.py --log-level DEBUG
        """
    )
    
    # Note: Data paths are now handled by DataLoader configuration
    parser.add_argument(
        "--demand-frequency",
        choices=["d", "w", "m"],
        default="d",
        help="Expected demand frequency: d=daily, w=weekly, m=monthly (default: d)"
    )
    
    # Validation options
    parser.add_argument(
        "--no-schema",
        action="store_true",
        help="Disable schema validation"
    )
    parser.add_argument(
        "--no-completeness",
        action="store_true",
        help="Disable completeness validation"
    )
    parser.add_argument(
        "--no-quality",
        action="store_true",
        help="Disable data quality validation"
    )
    parser.add_argument(
        "--no-coverage",
        action="store_true",
        help="Disable coverage validation"
    )
    parser.add_argument(
        "--strict-mode",
        action="store_true",
        help="Treat warnings as errors"
    )
    
    # Output options
    parser.add_argument(
        "--output-report",
        help="Path to save validation report JSON file"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Run validation
    success = run_data_validation(
        demand_frequency=args.demand_frequency,
        validate_schema=not args.no_schema,
        validate_completeness=not args.no_completeness,
        validate_quality=not args.no_quality,
        validate_coverage=not args.no_coverage,
        strict_mode=args.strict_mode,
        output_report=args.output_report,
        log_level=args.log_level
    )
    
    # Exit with appropriate code
    if success:
        print("\n✅ Data validation completed successfully!")
        print("Your data is ready for forecasting.")
        sys.exit(0)
    else:
        print("\n❌ Data validation failed!")
        print("Please address the issues above before proceeding with forecasting.")
        sys.exit(1)


if __name__ == "__main__":
    main() 