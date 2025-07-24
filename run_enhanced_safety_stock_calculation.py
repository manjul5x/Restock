#!/usr/bin/env python3
"""
Enhanced Safety Stock Calculation Script

This script runs safety stock calculation on enhanced backtest results instead of regular customer backtest.
It automatically uses the forecast_comparison.csv from the enhanced backtest output.
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from run_safety_stock_calculation import run_safety_stock_calculation


def main():
    """Run safety stock calculation on enhanced backtest results."""

    # Default paths for enhanced backtest (relative to restock directory)
    enhanced_backtest_dir = "output/enhanced_backtest"
    forecast_comparison_file = f"{enhanced_backtest_dir}/forecast_comparison.csv"
    product_master_file = "forecaster/data/customer_product_master.csv"
    output_dir = "output/enhanced_safety_stocks"

    # Check if enhanced backtest results exist
    if not Path(forecast_comparison_file).exists():
        print(f"❌ Enhanced backtest results not found: {forecast_comparison_file}")
        print("Please run enhanced backtest first:")
        print("  python run_enhanced_backtest.py")
        sys.exit(1)

    # Check if product master file exists
    if not Path(product_master_file).exists():
        print(f"❌ Product master file not found: {product_master_file}")
        sys.exit(1)

    print("🔍 Enhanced Safety Stock Calculation")
    print("=" * 50)
    print(f"Input: Enhanced backtest results from {enhanced_backtest_dir}")
    print(f"Output: {output_dir}")
    print()

    # Run safety stock calculation
    try:
        run_safety_stock_calculation(
            forecast_comparison_file=forecast_comparison_file,
            product_master_file=product_master_file,
            output_dir=output_dir,
        )

        print()
        print("✅ Enhanced safety stock calculation completed successfully!")
        print(f"📁 Results saved to: {output_dir}/safety_stock_results.csv")
        print()
        print("🎯 Next Steps:")
        print("  1. View results in the web dashboard")
        print("  2. Run simulation with enhanced safety stocks")
        print("  3. Compare with regular backtest safety stocks")

    except Exception as e:
        print(f"❌ Error during enhanced safety stock calculation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
