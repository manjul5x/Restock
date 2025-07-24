#!/usr/bin/env python3
"""
Enhanced Pipeline Runner

This script runs the complete enhanced pipeline:
1. Enhanced backtest
2. Enhanced safety stock calculation
3. Enhanced simulation
"""

import sys
import subprocess
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Running: {command}")

    try:
        result = subprocess.run(
            command, shell=True, check=True, capture_output=True, text=True
        )
        print("✅ Success!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {description}:")
        print(f"Exit code: {e.returncode}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False


def main():
    """Run the complete enhanced pipeline."""
    print("🔍 Enhanced Pipeline Runner")
    print("=" * 60)
    print("This will run the complete enhanced pipeline:")
    print("1. Enhanced backtest")
    print("2. Enhanced safety stock calculation")
    print("3. Enhanced simulation")
    print("=" * 60)

    # Check if we're in the right directory
    if not Path("run_enhanced_backtest.py").exists():
        print("❌ Error: Please run this script from the restock directory")
        sys.exit(1)

    # Step 1: Enhanced Backtest
    if not run_command(
        "python run_enhanced_backtest.py --start-date 2023-10-30 --end-date 2025-04-01 --output-dir output/enhanced_backtest",
        "Enhanced Backtest",
    ):
        print("❌ Enhanced backtest failed. Stopping pipeline.")
        sys.exit(1)

    # Step 2: Enhanced Safety Stock Calculation
    if not run_command(
        "python run_enhanced_safety_stock_calculation.py",
        "Enhanced Safety Stock Calculation",
    ):
        print("❌ Enhanced safety stock calculation failed. Stopping pipeline.")
        sys.exit(1)

    # Step 3: Enhanced Simulation
    if not run_command("python run_enhanced_simulation.py", "Enhanced Simulation"):
        print("❌ Enhanced simulation failed. Stopping pipeline.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("🎉 Enhanced Pipeline Completed Successfully!")
    print("=" * 60)
    print("\n📁 Output Files Created:")
    print("  • Enhanced backtest results: output/enhanced_backtest/")
    print("  • Enhanced safety stocks: output/enhanced_safety_stocks/")
    print("  • Enhanced simulation: output/enhanced_simulation/")
    print("\n🌐 Next Steps:")
    print("  1. Start the webapp: python webapp/app.py")
    print("  2. Visit http://localhost:5001")
    print("  3. Go to Safety Stocks tab to view enhanced results")
    print("  4. Compare with regular backtest results")


if __name__ == "__main__":
    main()
