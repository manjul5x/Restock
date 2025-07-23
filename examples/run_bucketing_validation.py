#!/usr/bin/env python3
"""
Bucketing Validation Script

This script runs comprehensive validation tests to ensure that:
1. First forecast bucket starts on the analysis date
2. All buckets (historic and forecast) follow on directly from each other
3. All buckets are exactly the risk period length (from product master)
4. No gaps or overlaps between buckets

This should be run as part of the forecasting pipeline to validate bucketing logic.
"""

import sys
from pathlib import Path
from datetime import date

# Add the forecaster package to the path
sys.path.append(str(Path(__file__).parent))

from forecaster.tests.test_bucketing_validation import run_bucketing_validation_tests


def main():
    """Run bucketing validation tests."""
    print("🧪 BUCKETING VALIDATION PIPELINE")
    print("=" * 60)
    print("This script validates the bucketing logic for forecasting models.")
    print("It ensures that:")
    print("  ✅ First forecast bucket starts on the analysis date")
    print("  ✅ All buckets follow on directly from each other")
    print("  ✅ All buckets are exactly the risk period length (from product master)")
    print("  ✅ No gaps or overlaps between buckets")
    print("=" * 60)
    
    # Run the validation tests
    success = run_bucketing_validation_tests()
    
    if success:
        print("\n🎉 All bucketing validation tests passed!")
        print("The forecasting pipeline bucketing logic is correct.")
        return True
    else:
        print("\n❌ Some bucketing validation tests failed!")
        print("Please review the errors above and fix the bucketing logic.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 