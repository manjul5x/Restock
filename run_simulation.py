#!/usr/bin/env python3
"""
Inventory Simulation Runner

Executes inventory simulations to test the impact of forecasting and safety stock recommendations.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add forecaster package to path
sys.path.append(str(Path(__file__).parent))

from forecaster.simulation import InventorySimulator, OrderPolicyFactory
from forecaster.utils.logger import get_logger

logger = get_logger(__name__)


def run_simulation(order_policy: str = "review_ordering",
                  max_workers: int = 8,
                  product_location_keys: list = None,
                  safety_stock_file: str = None,
                  forecast_comparison_file: str = None):
    """
    Run inventory simulation.
    
    Args:
        order_policy: Order policy to use
        max_workers: Maximum number of parallel workers
        product_location_keys: Specific product-location keys to simulate (None for all)
        safety_stock_file: Path to safety stock results file (optional)
        forecast_comparison_file: Path to forecast comparison file (optional)
    """
    
    print("🔍 Inventory Simulation")
    print("=" * 50)
    
    # Validate order policy
    available_policies = OrderPolicyFactory.list_policies()
    if order_policy not in available_policies:
        print(f"❌ Error: Unknown order policy '{order_policy}'")
        print(f"Available policies: {available_policies}")
        sys.exit(1)
    
    print(f"📋 Order Policy: {order_policy}")
    
    try:
        # Initialize simulator
        print("\n🔧 Initializing simulator...")
        simulator = InventorySimulator(
            default_policy=order_policy,
            safety_stock_file=safety_stock_file,
            forecast_comparison_file=forecast_comparison_file
        )
        
        # Run simulation
        print("🚀 Running simulation...")
        results = simulator.run_batch_simulation(
            product_location_keys=product_location_keys,
            max_workers=max_workers
        )
        
        if not results:
            print("❌ No simulation results generated")
            sys.exit(1)
        
        # Save results
        print("\n💾 Saving results...")
        simulator.save_results()
        
        # Display summary
        print("\n📊 Simulation Summary:")
        print(f"  Product-location combinations simulated: {len(results)}")
        
        # Calculate aggregate metrics
        all_metrics = [result['metrics'] for result in results.values()]
        
        if all_metrics:
            avg_service_level = sum(m['service_level'] for m in all_metrics) / len(all_metrics)
            avg_stockout_rate = sum(m['stockout_rate'] for m in all_metrics) / len(all_metrics)
            avg_inventory_turns = sum(m['inventory_turns'] for m in all_metrics) / len(all_metrics)
            
            print(f"  Average service level: {avg_service_level:.2%}")
            print(f"  Average stockout rate: {avg_stockout_rate:.2%}")
            print(f"  Average inventory turns: {avg_inventory_turns:.2f}")
        
        print("\n✅ Simulation completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during simulation: {e}")
        logger.error(f"Simulation failed: {e}")
        sys.exit(1)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run inventory simulation")
    # Note: Data paths are now handled by DataLoader configuration
    parser.add_argument("--order-policy", default="review_ordering",
                       choices=OrderPolicyFactory.list_policies(),
                       help="Order policy to use")
    parser.add_argument("--max-workers", type=int, default=8,
                       help="Maximum number of parallel workers")
    parser.add_argument("--product-location-keys", nargs="+",
                       help="Specific product-location keys to simulate")
    parser.add_argument("--safety-stock-file",
                       help="Path to safety stock results file")
    parser.add_argument("--forecast-comparison-file",
                       help="Path to forecast comparison file")
    
    args = parser.parse_args()
    

    # Run simulation
    run_simulation(
        order_policy=args.order_policy,
        max_workers=args.max_workers,
        product_location_keys=args.product_location_keys,
        safety_stock_file=args.safety_stock_file,
        forecast_comparison_file=args.forecast_comparison_file
    )


if __name__ == "__main__":
    main() 