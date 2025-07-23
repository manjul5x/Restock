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


def run_simulation(data_dir: str = "forecaster/data",
                  order_policy: str = "review_ordering",
                  output_dir: str = "output/simulation",
                  max_workers: int = None,
                  product_location_keys: list = None):
    """
    Run inventory simulation.
    
    Args:
        data_dir: Directory containing data files
        order_policy: Order policy to use
        output_dir: Directory to save results
        max_workers: Maximum number of parallel workers
        product_location_keys: Specific product-location keys to simulate (None for all)
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
    print(f"📁 Data Directory: {data_dir}")
    print(f"📊 Output Directory: {output_dir}")
    
    try:
        # Initialize simulator
        print("\n🔧 Initializing simulator...")
        simulator = InventorySimulator(data_dir=data_dir, default_policy=order_policy)
        
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
        output_files = simulator.save_results(output_dir)
        
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
        
        print(f"\n📁 Results saved to:")
        print(f"  Summary: {output_files['summary_file']}")
        print(f"  Detailed: {output_files['detailed_dir']}")
        
        print("\n✅ Simulation completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during simulation: {e}")
        logger.error(f"Simulation failed: {e}")
        sys.exit(1)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run inventory simulation")
    parser.add_argument("--data-dir", default="forecaster/data", 
                       help="Directory containing data files")
    parser.add_argument("--order-policy", default="review_ordering",
                       choices=OrderPolicyFactory.list_policies(),
                       help="Order policy to use")
    parser.add_argument("--output-dir", default="output/simulation",
                       help="Directory to save results")
    parser.add_argument("--max-workers", type=int, default=None,
                       help="Maximum number of parallel workers")
    parser.add_argument("--product-location-keys", nargs="+",
                       help="Specific product-location keys to simulate")
    
    args = parser.parse_args()
    
    # Validate data directory
    if not Path(args.data_dir).exists():
        print(f"❌ Error: Data directory not found: {args.data_dir}")
        sys.exit(1)
    
    # Run simulation
    run_simulation(
        data_dir=args.data_dir,
        order_policy=args.order_policy,
        output_dir=args.output_dir,
        max_workers=args.max_workers,
        product_location_keys=args.product_location_keys
    )


if __name__ == "__main__":
    main() 