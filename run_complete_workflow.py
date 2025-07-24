#!/usr/bin/env python3
"""
Complete Workflow Runner

This script runs the complete inventory analysis workflow:
1. Data validation
2. Backtesting (for historical analysis)
3. Safety stock calculation
4. Inventory simulation
5. Web interface startup (optional)

This replaces the need to run multiple scripts separately.
"""

import sys
import argparse
import subprocess
from pathlib import Path
from datetime import date, timedelta
import time

# Add forecaster package to path
sys.path.append(str(Path(__file__).parent))

from forecaster.utils.logger import get_logger

logger = get_logger(__name__)


def run_command(command, description, check=True):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Running: {command}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            command, shell=True, check=check, capture_output=True, text=True
        )
        end_time = time.time()
        
        if result.returncode == 0:
            print(f"✅ {description} completed successfully!")
            print(f"⏱️  Execution time: {end_time - start_time:.2f} seconds")
            if result.stdout:
                print("Output:")
                print(result.stdout)
            return True
        else:
            print(f"⚠️  {description} completed with warnings")
            if result.stdout:
                print("STDOUT:", result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            return True if not check else False
            
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        print(f"❌ {description} failed:")
        print(f"Exit code: {e.returncode}")
        print(f"Execution time: {end_time - start_time:.2f} seconds")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False


def generate_review_dates(analysis_start_date: str, analysis_end_date: str, review_interval_days: int = 30) -> str:
    """
    Generate review dates for safety stock calculation.
    
    Args:
        analysis_start_date: Analysis start date (YYYY-MM-DD)
        analysis_end_date: Analysis end date (YYYY-MM-DD)
        review_interval_days: Days between review dates
        
    Returns:
        Comma-separated string of review dates
    """
    start_date = date.fromisoformat(analysis_start_date)
    end_date = date.fromisoformat(analysis_end_date)
    
    review_dates = []
    current_date = start_date
    
    while current_date <= end_date:
        review_dates.append(current_date.isoformat())
        current_date = current_date + timedelta(days=review_interval_days)
    
    return ",".join(review_dates)


def run_complete_workflow(
    data_dir: str = "forecaster/data",
    demand_file: str = "customer_demand.csv",
    product_master_file: str = "customer_product_master.csv",
    output_dir: str = "output/complete_workflow",
    analysis_start_date: str = "2024-01-01",
    analysis_end_date: str = "2024-12-01",
    demand_frequency: str = "d",
    batch_size: int = 10,
    max_workers: int = 8,
    outlier_enabled: bool = True,
    backtesting_enabled: bool = True,
    safety_stock_enabled: bool = True,
    simulation_enabled: bool = True,
    web_interface: bool = False,
    log_level: str = "INFO",
    review_interval_days: int = 30,
    review_dates: str = None
):
    """
    Run the complete inventory analysis workflow.
    
    Args:
        data_dir: Directory containing data files
        demand_file: Demand data file name
        product_master_file: Product master file name
        output_dir: Output directory for results
        analysis_start_date: Analysis start date (YYYY-MM-DD format)
        analysis_end_date: Analysis end date (YYYY-MM-DD format)
        demand_frequency: Demand frequency ('d', 'w', 'm')
        batch_size: Batch size for processing
        max_workers: Maximum number of parallel workers
        outlier_enabled: Whether to enable outlier handling
        backtesting_enabled: Whether to enable backtesting
        safety_stock_enabled: Whether to enable safety stock calculation
        simulation_enabled: Whether to enable simulation
        web_interface: Whether to start web interface
        log_level: Logging level
        review_interval_days: Days between review dates for safety stock calculation
    """
    
    print("🔍 Complete Inventory Analysis Workflow")
    print("=" * 60)
    print("This workflow runs the complete inventory analysis pipeline:")
    print("1. Data validation")
    print("2. Backtesting (historical analysis)")
    print("3. Safety stock calculation")
    print("4. Inventory simulation")
    if web_interface:
        print("5. Web interface startup")
    print("=" * 60)
    
    # Validate data directory
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"❌ Error: Data directory not found: {data_dir}")
        sys.exit(1)
    
    # Validate data files
    demand_path = data_path / demand_file
    product_master_path = data_path / product_master_file
    
    if not demand_path.exists():
        print(f"❌ Error: Demand file not found: {demand_path}")
        sys.exit(1)
    
    if not product_master_path.exists():
        print(f"❌ Error: Product master file not found: {product_master_path}")
        sys.exit(1)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Data Directory: {data_dir}")
    print(f"📊 Demand File: {demand_file}")
    print(f"📋 Product Master File: {product_master_file}")
    print(f"📁 Output Directory: {output_dir}")
    print(f"📅 Analysis Period: {analysis_start_date} to {analysis_end_date}")
    print(f"🔄 Demand Frequency: {demand_frequency}")
    print(f"⚙️ Batch Size: {batch_size}")
    print(f"🚀 Max Workers: {max_workers}")
    print(f"🔍 Outlier Handling: {'Enabled' if outlier_enabled else 'Disabled'}")
    print(f"📈 Backtesting: {'Enabled' if backtesting_enabled else 'Disabled'}")
    print(f"🛡️ Safety Stock Calculation: {'Enabled' if safety_stock_enabled else 'Disabled'}")
    print(f"🎮 Simulation: {'Enabled' if simulation_enabled else 'Disabled'}")
    print(f"🌐 Web Interface: {'Enabled' if web_interface else 'Disabled'}")
    print(f"📅 Review Interval: {review_interval_days} days")
    print()
    
    workflow_start_time = time.time()
    success_count = 0
    total_steps = 0
    
    try:
        # Step 1: Data Validation
        total_steps += 1
        print("🔄 Step 1/5: Running data validation...")
        
        validation_cmd = f"python run_data_validation.py --data-dir {data_dir} --demand-file {demand_file} --product-master-file {product_master_file}"
        
        if demand_frequency:
            validation_cmd += f" --demand-frequency {demand_frequency}"
        
        if log_level:
            validation_cmd += f" --log-level {log_level}"
        
        if run_command(validation_cmd, "Data Validation", check=True):
            success_count += 1
            print("✅ Data validation completed successfully!")
        else:
            print("❌ Data validation failed. Please fix the issues before proceeding.")
            sys.exit(1)
        
        # Step 2: Backtesting
        if backtesting_enabled:
            total_steps += 1
            print("\n🔄 Step 2/5: Running backtesting...")
            
            backtest_cmd = f"python run_unified_backtest.py --data-dir {data_dir} --demand-file {demand_file} --product-master-file {product_master_file} --output-dir {output_dir}/backtesting --analysis-start-date {analysis_start_date} --analysis-end-date {analysis_end_date}"
            
            backtest_cmd += f" --demand-frequency {demand_frequency} --batch-size {batch_size} --max-workers {max_workers}"
            
            if not outlier_enabled:
                backtest_cmd += " --no-outliers"
            
            if run_command(backtest_cmd, "Backtesting", check=True):
                success_count += 1
                print("✅ Backtesting completed successfully!")
            else:
                print("❌ Backtesting failed. Stopping workflow.")
                sys.exit(1)
        
        # Step 3: Safety Stock Calculation
        if safety_stock_enabled and backtesting_enabled:
            total_steps += 1
            print("\n🔄 Step 3/5: Running safety stock calculation...")
            
            # Check if forecast comparison file exists
            forecast_comparison_file = f"{output_dir}/backtesting/forecast_comparison.csv"
            if not Path(forecast_comparison_file).exists():
                print(f"⚠️  Forecast comparison file not found: {forecast_comparison_file}")
                print("Skipping safety stock calculation...")
            else:
                # Use provided review dates or default to hardcoded dates (1st, 8th, 15th, 22nd of every month in 2024)
                if review_dates is None:
                    # Hardcoded review dates: 1st, 8th, 15th, 22nd of every month in 2024
                    hardcoded_review_dates = [
                        "2024-01-01", "2024-01-08", "2024-01-15", "2024-01-22",
                        "2024-02-01", "2024-02-08", "2024-02-15", "2024-02-22",
                        "2024-03-01", "2024-03-08", "2024-03-15", "2024-03-22",
                        "2024-04-01", "2024-04-08", "2024-04-15", "2024-04-22",
                        "2024-05-01", "2024-05-08", "2024-05-15", "2024-05-22",
                        "2024-06-01", "2024-06-08", "2024-06-15", "2024-06-22",
                        "2024-07-01", "2024-07-08", "2024-07-15", "2024-07-22",
                        "2024-08-01", "2024-08-08", "2024-08-15", "2024-08-22",
                        "2024-09-01", "2024-09-08", "2024-09-15", "2024-09-22",
                        "2024-10-01", "2024-10-08", "2024-10-15", "2024-10-22",
                        "2024-11-01", "2024-11-08", "2024-11-15", "2024-11-22",
                        "2024-12-01", "2024-12-08", "2024-12-15", "2024-12-22"
                    ]
                    review_dates_str = ",".join(hardcoded_review_dates)
                else:
                    review_dates_str = review_dates
                
                safety_stock_cmd = f"python run_safety_stock_calculation.py {forecast_comparison_file} {data_dir}/{product_master_file} --output-dir {output_dir}/safety_stocks --review-dates \"{review_dates_str}\""
                
                if run_command(safety_stock_cmd, "Safety Stock Calculation", check=False):
                    success_count += 1
                    print("✅ Safety stock calculation completed successfully!")
                else:
                    print("⚠️  Safety stock calculation failed, but continuing workflow...")
        
        # Step 4: Inventory Simulation
        if simulation_enabled and safety_stock_enabled:
            total_steps += 1
            print("\n🔄 Step 4/5: Running inventory simulation...")
            
            # Check if safety stock results exist
            safety_stock_file = f"{output_dir}/safety_stocks/safety_stock_results.csv"
            forecast_comparison_file = f"{output_dir}/backtesting/forecast_comparison.csv"
            
            if not Path(safety_stock_file).exists():
                print(f"⚠️  Safety stock results not found: {safety_stock_file}")
                print("Skipping simulation...")
            elif not Path(forecast_comparison_file).exists():
                print(f"⚠️  Forecast comparison file not found: {forecast_comparison_file}")
                print("Skipping simulation...")
            else:
                simulation_cmd = f"python run_simulation.py --data-dir {data_dir} --safety-stock-file {safety_stock_file} --forecast-comparison-file {forecast_comparison_file} --output-dir {output_dir}/simulation --max-workers {max_workers}"
                
                if run_command(simulation_cmd, "Inventory Simulation", check=False):
                    success_count += 1
                    print("✅ Inventory simulation completed successfully!")
                else:
                    print("⚠️  Inventory simulation failed, but continuing workflow...")
        
        # Step 5: Web Interface (optional)
        if web_interface:
            total_steps += 1
            print("\n🔄 Step 5/5: Starting web interface...")
            
            web_cmd = f"python webapp/app.py --port 5001"
            
            print("🌐 Starting web interface in background...")
            print("📱 You can access the web interface at: http://localhost:5001")
            print("🛑 To stop the web interface, press Ctrl+C")
            
            # Start web interface in background
            try:
                web_process = subprocess.Popen(web_cmd, shell=True)
                success_count += 1
                print("✅ Web interface started successfully!")
                print(f"🌐 Web interface running at: http://localhost:5001")
                print("🔄 Workflow completed! Web interface will continue running.")
                print("🛑 To stop the web interface, press Ctrl+C")
                
                # Keep the process running
                web_process.wait()
                
            except KeyboardInterrupt:
                print("\n🛑 Web interface stopped by user.")
                if web_process:
                    web_process.terminate()
        
        # Workflow Summary
        workflow_end_time = time.time()
        total_workflow_time = workflow_end_time - workflow_start_time
        
        print(f"\n{'='*60}")
        print("🎉 Complete Workflow Summary")
        print(f"{'='*60}")
        print(f"✅ Steps Completed: {success_count}/{total_steps}")
        print(f"⏱️  Total Workflow Time: {total_workflow_time:.2f} seconds")
        print(f"📁 Output Directory: {output_dir}")
        print()
        print("📁 Generated Files:")
        if backtesting_enabled:
            print(f"  • Backtesting Results: {output_dir}/backtesting/")
        if safety_stock_enabled:
            print(f"  • Safety Stock Results: {output_dir}/safety_stocks/")
        if simulation_enabled:
            print(f"  • Simulation Results: {output_dir}/simulation/")
        print()
        
        if success_count == total_steps:
            print("🎉 All steps completed successfully!")
        else:
            print(f"⚠️  {total_steps - success_count} step(s) had issues, but workflow completed.")
        
        if web_interface:
            print("\n🌐 Web Interface:")
            print("  • URL: http://localhost:5001")
            print("  • Navigate to different tabs to view results")
        else:
            print("\n🌐 To view results in web interface:")
            print("  • Run: python webapp/app.py")
            print("  • Visit: http://localhost:5001")
        
    except KeyboardInterrupt:
        print("\n🛑 Workflow interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Workflow failed with error: {e}")
        logger.error(f"Complete workflow failed: {e}")
        sys.exit(1)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run complete inventory analysis workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python run_complete_workflow.py
  
  # Run with custom data directory and files
  python run_complete_workflow.py --data-dir forecaster/data --demand-file my_demand.csv --product-master-file my_product_master.csv
  
  # Run with specific analysis period
  python run_complete_workflow.py --analysis-start-date 2024-01-01 --analysis-end-date 2024-12-01
  
  # Run with custom processing settings
  python run_complete_workflow.py --batch-size 20 --max-workers 8 --log-level DEBUG
  
  # Run without simulation
  python run_complete_workflow.py --no-simulation
  
  # Run with web interface
  python run_complete_workflow.py --web-interface
  
  # Run with custom review interval for safety stock
  python run_complete_workflow.py --review-interval 14
  
  # Run with custom review dates for safety stock
  python run_complete_workflow.py --review-dates "2024-01-01,2024-01-15,2024-02-01,2024-02-15"
        """
    )
    
    # Data configuration
    parser.add_argument("--data-dir", default="forecaster/data",
                       help="Directory containing data files (default: forecaster/data)")
    parser.add_argument("--demand-file", default="customer_demand.csv",
                       help="Demand data file name (default: customer_demand.csv)")
    parser.add_argument("--product-master-file", default="customer_product_master.csv",
                       help="Product master file name (default: customer_product_master.csv)")
    
    # Output configuration
    parser.add_argument("--output-dir", default="output/complete_workflow",
                       help="Output directory (default: output/complete_workflow)")
    
    # Analysis period configuration
    parser.add_argument("--analysis-start-date", default="2024-01-01",
                       help="Analysis start date (YYYY-MM-DD format, default: 2024-01-01)")
    parser.add_argument("--analysis-end-date", default="2024-12-01",
                       help="Analysis end date (YYYY-MM-DD format, default: 2024-12-01)")
    
    # Processing configuration
    parser.add_argument("--demand-frequency", default="d", choices=["d", "w", "m"],
                       help="Demand frequency: 'd' for daily, 'w' for weekly, 'm' for monthly (default: d)")
    parser.add_argument("--batch-size", type=int, default=10,
                       help="Batch size for processing (default: 10)")
    parser.add_argument("--max-workers", type=int, default=8,
                       help="Maximum number of parallel workers (default: 8)")
    
    # Safety stock configuration
    parser.add_argument("--review-interval", type=int, default=30,
                       help="Days between review dates for safety stock calculation (default: 30)")
    parser.add_argument("--review-dates",
                       help="Comma-separated list of review dates (YYYY-MM-DD format). If not provided, uses default dates: 1st, 8th, 15th, 22nd of every month in 2024")
    
    # Feature flags
    parser.add_argument("--no-outliers", action="store_true",
                       help="Disable outlier handling")
    parser.add_argument("--no-backtesting", action="store_true",
                       help="Disable backtesting")
    parser.add_argument("--no-safety-stock", action="store_true",
                       help="Disable safety stock calculation")
    parser.add_argument("--no-simulation", action="store_true",
                       help="Disable inventory simulation")
    parser.add_argument("--web-interface", action="store_true",
                       help="Start web interface after workflow completion")
    
    # Logging
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level (default: INFO)")
    
    args = parser.parse_args()
    
    # Convert feature flags
    outlier_enabled = not args.no_outliers
    backtesting_enabled = not args.no_backtesting
    safety_stock_enabled = not args.no_safety_stock
    simulation_enabled = not args.no_simulation
    
    run_complete_workflow(
        data_dir=args.data_dir,
        demand_file=args.demand_file,
        product_master_file=args.product_master_file,
        output_dir=args.output_dir,
        analysis_start_date=args.analysis_start_date,
        analysis_end_date=args.analysis_end_date,
        demand_frequency=args.demand_frequency,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        outlier_enabled=outlier_enabled,
        backtesting_enabled=backtesting_enabled,
        safety_stock_enabled=safety_stock_enabled,
        simulation_enabled=simulation_enabled,
        web_interface=args.web_interface,
        log_level=args.log_level,
        review_interval_days=args.review_interval,
        review_dates=args.review_dates
    )


if __name__ == "__main__":
    main() 