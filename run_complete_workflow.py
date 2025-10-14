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

from forecaster.utils.logger import configure_workflow_logging, get_logger
from datetime import date, timedelta
import pandas as pd





def run_command(command, description, logger, check=True, real_time_output=False):
    """Run a command and handle errors."""
    logger.info(f"🚀 {description}")
    logger.info(f"Running: {command}")

    start_time = time.time()

    try:
        if real_time_output:
            # Run with real-time output for progress tracking
            result = subprocess.run(command, shell=True, check=check, text=True)
        else:
            # Run with captured output for other commands
            result = subprocess.run(
                command, shell=True, check=check, text=True, capture_output=True
            )
            if result.stdout:
                logger.info(f"Output: {result.stdout.strip()}")

        execution_time = time.time() - start_time
        logger.log_step_completion(description, execution_time)
        return result

    except subprocess.CalledProcessError as e:
        execution_time = time.time() - start_time
        logger.log_error_with_context(e, f"Command failed: {command}")
        
        # Display the full output for validation errors
        if e.stdout:
            print(f"\n📋 Command Output:")
            print(e.stdout)
        if e.stderr:
            print(f"\n❌ Command Error:")
            print(e.stderr)
        
        # Also log for debugging
        if e.stdout:
            logger.info(f"stdout: {e.stdout}")
        if e.stderr:
            logger.error(f"stderr: {e.stderr}")
        raise





def run_complete_workflow(
    analysis_start_date: str = None,
    analysis_end_date: str = None,
    demand_frequency: str = "d",
    batch_size: int = 10,
    max_workers: int = 8,
    backtesting_enabled: bool = True,
    safety_stock_enabled: bool = True,
    simulation_enabled: bool = True,
    web_interface: bool = False,
    log_level: str = "INFO",
    review_dates: str = None,
    skip_validation: bool = False,
    logger=None,
):
    """
    Run the complete inventory analysis workflow.

    Args:
        analysis_start_date: Analysis start date (YYYY-MM-DD format). If None, will be calculated based on max ss_window_length and first review date. Required if calculation fails.
        analysis_end_date: Analysis end date (YYYY-MM-DD format). If None, will be set to the last review date. Required if calculation fails.
        demand_frequency: Demand frequency ('d', 'w', 'm')
        batch_size: Batch size for processing
        max_workers: Maximum number of parallel workers
        backtesting_enabled: Whether to enable backtesting
        safety_stock_enabled: Whether to enable safety stock calculation
        simulation_enabled: Whether to enable simulation
        web_interface: Whether to start web interface
        log_level: Logging level
        review_dates: Comma-separated list of review dates (YYYY-MM-DD format). If not provided, will use dates from config file.
        logger: Logger instance for logging workflow steps.
    """

    if logger is None:
        logger = get_logger(__name__)

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

    # Initialize DataLoader to validate configuration
    try:
        from data.loader import DataLoader
        loader = DataLoader()
        print("✅ DataLoader initialized successfully")
        print(f"📁 Data configuration loaded from: data/config/data_config.yaml")
    except Exception as e:
        print(f"❌ Error initializing DataLoader: {e}")
        print("Please check your data/config/data_config.yaml configuration")
        sys.exit(1)

    # Pass analysis dates to backtesting (None triggers auto-calculation in backtester)
    if analysis_start_date is not None:
        print(f"📅 Using provided analysis start date: {analysis_start_date}")
    if analysis_end_date is not None:
        print(f"📅 Using provided analysis end date: {analysis_end_date}")
    if analysis_start_date is None and analysis_end_date is None:
        print("📅 Analysis dates not provided - will be auto-calculated in backtester")

    print(f"📅 Analysis Period: {analysis_start_date} to {analysis_end_date}")
    print(f"🔄 Demand Frequency: {demand_frequency}")
    print(f"⚙️ Batch Size: {batch_size}")
    print(f"🚀 Max Workers: {max_workers}")
    print(f"📈 Backtesting: {'Enabled' if backtesting_enabled else 'Disabled'}")
    print(
        f"🛡️ Safety Stock Calculation: {'Enabled' if safety_stock_enabled else 'Disabled'}"
    )
    print(f"🎮 Simulation: {'Enabled' if simulation_enabled else 'Disabled'}")
    print(f"🌐 Web Interface: {'Enabled' if web_interface else 'Disabled'}")

    print()

    workflow_start_time = time.time()
    success_count = 0
    total_steps = 0

    try:
        # Step 1: Data Validation (conditional)
        if not skip_validation:
            total_steps += 1
            print("🔄 Step 1/5: Running data validation...")

            validation_cmd = f"python run_data_validation.py"

            if demand_frequency:
                validation_cmd += f" --demand-frequency {demand_frequency}"

            if log_level:
                validation_cmd += f" --log-level {log_level}"

            if run_command(validation_cmd, "Data Validation", logger, check=True):
                success_count += 1
                print("✅ Data validation completed successfully!")
            else:
                print("❌ Data validation failed. Please fix the issues before proceeding.")
                sys.exit(1)
        else:
            print("⏭️  Skipping data validation as requested...")
            success_count += 1

        # Step 2: Backtesting
        if backtesting_enabled:
            total_steps += 1
            step_number = total_steps
            print(f"\n🔄 Step {step_number}/5: Running new backtesting pipeline...")
            print("📊 Progress tracking and logging will be displayed in real-time")
            print(
                "⏱️  This step may take significant time depending on data size and workers"
            )

            backtest_cmd = f"uv run python run_backtesting.py"
            
            # Add analysis dates if provided, otherwise let backtester auto-calculate
            if analysis_start_date is not None:
                backtest_cmd += f" --analysis-start-date {analysis_start_date}"
            if analysis_end_date is not None:
                backtest_cmd += f" --analysis-end-date {analysis_end_date}"

            # Add demand frequency and max workers (batch-size is not used in new pipeline)
            backtest_cmd += f" --demand-frequency {demand_frequency} --max-workers {max_workers}"

            # Ensure log level is passed for progress tracking
            if log_level:
                backtest_cmd += f" --log-level {log_level}"

            if run_command(
                backtest_cmd, "Backtesting", logger, check=True, real_time_output=True
            ):
                success_count += 1
                print("✅ Backtesting completed successfully!")
            else:
                print("❌ Backtesting failed. Stopping workflow.")
                sys.exit(1)

        # Step 3: Safety Stock Calculation
        if safety_stock_enabled:
            total_steps += 1
            step_number = total_steps
            print(f"\n🔄 Step {step_number}/5: Running safety stock calculation...")
            print("📊 Progress tracking will be displayed in real-time")

            # Check if forecast comparison file exists
            filename = loader.config['paths']['output_files']['forecast_comparison']
            forecast_comparison_file = str(loader.get_output_path("backtesting", filename))
            if not Path(forecast_comparison_file).exists():
                print(
                    f"⚠️  Forecast comparison file not found: {forecast_comparison_file}"
                )
                print("Skipping safety stock calculation...")
            else:
                # Use provided review dates or get from config
                if review_dates is None:
                    # Get review dates from config
                    try:
                        config_review_dates = loader.config.get('safety_stock', {}).get('review_dates', [])
                        if not config_review_dates:
                            print("❌ No review dates found in config. Please add review_dates to data/config/data_config.yaml under safety_stock section.")
                            print("Skipping safety stock calculation...")
                        else:
                            review_dates_str = ",".join(config_review_dates)
                            print(f"📅 Using {len(config_review_dates)} review dates from config")
                    except Exception as e:
                        print(f"❌ Error reading review dates from config: {e}")
                        print("Skipping safety stock calculation...")
                        config_review_dates = []
                    
                    if not config_review_dates:
                        # Skip safety stock calculation if no review dates available
                        pass
                    else:
                        safety_stock_cmd = f'python run_safety_stock_calculation.py {forecast_comparison_file} --log-level {log_level}'

                        if run_command(
                            safety_stock_cmd,
                            "Safety Stock Calculation",
                            logger,
                            check=False,
                            real_time_output=True,
                        ):
                            success_count += 1
                            print("✅ Safety stock calculation completed successfully!")
                        else:
                            print(
                                "⚠️  Safety stock calculation failed, but continuing workflow..."
                            )
                else:
                    review_dates_str = review_dates
                    safety_stock_cmd = f'python run_safety_stock_calculation.py {forecast_comparison_file} --review-dates "{review_dates_str}" --log-level {log_level}'

                    if run_command(
                        safety_stock_cmd,
                        "Safety Stock Calculation",
                        logger,
                        check=False,
                        real_time_output=True,
                    ):
                        success_count += 1
                        print("✅ Safety stock calculation completed successfully!")
                    else:
                        print(
                            "⚠️  Safety stock calculation failed, but continuing workflow..."
                        )

        # Step 4: Inventory Simulation
        if simulation_enabled and safety_stock_enabled:
            total_steps += 1
            step_number = total_steps
            print(f"\n🔄 Step {step_number}/5: Running inventory simulation...")
            print("📊 Progress tracking will be displayed in real-time")

            # Check if safety stock results exist
            safety_filename = loader.config['paths']['output_files']['safety_stocks']
            safety_stock_file = str(loader.get_output_path("safety_stocks", safety_filename))
            forecast_filename = loader.config['paths']['output_files']['forecast_comparison']
            forecast_comparison_file = str(loader.get_output_path("backtesting", forecast_filename))

            if not Path(safety_stock_file).exists():
                print(f"⚠️  Safety stock results not found: {safety_stock_file}")
                print("Skipping simulation...")
            elif not Path(forecast_comparison_file).exists():
                print(
                    f"⚠️  Forecast comparison file not found: {forecast_comparison_file}"
                )
                print("Skipping simulation...")
            else:
                simulation_cmd = f"python run_simulation.py --safety-stock-file {safety_stock_file} --forecast-comparison-file {forecast_comparison_file} --max-workers {max_workers} --log-level {log_level}"

                if run_command(
                    simulation_cmd,
                    "Inventory Simulation",
                    logger,
                    check=False,
                    real_time_output=True,
                ):
                    success_count += 1
                    print("✅ Inventory simulation completed successfully!")
                else:
                    print("⚠️  Inventory simulation failed, but continuing workflow...")

        # Step 5: Web Interface (optional)
        if web_interface:
            total_steps += 1
            step_number = total_steps
            print(f"\n🔄 Step {step_number}/5: Starting web interface...")

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
        print()
        print("📁 Generated Files:")
        if backtesting_enabled:
            print(f"  • Backtesting Results: {loader.get_output_path('backtesting', '')}")
        if safety_stock_enabled:
            print(f"  • Safety Stock Results: {loader.get_output_path('safety_stocks', '')}")
        if simulation_enabled:
            print(f"  • Simulation Results: {loader.get_output_path('simulation', '')}")
        print()

        if success_count == total_steps:
            print("🎉 All steps completed successfully!")
        else:
            print(
                f"⚠️  {total_steps - success_count} step(s) had issues, but workflow completed."
            )

        if web_interface:
            print("\n🌐 Starting web interface...")
            web_cmd = "python webapp/app.py"
            if run_command(web_cmd, "Web Interface", logger, check=True, real_time_output=True):
                print("\n🌐 Web Interface:")
                print("  • URL: http://localhost:5001")
                print("  • Navigate to different tabs to view results")
            else:
                print("\n❌ Failed to start web interface")
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
    """Main entry point for the complete workflow."""
    parser = argparse.ArgumentParser(
        description="Run the complete inventory analysis workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete workflow with default settings
  python run_complete_workflow.py
  
  # Run with custom analysis period
  python run_complete_workflow.py --analysis-start-date 2023-01-01 --analysis-end-date 2024-12-31
  
  # Run with custom demand frequency
  python run_complete_workflow.py --demand-frequency w
  
  # Run with custom processing settings
  python run_complete_workflow.py --batch-size 20 --max-workers 16
  
  # Run without simulation
  python run_complete_workflow.py --no-simulation
  
  # Run with web interface
  python run_complete_workflow.py --web-interface
  
  # Run with custom review dates for safety stock
  python run_complete_workflow.py --review-dates "2024-01-01,2024-01-15,2024-02-01,2024-02-15"
        """,
    )

    # Output configuration
    # Note: Output directory is now handled by DataLoader configuration

    # Analysis period configuration
    parser.add_argument(
        "--analysis-start-date",
        default=None,
        help="Analysis start date (YYYY-MM-DD format). If not provided, will be calculated based on max ss_window_length and first review date. Required if calculation fails.",
    )
    parser.add_argument(
        "--analysis-end-date",
        default=None,
        help="Analysis end date (YYYY-MM-DD format). If not provided, will be set to the last review date. Required if calculation fails.",
    )

    # Processing configuration
    parser.add_argument(
        "--demand-frequency",
        default="d",
        choices=["d", "w", "m"],
        help="Demand frequency: 'd' for daily, 'w' for weekly, 'm' for monthly (default: d)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for processing (default: 10)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Maximum number of parallel workers (default: 8)",
    )

    # Safety stock configuration
    parser.add_argument(
        "--review-dates",
        help="Comma-separated list of review dates (YYYY-MM-DD format). If not provided, will use dates from config file.",
    )

    # Feature flags
    parser.add_argument(
        "--skip-validation", action="store_true", help="Skip data validation step"
    )
    parser.add_argument(
        "--no-backtesting", action="store_true", help="Disable backtesting"
    )
    parser.add_argument(
        "--no-safety-stock",
        action="store_true",
        help="Disable safety stock calculation",
    )
    parser.add_argument(
        "--no-simulation", action="store_true", help="Disable inventory simulation"
    )
    parser.add_argument(
        "--web-interface",
        action="store_true",
        help="Start web interface after workflow completion",
    )

    # Logging
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: WARNING)",
    )

    args = parser.parse_args()

    # Convert feature flags
    backtesting_enabled = not args.no_backtesting
    safety_stock_enabled = not args.no_safety_stock
    simulation_enabled = not args.no_simulation

    # Setup logging for the workflow
    logger = configure_workflow_logging(
        workflow_name="complete_workflow",
        log_level=args.log_level,
        log_dir="output/logs"
    )

    try:
        run_complete_workflow(
            analysis_start_date=args.analysis_start_date if args.analysis_start_date else None,
            analysis_end_date=args.analysis_end_date if args.analysis_end_date else None,
            demand_frequency=args.demand_frequency,
            batch_size=args.batch_size,
            max_workers=args.max_workers,
            backtesting_enabled=backtesting_enabled,
            safety_stock_enabled=safety_stock_enabled,
            simulation_enabled=simulation_enabled,
            web_interface=args.web_interface,
            log_level=args.log_level,
            review_dates=args.review_dates,
            skip_validation=args.skip_validation,
            logger=logger
        )
    except Exception as e:
        logger.log_error_with_context(e, "Complete workflow failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
