#!/usr/bin/env python3
"""
New Backtesting Pipeline Runner

This is a completely new backtesting pipeline that replaces the legacy system.
It provides aggregated risk-period logic, flexible regressors, and chunked persistence.

Usage:
    uv run python run_backtesting.py --analysis-start-date 2023-01-01 --analysis-end-date 2023-12-31
    uv run python run_backtesting.py --run-id 20250101_120000 --resume
"""

import argparse
import logging
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional, Tuple

from forecaster.utils.logger import configure_workflow_logging
from forecaster.backtesting.full_backtesting_pipeline import full_backtesting_pipeline


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="New Backtesting Pipeline - Aggregated Risk-Period Logic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full run with specific dates
  uv run python run_backtesting.py --analysis-start-date 2023-01-01 --analysis-end-date 2023-12-31 --demand-frequency d --max-workers 8
  
  # Resume existing run
  uv run python run_backtesting.py --run-id 20250101_120000 --resume
  
  # Use default dates from config
  uv run python run_backtesting.py --demand-frequency w --max-workers 4
        """
    )
    
    # Date arguments
    parser.add_argument(
        "--analysis-start-date",
        type=str,
        help="Analysis start date (YYYY-MM-DD). If omitted, computed from config."
    )
    parser.add_argument(
        "--analysis-end-date", 
        type=str,
        help="Analysis end date (YYYY-MM-DD). If omitted, computed from config."
    )
    
    # Configuration arguments
    parser.add_argument(
        "--demand-frequency",
        choices=["d", "w", "m"],
        default="d",
        help="Demand frequency: d=daily, w=weekly, m=monthly (default: d)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Maximum number of parallel workers (default: 8)"
    )
    
    # Logging and control
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--run-id",
        type=str,
        help="Unique run identifier. If omitted, auto-generated as YYYYMMDD_HHMMSS"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume existing run (append to existing chunks)"
    )
    
    # Dry run for testing
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be executed without running the pipeline"
    )
    
    # Performance profiling
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable performance profiling (saves profile data to output directory)"
    )
    
    return parser.parse_args()


def resolve_analysis_dates(args: argparse.Namespace) -> Tuple[date, date]:
    """
    Resolve analysis start and end dates.
    
    If dates are provided via CLI, use them. Otherwise, compute from config:
    - Read safety_stock.review_dates from config
    - Read product master to get max ss_window_length
    - Compute start_date = first_review_date - days_to_subtract
    - Compute end_date = last_review_date
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Tuple of (analysis_start_date, analysis_end_date)
    """
    from data.loader import DataLoader
    from forecaster.validation.product_master_schema import ProductMasterSchema
    
    # If dates are provided, parse and return them
    if args.analysis_start_date and args.analysis_end_date:
        try:
            start_date = datetime.strptime(args.analysis_start_date, "%Y-%m-%d").date()
            end_date = datetime.strptime(args.analysis_end_date, "%Y-%m-%d").date()
            
            if start_date >= end_date:
                raise ValueError("Analysis start date must be before end date")
                
            return start_date, end_date
            
        except ValueError as e:
            raise ValueError(f"Invalid date format. Use YYYY-MM-DD: {e}")
    
    # Otherwise, compute from config
    loader = DataLoader()
    
    # Read review dates from config
    config = loader.config
    review_dates = config.get('safety_stock', {}).get('review_dates', [])
    
    if not review_dates:
        raise ValueError("No review dates found in config. Please provide --analysis-start-date and --analysis-end-date")
    
    # Parse review dates
    try:
        review_dates_parsed = [datetime.strptime(d, "%Y-%m-%d").date() for d in review_dates]
        first_review_date = min(review_dates_parsed)
        last_review_date = max(review_dates_parsed)
    except ValueError as e:
        raise ValueError(f"Invalid review date format in config: {e}")
    
    # Read product master to get max ss_window_length
    product_master = loader.load_product_master()
    if 'ss_window_length' not in product_master.columns:
        raise ValueError("Product master missing 'ss_window_length' column")
    
    max_ss_window_length = product_master['ss_window_length'].max()
    
    # Compute days to subtract based on demand frequency
    if args.demand_frequency == "d":
        days_to_subtract = int(max_ss_window_length)
    elif args.demand_frequency == "w":
        days_to_subtract = int(max_ss_window_length * 7)
    elif args.demand_frequency == "m":
        days_to_subtract = int(max_ss_window_length * 30)
    else:
        raise ValueError(f"Invalid demand frequency: {args.demand_frequency}")
    
    # Compute analysis dates
    analysis_start_date = first_review_date - timedelta(days=days_to_subtract)
    analysis_end_date = last_review_date
    
    return analysis_start_date, analysis_end_date


def main():
    """Main entry point"""
    # Parse arguments
    args = parse_arguments()
    
    # Resolve analysis dates
    try:
        analysis_start_date, analysis_end_date = resolve_analysis_dates(args)
    except ValueError as e:
        print(f"❌ Error resolving analysis dates: {e}")
        return 1
    
    # Generate run_id if not provided
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Configure logging
    logger = configure_workflow_logging('backtesting', args.log_level)
    
    # Capture warnings to logs
    logging.captureWarnings(True)
    
    # Log configuration
    logger.info("🚀 New Backtesting Pipeline Starting")
    logger.info(f"📅 Analysis period: {analysis_start_date} to {analysis_end_date}")
    logger.info(f"🔄 Demand frequency: {args.demand_frequency}")
    logger.info(f"👥 Max workers: {args.max_workers}")
    logger.info(f"🆔 Run ID: {run_id}")
    logger.info(f"📝 Resume mode: {args.resume}")
    logger.info(f"📊 Profiling enabled: {args.profile}")
    
    # Dry run mode
    if args.dry_run:
        logger.info("🔍 DRY RUN MODE - No actual execution")
        logger.info("Configuration validated successfully")
        return 0
    
    # Execute pipeline
    try:
        logger.info("⚙️ Executing backtesting pipeline...")
        
        # Execute the pipeline
        result = full_backtesting_pipeline(
            analysis_start_date=analysis_start_date,
            analysis_end_date=analysis_end_date,
            demand_frequency=args.demand_frequency,
            max_workers=args.max_workers,
            run_id=run_id,
            resume=args.resume,
            log_level=args.log_level,
            profile=args.profile
        )
        
        # Log execution summary
        if result and 'execution_time' in result:
            logger.info(f"✅ Backtesting pipeline completed successfully in {result['execution_time']:.2f} seconds")
        else:
            logger.info("✅ Backtesting pipeline completed successfully")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Backtesting pipeline failed: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    exit(main())
