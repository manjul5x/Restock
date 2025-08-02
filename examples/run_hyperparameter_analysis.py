#!/usr/bin/env python3
"""
Example script for running hyperparameter analysis on Prophet models.

This script demonstrates how to use the HyperparameterAnalyzer to find
optimal Prophet parameters for demand forecasting.
"""

import sys
from pathlib import Path
from datetime import date

# Add the forecaster package to the path
sys.path.append(str(Path(__file__).parent.parent))

from forecaster.backtesting.hyperparameter_analyzer import (
    HyperparameterAnalyzer,
    run_hyperparameter_analysis,
)
from forecaster.backtesting.config import BacktestConfig


def main():
    """Run hyperparameter analysis example."""
    print("🔍 Prophet Hyperparameter Analysis")
    print("=" * 50)

    # Create configuration with smaller scope to see progress
    config = BacktestConfig(
        output_dir="output/hyperparameter_analysis",
        historic_start_date=date(2022, 1, 1),
        analysis_start_date=date(2023, 6, 1),
        analysis_end_date=date(2023, 7, 1),  # Shorter period
        demand_frequency="d",
        forecast_model="prophet",
        default_horizon=1,
        max_workers=1,  # Single worker to see progress clearly
        log_level="INFO",  # Reduce logging noise
    )

    print(f"📊 Configuration:")
    print(
        f"   - Analysis period: {config.analysis_start_date} to {config.analysis_end_date}"
    )
    print(f"   - Max workers: {config.max_workers}")
    print()

    try:
        # Run comprehensive analysis
        print("🚀 Starting hyperparameter analysis...")
        print("⏳ This may take several minutes depending on configuration...")
        print("📊 Progress will be shown with detailed statistics...")
        print("-" * 60)

        results = run_hyperparameter_analysis(config)

        print("-" * 60)
        print("🎉 Analysis completed! Processing results...")

        # Debug: Print raw results
        print(
            f"Raw results count: {len(results.get('analysis_results', {}).get('all_param_metrics', {}))}"
        )
        print(f"Results keys: {list(results.keys())}")

        # Display results
        analysis_summary = results["analysis_results"]

        print("✅ Analysis completed successfully!")
        print()

        # Display summary statistics
        summary_stats = analysis_summary.get("summary_stats", {})
        print("📈 Summary Statistics:")
        print(f"   - Total tests: {summary_stats.get('total_tests', 'N/A')}")
        print(f"   - Successful tests: {summary_stats.get('successful_tests', 'N/A')}")
        print(f"   - Success rate: {summary_stats.get('success_rate', 0) * 100:.1f}%")
        print(f"   - Best MAPE: {summary_stats.get('best_mape', 'N/A')}%")
        print()

        # Display best parameters
        best_params = analysis_summary.get("best_parameters", {})
        if best_params:
            print("🏆 Best Parameters Found:")
            for param, value in best_params.items():
                formatted_value = (
                    "Yes" if value is True else "No" if value is False else str(value)
                )
                print(f"   - {param}: {formatted_value}")
            print()

        # Display parameter effects
        param_analysis = analysis_summary.get("parameter_analysis", {})
        if param_analysis:
            print("📊 Parameter Effects Analysis:")
            for param, effects in param_analysis.items():
                print(f"   {param}:")
                for value, metrics in effects.items():
                    avg_mape = metrics.get("avg_mape", "N/A")
                    if avg_mape != "N/A":
                        print(f"     - {value}: {avg_mape:.2f}% MAPE")
                print()

        # Display plot files
        plot_files = results.get("plot_files", {})
        if plot_files:
            print("📊 Generated Visualization Plots:")
            for plot_name, plot_path in plot_files.items():
                if plot_path:
                    print(f"   - {plot_name}: {plot_path}")
            print()

        print("🎉 Analysis complete! Check the generated plots for detailed insights.")

    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    from datetime import date

    exit(main())
