"""
Seasonality Analysis Module for Prophet Forecasting

This module provides comprehensive analysis of seasonality components in time series data,
including Fourier term analysis, seasonality strength assessment, and recommendations
for Prophet model configuration.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import date, timedelta
import warnings
from prophet import Prophet
from ..utils.logger import get_logger


class SeasonalityAnalyzer:
    """
    Comprehensive seasonality analyzer for Prophet forecasting models.

    This class analyzes time series data to determine optimal seasonality components
    for Prophet models, including Fourier term analysis and strength assessment.
    """

    def __init__(self, log_level: str = "INFO"):
        """
        Initialize the seasonality analyzer.
        
        Args:
            log_level: Logging level to use for analysis output
        """
        self.analysis_results = {}
        self.recommendations = {}
        self.log_level = log_level
        # Create logger instance once during initialization
        self.logger = get_logger(__name__, level=self.log_level)

    def analyze_seasonality_components(
        self, data: pd.DataFrame, model: Prophet, fitted_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Perform comprehensive seasonality analysis on fitted Prophet model.

        Args:
            data: Original input data
            model: Fitted Prophet model
            fitted_data: Data used for fitting (after preprocessing)

        Returns:
            Dictionary containing comprehensive seasonality analysis
        """
        self.logger.info("=" * 80)
        self.logger.info("FOURIER TERMS ANALYSIS - SEASONALITY ASSESSMENT")
        self.logger.info("=" * 80)

        analysis_results = {"seasonalities": {}, "summary": {}, "recommendations": {}}

        # Analyze each seasonality component
        seasonality_components = self._get_seasonality_components(model)
        self.logger.debug(
            f"Found {len(seasonality_components)} seasonality components: {list(seasonality_components.keys())}"
        )

        for seasonality_name, seasonality_info in seasonality_components.items():
            self.logger.info(f"\nSeasonality: {seasonality_name}")
            self.logger.debug(f"Analyzing {seasonality_name} with info: {seasonality_info}")

            try:
                # Analyze Fourier terms for this seasonality
                fourier_analysis = self._analyze_fourier_terms(
                    model, seasonality_name, seasonality_info, fitted_data
                )

                analysis_results["seasonalities"][seasonality_name] = fourier_analysis

                # Log detailed analysis
                self._log_seasonality_analysis(seasonality_name, fourier_analysis)
            except Exception as e:
                self.logger.error(f"Error analyzing {seasonality_name}: {e}")
                continue

        # Generate summary and recommendations
        summary = self._generate_seasonality_summary(analysis_results["seasonalities"])
        recommendations = self._generate_recommendations(summary)

        analysis_results["summary"] = summary
        analysis_results["recommendations"] = recommendations

        # Log summary
        self._log_seasonality_summary(summary, recommendations)

        return analysis_results

    def _get_seasonality_components(self, model: Prophet) -> Dict[str, Dict]:
        """
        Extract seasonality components from fitted Prophet model.

        Args:
            model: Fitted Prophet model

        Returns:
            Dictionary of seasonality components and their parameters
        """
        seasonalities = {}

        # Debug logging
        self.logger.debug(
            f"Model seasonalities attribute: {hasattr(model, 'seasonalities')}"
        )
        if hasattr(model, "seasonalities"):
            self.logger.debug(f"Model seasonalities: {model.seasonalities}")

        # Built-in seasonalities
        if hasattr(model, "seasonalities") and model.seasonalities:
            for name, seasonality in model.seasonalities.items():
                try:
                    # Handle case where seasonality is a dictionary
                    if isinstance(seasonality, dict):
                        seasonalities[name] = {
                            "period": seasonality.get("period", 365.25),
                            "fourier_order": seasonality.get("fourier_order", 10),
                            "type": "custom",
                        }
                    else:
                        # Handle case where seasonality is an object with attributes
                        seasonalities[name] = {
                            "period": seasonality.period,
                            "fourier_order": seasonality.fourier_order,
                            "type": "custom",
                        }
                except AttributeError as e:
                    self.logger.warning(
                        f"Could not access attributes for seasonality {name}: {e}"
                    )
                    # Try to get attributes from the seasonality object
                    if hasattr(seasonality, "period"):
                        period = seasonality.period
                    else:
                        period = 365.25  # Default

                    if hasattr(seasonality, "fourier_order"):
                        fourier_order = seasonality.fourier_order
                    else:
                        fourier_order = 10  # Default

                    seasonalities[name] = {
                        "period": period,
                        "fourier_order": fourier_order,
                        "type": "custom",
                    }

        # Default seasonalities
        if model.yearly_seasonality:
            seasonalities["yearly"] = {
                "period": 365.25,
                "fourier_order": 10,
                "type": "default",
            }

        if model.weekly_seasonality:
            seasonalities["weekly"] = {
                "period": 7,
                "fourier_order": 3,
                "type": "default",
            }

        if model.daily_seasonality:
            seasonalities["daily"] = {
                "period": 1,
                "fourier_order": 4,
                "type": "default",
            }

        # Add custom seasonalities that we know are added
        # These are added in the Prophet forecaster but may not be in model.seasonalities
        if not "monthly" in seasonalities:
            seasonalities["monthly"] = {
                "period": 30.5,
                "fourier_order": 5,
                "type": "custom",
            }

        if not "quarterly" in seasonalities:
            seasonalities["quarterly"] = {
                "period": 91.25,
                "fourier_order": 8,
                "type": "custom",
            }

        # Ensure correct periods are used (override any fallback values)
        if "monthly" in seasonalities:
            seasonalities["monthly"]["period"] = 30.5
        if "quarterly" in seasonalities:
            seasonalities["quarterly"]["period"] = 91.25
        if "yearly" in seasonalities:
            seasonalities["yearly"]["period"] = 365.25
        if "weekly" in seasonalities:
            seasonalities["weekly"]["period"] = 7
        if "daily" in seasonalities:
            seasonalities["daily"]["period"] = 1

        self.logger.debug(f"Final seasonalities: {seasonalities}")
        return seasonalities

    def _analyze_fourier_terms(
        self,
        model: Prophet,
        seasonality_name: str,
        seasonality_info: Dict,
        fitted_data: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Analyze Fourier terms for a specific seasonality component.

        Args:
            model: Fitted Prophet model
            seasonality_name: Name of the seasonality component
            seasonality_info: Seasonality parameters
            fitted_data: Data used for fitting

        Returns:
            Dictionary containing Fourier term analysis
        """
        period = seasonality_info["period"]
        fourier_order = seasonality_info["fourier_order"]

        # Calculate Fourier terms
        fourier_terms = self._calculate_fourier_terms(
            fitted_data, period, fourier_order
        )

        # Analyze term magnitudes
        magnitudes = np.abs(fourier_terms)
        max_magnitude = np.max(magnitudes)
        mean_magnitude = np.mean(magnitudes)
        std_magnitude = np.std(magnitudes)

        # Determine seasonality strength
        strength = self._determine_seasonality_strength(max_magnitude)

        # Find significant terms (magnitude > 0.01)
        significant_terms = []
        for i, magnitude in enumerate(magnitudes):
            if magnitude > 0.01:
                significant_terms.append(
                    {"term": i, "value": fourier_terms[i], "magnitude": magnitude}
                )

        return {
            "period": period,
            "fourier_order": fourier_order,
            "num_terms": len(fourier_terms),
            "max_magnitude": max_magnitude,
            "mean_magnitude": mean_magnitude,
            "std_magnitude": std_magnitude,
            "strength": strength,
            "significant_terms": significant_terms,
            "all_magnitudes": magnitudes.tolist(),
        }

    def _calculate_fourier_terms(
        self, data: pd.DataFrame, period: float, fourier_order: int
    ) -> np.ndarray:
        """
        Calculate Fourier terms for a given period and order.

        Args:
            data: Time series data
            period: Seasonality period
            fourier_order: Fourier order

        Returns:
            Array of Fourier term coefficients
        """
        # Convert dates to numeric for calculation
        dates = pd.to_datetime(data["ds"])
        t = (dates - dates.min()).dt.total_seconds() / (24 * 3600)  # Days since start

        # Calculate Fourier terms
        fourier_terms = np.zeros(2 * fourier_order)

        for k in range(1, fourier_order + 1):
            # Cosine term
            cos_term = np.cos(2 * np.pi * k * t / period)
            fourier_terms[2 * k - 2] = np.mean(cos_term * data["y"])

            # Sine term
            sin_term = np.sin(2 * np.pi * k * t / period)
            fourier_terms[2 * k - 1] = np.mean(sin_term * data["y"])

        return fourier_terms

    def _determine_seasonality_strength(self, max_magnitude: float) -> str:
        """
        Determine seasonality strength based on maximum magnitude.

        Args:
            max_magnitude: Maximum Fourier term magnitude

        Returns:
            Strength classification
        """
        if max_magnitude > 0.5:
            return "VERY_STRONG"
        elif max_magnitude > 0.2:
            return "STRONG"
        elif max_magnitude > 0.1:
            return "MODERATE"
        elif max_magnitude > 0.05:
            return "WEAK"
        else:
            return "VERY_WEAK"

    def _log_seasonality_analysis(self, seasonality_name: str, analysis: Dict):
        """
        Log detailed seasonality analysis.

        Args:
            seasonality_name: Name of the seasonality
            analysis: Analysis results
        """
        self.logger.info(f"  Period: {analysis['period']} days")
        self.logger.info(f"  Fourier Order: {analysis['fourier_order']}")
        self.logger.info(f"  Number of Fourier Terms: {analysis['num_terms']}")
        self.logger.info(f"  Max Magnitude: {analysis['max_magnitude']:.6f}")
        self.logger.info(f"  Mean Magnitude: {analysis['mean_magnitude']:.6f}")
        self.logger.info(f"  Std Magnitude: {analysis['std_magnitude']:.6f}")
        self.logger.info(f"  Seasonality Strength: {analysis['strength']}")

        # Log significant terms
        self.logger.info(f"  Significant Fourier Terms (magnitude > 0.01):")
        for term in analysis["significant_terms"]:
            self.logger.info(
                f"    Term {term['term']}: {term['value']:.6f} (magnitude: {term['magnitude']:.6f})"
            )

        # Add business interpretation
        self._log_business_interpretation(seasonality_name, analysis)

    def _log_business_interpretation(self, seasonality_name: str, analysis: Dict):
        """
        Log business interpretation of seasonality patterns.

        Args:
            seasonality_name: Name of the seasonality
            analysis: Analysis results
        """
        if seasonality_name == "quarterly":
            self.logger.info(f"  Quarterly Seasonality Analysis:")
            self.logger.info(f"    Expected business cycle patterns every ~3 months")
        elif seasonality_name == "monthly":
            self.logger.info(f"  Monthly Seasonality Analysis:")
            self.logger.info(f"    Expected monthly patterns")
        elif seasonality_name == "yearly":
            self.logger.info(f"  Yearly Seasonality Analysis:")
            self.logger.info(f"    Expected annual patterns")
        elif seasonality_name == "weekly":
            self.logger.info(f"  Weekly Seasonality Analysis:")
            self.logger.info(f"    Expected weekly patterns")

    def _generate_seasonality_summary(self, seasonalities: Dict) -> Dict[str, Any]:
        """
        Generate summary of all seasonality components.

        Args:
            seasonalities: Dictionary of seasonality analyses

        Returns:
            Summary statistics and rankings
        """
        # Rank seasonalities by strength
        ranked_seasonalities = []
        magnitudes = []

        for name, analysis in seasonalities.items():
            ranked_seasonalities.append(
                {
                    "name": name,
                    "max_magnitude": analysis["max_magnitude"],
                    "strength": analysis["strength"],
                }
            )
            magnitudes.append(analysis["max_magnitude"])

        # Sort by magnitude (descending)
        ranked_seasonalities.sort(key=lambda x: x["max_magnitude"], reverse=True)

        # Calculate summary statistics
        max_magnitudes = [s["max_magnitude"] for s in ranked_seasonalities]

        summary = {
            "ranked_seasonalities": ranked_seasonalities,
            "strongest": ranked_seasonalities[0] if ranked_seasonalities else None,
            "weakest": ranked_seasonalities[-1] if ranked_seasonalities else None,
            "mean_max_magnitude": np.mean(max_magnitudes) if max_magnitudes else 0,
            "std_max_magnitude": np.std(max_magnitudes) if max_magnitudes else 0,
            "total_seasonalities": len(ranked_seasonalities),
        }

        return summary

    def _generate_recommendations(self, summary: Dict) -> Dict[str, Any]:
        """
        Generate recommendations based on seasonality analysis.

        Args:
            summary: Seasonality summary

        Returns:
            Dictionary of recommendations
        """
        recommendations = {
            "very_strong_seasonalities": [],
            "regularization_needed": [],
            "component_suggestions": [],
            "best_model_parameters": {},
            "weak_seasonalities": [],
            "unused_seasonalities": [],
        }

        # Identify very strong seasonalities that may need regularization
        for seasonality in summary["ranked_seasonalities"]:
            if seasonality["strength"] in ["VERY_STRONG", "STRONG"]:
                recommendations["very_strong_seasonalities"].append(seasonality["name"])

                if seasonality["max_magnitude"] > 0.3:
                    recommendations["regularization_needed"].append(seasonality["name"])

            # Identify weak seasonalities
            elif seasonality["strength"] in ["WEAK", "VERY_WEAK"]:
                recommendations["weak_seasonalities"].append(seasonality["name"])

        # Component suggestions
        if summary["mean_max_magnitude"] > 0.2:
            recommendations["component_suggestions"].append(
                "Consider reducing seasonality_prior_scale for regularization"
            )

        if len(recommendations["very_strong_seasonalities"]) > 2:
            recommendations["component_suggestions"].append(
                "Multiple strong seasonalities detected - consider feature selection"
            )

        # Generate best model parameters based on analysis
        recommendations["best_model_parameters"] = self._generate_best_model_parameters(
            summary, recommendations
        )

        # Identify unused seasonalities (potential components not currently used)
        recommendations["unused_seasonalities"] = self._identify_unused_seasonalities(
            summary
        )

        return recommendations

    def _generate_best_model_parameters(
        self, summary: Dict, recommendations: Dict
    ) -> Dict[str, Any]:
        """
        Generate optimal model parameters based on seasonality analysis.

        Args:
            summary: Seasonality summary
            recommendations: Current recommendations

        Returns:
            Dictionary of optimal model parameters
        """
        best_params = {
            "changepoint_prior_scale": 0.05,
            "seasonality_prior_scale": 10.0,
            "holidays_prior_scale": 10.0,
            "seasonality_mode": "multiplicative",
            "weekly_seasonality": True,
            "daily_seasonality": False,
            "include_quarterly_effects": True,
            "include_monthly_effects": True,
            "include_indian_holidays": True,
            "include_festival_seasons": True,
            "include_monsoon_effect": True,
            "n_changepoints": 25,
            "changepoint_range": 0.8,
        }

        # Adjust parameters based on seasonality strength
        if recommendations["regularization_needed"]:
            best_params["seasonality_prior_scale"] = 5.0
            best_params["holidays_prior_scale"] = 5.0
            best_params["changepoint_prior_scale"] = 0.01

        # Adjust based on mean magnitude
        if summary.get("mean_max_magnitude", 0) > 0.3:
            best_params["seasonality_prior_scale"] = 3.0
            best_params["changepoint_prior_scale"] = 0.005

        # Adjust based on number of strong seasonalities
        strong_count = len(recommendations["very_strong_seasonalities"])
        if strong_count > 2:
            best_params["seasonality_prior_scale"] = 2.0
            best_params["n_changepoints"] = 15

        # Adjust seasonality mode based on data characteristics
        if summary.get("mean_max_magnitude", 0) < 0.1:
            best_params["seasonality_mode"] = "additive"

        # Component-specific recommendations
        if "yearly" in recommendations["very_strong_seasonalities"]:
            best_params["include_indian_holidays"] = True
            best_params["include_festival_seasons"] = True

        if "quarterly" in recommendations["very_strong_seasonalities"]:
            best_params["include_quarterly_effects"] = True

        if "monthly" in recommendations["very_strong_seasonalities"]:
            best_params["include_monthly_effects"] = True

        # Disable weak components
        for weak_seasonality in recommendations["weak_seasonalities"]:
            if weak_seasonality == "weekly":
                best_params["weekly_seasonality"] = False
            elif weak_seasonality == "daily":
                best_params["daily_seasonality"] = False
            elif weak_seasonality == "monthly":
                best_params["include_monthly_effects"] = False
            elif weak_seasonality == "quarterly":
                best_params["include_quarterly_effects"] = False

        return best_params

    def _identify_unused_seasonalities(self, summary: Dict) -> List[str]:
        """
        Identify potential seasonality components that could be added.

        Args:
            summary: Seasonality summary

        Returns:
            List of unused seasonality components
        """
        used_seasonalities = [s["name"] for s in summary["ranked_seasonalities"]]
        all_possible_seasonalities = [
            "yearly",
            "quarterly",
            "monthly",
            "weekly",
            "daily",
        ]

        unused = []
        for seasonality in all_possible_seasonalities:
            if seasonality not in used_seasonalities:
                unused.append(seasonality)

        return unused

    def _log_seasonality_summary(self, summary: Dict, recommendations: Dict):
        """
        Log seasonality summary and recommendations.

        Args:
            summary: Seasonality summary
            recommendations: Recommendations
        """
        self.logger.info("=" * 80)
        self.logger.info("SEASONALITY STRENGTH SUMMARY")
        self.logger.info("=" * 80)

        self.logger.info("Seasonalities ranked by strength (max magnitude):")
        for i, seasonality in enumerate(summary["ranked_seasonalities"], 1):
            self.logger.info(
                f"   {i}. {seasonality['name']}: {seasonality['max_magnitude']:.6f} ({seasonality['strength']})"
            )

        self.logger.info(f"\nOverall Statistics:")
        self.logger.info(
            f"  Strongest Seasonality: {summary['strongest']['name']} ({summary['strongest']['max_magnitude']:.6f})"
        )
        self.logger.info(
            f"  Weakest Seasonality: {summary['weakest']['name']} ({summary['weakest']['max_magnitude']:.6f})"
        )
        self.logger.info(f"  Mean Max Magnitude: {summary['mean_max_magnitude']:.6f}")
        self.logger.info(f"  Std Max Magnitude: {summary['std_max_magnitude']:.6f}")

        self.logger.info(f"\nRecommendations:")
        if recommendations["very_strong_seasonalities"]:
            self.logger.info(
                f"  Very strong seasonalities (may need regularization): {recommendations['very_strong_seasonalities']}"
            )
        else:
            self.logger.info(f"  No very strong seasonalities detected")

        if recommendations["weak_seasonalities"]:
            self.logger.info(
                f"  Weak seasonalities (consider disabling): {recommendations['weak_seasonalities']}"
            )

        if recommendations["unused_seasonalities"]:
            self.logger.info(
                f"  Unused seasonalities (potential components): {recommendations['unused_seasonalities']}"
            )

        # Log best model parameters
        if recommendations.get("best_model_parameters"):
            self.logger.info(f"\nBest Model Parameters:")
            best_params = recommendations["best_model_parameters"]
            self.logger.info(
                f"  Changepoint Prior Scale: {best_params.get('changepoint_prior_scale', 'N/A')}"
            )
            self.logger.info(
                f"  Seasonality Prior Scale: {best_params.get('seasonality_prior_scale', 'N/A')}"
            )
            self.logger.info(
                f"  Holidays Prior Scale: {best_params.get('holidays_prior_scale', 'N/A')}"
            )
            self.logger.info(
                f"  Seasonality Mode: {best_params.get('seasonality_mode', 'N/A')}"
            )
            self.logger.info(
                f"  Weekly Seasonality: {best_params.get('weekly_seasonality', 'N/A')}"
            )
            self.logger.info(
                f"  Daily Seasonality: {best_params.get('daily_seasonality', 'N/A')}"
            )
            self.logger.info(
                f"  Include Quarterly Effects: {best_params.get('include_quarterly_effects', 'N/A')}"
            )
            self.logger.info(
                f"  Include Monthly Effects: {best_params.get('include_monthly_effects', 'N/A')}"
            )

        self.logger.info("=" * 80)

    def get_optimal_components(self, analysis_results: Dict) -> Dict[str, Any]:
        """
        Determine optimal seasonality components based on analysis.

        Args:
            analysis_results: Results from seasonality analysis

        Returns:
            Dictionary of optimal components and their configurations
        """
        optimal_components = {
            "selected_components": [],
            "regularization_settings": {},
            "component_details": {},
        }

        summary = analysis_results["summary"]
        recommendations = analysis_results["recommendations"]

        # Select components based on strength
        for seasonality in summary["ranked_seasonalities"]:
            if seasonality["strength"] in ["STRONG", "VERY_STRONG", "MODERATE"]:
                optimal_components["selected_components"].append(seasonality["name"])

                # Add component details
                seasonality_data = analysis_results["seasonalities"][
                    seasonality["name"]
                ]
                optimal_components["component_details"][seasonality["name"]] = {
                    "period": seasonality_data["period"],
                    "fourier_order": seasonality_data["fourier_order"],
                    "strength": seasonality_data["strength"],
                    "max_magnitude": seasonality_data["max_magnitude"],
                    "significant_terms": len(seasonality_data["significant_terms"]),
                }

        # Determine regularization settings
        if recommendations["regularization_needed"]:
            optimal_components["regularization_settings"] = {
                "seasonality_prior_scale": 5.0,  # Reduced from default 10.0
                "holidays_prior_scale": 5.0,  # Reduced from default 10.0
                "changepoint_prior_scale": 0.01,  # Reduced for more stable trends
            }
        else:
            optimal_components["regularization_settings"] = {
                "seasonality_prior_scale": 10.0,
                "holidays_prior_scale": 10.0,
                "changepoint_prior_scale": 0.05,
            }

        return optimal_components
