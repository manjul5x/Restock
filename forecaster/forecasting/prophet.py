"""
Prophet forecasting implementation.

This module provides Prophet-based forecasting with comprehensive Indian market
seasonality, holiday effects, and integration with the existing forecasting framework.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from datetime import date, timedelta
import warnings
from .base import BaseForecaster, ForecastingError, validate_forecast_parameters

# Try to import Prophet, handle gracefully if not available
try:
    from prophet import Prophet
    from prophet.diagnostics import cross_validation, performance_metrics

    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    warnings.warn("Prophet not available. Install with: pip install prophet")

# Try to import holidays for Indian holidays
try:
    import holidays

    HOLIDAYS_AVAILABLE = True
except ImportError:
    HOLIDAYS_AVAILABLE = False
    warnings.warn("Holidays package not available. Install with: pip install holidays")

# Import seasonality analyzer
from .seasonality_analyzer import SeasonalityAnalyzer


class ProphetForecaster(BaseForecaster):
    """
    Prophet forecaster with comprehensive Indian market seasonality and holiday effects.
    """

    def __init__(
        self,
        changepoint_range: float = 0.8,
        n_changepoints: int = 25,
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        holidays_prior_scale: float = 10.0,
        seasonality_mode: str = "multiplicative",
        weekly_seasonality: bool = True,
        daily_seasonality: bool = False,
        include_indian_holidays: bool = True,
        include_regional_holidays: bool = False,
        include_quarterly_effects: bool = True,
        include_monthly_effects: bool = True,
        include_festival_seasons: bool = True,
        include_monsoon_effect: bool = True,
        min_data_points: int = 10,
        window_length: Optional[int] = None,
    ):
        """
        Initialize Prophet forecaster

        Args:
            changepoint_range: Proportion of history in which trend changepoints will be estimated
            n_changepoints: Number of changepoints to be estimated
            changepoint_prior_scale: Flexibility of the changepoint
            seasonality_prior_scale: Flexibility of the seasonality
            holidays_prior_scale: Flexibility of the holiday effects
            seasonality_mode: 'additive' or 'multiplicative'
            weekly_seasonality: Whether to include weekly seasonality
            daily_seasonality: Whether to include daily seasonality
            include_indian_holidays: Whether to include Indian national holidays
            include_regional_holidays: Whether to include regional holidays
            include_quarterly_effects: Whether to include quarterly seasonality
            include_monthly_effects: Whether to include monthly seasonality
            include_festival_seasons: Whether to include festival season effects
            include_monsoon_effect: Whether to include monsoon season effects
            min_data_points: Minimum data points required
        """
        super().__init__(name="prophet")

        if not PROPHET_AVAILABLE:
            raise ImportError(
                "Prophet is not available. Install with: pip install prophet"
            )

        self.changepoint_range = changepoint_range
        self.n_changepoints = n_changepoints
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.holidays_prior_scale = holidays_prior_scale
        self.seasonality_mode = seasonality_mode
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.include_indian_holidays = include_indian_holidays
        self.include_regional_holidays = include_regional_holidays
        self.include_quarterly_effects = include_quarterly_effects
        self.include_monthly_effects = include_monthly_effects
        self.include_festival_seasons = include_festival_seasons
        self.include_monsoon_effect = include_monsoon_effect
        self.min_data_points = min_data_points
        self.window_length = window_length

        self.data = None
        self.model = None
        self.is_fitted = False
        self.holidays_df = None
        self.risk_period_days = None  # Will be set during fit
        self.fallback_forecaster = None  # Moving Average fallback

    def _determine_risk_period(self, data: pd.DataFrame, **kwargs) -> int:
        """
        Determine the risk period in days from data or parameters.

        Args:
            data: Input data
            **kwargs: Additional arguments that may contain risk_period

        Returns:
            Risk period in days
        """
        # First check if risk_period is provided in kwargs
        if "risk_period" in kwargs:
            risk_period = kwargs["risk_period"]
            if isinstance(risk_period, int):
                return risk_period
            elif isinstance(risk_period, str):
                # Try to parse as days
                try:
                    return int(risk_period)
                except ValueError:
                    pass

        # Check if risk_period_days is provided in kwargs
        if "risk_period_days" in kwargs:
            return int(kwargs["risk_period_days"])

        # Try to determine from data frequency
        if len(data) >= 2:
            dates = pd.to_datetime(data["date"])
            date_diffs = dates.diff().dropna()
            if len(date_diffs) > 0:
                # Get the most common interval
                most_common_interval = date_diffs.mode().iloc[0]
                if pd.notna(most_common_interval):
                    days = most_common_interval.days
                    if days > 0:
                        return days

        # Default to 14 days if we can't determine
        return 14

    def _create_indian_holidays(self) -> pd.DataFrame:
        """
        Create Indian holidays dataframe for Prophet

        Returns:
            DataFrame with holiday information
        """
        if not HOLIDAYS_AVAILABLE:
            return pd.DataFrame()

        holidays_list = []

        # National holidays
        if self.include_indian_holidays:
            indian_holidays = {
                "Republic Day": "2025-01-26",
                "Independence Day": "2025-08-15",
                "Gandhi Jayanti": "2025-10-02",
                "Republic Day": "2024-01-26",
                "Independence Day": "2024-08-15",
                "Gandhi Jayanti": "2024-10-02",
                "Republic Day": "2023-01-26",
                "Independence Day": "2023-08-15",
                "Gandhi Jayanti": "2023-10-02",
                "Republic Day": "2022-01-26",
                "Independence Day": "2022-08-15",
                "Gandhi Jayanti": "2022-10-02",
                "Republic Day": "2021-01-26",
                "Independence Day": "2021-08-15",
                "Gandhi Jayanti": "2021-10-02",
                "Republic Day": "2020-01-26",
                "Independence Day": "2020-08-15",
                "Gandhi Jayanti": "2020-10-02",
            }

            for holiday_name, holiday_date in indian_holidays.items():
                holidays_list.append(
                    {
                        "holiday": holiday_name,
                        "ds": pd.to_datetime(holiday_date),
                        "lower_window": -1,
                        "upper_window": 1,
                    }
                )

        # Festival seasons (extended periods)
        if self.include_festival_seasons:
            festival_periods = [
                # Diwali period (typically October-November)
                {"name": "Diwali", "start": "2025-10-20", "end": "2025-11-15"},
                {"name": "Diwali", "start": "2024-10-20", "end": "2024-11-15"},
                {"name": "Diwali", "start": "2023-10-20", "end": "2023-11-15"},
                {"name": "Diwali", "start": "2022-10-20", "end": "2022-11-15"},
                {"name": "Diwali", "start": "2021-10-20", "end": "2021-11-15"},
                {"name": "Diwali", "start": "2020-10-20", "end": "2020-11-15"},
                # Holi period (typically March)
                {"name": "Holi", "start": "2025-03-20", "end": "2025-03-30"},
                {"name": "Holi", "start": "2024-03-20", "end": "2024-03-30"},
                {"name": "Holi", "start": "2023-03-20", "end": "2023-03-30"},
                {"name": "Holi", "start": "2022-03-20", "end": "2022-03-30"},
                {"name": "Holi", "start": "2021-03-20", "end": "2021-03-30"},
                {"name": "Holi", "start": "2020-03-20", "end": "2020-03-30"},
                # Eid periods
                {"name": "Eid_al_Fitr", "start": "2025-04-10", "end": "2025-04-12"},
                {"name": "Eid_al_Fitr", "start": "2024-04-10", "end": "2024-04-12"},
                {"name": "Eid_al_Fitr", "start": "2023-04-21", "end": "2023-04-23"},
                {"name": "Eid_al_Fitr", "start": "2022-05-02", "end": "2022-05-04"},
                {"name": "Eid_al_Fitr", "start": "2021-05-13", "end": "2021-05-15"},
                {"name": "Eid_al_Fitr", "start": "2020-05-24", "end": "2020-05-26"},
                # Christmas and New Year
                {
                    "name": "Christmas_NewYear",
                    "start": "2025-12-20",
                    "end": "2026-01-05",
                },
                {
                    "name": "Christmas_NewYear",
                    "start": "2024-12-20",
                    "end": "2025-01-05",
                },
                {
                    "name": "Christmas_NewYear",
                    "start": "2023-12-20",
                    "end": "2024-01-05",
                },
                {
                    "name": "Christmas_NewYear",
                    "start": "2022-12-20",
                    "end": "2023-01-05",
                },
                {
                    "name": "Christmas_NewYear",
                    "start": "2021-12-20",
                    "end": "2022-01-05",
                },
                {
                    "name": "Christmas_NewYear",
                    "start": "2020-12-20",
                    "end": "2021-01-05",
                },
            ]

            for festival in festival_periods:
                start_date = pd.to_datetime(festival["start"])
                end_date = pd.to_datetime(festival["end"])
                current_date = start_date

                while current_date <= end_date:
                    holidays_list.append(
                        {
                            "holiday": festival["name"],
                            "ds": current_date,
                            "lower_window": 0,
                            "upper_window": 0,
                        }
                    )
                    current_date += timedelta(days=1)

        # Monsoon season effects
        if self.include_monsoon_effect:
            monsoon_periods = [
                # Monsoon season (June to September)
                {"name": "Monsoon", "start": "2025-06-01", "end": "2025-09-30"},
                {"name": "Monsoon", "start": "2024-06-01", "end": "2024-09-30"},
                {"name": "Monsoon", "start": "2023-06-01", "end": "2023-09-30"},
                {"name": "Monsoon", "start": "2022-06-01", "end": "2022-09-30"},
                {"name": "Monsoon", "start": "2021-06-01", "end": "2021-09-30"},
                {"name": "Monsoon", "start": "2020-06-01", "end": "2020-09-30"},
            ]

            for monsoon in monsoon_periods:
                start_date = pd.to_datetime(monsoon["start"])
                end_date = pd.to_datetime(monsoon["end"])
                current_date = start_date

                while current_date <= end_date:
                    holidays_list.append(
                        {
                            "holiday": monsoon["name"],
                            "ds": current_date,
                            "lower_window": 0,
                            "upper_window": 0,
                        }
                    )
                    current_date += timedelta(days=1)

        if holidays_list:
            return pd.DataFrame(holidays_list)
        else:
            return pd.DataFrame()

    def _prepare_data_for_prophet(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data in Prophet format (ds for dates, y for values)

        Args:
            data: Input data with 'date' and 'demand' columns

        Returns:
            DataFrame in Prophet format
        """
        prophet_data = data.copy()

        # Ensure date column is datetime
        prophet_data["ds"] = pd.to_datetime(prophet_data["date"])

        # Ensure demand column is numeric and handle any missing values
        prophet_data["y"] = pd.to_numeric(prophet_data["demand"], errors="coerce")

        # Remove rows with missing values
        prophet_data = prophet_data.dropna(subset=["y"])

        # Sort by date
        prophet_data = prophet_data.sort_values("ds").reset_index(drop=True)

        return prophet_data[["ds", "y"]]

    def _create_prophet_model(self) -> Prophet:
        """
        Create Prophet model with configured parameters

        Returns:
            Configured Prophet model
        """
        model = Prophet(
            changepoint_range=self.changepoint_range,
            n_changepoints=self.n_changepoints,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            holidays_prior_scale=self.holidays_prior_scale,
            seasonality_mode=self.seasonality_mode,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
        )

        # Add holidays if available
        if (
            self.include_indian_holidays
            or self.include_festival_seasons
            or self.include_monsoon_effect
        ):
            holidays_df = self._create_indian_holidays()
            if not holidays_df.empty:
                model.add_country_holidays(country_name="IN")
                model.holidays = holidays_df

        # Add custom seasonalities
        if self.include_monthly_effects:
            model.add_seasonality(name="monthly", period=30.5, fourier_order=5)

        if self.include_quarterly_effects:
            model.add_seasonality(name="quarterly", period=91.25, fourier_order=8)

        return model

    def fit(self, data: pd.DataFrame, **kwargs) -> "ProphetForecaster":
        """
        Fit the Prophet model to the data

        Args:
            data: DataFrame with 'date' and 'demand' columns
            **kwargs: Additional arguments

        Returns:
            Self for chaining
        """
        # Validate data
        self.validate_data(data)

        # Prepare data for Prophet (this data is already aggregated into buckets)
        prophet_data = self._prepare_data_for_prophet(data)

        # Apply rolling window if specified (after bucketing)
        if self.window_length is not None and len(prophet_data) > self.window_length:
            # Sort by date and take the most recent window_length data points
            prophet_data = (
                prophet_data.sort_values("ds")
                .tail(self.window_length)
                .reset_index(drop=True)
            )
        # If window_length is None, use all available data (no windowing applied)

        # Check if we have enough data for Prophet
        if len(prophet_data) < self.min_data_points:
            print(
                f"⚠️  Insufficient data points for Prophet ({len(prophet_data)} < {self.min_data_points}). Falling back to Moving Average."
            )
            return self._fallback_to_moving_average(data, **kwargs)

        # Determine risk period
        self.risk_period_days = self._determine_risk_period(data, **kwargs)

        # Create and fit model
        self.model = self._create_prophet_model()

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model.fit(prophet_data)
        except Exception as e:
            print(
                f"⚠️  Failed to fit Prophet model: {str(e)}. Falling back to Moving Average."
            )
            return self._fallback_to_moving_average(data, **kwargs)

        self.data = data.copy()
        if self.window_length is not None and len(data) > self.window_length:
            # Update self.data to reflect the rolling window
            self.data = (
                self.data.sort_values("date")
                .tail(self.window_length)
                .reset_index(drop=True)
            )
        # If window_length is None, keep all data (no windowing applied)

        self.is_fitted = True

        # Perform seasonality analysis if model is fitted successfully
        if self.is_fitted and self.model is not None:
            self._perform_seasonality_analysis(data, prophet_data)

        return self

    def _perform_seasonality_analysis(
        self, original_data: pd.DataFrame, fitted_data: pd.DataFrame
    ):
        """
        Perform comprehensive seasonality analysis on the fitted Prophet model.

        Args:
            original_data: Original input data
            fitted_data: Data used for fitting (after preprocessing)
        """
        try:
            # Initialize seasonality analyzer
            analyzer = SeasonalityAnalyzer()

            # Perform analysis
            analysis_results = analyzer.analyze_seasonality_components(
                original_data, self.model, fitted_data
            )

            # Store analysis results for later use
            self.seasonality_analysis = analysis_results

            # Get optimal components and apply regularization if needed
            optimal_components = analyzer.get_optimal_components(analysis_results)

            # Apply regularization settings if recommended
            if optimal_components["regularization_settings"]:
                self._apply_regularization_settings(
                    optimal_components["regularization_settings"]
                )

        except Exception as e:
            print(f"⚠️  Seasonality analysis failed: {str(e)}")
            self.seasonality_analysis = None

    def _apply_regularization_settings(self, regularization_settings: Dict):
        """
        Apply regularization settings to the model if needed.

        Args:
            regularization_settings: Dictionary of regularization parameters
        """
        try:
            # Update model parameters for regularization
            if "seasonality_prior_scale" in regularization_settings:
                self.seasonality_prior_scale = regularization_settings[
                    "seasonality_prior_scale"
                ]
                print(
                    f"Applied regularization: seasonality_prior_scale = {self.seasonality_prior_scale}"
                )

            if "holidays_prior_scale" in regularization_settings:
                self.holidays_prior_scale = regularization_settings[
                    "holidays_prior_scale"
                ]
                print(
                    f"Applied regularization: holidays_prior_scale = {self.holidays_prior_scale}"
                )

            if "changepoint_prior_scale" in regularization_settings:
                self.changepoint_prior_scale = regularization_settings[
                    "changepoint_prior_scale"
                ]
                print(
                    f"Applied regularization: changepoint_prior_scale = {self.changepoint_prior_scale}"
                )

        except Exception as e:
            print(f"⚠️  Failed to apply regularization settings: {str(e)}")

    def get_seasonality_analysis(self) -> Optional[Dict]:
        """
        Get the seasonality analysis results.

        Returns:
            Seasonality analysis results or None if not available
        """
        return getattr(self, "seasonality_analysis", None)

    def _fallback_to_moving_average(
        self, data: pd.DataFrame, **kwargs
    ) -> "ProphetForecaster":
        """
        Fallback to Moving Average when Prophet cannot be used

        Args:
            data: Input data
            **kwargs: Additional arguments

        Returns:
            Self for chaining
        """
        print("🔄 Switching to Moving Average forecaster...")

        # Import here to avoid circular imports
        from .moving_average import MovingAverageForecaster

        # Create Moving Average forecaster with same window_length
        self.fallback_forecaster = MovingAverageForecaster(
            window_length=self.window_length, horizon=1  # Default horizon for fallback
        )

        # Fit the fallback forecaster
        self.fallback_forecaster.fit(data, **kwargs)

        # Store the data that was actually used for fitting
        self.data = self.fallback_forecaster.data.copy()
        self.risk_period_days = self.fallback_forecaster.risk_period_days
        self.is_fitted = True

        print("✅ Successfully switched to Moving Average forecaster")
        return self

    def forecast(self, steps: Optional[int] = None, **kwargs) -> pd.Series:
        """
        Generate forecast for specified number of steps

        Args:
            steps: Number of steps to forecast (if None, uses default horizon)
            **kwargs: Additional arguments

        Returns:
            Series with forecast values
        """
        if not self.is_fitted:
            raise ForecastingError("Model must be fitted before forecasting")

        # Check if we're using fallback forecaster
        if self.fallback_forecaster is not None:
            print("📊 Using Moving Average fallback for forecasting")
            return self.fallback_forecaster.forecast(steps, **kwargs)

        if steps is None:
            # Use the risk period determined during fit, or default to 14 days
            steps = self.risk_period_days if self.risk_period_days is not None else 14

        # Create future dataframe
        last_date = self.data["date"].max()
        if isinstance(last_date, pd.Timestamp):
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1), periods=steps, freq="D"
            )
        else:
            future_dates = pd.date_range(
                start=last_date + timedelta(days=1), periods=steps, freq="D"
            )

        future_df = pd.DataFrame({"ds": future_dates})

        # Generate forecast
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                forecast = self.model.predict(future_df)
        except Exception as e:
            print(
                f"⚠️  Failed to generate Prophet forecast: {str(e)}. Using Moving Average fallback."
            )
            # Create fallback forecaster on the fly if not already created
            if self.fallback_forecaster is None:
                self._fallback_to_moving_average(self.data, **kwargs)
            return self.fallback_forecaster.forecast(steps, **kwargs)

        # Extract forecast values
        forecast_values = forecast["yhat"].values

        # Handle negative values (demand can't be negative)
        forecast_values = np.maximum(forecast_values, 0)

        return pd.Series(forecast_values, index=future_dates)

    def update(self, new_data: pd.DataFrame) -> "ProphetForecaster":
        """
        Update the model with new data

        Args:
            new_data: New data to add

        Returns:
            Self for chaining
        """
        if self.data is None:
            return self.fit(new_data)

        # Combine existing and new data
        combined_data = pd.concat([self.data, new_data], ignore_index=True)
        combined_data = combined_data.drop_duplicates(subset=["date"]).sort_values(
            "date"
        )

        return self.fit(combined_data)

    def get_parameters(self) -> Dict:
        """
        Get model parameters

        Returns:
            Dictionary of parameters
        """
        params = {
            "changepoint_range": self.changepoint_range,
            "n_changepoints": self.n_changepoints,
            "changepoint_prior_scale": self.changepoint_prior_scale,
            "seasonality_prior_scale": self.seasonality_prior_scale,
            "holidays_prior_scale": self.holidays_prior_scale,
            "seasonality_mode": self.seasonality_mode,
            "weekly_seasonality": self.weekly_seasonality,
            "daily_seasonality": self.daily_seasonality,
            "include_indian_holidays": self.include_indian_holidays,
            "include_regional_holidays": self.include_regional_holidays,
            "include_quarterly_effects": self.include_quarterly_effects,
            "include_monthly_effects": self.include_monthly_effects,
            "include_festival_seasons": self.include_festival_seasons,
            "include_monsoon_effect": self.include_monsoon_effect,
            "min_data_points": self.min_data_points,
            "window_length": self.window_length,
            "risk_period_days": self.risk_period_days,
            "is_fitted": self.is_fitted,
            "using_fallback": self.fallback_forecaster is not None,
        }

        # Add fallback forecaster parameters if using fallback
        if self.fallback_forecaster is not None:
            params["fallback_parameters"] = self.fallback_forecaster.get_parameters()

        return params

    def set_parameters(self, parameters: Dict) -> "ProphetForecaster":
        """
        Set model parameters

        Args:
            parameters: Dictionary of parameters

        Returns:
            Self for chaining
        """
        if "changepoint_range" in parameters:
            self.changepoint_range = parameters["changepoint_range"]
        if "n_changepoints" in parameters:
            self.n_changepoints = parameters["n_changepoints"]
        if "changepoint_prior_scale" in parameters:
            self.changepoint_prior_scale = parameters["changepoint_prior_scale"]
        if "seasonality_prior_scale" in parameters:
            self.seasonality_prior_scale = parameters["seasonality_prior_scale"]
        if "holidays_prior_scale" in parameters:
            self.holidays_prior_scale = parameters["holidays_prior_scale"]
        if "seasonality_mode" in parameters:
            self.seasonality_mode = parameters["seasonality_mode"]
        if "weekly_seasonality" in parameters:
            self.weekly_seasonality = parameters["weekly_seasonality"]
        if "daily_seasonality" in parameters:
            self.daily_seasonality = parameters["daily_seasonality"]
        if "include_indian_holidays" in parameters:
            self.include_indian_holidays = parameters["include_indian_holidays"]
        if "include_regional_holidays" in parameters:
            self.include_regional_holidays = parameters["include_regional_holidays"]
        if "include_quarterly_effects" in parameters:
            self.include_quarterly_effects = parameters["include_quarterly_effects"]
        if "include_monthly_effects" in parameters:
            self.include_monthly_effects = parameters["include_monthly_effects"]
        if "include_festival_seasons" in parameters:
            self.include_festival_seasons = parameters["include_festival_seasons"]
        if "include_monsoon_effect" in parameters:
            self.include_monsoon_effect = parameters["include_monsoon_effect"]
        if "min_data_points" in parameters:
            self.min_data_points = parameters["min_data_points"]
        if "window_length" in parameters:
            self.window_length = parameters["window_length"]
        if "risk_period_days" in parameters:
            self.risk_period_days = parameters["risk_period_days"]

        return self


def create_prophet_forecaster(parameters: Dict) -> ProphetForecaster:
    """
    Create a Prophet forecaster with given parameters

    Args:
        parameters: Dictionary of parameters

    Returns:
        ProphetForecaster instance
    """
    validate_forecast_parameters(parameters)

    forecaster = ProphetForecaster(
        changepoint_range=parameters.get("changepoint_range", 0.8),
        n_changepoints=parameters.get("n_changepoints", 25),
        changepoint_prior_scale=parameters.get("changepoint_prior_scale", 0.05),
        seasonality_prior_scale=parameters.get("seasonality_prior_scale", 10.0),
        holidays_prior_scale=parameters.get("holidays_prior_scale", 10.0),
        seasonality_mode=parameters.get("seasonality_mode", "multiplicative"),
        weekly_seasonality=parameters.get("weekly_seasonality", True),
        daily_seasonality=parameters.get("daily_seasonality", False),
        include_indian_holidays=parameters.get("include_indian_holidays", True),
        include_regional_holidays=parameters.get("include_regional_holidays", False),
        include_quarterly_effects=parameters.get("include_quarterly_effects", True),
        include_monthly_effects=parameters.get("include_monthly_effects", True),
        include_festival_seasons=parameters.get("include_festival_seasons", True),
        include_monsoon_effect=parameters.get("include_monsoon_effect", True),
        min_data_points=parameters.get("min_data_points", 10),
        window_length=parameters.get("window_length"),
    )

    # Set risk period if provided
    if "risk_period_days" in parameters:
        forecaster.risk_period_days = parameters["risk_period_days"]

    return forecaster


def forecast_product_location(
    data: pd.DataFrame,
    product_id: str,
    location_id: str,
    parameters: Dict,
    forecast_date: Optional[date] = None,
) -> Dict:
    """
    Forecast demand for a specific product-location combination using Prophet

    Args:
        data: Full demand dataset
        product_id: Product ID to forecast
        location_id: Location ID to forecast
        parameters: Forecasting parameters
        forecast_date: Date to forecast from (defaults to latest date in data)

    Returns:
        Dictionary with forecast results
    """
    # Filter data for specific product-location
    product_data = data[
        (data["product_id"] == product_id) & (data["location_id"] == location_id)
    ].copy()

    if len(product_data) == 0:
        raise ForecastingError(
            f"No data found for product {product_id} at location {location_id}"
        )

    # Determine forecast date
    if forecast_date is None:
        forecast_date = product_data["date"].max()

    # Filter data up to forecast date
    historical_data = product_data[product_data["date"] <= forecast_date].copy()

    if len(historical_data) < parameters.get("min_data_points", 10):
        raise ForecastingError(f"Insufficient historical data for forecasting")

    # Create and fit forecaster
    forecaster = create_prophet_forecaster(parameters)

    # Pass risk period information if available
    fit_kwargs = {}
    if "risk_period" in parameters:
        fit_kwargs["risk_period"] = parameters["risk_period"]
    elif "risk_period_days" in parameters:
        fit_kwargs["risk_period_days"] = parameters["risk_period_days"]

    forecaster.fit(historical_data, **fit_kwargs)

    # Generate forecast
    forecast_values = forecaster.forecast()

    # Create forecast dates
    last_date = historical_data["date"].max()
    forecast_dates = []
    for i in range(len(forecast_values)):
        if isinstance(last_date, pd.Timestamp):
            forecast_date = last_date + pd.Timedelta(days=i)
        else:
            forecast_date = last_date + timedelta(days=i)
        forecast_dates.append(forecast_date)

    return {
        "product_id": product_id,
        "location_id": location_id,
        "forecast_date": forecast_date,
        "forecast_values": forecast_values.tolist(),
        "forecast_dates": [
            d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)
            for d in forecast_dates
        ],
        "model": "prophet",
        "parameters": forecaster.get_parameters(),
    }
