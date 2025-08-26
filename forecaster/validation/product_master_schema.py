"""
Schema definitions for product master data.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, Literal, List
from datetime import date
import pandas as pd


class ProductMasterRecord(BaseModel):
    """Schema for a single product master record"""

    product_id: str = Field(..., description="Unique product identifier")
    location_id: str = Field(..., description="Unique location identifier")
    product_category: str = Field(..., description="Product category")
    demand_frequency: Literal["d", "w", "m"] = Field(
        ..., description="Demand frequency: 'd'=daily, 'w'=weekly, 'm'=monthly"
    )
    risk_period: int = Field(
        ..., gt=0, description="Risk period as integer multiple of demand frequency"
    )
    outlier_method: Optional[str] = Field(
        "iqr",
        description="Outlier detection method: 'iqr', 'zscore', 'mad', 'rolling', 'no'",
    )
    outlier_threshold: Optional[float] = Field(
        1.5, description="Outlier detection threshold"
    )
    forecast_window_length: int = Field(
        ..., description="Forecasting window length in days"
    )
    forecast_horizon: int = Field(
        ..., description="Forecasting horizon in days"
    )
    forecast_methods: Optional[str] = Field(
        "moving_average",
        description="Comma-separated forecasting methods: 'moving_average', 'prophet', 'arima'",
    )
    distribution: Optional[str] = Field(
        "kde", description="Safety stock distribution type: 'kde', 'normal'"
    )
    service_level: Optional[float] = Field(
        0.95, ge=0.0, le=1.0, description="Service level percentage (0.0 to 1.0)"
    )
    ss_window_length: Optional[int] = Field(
        180,
        gt=0,
        description="Rolling window length for safety stock calculation in demand frequency units",
    )
    leadtime: int = Field(..., gt=0, description="Lead time in demand frequency units")
    inventory_cost: Optional[float] = Field(
        0.0, ge=0.0, description="Unit cost of inventory"
    )
    moq: Optional[float] = Field(
        0.0, ge=0.0, description="Minimum order quantity"
    )
    min_safety_stock: Optional[float] = Field(
        0.0, ge=0.0, description="Minimum safety stock level (cannot be negative)"
    )
    sunset_date: Optional[date] = Field(
        None, description="Date when product is sunset (empty string = None = not sunset)"
    )

    @validator("risk_period")
    def validate_risk_period(cls, v, values):
        """Validate risk period based on frequency"""
        frequency = values.get("demand_frequency")
        if frequency == "d" and v > 365:
            raise ValueError("Daily risk period cannot exceed 365 days")
        elif frequency == "w" and v > 52:
            raise ValueError("Weekly risk period cannot exceed 52 weeks")
        elif frequency == "m" and v > 12:
            raise ValueError("Monthly risk period cannot exceed 12 months")
        return v

    @validator("forecast_methods")
    def validate_forecast_methods(cls, v):
        """Validate forecast methods are valid"""
        if v is None:
            return "moving_average"
        
        valid_methods = {"moving_average", "prophet", "arima"}
        methods = [method.strip() for method in v.split(",")]
        
        invalid_methods = set(methods) - valid_methods
        if invalid_methods:
            raise ValueError(f"Invalid forecast methods: {invalid_methods}")
        
        return v

    def get_forecast_methods_list(self) -> List[str]:
        """Get list of forecast methods"""
        if not self.forecast_methods:
            return ["moving_average"]
        return [method.strip() for method in self.forecast_methods.split(",")]

    @validator("min_safety_stock", pre=True)
    def validate_min_safety_stock(cls, v):
        """Handle empty strings and convert to float"""
        if v == "" or v is None:
            return 0.0
        if isinstance(v, str):
            # Handle whitespace-only strings
            if v.strip() == "":
                return 0.0
            try:
                return float(v)
            except ValueError:
                raise ValueError(f"Invalid min_safety_stock value: {v}")
        return v

    @validator("sunset_date", pre=True)
    def validate_sunset_date(cls, v):
        """Handle empty strings and convert to date or None"""
        if v == "" or v is None:
            return None
        if isinstance(v, str):
            # Handle whitespace-only strings
            if v.strip() == "":
                return None
            try:
                # Try to parse the date string
                from datetime import datetime
                return datetime.strptime(v.strip(), "%Y-%m-%d").date()
            except ValueError:
                raise ValueError(f"Invalid sunset_date format: {v}. Expected YYYY-MM-DD or empty string.")
        return v


class ProductMasterSchema:
    """Schema validation and conversion utilities for product master data"""

    REQUIRED_COLUMNS = [
        "product_id",
        "location_id",
        "product_category",
        "demand_frequency",
        "risk_period",
        "leadtime",
        "inventory_cost",
        "moq",
    ]
    VALID_FREQUENCIES = ["d", "w", "m"]
    VALID_FORECAST_METHODS = {"moving_average", "prophet", "arima"}

    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> bool:
        """Validate that a dataframe matches the required schema"""
        # Check required columns exist
        missing_cols = set(ProductMasterSchema.REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Check data types
        if not pd.api.types.is_object_dtype(df["product_id"]):
            raise ValueError("'product_id' column must be string type")

        if not pd.api.types.is_object_dtype(df["location_id"]):
            raise ValueError("'location_id' column must be string type")

        if not pd.api.types.is_object_dtype(df["product_category"]):
            raise ValueError("'product_category' column must be string type")

        if not pd.api.types.is_object_dtype(df["demand_frequency"]):
            raise ValueError("'demand_frequency' column must be string type")

        if not pd.api.types.is_numeric_dtype(df["risk_period"]):
            raise ValueError("'risk_period' column must be numeric")

        if not pd.api.types.is_numeric_dtype(df["leadtime"]):
            raise ValueError("'leadtime' column must be numeric")

        # Check for valid frequencies
        invalid_frequencies = set(df["demand_frequency"].unique()) - set(
            ProductMasterSchema.VALID_FREQUENCIES
        )
        if invalid_frequencies:
            raise ValueError(f"Invalid demand frequencies: {invalid_frequencies}")

        # Check for positive risk periods
        if (df["risk_period"] <= 0).any():
            raise ValueError("Risk period values must be positive")

        # Check for positive leadtimes
        if (df["leadtime"] <= 0).any():
            raise ValueError("Leadtime values must be positive")

        # Check for non-negative inventory costs
        if "inventory_cost" in df.columns and (df["inventory_cost"] < 0).any():
            raise ValueError("Inventory cost values must be non-negative")

        # Check for non-negative MOQ values
        if "moq" in df.columns and (df["moq"] < 0).any():
            raise ValueError("MOQ values must be non-negative")

        # Check for non-negative min_safety_stock values
        if "min_safety_stock" in df.columns and (df["min_safety_stock"] < 0).any():
            raise ValueError("Minimum safety stock values must be non-negative")

        # Check for reasonable risk period limits
        daily_risk = df[df["demand_frequency"] == "d"]["risk_period"]
        if (daily_risk > 365).any():
            raise ValueError("Daily risk period cannot exceed 365 days")

        weekly_risk = df[df["demand_frequency"] == "w"]["risk_period"]
        if (weekly_risk > 52).any():
            raise ValueError("Weekly risk period cannot exceed 52 weeks")

        monthly_risk = df[df["demand_frequency"] == "m"]["risk_period"]
        if (monthly_risk > 12).any():
            raise ValueError("Monthly risk period cannot exceed 12 months")

        # Validate forecast methods if present
        if "forecast_methods" in df.columns:
            for methods_str in df["forecast_methods"].dropna():
                methods = [method.strip() for method in methods_str.split(",")]
                invalid_methods = set(methods) - ProductMasterSchema.VALID_FORECAST_METHODS
                if invalid_methods:
                    raise ValueError(f"Invalid forecast methods: {invalid_methods}")

        # Validate sunset_date if present
        if "sunset_date" in df.columns:
            for i, sunset_val in enumerate(df["sunset_date"]):
                if pd.notna(sunset_val) and isinstance(sunset_val, str) and sunset_val.strip() != "":
                    try:
                        from datetime import datetime
                        datetime.strptime(sunset_val.strip(), "%Y-%m-%d")
                    except ValueError:
                        raise ValueError(f"Invalid sunset_date format at row {i}: '{sunset_val}'. Expected YYYY-MM-DD or empty string.")

        return True

    @staticmethod
    def standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Convert dataframe to standard format"""
        # Ensure string columns are string
        df["product_id"] = df["product_id"].astype(str)
        df["location_id"] = df["location_id"].astype(str)
        df["product_category"] = df["product_category"].astype(str)
        df["demand_frequency"] = df["demand_frequency"].astype(str)

        # Ensure numeric columns are integer
        df["risk_period"] = df["risk_period"].astype(int)
        df["leadtime"] = df["leadtime"].astype(int)

        # Handle optional outlier columns
        if "outlier_method" not in df.columns:
            df["outlier_method"] = "iqr"
        else:
            df["outlier_method"] = df["outlier_method"].fillna("iqr").astype(str)

        if "outlier_threshold" not in df.columns:
            df["outlier_threshold"] = 1.5
        else:
            df["outlier_threshold"] = df["outlier_threshold"].fillna(1.5).astype(float)

        # Ensure forecast columns are integer
        if "forecast_window_length" in df.columns:
            df["forecast_window_length"] = df["forecast_window_length"].astype(int)

        if "forecast_horizon" in df.columns:
            df["forecast_horizon"] = df["forecast_horizon"].astype(int)

        # Handle optional forecast methods column (support both old and new format)
        if "forecast_methods" not in df.columns:
            if "forecast_method" in df.columns:
                # Migrate from old single method to new multiple methods format
                df["forecast_methods"] = df["forecast_method"].fillna("moving_average")
                df = df.drop(columns=["forecast_method"])
            else:
                df["forecast_methods"] = "moving_average"
        else:
            df["forecast_methods"] = df["forecast_methods"].fillna("moving_average").astype(str)

        # Handle optional safety stock columns
        if "distribution" not in df.columns:
            df["distribution"] = "kde"
        else:
            df["distribution"] = df["distribution"].fillna("kde").astype(str)

        if "service_level" not in df.columns:
            df["service_level"] = 0.95
        else:
            df["service_level"] = df["service_level"].fillna(0.95).astype(float)

        if "ss_window_length" not in df.columns:
            df["ss_window_length"] = 180
        else:
            df["ss_window_length"] = df["ss_window_length"].fillna(180).astype(int)

        # Handle optional MOQ column
        if "moq" not in df.columns:
            df["moq"] = 1.0
        else:
            df["moq"] = df["moq"].fillna(1.0).astype(float)

        # Handle optional min_safety_stock column
        if "min_safety_stock" not in df.columns:
            df["min_safety_stock"] = 0.0
        else:
            # Handle empty strings and whitespace by cleaning the data
            def clean_min_safety_stock(val):
                if pd.isna(val):
                    return 0.0
                if isinstance(val, str):
                    if val.strip() == "":
                        return 0.0
                    try:
                        return float(val)
                    except ValueError:
                        return 0.0
                return val
            
            df["min_safety_stock"] = df["min_safety_stock"].apply(clean_min_safety_stock).astype(float)

        # Handle optional sunset_date column
        if "sunset_date" not in df.columns:
            df["sunset_date"] = None
        else:
            # Handle empty strings and convert to dates or None
            def clean_sunset_date(val):
                if pd.isna(val):
                    return None
                if isinstance(val, str):
                    if val.strip() == "":
                        return None
                    try:
                        from datetime import datetime
                        return datetime.strptime(val.strip(), "%Y-%m-%d").date()
                    except ValueError:
                        return None  # Invalid date format becomes None
                return val
            
            df["sunset_date"] = df["sunset_date"].apply(clean_sunset_date)

        # Sort for consistency
        df = df.sort_values(
            ["product_id", "location_id", "demand_frequency"]
        ).reset_index(drop=True)

        return df

    @staticmethod
    def expand_product_master_for_methods(df: pd.DataFrame) -> pd.DataFrame:
        """
        Expand product master to create separate rows for each forecast method.
        This creates the product × location × method combinations.
        """
        expanded_rows = []
        
        for _, row in df.iterrows():
            methods = [method.strip() for method in row["forecast_methods"].split(",")]
            
            for method in methods:
                new_row = row.copy()
                new_row["forecast_method"] = method  # Single method for this row
                expanded_rows.append(new_row)
        
        expanded_df = pd.DataFrame(expanded_rows)
        return expanded_df.reset_index(drop=True)

    @staticmethod
    def get_risk_period_days(frequency: str, risk_period: int) -> int:
        """
        Convert frequency and risk period to total days

        Args:
            frequency: 'd', 'w', or 'm'
            risk_period: Integer multiple of frequency

        Returns:
            Total days
        """
        if frequency == "d":
            return risk_period
        elif frequency == "w":
            return risk_period * 7
        elif frequency == "m":
            return risk_period * 30  # Approximate
        else:
            raise ValueError(f"Invalid frequency: {frequency}")

    @staticmethod
    def get_frequency_description(frequency: str) -> str:
        """Get human-readable frequency description"""
        descriptions = {"d": "Daily", "w": "Weekly", "m": "Monthly"}
        return descriptions.get(frequency, "Unknown")
