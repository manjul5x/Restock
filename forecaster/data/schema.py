from pydantic import BaseModel, Field
from typing import Optional
import datetime
import pandas as pd


class DemandRecord(BaseModel):
    """Schema for a single demand record"""

    product_id: str = Field(..., description="Unique product identifier")
    product_category: str = Field(..., description="Product category")
    location_id: str = Field(..., description="Unique location identifier")
    date: datetime.date = Field(..., description="Date of the demand record")
    demand: float = Field(..., ge=0, description="Demand quantity")
    stock_level: float = Field(
        ..., ge=0, description="Stock level at the end of the day"
    )
    incoming_inventory: Optional[float] = Field(
        None, ge=0, description="Incoming inventory quantity"
    )

    class Config:
        json_encoders = {datetime.date: lambda v: v.isoformat()}


class DemandSchema:
    """Schema validation and conversion utilities for demand data"""

    REQUIRED_COLUMNS = [
        "product_id",
        "product_category",
        "location_id",
        "date",
        "demand",
        "stock_level",
    ]
    OPTIONAL_COLUMNS = ["incoming_inventory"]

    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> bool:
        """Validate that a dataframe matches the required schema"""
        # Check required columns exist
        missing_cols = set(DemandSchema.REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Check data types - convert to date if needed
        if not pd.api.types.is_datetime64_any_dtype(df["date"]):
            # Try to convert to datetime first, then to date
            try:
                df["date"] = pd.to_datetime(df["date"]).dt.date
            except Exception as e:
                raise ValueError(f"'date' column cannot be converted to date type: {e}")

        if not pd.api.types.is_numeric_dtype(df["demand"]):
            raise ValueError("'demand' column must be numeric")

        if not pd.api.types.is_numeric_dtype(df["stock_level"]):
            raise ValueError("'stock_level' column must be numeric")

        # Check optional columns if they exist
        if "incoming_inventory" in df.columns:
            if not pd.api.types.is_numeric_dtype(df["incoming_inventory"]):
                raise ValueError("'incoming_inventory' column must be numeric")
            if (df["incoming_inventory"] < 0).any():
                raise ValueError("Incoming inventory values cannot be negative")

        # Check for negative values
        if (df["demand"] < 0).any():
            raise ValueError("Demand values cannot be negative")

        if (df["stock_level"] < 0).any():
            raise ValueError("Stock level values cannot be negative")

        return True

    @staticmethod
    def standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Convert dataframe to standard format"""
        # Ensure date column is date (not datetime)
        df["date"] = pd.to_datetime(df["date"]).dt.date

        # Ensure numeric columns are float
        df["demand"] = df["demand"].astype(float)
        df["stock_level"] = df["stock_level"].astype(float)

        # Handle optional incoming_inventory column
        if "incoming_inventory" in df.columns:
            df["incoming_inventory"] = df["incoming_inventory"].astype(float)
        else:
            # Add incoming_inventory column with zeros if it doesn't exist
            df["incoming_inventory"] = 0.0

        # Handle negative demand values (convert to 0)
        negative_demand_count = (df["demand"] < 0).sum()
        if negative_demand_count > 0:
            print(
                f"Warning: Found {negative_demand_count} negative demand values, converting to 0"
            )
            df.loc[df["demand"] < 0, "demand"] = 0

        # Handle negative stock values (convert to 0)
        negative_stock_count = (df["stock_level"] < 0).sum()
        if negative_stock_count > 0:
            print(
                f"Warning: Found {negative_stock_count} negative stock values, converting to 0"
            )
            df.loc[df["stock_level"] < 0, "stock_level"] = 0

        # Handle negative incoming inventory values (convert to 0)
        negative_incoming_count = (df["incoming_inventory"] < 0).sum()
        if negative_incoming_count > 0:
            print(
                f"Warning: Found {negative_incoming_count} negative incoming inventory values, converting to 0"
            )
            df.loc[df["incoming_inventory"] < 0, "incoming_inventory"] = 0

        # Ensure string columns are string
        df["product_id"] = df["product_id"].astype(str)
        df["product_category"] = df["product_category"].astype(str)
        df["location_id"] = df["location_id"].astype(str)

        # Sort by date, product, location for consistency
        df = df.sort_values(
            ["date", "product_id", "product_category", "location_id"]
        ).reset_index(drop=True)

        return df
