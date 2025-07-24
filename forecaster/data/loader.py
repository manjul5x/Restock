import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Optional
from .schema import DemandSchema, DemandRecord
from .product_master_schema import ProductMasterSchema, ProductMasterRecord


class DemandDataLoader:
    """Loader for demand data with schema validation"""

    def __init__(self, data_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the data loader

        Args:
            data_dir: Directory containing data files. If None, uses default dummy data location
        """
        if data_dir is None:
            # Default to the main data directory (not just dummy)
            self.data_dir = Path(__file__).parent
        else:
            self.data_dir = Path(data_dir)

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

    def load_csv(self, filename: str, validate: bool = True) -> pd.DataFrame:
        """
        Load demand data from CSV file

        Args:
            filename: Name of the CSV file
            validate: Whether to validate against schema

        Returns:
            DataFrame with demand data
        """
        file_path = self.data_dir / filename

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Load CSV
        df = pd.read_csv(file_path)

        # Determine schema based on filename or columns
        if filename in ["product_master_daily.csv", "product_master_weekly.csv"] or (
            "demand_frequency" in df.columns and "risk_period" in df.columns
        ):
            # Product master schema
            df = ProductMasterSchema.standardize_dataframe(df)
            if validate:
                ProductMasterSchema.validate_dataframe(df)
        else:
            # Demand data schema
            df = DemandSchema.standardize_dataframe(df)
            if validate:
                DemandSchema.validate_dataframe(df)

        return df

    def load_dummy_data(self, frequency: str = "daily") -> pd.DataFrame:
        """
        Load dummy data with specified frequency

        Args:
            frequency: 'daily' or 'weekly'

        Returns:
            DataFrame with dummy demand data
        """
        filename = f"sku_demand_{frequency}.csv"
        return self.load_csv(filename)

    def get_available_files(self) -> list:
        """Get list of available data files"""
        csv_files = list(self.data_dir.glob("*.csv"))
        return [f.name for f in csv_files]

    def validate_file(self, filename: str) -> bool:
        """
        Validate a CSV file against the schema without loading it

        Args:
            filename: Name of the CSV file

        Returns:
            True if valid, raises exception if invalid
        """
        return self.load_csv(filename, validate=True) is not None

    def load_product_master_daily(self, validate: bool = True) -> pd.DataFrame:
        """
        Load daily product master data

        Args:
            validate: Whether to validate against schema

        Returns:
            DataFrame with daily product master data
        """
        return self.load_csv("product_master_daily.csv", validate=validate)

    def load_product_master_weekly(self, validate: bool = True) -> pd.DataFrame:
        """
        Load weekly product master data

        Args:
            validate: Whether to validate against schema

        Returns:
            DataFrame with weekly product master data
        """
        return self.load_csv("product_master_weekly.csv", validate=validate)

    def get_product_master_summary(self, frequency: str = "daily") -> dict:
        """
        Get summary statistics for product master data

        Args:
            frequency: 'daily' or 'weekly'

        Returns:
            Dictionary with summary statistics
        """
        try:
            if frequency == "daily":
                df = self.load_product_master_daily()
            elif frequency == "weekly":
                df = self.load_product_master_weekly()
            else:
                raise ValueError("Frequency must be 'daily' or 'weekly'")

            summary = {
                "frequency": frequency,
                "total_records": len(df),
                "products": df["product_id"].nunique(),
                "locations": df["location_id"].nunique(),
                "categories": df["product_category"].nunique(),
                "demand_frequency": df["demand_frequency"].iloc[0],
                "risk_period": df["risk_period"].iloc[0],
                "product_location_combinations": df.groupby(
                    ["product_id", "location_id"]
                )
                .size()
                .count(),
            }

            return summary

        except Exception as e:
            return {"error": str(e)}

    def validate_product_master_coverage(self, frequency: str = "daily") -> dict:
        """
        Validate that all product-location combinations in demand data exist in product master

        Args:
            frequency: 'daily' or 'weekly'

        Returns:
            Dictionary with validation results
        """
        try:
            # Load demand data
            demand_df = self.load_dummy_data(frequency=frequency)

            # Load product master
            if frequency == "daily":
                master_df = self.load_product_master_daily()
            else:
                master_df = self.load_product_master_weekly()

            # Get unique combinations from demand data
            demand_combinations = set(
                demand_df[["product_id", "location_id"]].apply(tuple, axis=1)
            )

            # Get unique combinations from product master
            master_combinations = set(
                master_df[["product_id", "location_id"]].apply(tuple, axis=1)
            )

            # Find missing combinations
            missing_combinations = demand_combinations - master_combinations
            extra_combinations = master_combinations - demand_combinations

            validation_result = {
                "frequency": frequency,
                "demand_combinations": len(demand_combinations),
                "master_combinations": len(master_combinations),
                "missing_combinations": len(missing_combinations),
                "extra_combinations": len(extra_combinations),
                "coverage_percentage": (
                    len(demand_combinations - missing_combinations)
                    / len(demand_combinations)
                )
                * 100,
                "is_valid": len(missing_combinations) == 0,
                "missing_details": (
                    list(missing_combinations) if missing_combinations else []
                ),
                "extra_details": list(extra_combinations) if extra_combinations else [],
            }

            return validation_result

        except Exception as e:
            return {"error": str(e)}

    def load_customer_demand(self, validate: bool = True) -> pd.DataFrame:
        """
        Load customer demand data from customer_demand.csv

        Args:
            validate: Whether to validate against schema

        Returns:
            DataFrame with customer demand data
        """
        return self.load_csv("customer_demand.csv", validate=validate)

    def load_customer_product_master(self, validate: bool = True) -> pd.DataFrame:
        """
        Load customer product master data from customer_product_master.csv

        Args:
            validate: Whether to validate against schema

        Returns:
            DataFrame with customer product master data
        """
        return self.load_csv("customer_product_master.csv", validate=validate)


# Standalone functions for convenience
def load_csv(file_path: Union[str, Path], validate: bool = True) -> pd.DataFrame:
    """
    Load CSV file with automatic schema detection and validation.

    Args:
        file_path: Path to the CSV file
        validate: Whether to validate against schema

    Returns:
        DataFrame with loaded and validated data
    """
    loader = DemandDataLoader(Path(file_path).parent)
    return loader.load_csv(Path(file_path).name, validate=validate)


def load_product_master_daily(
    file_path: Union[str, Path], validate: bool = True
) -> pd.DataFrame:
    """
    Load daily product master data.

    Args:
        file_path: Path to the CSV file
        validate: Whether to validate against schema

    Returns:
        DataFrame with daily product master data
    """
    loader = DemandDataLoader(Path(file_path).parent)
    return loader.load_product_master_daily(validate=validate)


def load_product_master_weekly(
    file_path: Union[str, Path], validate: bool = True
) -> pd.DataFrame:
    """
    Load weekly product master data.

    Args:
        file_path: Path to the CSV file
        validate: Whether to validate against schema

    Returns:
        DataFrame with weekly product master data
    """
    loader = DemandDataLoader(Path(file_path).parent)
    return loader.load_product_master_weekly(validate=validate)


def validate_product_master_coverage(
    demand_data: pd.DataFrame, product_master_data: pd.DataFrame
) -> dict:
    """
    Validate that all product-location combinations in demand data exist in product master.

    Args:
        demand_data: DataFrame with demand data
        product_master_data: DataFrame with product master data

    Returns:
        Dictionary with validation results
    """
    # Get unique combinations from demand data
    demand_combinations = set(
        demand_data[["product_id", "location_id"]].apply(tuple, axis=1)
    )

    # Get unique combinations from product master
    master_combinations = set(
        product_master_data[["product_id", "location_id"]].apply(tuple, axis=1)
    )

    # Find missing combinations
    missing_combinations = demand_combinations - master_combinations
    extra_combinations = master_combinations - demand_combinations

    validation_result = {
        "demand_combinations": len(demand_combinations),
        "master_combinations": len(master_combinations),
        "missing_combinations": len(missing_combinations),
        "extra_combinations": len(extra_combinations),
        "coverage_percentage": (
            len(demand_combinations - missing_combinations) / len(demand_combinations)
        )
        * 100,
        "is_valid": len(missing_combinations) == 0,
        "missing_details": list(missing_combinations) if missing_combinations else [],
        "extra_details": list(extra_combinations) if extra_combinations else [],
    }

    return validation_result
