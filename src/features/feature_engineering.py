"""
Feature engineering module for time series data.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import logging
from typing import Tuple, Dict, Any, List, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Class for handling feature engineering tasks."""

    def __init__(self):
        self.numeric_scaler = StandardScaler()
        self.categorical_encoder = OneHotEncoder(
            sparse_output=False, handle_unknown="ignore"
        )
        self.numeric_features = None
        self.categorical_features = None

        # Time series specific parameters
        self.lag_features = []
        self.rolling_features = []
        self.seasonal_features = []
        self.date_features = []

    def create_time_series_features(
        self,
        df: pd.DataFrame,
        date_column: str,
        target_column: str,
        lag_periods: List[int] = [1, 7, 30],
        rolling_windows: List[int] = [7, 30, 90],
        include_holidays: bool = True,
    ) -> pd.DataFrame:
        """
        Create time series specific features.

        Args:
            df (pd.DataFrame): Input DataFrame
            date_column (str): Name of the date column
            target_column (str): Name of the target column
            lag_periods (List[int]): Periods for lag features
            rolling_windows (List[int]): Windows for rolling statistics
            include_holidays (bool): Whether to include holiday features

        Returns:
            pd.DataFrame: DataFrame with time series features
        """
        try:
            logger.info("Creating time series features")

            # Ensure date column is datetime
            df[date_column] = pd.to_datetime(df[date_column])

            # Create date-based features
            df = self._create_date_features(df, date_column)

            # Create lag features
            df = self._create_lag_features(df, target_column, lag_periods)

            # Create rolling statistics
            df = self._create_rolling_features(df, target_column, rolling_windows)

            # Create seasonal features
            df = self._create_seasonal_features(df, date_column)

            # Add holiday features if requested
            if include_holidays:
                df = self._add_holiday_features(df, date_column)

            logger.info(f"Created time series features, new shape: {df.shape}")
            return df

        except Exception as e:
            logger.error(f"Error creating time series features: {str(e)}")
            raise

    def _create_date_features(self, df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """Create date-based features."""
        df["year"] = df[date_column].dt.year
        df["month"] = df[date_column].dt.month
        df["day"] = df[date_column].dt.day
        df["dayofweek"] = df[date_column].dt.dayofweek
        df["quarter"] = df[date_column].dt.quarter
        df["is_weekend"] = df[date_column].dt.dayofweek.isin([5, 6]).astype(int)
        df["is_month_start"] = df[date_column].dt.is_month_start.astype(int)
        df["is_month_end"] = df[date_column].dt.is_month_end.astype(int)

        self.date_features = [
            "year",
            "month",
            "day",
            "dayofweek",
            "quarter",
            "is_weekend",
            "is_month_start",
            "is_month_end",
        ]
        return df

    def _create_lag_features(
        self, df: pd.DataFrame, target_column: str, lag_periods: List[int]
    ) -> pd.DataFrame:
        """Create lag features."""
        for lag in lag_periods:
            col_name = f"lag_{lag}"
            df[col_name] = df[target_column].shift(lag)
            self.lag_features.append(col_name)
        return df

    def _create_rolling_features(
        self, df: pd.DataFrame, target_column: str, windows: List[int]
    ) -> pd.DataFrame:
        """Create rolling statistics features."""
        for window in windows:
            # Mean
            df[f"rolling_mean_{window}"] = df[target_column].rolling(window).mean()
            # Standard deviation
            df[f"rolling_std_{window}"] = df[target_column].rolling(window).std()
            # Min and Max
            df[f"rolling_min_{window}"] = df[target_column].rolling(window).min()
            df[f"rolling_max_{window}"] = df[target_column].rolling(window).max()

            self.rolling_features.extend(
                [
                    f"rolling_mean_{window}",
                    f"rolling_std_{window}",
                    f"rolling_min_{window}",
                    f"rolling_max_{window}",
                ]
            )
        return df

    def _create_seasonal_features(
        self, df: pd.DataFrame, date_column: str
    ) -> pd.DataFrame:
        """Create seasonal features."""
        # Yearly seasonality
        df["day_of_year"] = df[date_column].dt.dayofyear
        df["day_of_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
        df["day_of_year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

        # Weekly seasonality
        df["day_of_week_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
        df["day_of_week_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

        self.seasonal_features = [
            "day_of_year",
            "day_of_year_sin",
            "day_of_year_cos",
            "day_of_week_sin",
            "day_of_week_cos",
        ]
        return df

    def _add_holiday_features(self, df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """Add holiday-related features."""
        try:
            import holidays

            us_holidays = holidays.US()

            # Create holiday flags
            df["is_holiday"] = (
                df[date_column].map(lambda x: x in us_holidays).astype(int)
            )
            df["days_to_holiday"] = df[date_column].map(
                lambda x: min((h - x).days for h in us_holidays if h > x)
            )
            df["days_from_holiday"] = df[date_column].map(
                lambda x: min((x - h).days for h in us_holidays if h < x)
            )

            self.seasonal_features.extend(
                ["is_holiday", "days_to_holiday", "days_from_holiday"]
            )
            return df

        except ImportError:
            logger.warning("holidays package not installed, skipping holiday features")
            return df

    def identify_feature_types(self, df: pd.DataFrame) -> Tuple[list, list]:
        """
        Identify numeric and categorical features in the dataset.

        Args:
            df (pd.DataFrame): Input DataFrame

        Returns:
            Tuple[list, list]: Lists of numeric and categorical feature names
        """
        try:
            logger.info("Identifying feature types")

            # Identify numeric and categorical columns
            numeric_features = df.select_dtypes(
                include=["int64", "float64"]
            ).columns.tolist()
            categorical_features = df.select_dtypes(include=["object"]).columns.tolist()

            # Remove date columns from categorical features
            date_columns = ["Order Date", "Ship Date"]
            categorical_features = [
                col for col in categorical_features if col not in date_columns
            ]

            self.numeric_features = numeric_features
            self.categorical_features = categorical_features

            logger.info(
                f"Found {len(numeric_features)} numeric features and {len(categorical_features)} categorical features"
            )
            return numeric_features, categorical_features

        except Exception as e:
            logger.error(f"Error identifying feature types: {str(e)}")
            raise

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform the data with feature engineering.

        Args:
            df (pd.DataFrame): Input DataFrame

        Returns:
            pd.DataFrame: Transformed DataFrame
        """
        try:
            logger.info("Fitting and transforming features")

            # Identify feature types if not already done
            if self.numeric_features is None or self.categorical_features is None:
                self.identify_feature_types(df)

            # Transform numeric features
            numeric_df = pd.DataFrame(
                self.numeric_scaler.fit_transform(df[self.numeric_features]),
                columns=self.numeric_features,
                index=df.index,
            )

            # Transform categorical features
            categorical_df = pd.DataFrame(
                self.categorical_encoder.fit_transform(df[self.categorical_features]),
                columns=self.categorical_encoder.get_feature_names_out(
                    self.categorical_features
                ),
                index=df.index,
            )

            # Combine transformed features
            transformed_df = pd.concat([numeric_df, categorical_df], axis=1)

            logger.info(
                f"Successfully transformed data with shape {transformed_df.shape}"
            )
            return transformed_df

        except Exception as e:
            logger.error(f"Error in feature transformation: {str(e)}")
            raise

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted transformers.

        Args:
            df (pd.DataFrame): Input DataFrame

        Returns:
            pd.DataFrame: Transformed DataFrame
        """
        try:
            logger.info("Transforming features using fitted transformers")

            # Transform numeric features
            numeric_df = pd.DataFrame(
                self.numeric_scaler.transform(df[self.numeric_features]),
                columns=self.numeric_features,
                index=df.index,
            )

            # Transform categorical features
            categorical_df = pd.DataFrame(
                self.categorical_encoder.transform(df[self.categorical_features]),
                columns=self.categorical_encoder.get_feature_names_out(
                    self.categorical_features
                ),
                index=df.index,
            )

            # Combine transformed features
            transformed_df = pd.concat([numeric_df, categorical_df], axis=1)

            logger.info(
                f"Successfully transformed data with shape {transformed_df.shape}"
            )
            return transformed_df

        except Exception as e:
            logger.error(f"Error in feature transformation: {str(e)}")
            raise

    def save_transformers(self, output_dir: str = "../data/models") -> None:
        """
        Save fitted transformers to disk.

        Args:
            output_dir (str): Directory to save transformers
        """
        try:
            logger.info(f"Saving transformers to {output_dir}")

            import joblib

            # Create output directory if it doesn't exist
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            # Save transformers
            joblib.dump(self.numeric_scaler, f"{output_dir}/numeric_scaler.joblib")
            joblib.dump(
                self.categorical_encoder, f"{output_dir}/categorical_encoder.joblib"
            )

            # Save feature lists
            feature_lists = {
                "numeric_features": self.numeric_features,
                "categorical_features": self.categorical_features,
            }
            joblib.dump(feature_lists, f"{output_dir}/feature_lists.joblib")

            logger.info("Successfully saved transformers")

        except Exception as e:
            logger.error(f"Error saving transformers: {str(e)}")
            raise

    def load_transformers(self, input_dir: str = "../data/models") -> None:
        """
        Load fitted transformers from disk.

        Args:
            input_dir (str): Directory containing saved transformers
        """
        try:
            logger.info(f"Loading transformers from {input_dir}")

            import joblib

            # Load transformers
            self.numeric_scaler = joblib.load(f"{input_dir}/numeric_scaler.joblib")
            self.categorical_encoder = joblib.load(
                f"{input_dir}/categorical_encoder.joblib"
            )

            # Load feature lists
            feature_lists = joblib.load(f"{input_dir}/feature_lists.joblib")
            self.numeric_features = feature_lists["numeric_features"]
            self.categorical_features = feature_lists["categorical_features"]

            logger.info("Successfully loaded transformers")

        except Exception as e:
            logger.error(f"Error loading transformers: {str(e)}")
            raise
