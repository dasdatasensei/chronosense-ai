import pandas as pd
import numpy as np
import logging


class TimeSeriesPreprocessor:
    """Class for preprocessing time series data."""

    def __init__(self):
        """Initialize the TimeSeriesPreprocessor."""
        self.logger = logging.getLogger(__name__)

    def process(self, series: pd.Series) -> pd.Series:
        """
        Preprocess time series data.

        Args:
            series (pd.Series): Input time series data

        Returns:
            pd.Series: Preprocessed time series data
        """
        try:
            self.logger.info("Starting time series preprocessing")

            # Sort index
            series = series.sort_index()

            # Convert to daily frequency
            series = series.asfreq("D")

            # Handle missing values
            series = self._handle_missing_values(series)

            # Handle outliers
            series = self._handle_outliers(series)

            self.logger.info("Time series preprocessing completed successfully")
            return series

        except Exception as e:
            self.logger.error(f"Error preprocessing time series: {str(e)}")
            raise

    def _handle_missing_values(self, series: pd.Series) -> pd.Series:
        """Handle missing values in the time series."""
        # Fill missing values with forward fill then backward fill
        return series.ffill().bfill()

    def _handle_outliers(self, series: pd.Series) -> pd.Series:
        """Handle outliers in the time series using IQR method."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Create a copy of the series
        processed_series = series.copy()

        # Replace outliers with bounds
        processed_series[processed_series < lower_bound] = lower_bound
        processed_series[processed_series > upper_bound] = upper_bound

        return processed_series
