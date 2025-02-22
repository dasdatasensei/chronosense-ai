"""
Data loading and preprocessing module.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_raw_data(data_path: str = "../data/raw/train.csv") -> pd.DataFrame:
    """
    Load raw data from CSV file.

    Args:
        data_path (str): Path to the raw data file

    Returns:
        pd.DataFrame: Raw data as a pandas DataFrame
    """
    try:
        logger.info(f"Loading raw data from {data_path}")
        df = pd.read_csv(data_path)
        logger.info(f"Successfully loaded data with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def process_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process date columns in the dataset.

    Args:
        df (pd.DataFrame): Input DataFrame with date columns

    Returns:
        pd.DataFrame: DataFrame with processed date columns
    """
    try:
        logger.info("Processing date columns")

        # Convert date columns to datetime
        date_columns = ["Order Date", "Ship Date"]
        for col in date_columns:
            df[col] = pd.to_datetime(df[col], format="%d/%m/%Y")

        # Extract additional date features
        for col in date_columns:
            col_prefix = col.split()[0].lower()
            df[f"{col_prefix}_year"] = df[col].dt.year
            df[f"{col_prefix}_month"] = df[col].dt.month
            df[f"{col_prefix}_day"] = df[col].dt.day
            df[f"{col_prefix}_dayofweek"] = df[col].dt.dayofweek

        # Calculate shipping duration
        df["shipping_duration"] = (df["Ship Date"] - df["Order Date"]).dt.days

        logger.info("Successfully processed date columns")
        return df
    except Exception as e:
        logger.error(f"Error processing date columns: {str(e)}")
        raise


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.2,
    date_column: str = "Order Date",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets based on time.

    Args:
        df (pd.DataFrame): Input DataFrame
        test_size (float): Proportion of data for test set
        val_size (float): Proportion of remaining data for validation set
        date_column (str): Column to use for time-based splitting

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Train, validation, and test sets
    """
    try:
        logger.info("Splitting data into train, validation, and test sets")

        # Sort by date
        df = df.sort_values(date_column)

        # Calculate split points
        total_size = len(df)
        test_idx = int(total_size * (1 - test_size))
        val_idx = int(test_idx * (1 - val_size))

        # Split data
        train_df = df.iloc[:val_idx]
        val_df = df.iloc[val_idx:test_idx]
        test_df = df.iloc[test_idx:]

        logger.info(f"Train set shape: {train_df.shape}")
        logger.info(
            f"Train date range: {train_df[date_column].min()} to {train_df[date_column].max()}"
        )
        logger.info(f"Validation set shape: {val_df.shape}")
        logger.info(
            f"Validation date range: {val_df[date_column].min()} to {val_df[date_column].max()}"
        )
        logger.info(f"Test set shape: {test_df.shape}")
        logger.info(
            f"Test date range: {test_df[date_column].min()} to {test_df[date_column].max()}"
        )

        return train_df, val_df, test_df

    except Exception as e:
        logger.error(f"Error splitting data: {str(e)}")
        raise


def save_processed_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str = "../data/processed",
) -> None:
    """
    Save processed datasets to CSV files.

    Args:
        train_df (pd.DataFrame): Training set
        val_df (pd.DataFrame): Validation set
        test_df (pd.DataFrame): Test set
        output_dir (str): Directory to save processed data
    """
    try:
        logger.info(f"Saving processed data to {output_dir}")

        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Save datasets
        train_df.to_csv(f"{output_dir}/train_set.csv", index=False)
        val_df.to_csv(f"{output_dir}/val_set.csv", index=False)
        test_df.to_csv(f"{output_dir}/test_set.csv", index=False)

        logger.info("Successfully saved processed data")
    except Exception as e:
        logger.error(f"Error saving processed data: {str(e)}")
        raise
