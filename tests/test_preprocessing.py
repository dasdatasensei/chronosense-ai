"""
Unit tests for preprocessing module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from src.preprocessing.data_loader import process_date_columns, split_data


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "Order Date": ["2021-01-01", "2021-01-02", "2021-01-03"],
            "Ship Date": ["2021-01-03", "2021-01-04", "2021-01-05"],
            "Sales": [100, 200, 300],
            "Category": ["A", "B", "A"],
        }
    )


def test_process_date_columns(sample_df):
    """Test date column processing."""
    processed_df = process_date_columns(sample_df)

    # Check if date columns are converted to datetime
    assert isinstance(processed_df["Order Date"].dtype, pd.DatetimeDtype)
    assert isinstance(processed_df["Ship Date"].dtype, pd.DatetimeDtype)

    # Check if new date features are created
    assert "order_year" in processed_df.columns
    assert "order_month" in processed_df.columns
    assert "order_day" in processed_df.columns
    assert "order_dayofweek" in processed_df.columns

    # Check shipping duration calculation
    assert "shipping_duration" in processed_df.columns
    assert processed_df["shipping_duration"].tolist() == [2, 2, 2]


def test_split_data(sample_df):
    """Test data splitting functionality."""
    train_df, val_df, test_df = split_data(
        sample_df, test_size=0.2, val_size=0.2, random_state=42
    )

    # Check if splits have correct proportions
    total_rows = len(sample_df)
    assert len(train_df) == pytest.approx(total_rows * 0.6, abs=1)
    assert len(val_df) == pytest.approx(total_rows * 0.2, abs=1)
    assert len(test_df) == pytest.approx(total_rows * 0.2, abs=1)

    # Check if splits contain different data
    assert not train_df.index.isin(val_df.index).any()
    assert not train_df.index.isin(test_df.index).any()
    assert not val_df.index.isin(test_df.index).any()

    # Check if all data is preserved
    assert len(train_df) + len(val_df) + len(test_df) == len(sample_df)
