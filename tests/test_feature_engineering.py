"""
Unit tests for feature engineering module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.features.feature_engineering import FeatureEngineer


@pytest.fixture
def sample_data():
    """Create sample time series data for testing."""
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
    values = np.random.normal(100, 10, len(dates))  # Random values with seasonality
    values += 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)  # Add yearly pattern
    values += 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)  # Add weekly pattern

    df = pd.DataFrame(
        {
            "date": dates,
            "value": values,
            "category": np.random.choice(["A", "B", "C"], len(dates)),
        }
    )
    return df


def test_feature_engineer_initialization():
    """Test FeatureEngineer initialization."""
    fe = FeatureEngineer()
    assert fe.numeric_scaler is not None
    assert fe.categorical_encoder is not None
    assert fe.numeric_features is None
    assert fe.categorical_features is None
    assert isinstance(fe.lag_features, list)
    assert isinstance(fe.rolling_features, list)
    assert isinstance(fe.seasonal_features, list)
    assert isinstance(fe.date_features, list)


def test_create_time_series_features(sample_data):
    """Test creation of time series features."""
    fe = FeatureEngineer()
    df_features = fe.create_time_series_features(
        df=sample_data, date_column="date", target_column="value"
    )

    # Check date features
    assert "year" in df_features.columns
    assert "month" in df_features.columns
    assert "day" in df_features.columns
    assert "is_weekend" in df_features.columns

    # Check lag features
    assert "lag_1" in df_features.columns
    assert "lag_7" in df_features.columns
    assert "lag_30" in df_features.columns

    # Check rolling features
    assert "rolling_mean_7" in df_features.columns
    assert "rolling_std_30" in df_features.columns
    assert "rolling_min_90" in df_features.columns
    assert "rolling_max_90" in df_features.columns

    # Check seasonal features
    assert "day_of_year_sin" in df_features.columns
    assert "day_of_year_cos" in df_features.columns
    assert "day_of_week_sin" in df_features.columns
    assert "day_of_week_cos" in df_features.columns


def test_feature_type_identification(sample_data):
    """Test identification of feature types."""
    fe = FeatureEngineer()
    numeric_features, categorical_features = fe.identify_feature_types(sample_data)

    assert "value" in numeric_features
    assert "category" in categorical_features
    assert "date" not in categorical_features  # Date column should be excluded


def test_fit_transform(sample_data):
    """Test fit_transform functionality."""
    fe = FeatureEngineer()

    # First create time series features
    df_features = fe.create_time_series_features(
        df=sample_data, date_column="date", target_column="value"
    )

    # Then transform all features
    transformed_df = fe.fit_transform(df_features)

    # Check that numeric features are scaled
    assert transformed_df["value"].mean() == pytest.approx(0, abs=1e-10)
    assert transformed_df["value"].std() == pytest.approx(1, abs=1e-10)

    # Check that categorical features are encoded
    assert any("category_" in col for col in transformed_df.columns)


def test_transform(sample_data):
    """Test transform functionality with fitted transformers."""
    fe = FeatureEngineer()

    # First create time series features
    df_features = fe.create_time_series_features(
        df=sample_data, date_column="date", target_column="value"
    )

    # Fit and transform training data
    train_transformed = fe.fit_transform(df_features)

    # Transform test data
    test_transformed = fe.transform(df_features)

    # Check that transformations are consistent
    assert test_transformed.shape[1] == train_transformed.shape[1]
    assert all(col in test_transformed.columns for col in train_transformed.columns)


def test_save_load_transformers(sample_data, tmp_path):
    """Test saving and loading of transformers."""
    fe = FeatureEngineer()

    # Create features and fit transformers
    df_features = fe.create_time_series_features(
        df=sample_data, date_column="date", target_column="value"
    )
    fe.fit_transform(df_features)

    # Save transformers
    fe.save_transformers(str(tmp_path))

    # Create new instance and load transformers
    fe_new = FeatureEngineer()
    fe_new.load_transformers(str(tmp_path))

    # Transform with both instances and compare results
    result1 = fe.transform(df_features)
    result2 = fe_new.transform(df_features)

    pd.testing.assert_frame_equal(result1, result2)


def test_holiday_features(sample_data):
    """Test holiday feature creation."""
    fe = FeatureEngineer()
    df_features = fe.create_time_series_features(
        df=sample_data, date_column="date", target_column="value", include_holidays=True
    )

    # Check holiday features
    assert "is_holiday" in df_features.columns
    assert "days_to_holiday" in df_features.columns
    assert "days_from_holiday" in df_features.columns


def test_error_handling():
    """Test error handling for invalid inputs."""
    fe = FeatureEngineer()

    # Test with invalid DataFrame
    with pytest.raises(Exception):
        fe.create_time_series_features(
            df=None, date_column="date", target_column="value"
        )

    # Test with invalid column names
    df = pd.DataFrame({"A": [1, 2, 3]})
    with pytest.raises(Exception):
        fe.create_time_series_features(
            df=df, date_column="non_existent", target_column="value"
        )
