"""
Main script for running the time series analysis pipeline.
"""

import logging
from pathlib import Path
from typing import Optional
import pandas as pd
import yaml
import os
import sys
import joblib

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.data.data_loader import DataLoader
from src.preprocessing.preprocessor import TimeSeriesPreprocessor
from src.features.feature_engineering import FeatureEngineer
from src.models.time_series_models import ProphetModel
from src.utils.evaluation import (
    calculate_metrics,
    plot_predictions,
    plot_residuals,
    cross_validate_time_series,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Set up base directory
BASE_DIR = Path(project_root)
DATA_DIR = BASE_DIR / "data"


class TimeSeriesPipeline:
    """Class to orchestrate the time series analysis pipeline."""

    def __init__(self):
        """Initialize the time series pipeline."""
        try:
            # Set up logging
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)

            # Create file handler
            fh = logging.FileHandler("pipeline.log")
            fh.setLevel(logging.INFO)

            # Create console handler
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)

            # Create formatter
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)

            # Add handlers to logger
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)

            # Initialize components
            self.data_loader = DataLoader()
            self.preprocessor = TimeSeriesPreprocessor()
            self.feature_engineer = FeatureEngineer()
            self.model = ProphetModel()

            self.logger.info("Pipeline initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing pipeline: {str(e)}")
            raise

    def run_pipeline(self):
        """Execute the complete time series pipeline."""
        try:
            self.logger.info("Starting pipeline execution")

            # Load data
            self.logger.info("Loading data...")
            train_series = self.data_loader.load_data("data/raw/train.csv")

            # Preprocess data
            self.logger.info("Preprocessing data...")
            train_series = self.preprocessor.process(train_series)

            # Engineer features
            self.logger.info("Engineering features...")
            train_series = self.feature_engineer.create_time_series_features(
                df=train_series.to_frame(name="value").reset_index(),
                date_column="Order Date",
                target_column="value",
                lag_periods=[1, 7, 14, 30],  # Daily, weekly, bi-weekly, monthly lags
                rolling_windows=[7, 14, 30, 90],  # Weekly to quarterly windows
                include_holidays=True,
            )

            # Scale features
            train_series = self.feature_engineer.fit_transform(train_series)

            # Train model
            self.logger.info("Training model...")
            self.model.fit(train_series)

            # Evaluate model
            self.logger.info("Evaluating model performance...")
            performance_metrics = self.model.evaluate_performance(train_series)

            # Save model and feature engineering transformers
            self.logger.info("Saving model and transformers...")
            joblib.dump(self.model, "data/models/prophet_model.joblib")
            self.feature_engineer.save_transformers("data/models")

            self.logger.info("Pipeline execution completed successfully")
            self.logger.info(f"Model performance metrics: {performance_metrics}")

        except Exception as e:
            self.logger.error(f"Error in pipeline execution: {str(e)}")
            raise


if __name__ == "__main__":
    # Set up base directory
    project_root = Path(__file__).resolve().parent.parent
    sys.path.append(str(project_root))

    # Run pipeline
    pipeline = TimeSeriesPipeline()
    pipeline.run_pipeline()
