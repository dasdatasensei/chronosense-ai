"""
Evaluation metrics and utilities for time series models.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """
    Calculate evaluation metrics for time series forecasting.

    Args:
        actual: Actual values
        predicted: Predicted values

    Returns:
        Dictionary containing MAPE, RMSE, and MAE
    """
    try:
        # Remove any NaN or infinite values
        mask = ~(
            np.isnan(actual)
            | np.isnan(predicted)
            | np.isinf(actual)
            | np.isinf(predicted)
        )
        actual = actual[mask]
        predicted = predicted[mask]

        if len(actual) == 0 or len(predicted) == 0:
            return {"mape": 0.0, "rmse": 0.0, "mae": 0.0}

        # Calculate MAPE
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100

        # Calculate RMSE
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))

        # Calculate MAE
        mae = np.mean(np.abs(actual - predicted))

        # Handle any remaining NaN or infinite values
        metrics = {
            "mape": float(mape) if not np.isnan(mape) and not np.isinf(mape) else 0.0,
            "rmse": float(rmse) if not np.isnan(rmse) and not np.isinf(rmse) else 0.0,
            "mae": float(mae) if not np.isnan(mae) and not np.isinf(mae) else 0.0,
        }

        return metrics

    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        return {"mape": 0.0, "rmse": 0.0, "mae": 0.0}


def plot_predictions(
    y_true: pd.Series,
    y_pred: pd.Series,
    title: str = "Time Series Predictions",
    figsize: tuple = (12, 6),
) -> None:
    """
    Plot actual vs predicted values.

    Args:
        y_true (pd.Series): True values
        y_pred (pd.Series): Predicted values
        title (str): Plot title
        figsize (tuple): Figure size
    """
    try:
        logger.info("Creating prediction plot")

        plt.figure(figsize=figsize)
        plt.plot(y_true.index, y_true.values, label="Actual", color="blue")
        plt.plot(
            y_pred.index, y_pred.values, label="Predicted", color="red", linestyle="--"
        )

        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)

        # Adjust layout to prevent label cutoff
        plt.tight_layout()

        logger.info("Successfully created plot")
        plt.show()

    except Exception as e:
        logger.error(f"Error creating plot: {str(e)}")
        raise


def plot_residuals(
    y_true: pd.Series, y_pred: pd.Series, figsize: tuple = (15, 5)
) -> None:
    """
    Create diagnostic plots for residuals.

    Args:
        y_true (pd.Series): True values
        y_pred (pd.Series): Predicted values
        figsize (tuple): Figure size
    """
    try:
        logger.info("Creating residual diagnostic plots")

        residuals = y_true - y_pred

        # Create subplot
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

        # Residuals over time
        ax1.plot(residuals.index, residuals.values)
        ax1.set_title("Residuals Over Time")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Residual")
        ax1.grid(True)

        # Residual histogram
        sns.histplot(residuals, kde=True, ax=ax2)
        ax2.set_title("Residual Distribution")
        ax2.set_xlabel("Residual")

        # Q-Q plot
        from scipy import stats

        stats.probplot(residuals, dist="norm", plot=ax3)
        ax3.set_title("Q-Q Plot")

        # Adjust layout
        plt.tight_layout()

        logger.info("Successfully created residual plots")
        plt.show()

    except Exception as e:
        logger.error(f"Error creating residual plots: {str(e)}")
        raise


def cross_validate_time_series(
    model: Any, data: pd.Series, n_splits: int = 5, train_size: float = 0.7
) -> Dict[str, list]:
    """
    Perform time series cross-validation.

    Args:
        model: Time series model object
        data (pd.Series): Time series data
        n_splits (int): Number of splits for cross-validation
        train_size (float): Proportion of data to use for training

    Returns:
        Dict[str, list]: Dictionary of metric lists for each fold
    """
    try:
        logger.info("Performing time series cross-validation")

        metrics_dict = {"mse": [], "rmse": [], "mae": [], "r2": [], "mape": []}

        # Calculate size of each split
        total_size = len(data)
        split_size = int(total_size / n_splits)

        for i in range(n_splits):
            # Calculate indices for this split
            start_idx = i * split_size
            split_point = start_idx + int(split_size * train_size)
            end_idx = start_idx + split_size

            # Extract train and test sets
            train_data = data[start_idx:split_point]
            test_data = data[split_point:end_idx]

            # Fit model and make predictions
            model.fit(train_data)
            predictions = model.predict(len(test_data))

            # Calculate metrics
            fold_metrics = calculate_metrics(test_data, predictions["forecast"])

            # Store metrics
            for metric, value in fold_metrics.items():
                metrics_dict[metric].append(value)

        logger.info("Successfully completed cross-validation")
        return metrics_dict

    except Exception as e:
        logger.error(f"Error in cross-validation: {str(e)}")
        raise
