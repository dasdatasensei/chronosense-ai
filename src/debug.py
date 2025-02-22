"""
Debug script to test data loading and processing.
"""

import os
import sys
import logging
from pathlib import Path

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.preprocessing.data_loader import load_raw_data, process_date_columns

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_data_loading():
    """Test data loading and processing functionality."""
    try:
        # Set up data path
        data_dir = Path(project_root) / "data"
        data_path = str(data_dir / "raw" / "train.csv")

        logger.info(f"Testing data loading from: {data_path}")

        # Load data
        df = load_raw_data(data_path)
        logger.info(f"Successfully loaded data with shape: {df.shape}")

        # Process dates
        df = process_date_columns(df)
        logger.info("Successfully processed date columns")

        # Display sample
        logger.info("\nSample of processed data:")
        logger.info(df.head())

        return True

    except Exception as e:
        logger.error(f"Error in test: {str(e)}")
        return False


if __name__ == "__main__":
    success = test_data_loading()
    if success:
        logger.info("All tests passed successfully!")
    else:
        logger.error("Tests failed!")
