"""Data loading component for the Streamlit app."""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def load_csv_data(uploaded_file):
    """Load and validate CSV data."""
    try:
        data = pd.read_csv(uploaded_file)
        if "Order Date" not in data.columns or "Sales" not in data.columns:
            st.error("CSV must contain 'Order Date' and 'Sales' columns")
            return None

        data["Order Date"] = pd.to_datetime(data["Order Date"])
        return data
    except Exception as e:
        st.error(f"Error loading CSV: {str(e)}")
        return None


def generate_demo_data():
    """Generate demo time series data."""
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
    np.random.seed(42)  # For reproducibility
    values = np.random.normal(1000, 100, len(dates))
    values = np.abs(values)  # Ensure positive values
    return pd.DataFrame({"Order Date": dates, "Sales": values})
