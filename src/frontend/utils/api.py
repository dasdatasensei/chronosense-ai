"""API utilities for making forecast requests."""

import requests
import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

API_HOST = os.getenv("API_HOST", "localhost")
API_PORT = os.getenv("API_PORT", "8000")
API_URL = f"http://{API_HOST}:{API_PORT}"


def check_api_status() -> str:
    """Check if the API is online."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return "Online" if response.status_code == 200 else "Offline"
    except:
        return "Offline"


def create_forecast(data: pd.DataFrame, horizon: int) -> dict:
    """Make API call to create forecast."""
    try:
        # Convert data to API format
        historical_data = [
            {"date": date.strftime("%Y-%m-%d"), "value": float(value)}
            for date, value in zip(data["Order Date"], data["Sales"])
            if pd.notna(date) and pd.notna(value)
        ]

        # Make request
        response = requests.post(
            f"{API_URL}/forecast",
            json={"historical_data": historical_data, "forecast_horizon": horizon},
            timeout=60,
        )

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error ({response.status_code}): {response.text}")
            return None

    except Exception as e:
        st.error(f"Error creating forecast: {str(e)}")
        return None
