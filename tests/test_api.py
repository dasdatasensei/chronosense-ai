"""
Tests for the FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from src.api.main import app

client = TestClient(app)


@pytest.fixture
def sample_forecast_request():
    """Create sample forecast request data."""
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
    values = np.random.normal(100, 10, len(dates))
    values += 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)  # Add yearly pattern

    data = [
        {"date": date.strftime("%Y-%m-%d"), "value": float(value)}
        for date, value in zip(dates, values)
    ]

    return {"historical_data": data, "forecast_horizon": 30, "include_components": True}


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_metrics():
    """Test metrics endpoint."""
    response = client.get("/metrics")
    assert response.status_code == 200


def test_create_forecast(sample_forecast_request):
    """Test forecast creation endpoint."""
    response = client.post("/forecast", json=sample_forecast_request)
    assert response.status_code == 200

    data = response.json()
    assert "forecast" in data
    assert len(data["forecast"]) == sample_forecast_request["forecast_horizon"]

    # Check forecast structure
    first_forecast = data["forecast"][0]
    assert "date" in first_forecast
    assert "value" in first_forecast

    # Check metrics if available
    if "metrics" in data:
        assert isinstance(data["metrics"], dict)

    # Check components if requested
    if sample_forecast_request["include_components"]:
        assert "components" in data
