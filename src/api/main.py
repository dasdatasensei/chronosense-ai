"""
ChronoSense AI API
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
from datetime import datetime, timedelta
import logging
from src.models.time_series_models import ProphetModel
from src.utils.evaluation import calculate_metrics
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ChronoSense AI API",
    description="Enterprise-grade time series forecasting API",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model
model = ProphetModel()


class DataPoint(BaseModel):
    """Single data point model."""

    date: str
    value: float


class ForecastRequest(BaseModel):
    """Forecast request model."""

    historical_data: List[DataPoint]
    forecast_horizon: int
    include_components: bool = False


class MetricsResponse(BaseModel):
    """Metrics response model."""

    mape: float = 0.0
    rmse: float = 0.0
    mae: float = 0.0


class ForecastResponse(BaseModel):
    """Forecast response model."""

    forecast: List[Dict[str, Any]]
    metrics: Optional[MetricsResponse] = None
    components: Optional[Dict[str, List[float]]] = None


@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API documentation."""
    return """
    <html>
        <head>
            <title>ChronoSense AI API</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                code { background-color: #f4f4f4; padding: 2px 5px; border-radius: 3px; }
            </style>
        </head>
        <body>
            <h1>🔮 ChronoSense AI API</h1>
            <p>Welcome to the ChronoSense AI API. Available endpoints:</p>
            <ul>
                <li><code>GET /</code> - This documentation</li>
                <li><code>GET /health</code> - Health check endpoint</li>
                <li><code>GET /metrics</code> - Prometheus metrics</li>
                <li><code>POST /forecast</code> - Create sales forecast</li>
            </ul>
            <p>For complete API documentation, visit <a href="/docs">/docs</a></p>
        </body>
    </html>
    """


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint."""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    from fastapi.responses import Response

    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/forecast", response_model=ForecastResponse)
async def create_forecast(request: ForecastRequest):
    """
    Create a forecast using historical data.
    """
    try:
        logger.info("Received forecast request")
        logger.info(f"Request data: {request}")

        # Convert request data to pandas Series with better error handling
        try:
            df = pd.DataFrame([d.dict() for d in request.historical_data])
            logger.info(f"Raw DataFrame: {df.head()}")

            df["date"] = pd.to_datetime(df["date"])
            logger.info(f"DataFrame with parsed dates: {df.head()}")

            series = pd.Series(data=df["value"].values, index=df["date"], name="value")
            logger.info(f"Created series with shape: {series.shape}")

        except Exception as e:
            logger.error(f"Error processing input data: {str(e)}")
            raise HTTPException(
                status_code=400, detail=f"Error processing input data: {str(e)}"
            )

        # Initialize model if not already initialized
        if not hasattr(model, "model") or model.model is None:
            logger.info("Initializing new Prophet model")
            model.__init__()

        # Fit model and make predictions
        try:
            model.fit(series)
            logger.info("Model fitting complete")
            forecast = model.predict(request.forecast_horizon)
            logger.info(f"Generated forecast with shape: {forecast.shape}")

        except Exception as e:
            logger.error(f"Error in model fitting/prediction: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Error in model fitting/prediction: {str(e)}"
            )

        # Calculate metrics if we have actual values
        metrics = None
        if len(series) > 0:
            try:
                # Get the last known values for comparison
                actual_values = series[-request.forecast_horizon :]

                # Make in-sample predictions for the same period
                in_sample_forecast = model.predict(
                    request.forecast_horizon, is_future=False
                )
                predicted_values = in_sample_forecast[-request.forecast_horizon :]

                metrics = calculate_metrics(
                    actual_values.values,  # Convert series to numpy array
                    predicted_values.values,
                )

                logger.info(f"Actual values shape: {actual_values.shape}")
                logger.info(f"Predicted values shape: {predicted_values.shape}")
                logger.info(f"Calculated metrics: {metrics}")

            except Exception as e:
                logger.warning(f"Error calculating metrics: {str(e)}")
                metrics = {"mape": 0.0, "rmse": 0.0, "mae": 0.0}

        # If metrics is still None, provide default values
        if metrics is None:
            metrics = {"mape": 0.0, "rmse": 0.0, "mae": 0.0}

        # Get components if requested
        components = None
        if request.include_components:
            try:
                raw_components = model.plot_components(series)
                # Convert numpy arrays to lists and handle NaN values
                components = {
                    k: [None if pd.isna(x) or np.isinf(x) else float(x) for x in v]
                    for k, v in raw_components.items()
                }
                logger.info("Components extraction complete")
            except Exception as e:
                logger.warning(f"Error extracting components: {str(e)}")
                components = None

        # Prepare response
        try:
            # Handle NaN/infinite values in forecast
            response = {
                "forecast": [
                    {
                        "date": date.strftime("%Y-%m-%d"),
                        "value": (
                            None if pd.isna(value) or np.isinf(value) else float(value)
                        ),
                    }
                    for date, value in zip(forecast.index, forecast.values)
                ],
                "metrics": metrics,
                "components": components,
            }
            logger.info("Response prepared successfully")
            return response

        except Exception as e:
            logger.error(f"Error preparing response: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Error preparing response: {str(e)}"
            )

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error creating forecast: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Unexpected error creating forecast: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
