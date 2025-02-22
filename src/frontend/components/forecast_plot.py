"""Plotting component for forecast visualization."""

import plotly.graph_objects as go
import streamlit as st


def create_forecast_plot(historical_data, forecast_data):
    """Create an interactive forecast plot."""
    fig = go.Figure()

    # Historical data
    fig.add_trace(
        go.Scatter(
            x=historical_data["Order Date"],
            y=historical_data["Sales"],
            name="Historical",
            line=dict(color="#4CAF50", width=2),
        )
    )

    # Forecast data
    fig.add_trace(
        go.Scatter(
            x=forecast_data["date"],
            y=forecast_data["value"],
            name="Forecast",
            line=dict(color="#81c784", width=2, dash="dash"),
        )
    )

    # Layout
    fig.update_layout(
        title="Sales Forecast",
        xaxis_title="Date",
        yaxis_title="Sales ($)",
        template="plotly_dark",
        height=500,
    )

    return fig


def display_metrics(metrics):
    """Display forecast metrics in columns."""
    col1, col2, col3 = st.columns(3)
    col1.metric("MAPE", f"{metrics.get('mape', 0):.2f}%")
    col2.metric("RMSE", f"${metrics.get('rmse', 0):,.2f}")
    col3.metric("MAE", f"${metrics.get('mae', 0):,.2f}")
