import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import pandas as pd
from datetime import datetime, timedelta

# Initialize the Dash app with a dark theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

# Layout
app.layout = html.Div(
    [
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H1(
                                    "🔮 ChronoSense AI - Forecast Visualization",
                                    className="text-center mb-4",
                                ),
                            ]
                        )
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dcc.Graph(
                                    id="forecast-plot",
                                    config={"displayModeBar": True},
                                    style={"height": "600px"},
                                )
                            ]
                        )
                    ]
                ),
                dbc.Row(
                    [dbc.Col([html.Div(id="metrics-container", className="mt-4")])]
                ),
            ],
            fluid=True,
        )
    ]
)


def create_forecast_plot(historical_df, forecast_df):
    """Create the forecast plot using Plotly."""
    fig = go.Figure()

    # Add historical data
    fig.add_trace(
        go.Scatter(
            x=historical_df["Order Date"],
            y=historical_df["Sales"],
            name="Historical",
            mode="lines",
            line=dict(color="#4CAF50", width=2),
            hovertemplate="Date: %{x}<br>Sales: $%{y:,.2f}<extra></extra>",
        )
    )

    # Add forecast data
    fig.add_trace(
        go.Scatter(
            x=forecast_df["date"],
            y=forecast_df["value"],
            name="Forecast",
            mode="lines",
            line=dict(color="#81c784", width=2, dash="dash"),
            hovertemplate="Date: %{x}<br>Forecast: $%{y:,.2f}<extra></extra>",
        )
    )

    # Update layout
    fig.update_layout(
        title=dict(
            text="Sales Forecast",
            x=0.5,
            y=0.95,
            xanchor="center",
            yanchor="top",
            font=dict(size=24, color="#ffffff"),
        ),
        plot_bgcolor="#1e2130",
        paper_bgcolor="#1e2130",
        font=dict(color="#ffffff"),
        xaxis=dict(
            title="Date",
            gridcolor="#2e3344",
            showgrid=True,
            tickfont=dict(color="#ffffff"),
            title_font=dict(color="#ffffff"),
        ),
        yaxis=dict(
            title="Sales ($)",
            gridcolor="#2e3344",
            showgrid=True,
            tickfont=dict(color="#ffffff"),
            title_font=dict(color="#ffffff"),
            tickformat="$,.0f",
        ),
        showlegend=True,
        legend=dict(font=dict(color="#ffffff"), bgcolor="rgba(0,0,0,0.5)"),
        hovermode="x unified",
    )

    return fig


# Callback to update the plot
@app.callback(Output("forecast-plot", "figure"), [Input("store-data", "data")])
def update_plot(data):
    if data is None:
        return go.Figure()

    historical_df = pd.DataFrame(data["historical"])
    forecast_df = pd.DataFrame(data["forecast"])

    return create_forecast_plot(historical_df, forecast_df)


if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
