"""
Streamlit frontend for ChronoSense AI.
"""

import streamlit as st

# Must be the first Streamlit command
st.set_page_config(
    page_title="ChronoSense AI",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

from components.data_loader import load_csv_data, generate_demo_data
from components.forecast_plot import create_forecast_plot, display_metrics
from utils.api import create_forecast, check_api_status

# Dark theme CSS
st.markdown(
    """
    <style>
    /* Main theme */
    .main {
        background-color: #0e1117;
        color: #fafafa;
    }
    [data-testid="stAppViewContainer"] {
        background-color: #0e1117;
        color: #fafafa;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1e2130;
        border-right: 1px solid #2e3344;
    }
    .sidebar-content {
        padding: 1rem;
    }

    /* Status indicator */
    .status-indicator {
        background-color: #1e2130;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        margin: 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .status-online {
        color: #4CAF50;
    }

    /* Section headers */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin: 1.5rem 0 1rem 0;
        color: #fafafa;
    }

    /* Feature cards */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin-top: 2rem;
    }
    .feature-card {
        background-color: #1e2130;
        border-radius: 10px;
        padding: 1.5rem;
        border: 1px solid #2e3344;
    }
    .feature-title {
        color: #4CAF50;
        font-size: 1.2rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Help section */
    .help-section {
        background-color: #1e2130;
        border-radius: 10px;
        margin-bottom: 2rem;
        padding: 1rem;
    }

    /* File upload */
    [data-testid="stUploadedFileInfo"] {
        background-color: #2e3344;
        border-radius: 8px;
        padding: 0.5rem;
    }

    /* Buttons and inputs */
    .stButton>button {
        background-color: #4CAF50 !important;
        color: white !important;
        border: none !important;
        padding: 0.5rem 1rem !important;
        border-radius: 8px !important;
    }
    .stButton>button:hover {
        background-color: #45a049 !important;
    }

    /* Hide label but keep for accessibility */
    .hide-label label {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def main():
    # Sidebar
    with st.sidebar:
        # Logo and title
        st.title("🔮 ChronoSense AI")
        st.caption("Enterprise Forecasting Platform")

        # API Status
        api_status = check_api_status()
        st.markdown(
            f"""
            <div class="status-indicator">
                <span class="status-online">✓</span>
                <span>API Status: {api_status}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Data Input Section
        st.subheader("📊 Data Input")
        st.info("Upload your time series data")

        data_source = st.radio(
            "Choose Data Source",
            ["Upload CSV", "Demo Data", "Sample Templates"],
            label_visibility="visible",
        )

        data = None
        if data_source == "Upload CSV":
            st.markdown("##### Upload your sales data (CSV)")
            uploaded_file = st.file_uploader(
                "Upload CSV file",
                type=["csv"],
                label_visibility="collapsed",
            )
            if uploaded_file:
                data = load_csv_data(uploaded_file)
        elif data_source == "Demo Data":
            if st.button("Load Demo Data"):
                data = generate_demo_data()

        # Model Settings
        st.subheader("⚙️ Model Settings")
        st.info("Configure forecast parameters")

        model = st.selectbox(
            "Forecasting Model",
            ["Prophet"],
            label_visibility="visible",
        )
        horizon = st.slider(
            "Forecast Horizon (Days)",
            7,
            365,
            30,
            label_visibility="visible",
        )

    # Main content
    # Help section
    with st.expander("📖 How to Use This App"):
        st.markdown(
            """
            1. Choose your data source from the sidebar
            2. Upload your CSV file or use demo data
            3. Configure forecast settings
            4. Generate your forecast
            """
        )

    # Title section
    st.markdown(
        """
        <div style="
            text-align: center;
            padding: 3rem;
            background: linear-gradient(120deg, #1e2130, #2c3e50);
            border-radius: 10px;
            margin: 2rem 0;
        ">
            <div style="font-size: 64px; margin-bottom: 1rem;">🔮</div>
            <h1 style="font-size: 3rem; margin-bottom: 1rem;">ChronoSense AI</h1>
            <p style="font-size: 1.5rem; color: #e0e0e0;">
                Transform your business with enterprise-grade sales forecasting
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Key Features
    st.markdown("## ⭐ Key Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-title">
                    🎯 High Accuracy
                </div>
                <p>Industry-leading<br>4.64% MAPE</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-title">
                    ⚡ Real-time Analysis
                </div>
                <p>Instant forecasting<br>and insights</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-title">
                    📊 Rich Analytics
                </div>
                <p>Comprehensive<br>business intelligence</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Forecast section
    if data is not None:
        st.markdown("<hr>", unsafe_allow_html=True)
        if st.button("🔮 Generate Forecast", use_container_width=True):
            with st.spinner("Creating forecast..."):
                result = create_forecast(data, horizon)
                if result:
                    forecast_df = pd.DataFrame(result["forecast"])
                    forecast_df["date"] = pd.to_datetime(forecast_df["date"])

                    fig = create_forecast_plot(data, forecast_df)
                    st.plotly_chart(fig, use_container_width=True)
                    display_metrics(result.get("metrics", {}))


if __name__ == "__main__":
    main()
