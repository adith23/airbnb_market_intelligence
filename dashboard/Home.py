"""Main entry point for the Streamlit dashboard."""

import sys
from pathlib import Path

import streamlit as st

# Ensure the root project directory is in the PYTHONPATH
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.components.ai_chat import render_ai_chat
from dashboard.config import LAYOUT, PAGE_ICON, PAGE_TITLE

# Must be the first Streamlit command
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state="expanded",
)

# Sidebar Toggle for AI Assistant
if "show_ai_chat" not in st.session_state:
    st.session_state.show_ai_chat = False

st.sidebar.markdown("---")
if st.sidebar.button("💬 Ask AI Database Assistant", use_container_width=True):
    st.session_state.show_ai_chat = not st.session_state.show_ai_chat

# Dynamic Layout Configuration
if st.session_state.show_ai_chat:
    col_main, col_chat = st.columns([2.5, 1.5])
else:
    col_main, col_chat = st.columns([1, 0.0001])  # Effectively hides the second column

with col_main:
    st.title(f"{PAGE_ICON} {PAGE_TITLE}")
    st.markdown(
        "Welcome to the **Market Intelligence Command Center**. Please select a module from the sidebar."
    )

    st.markdown("""
    ### Command Center Modules
    1. **Market Overview:** Explore macroeconomic KPIs and interactive WebGL pricing density maps.
    2. **Price Estimator:** Predict baseline property pricing using XGBoost ML models.
    3. **Explainability:** Deconstruct predicted prices and visualize Amenity ROI via SHAP.
    4. **MLOps Governance:** Review cross-city generalizability and geographic fairness biases.
    5. **Valuation & Arbitrage:** Identify undervalued assets and arbitrage opportunities based on reviews vs pricing.
    6. **Supply & Demand:** Analyze commercial host concentration and 365-day seasonal yield curves from calendar data.
    7. **Intervention Radar:** A decision engine that scores listings and generates prioritized operational queues.
    """)

if st.session_state.show_ai_chat:
    with col_chat:
        render_ai_chat()
