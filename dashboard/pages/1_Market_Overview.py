"""Market Overview & Geospatial Intelligence page."""

import sys
from pathlib import Path

import streamlit as st

# Ensure the root project directory is in the PYTHONPATH
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.components.charts import (
    render_neighbourhood_roi_chart,
    render_revenue_scatter,
    render_room_type_pie,
)
from dashboard.components.kpi_cards import render_kpi_row
from dashboard.components.map_renderer import render_pricing_heatmap
from dashboard.backend.data_service import (
    fetch_available_cities,
    fetch_executive_kpis,
    fetch_geospatial_data,
    fetch_neighbourhood_metrics,
    fetch_room_type_metrics,
)

st.set_page_config(page_title="Market Overview", page_icon="🏠", layout="wide")

st.title("🏠 Market Overview & Geospatial Intelligence")

# Global Filters
st.sidebar.header("Filters")
cities_map = fetch_available_cities()
options = ["All Cities"] + list(cities_map.values())
default_index = options.index("New York City") if "New York City" in options else 0
selected_city_name = st.sidebar.selectbox(
    "Select Market",
    options=options,
    index=default_index,
)

# Determine query key
city_key = None
if selected_city_name != "All Cities":
    city_key = [k for k, v in cities_map.items() if v == selected_city_name][0]

# Load Data
with st.spinner("Fetching market data (this queries millions of rows via DuckDB)..."):
    kpis = fetch_executive_kpis(city_key)
    geo_df = fetch_geospatial_data(city_key)
    hood_df = fetch_neighbourhood_metrics(city_key)
    room_df = fetch_room_type_metrics(city_key)

# Render KPIs
st.subheader("Executive KPIs")
render_kpi_row(kpis)

st.divider()

# Layout for Maps and Charts
col_map, col_charts = st.columns([1.5, 1])

with col_map:
    st.subheader("Interactive Pricing Heatmap")
    st.markdown(
        "*Use mouse to pan, zoom, and rotate (Right-click + Drag). Elevation represents aggregated price density.*"
    )
    render_pricing_heatmap(geo_df)

with col_charts:
    st.subheader("Top Neighbourhoods by Price")
    render_neighbourhood_roi_chart(hood_df)

st.divider()

col_pie, col_scatter = st.columns([1, 1.5])

with col_pie:
    st.subheader("Market Composition")
    render_room_type_pie(room_df)

with col_scatter:
    st.subheader("Occupancy vs Revenue (by Neighbourhood)")
    render_revenue_scatter(hood_df)
