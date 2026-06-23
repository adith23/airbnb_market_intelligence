"""UI components for KPI rendering."""

import streamlit as st


def render_kpi_row(kpi_data: dict):
    """Render the executive KPIs in a row of columns."""
    if not kpi_data:
        st.warning("No data available for KPIs.")
        return

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Active Listings",
            value=f"{kpi_data.get('total_listings', 0):,.0f}",
        )

    with col2:
        st.metric(
            label="Average Daily Rate",
            value=f"${kpi_data.get('avg_daily_rate_usd', 0):,.2f}",
        )

    with col3:
        occ = kpi_data.get("avg_occupancy_pct")
        st.metric(label="Est. Occupancy Rate", value=f"{occ:.1f}%" if occ else "N/A")

    with col4:
        prof_pct = kpi_data.get("professional_host_pct")
        st.metric(
            label="Professional Host % (>2)",
            value=f"{prof_pct:.1f}%" if prof_pct else "N/A",
        )
