"""UI components for analytical charts."""

import altair as alt
import pandas as pd
import streamlit as st

from dashboard.config import PRIMARY_COLOR


def render_neighbourhood_roi_chart(df: pd.DataFrame):
    """Bar chart showing top neighbourhoods by median price."""
    if df.empty:
        return st.warning("No neighbourhood data available.")

    chart = (
        alt.Chart(df)
        .mark_bar(color=PRIMARY_COLOR, opacity=0.8)
        .encode(
            x=alt.X("median_price_usd:Q", title="Median Daily Price (USD)"),
            y=alt.Y("neighbourhood_name:N", sort="-x", title="Neighbourhood"),
            tooltip=[
                alt.Tooltip("neighbourhood_name", title="Neighbourhood"),
                alt.Tooltip("listing_count", title="Active Listings"),
                alt.Tooltip("median_price_usd", title="Median Price ($)"),
                alt.Tooltip("avg_occupancy_pct", title="Occupancy Rate (%)", format=".1f"),
            ],
        )
        .properties(height=400)
    )
    st.altair_chart(chart, use_container_width=True)


def render_room_type_pie(df: pd.DataFrame):
    """Donut chart for room type distribution."""
    if df.empty:
        return st.warning("No room type data available.")

    chart = (
        alt.Chart(df)
        .mark_arc(innerRadius=60)
        .encode(
            theta=alt.Theta(field="listing_count", type="quantitative"),
            color=alt.Color(
                field="room_type",
                type="nominal",
                scale=alt.Scale(scheme="reds"),
                title="Room Type",
            ),
            tooltip=[
                alt.Tooltip("room_type", title="Room Type"),
                alt.Tooltip("listing_count", title="Listings"),
                alt.Tooltip("median_price_usd", title="Median Price ($)"),
            ],
        )
        .properties(height=400)
    )
    st.altair_chart(chart, use_container_width=True)


def render_revenue_scatter(df: pd.DataFrame):
    """Scatter plot of occupancy vs monthly revenue."""
    if df.empty:
        return st.warning("No revenue data available.")

    chart = (
        alt.Chart(df)
        .mark_circle(color="teal", opacity=0.6)
        .encode(
            x=alt.X(
                "avg_occupancy_pct:Q",
                title="Avg Occupancy Rate (%)",
                scale=alt.Scale(zero=False),
            ),
            y=alt.Y("median_monthly_revenue:Q", title="Median Monthly Revenue (USD)"),
            size=alt.Size(
                "listing_count:Q",
                title="Market Size",
                scale=alt.Scale(range=[50, 1000]),
            ),
            tooltip=[
                alt.Tooltip("neighbourhood_name", title="Neighbourhood"),
                alt.Tooltip("median_monthly_revenue", title="Monthly Revenue ($)", format=",.0f"),
                alt.Tooltip("avg_occupancy_pct", title="Occupancy (%)", format=".1f"),
                alt.Tooltip("listing_count", title="Listings"),
            ],
        )
        .properties(height=400)
    )

    st.altair_chart(chart, use_container_width=True)
