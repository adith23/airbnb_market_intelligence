"""Geospatial mapping components using PyDeck for high performance."""

import pandas as pd
import pydeck as pdk
import streamlit as st


def render_pricing_heatmap(df: pd.DataFrame):
    """Render a PyDeck WebGL HexagonLayer map for pricing densities."""
    if df.empty:
        st.warning("No geospatial data to display.")
        return

    # Default view centered on the mean coordinates
    view_state = pdk.ViewState(
        latitude=df["latitude"].mean(),
        longitude=df["longitude"].mean(),
        zoom=11,
        pitch=45,
    )

    # HexagonLayer automatically aggregates data into bins
    layer = pdk.Layer(
        "HexagonLayer",
        data=df,
        get_position=["longitude", "latitude"],
        get_weight="price_usd",
        get_elevation_weight="price_usd",
        elevation_scale=50,
        elevation_range=[0, 3000],
        pickable=True,
        extruded=True,
        auto_highlight=True,
        color_range=[
            [255, 255, 204],
            [255, 237, 160],
            [254, 217, 118],
            [254, 178, 76],
            [253, 141, 60],
            [240, 59, 32],
            [189, 0, 38],
        ],
    )

    # Tooltip configuration
    tooltip = {
        "html": "<b>Hexagon Density/Avg Price</b><br/>Elevation reflects local price density.",
        "style": {"backgroundColor": "steelblue", "color": "white"},
    }

    r = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style="dark",
    )

    st.pydeck_chart(r)
