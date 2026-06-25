"""Intervention Radar decision engine module."""

import altair as alt
import streamlit as st

from dashboard.backend.data_service import (
    fetch_available_cities,
    fetch_intervention_radar,
    fetch_neighbourhood_interventions,
)

st.set_page_config(page_title="Intervention Radar", page_icon="🎯", layout="wide")
st.title("🎯 Market Health & Intervention Radar")

st.markdown("""
This is a **Decision Engine**, not just a dashboard. 
It continuously scores listings and neighbourhoods on actionable business dimensions (pricing competitiveness, review quality, occupancy proxies) and generates a ranked Intervention Queue for operations leads and revenue strategists.
""")

# Global Filters
st.sidebar.header("Filters")
cities_map = fetch_available_cities()
selected_city_name = st.sidebar.selectbox(
    "Select Market",
    options=["All Cities"] + list(cities_map.values()),
    key="radar_city"
)
city_key = None
if selected_city_name != "All Cities":
    city_key = [k for k, v in cities_map.items() if v == selected_city_name][0]

st.divider()

col1, col2 = st.columns([1, 1])

with st.spinner("Scoring thousands of entities using composite business logic..."):
    df_radar = fetch_intervention_radar(city_key)
    df_hoods = fetch_neighbourhood_interventions(city_key)

with col1:
    st.subheader("Neighbourhood Opportunity Zones")
    st.info("Ranked map showing high demand areas, premium pockets, and over-saturated zones.")
    if not df_hoods.empty:
        # Filter out neutral for impact
        df_zones = df_hoods[df_hoods["opportunity_zone"] != "Neutral"]
        st.dataframe(
            df_zones.style.format(
                {
                    "avg_price": "${:.0f}",
                    "avg_occupancy_pct": "{:.1f}%",
                    "avg_rating": "{:.2f}",
                }
            ),
            width="stretch",
        )

with col2:
    st.subheader("Intervention Queue Breakdown")
    st.info(
        "Aggregated volume of listings requiring specific operational or pricing interventions."
    )
    if not df_radar.empty:
        action_counts = df_radar["priority_action"].value_counts().reset_index()
        action_counts.columns = ["priority_action", "count"]

        chart = (
            alt.Chart(action_counts)
            .mark_bar()
            .encode(
                x=alt.X("count:Q", title="Number of Listings"),
                y=alt.Y(
                    "priority_action:N",
                    title="",
                    sort="-x",
                    axis=alt.Axis(labelLimit=300),
                ),
                color=alt.Color(
                    "priority_action:N",
                    legend=None,
                    scale=alt.Scale(scheme="tableau10"),
                ),
                tooltip=["priority_action", "count"],
            )
            .properties(height=250)
        )
        st.altair_chart(chart, use_container_width=True)

st.divider()

st.subheader("Detailed Listing Intervention Table")
st.markdown("Actionable list ranked by severity (lowest Health Score first).")

if not df_radar.empty:
    st.dataframe(
        df_radar[
            [
                "name",
                "neighbourhood_name",
                "price_usd",
                "rating",
                "occupancy_pct",
                "health_score",
                "priority_action",
            ]
        ]
        .style.background_gradient(subset=["health_score"], cmap="RdYlGn")
        .format(
            {
                "price_usd": "${:.0f}",
                "rating": "{:.2f}",
                "occupancy_pct": "{:.1f}%",
                "health_score": "{:.1f}/10",
            }
        ),
        width="stretch",
    )
else:
    st.success("No critical interventions required for this market segment!")
