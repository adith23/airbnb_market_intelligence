"""Supply Concentration and Temporal Trends page."""

import altair as alt
import streamlit as st

from dashboard.backend.data_service import (
    fetch_available_cities,
    fetch_host_concentration,
    fetch_temporal_trends,
)

st.set_page_config(page_title="Supply & Demand", page_icon="📈", layout="wide")
st.title("📈 Supply Concentration & Temporal Trends")

# Global Filters
st.sidebar.header("Filters")
cities_map = fetch_available_cities()
selected_city_name = st.sidebar.selectbox(
    "Select Market",
    options=["All Cities"] + list(cities_map.values()),
    key="sup_city"
)
city_key = None
if selected_city_name != "All Cities":
    city_key = [k for k, v in cities_map.items() if v == selected_city_name][0]

col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("Commercial vs. Casual Split")
    st.markdown("Pricing strategy divergence based on portfolio size.")
    with st.spinner("Segmenting host data..."):
        df_host = fetch_host_concentration(city_key)

    if not df_host.empty:
        chart = (
            alt.Chart(df_host)
            .mark_bar()
            .encode(
                x=alt.X("host_segment:N", title="", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("median_price_usd:Q", title="Median Price (USD)"),
                color=alt.Color("host_segment:N", legend=None, scale=alt.Scale(scheme="set2")),
                tooltip=[
                    "host_segment",
                    "listing_count",
                    "median_price_usd",
                    "avg_occupancy_pct",
                ],
            )
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)

        st.dataframe(
            df_host.style.format(
                {
                    "median_price_usd": "${:.0f}",
                    "avg_occupancy_pct": "{:.1f}%",
                    "median_revenue": "${:,.0f}",
                }
            ),
            use_container_width=True,
        )
    else:
        st.warning("⚠️ No host concentration data available for the selected market.")

with col2:
    st.subheader("Seasonal Yield Curves & Forward Demand")
    st.markdown("Dynamic constraint tracking from forward calendar availability (next 365 days).")
    with st.spinner("Querying millions of calendar records for temporal trends..."):
        df_temporal = fetch_temporal_trends(city_key)

    if not df_temporal.empty:
        # Create dual axis chart
        base = alt.Chart(df_temporal).encode(x=alt.X("date:T", title="Date"))

        line_price = base.mark_line(color="#FF5A5F", strokeWidth=3).encode(
            y=alt.Y(
                "avg_price:Q",
                title="Avg Daily Price (USD)",
                scale=alt.Scale(zero=False),
            ),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("avg_price:Q", title="Avg Price", format="$.0f"),
            ],
        )

        area_occ = base.mark_area(opacity=0.2, color="teal").encode(
            y=alt.Y("booked_occupancy_rate:Q", title="Booked Occupancy (%)"),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("booked_occupancy_rate:Q", title="Occupancy", format=".1f%"),
            ],
        )

        chart = (
            alt.layer(area_occ, line_price).resolve_scale(y="independent").properties(height=450)
        )

        st.altair_chart(chart, use_container_width=True)
    else:
        st.warning("⚠️ No forward calendar demand data available for the selected market. Run the calendar ingest/model pipelines to generate predictions.")
