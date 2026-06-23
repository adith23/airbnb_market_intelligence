"""Revenue and Valuation Intelligence page."""

import streamlit as st

from src.platform.data_engineering.storage.data_client import (
    fetch_available_cities,
    fetch_valuation_arbitrage,
)

st.set_page_config(page_title="Valuation & Arbitrage", page_icon="💰", layout="wide")
st.title("💰 Revenue & Valuation Intelligence")

st.markdown("""
Identify intrinsic value discrepancies and arbitrage opportunities across the market.
""")

# Global Filters
st.sidebar.header("Filters")
cities = ["All Cities"] + fetch_available_cities()
selected_city = st.sidebar.selectbox("Select Market", cities, key="val_city")
city_key = None if selected_city == "All Cities" else selected_city

st.subheader("The 'Undervalued' Index")
st.info(
    "Listings with exceptional reviews (>4.8 rating, 30+ reviews) but priced below the city median. These represent prime acquisition or optimization targets for property managers."
)

with st.spinner("Calculating valuation metrics across the database..."):
    df_arb = fetch_valuation_arbitrage(city_key)

if not df_arb.empty:
    st.dataframe(
        df_arb.style.highlight_max(subset=["estimated_monthly_revenue"], color="lightgreen").format(
            {
                "price_usd": "${:.0f}",
                "price_per_bedroom": "${:.0f}",
                "estimated_monthly_revenue": "${:,.0f}",
            }
        ),
        width="stretch",
    )
else:
    st.warning("No undervalued listings found matching the strict criteria.")
