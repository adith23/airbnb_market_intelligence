import streamlit as st

from src.platform.mlops.serving.ml_client import predict_price

st.set_page_config(page_title="Price Estimator", page_icon="🤖", layout="wide")

st.title("🤖 Interactive Price Estimator")

st.markdown("""
This tool uses our trained **XGBoost** model and **Quantile Regression** models to estimate the optimal baseline nightly price 
based on property features.
""")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Property Configuration")
    with st.container(border=True):
        city = st.selectbox("City Context", ["Paris", "New York City", "Barcelona", "Amsterdam"])
        room_type = st.selectbox(
            "Room Type",
            ["Entire home/apt", "Private room", "Hotel room", "Shared room"],
        )

        col_a, col_b = st.columns(2)
        with col_a:
            accommodates = st.number_input("Accommodates", 1, 16, 2)
            bedrooms = st.number_input("Bedrooms", 1, 10, 1)
        with col_b:
            bathrooms = st.number_input("Bathrooms", 1.0, 10.0, 1.0, step=0.5)
            beds = st.number_input("Beds", 1, 15, 1)

        st.divider()
        st.markdown("**Premium Amenities**")
        col_c, col_d = st.columns(2)
        with col_c:
            ac = st.checkbox("Air Conditioning", value=True)
            pool = st.checkbox("Swimming Pool")
            gym = st.checkbox("Gym Access")
        with col_d:
            parking = st.checkbox("Free Parking")
            workspace = st.checkbox("Dedicated Workspace")
            tv = st.checkbox("TV / Streaming")

        user_inputs = {
            "accommodates": accommodates,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "beds": beds,
            "has_air_conditioning": 1 if ac else 0,
            "has_pool": 1 if pool else 0,
            "has_gym": 1 if gym else 0,
            "has_parking": 1 if parking else 0,
            "has_workspace": 1 if workspace else 0,
            "has_tv": 1 if tv else 0,
        }

        # Save to session state for Explainability tab
        st.session_state["user_inputs"] = user_inputs

        # City specific coordinates and OHE mapping
        # XGBoost splits heavily on latitude/longitude, so we MUST provide realistic
        # coordinates for the selected city, otherwise the model ignores the city OHE flag!
        city_coords = {
            "Paris": {"lat": 48.8566, "lon": 2.3522},
            "New York City": {"lat": 40.7128, "lon": -74.0060},
            "Barcelona": {"lat": 41.3851, "lon": 2.1734},
            "Amsterdam": {"lat": 52.3676, "lon": 4.9041},
        }

        user_inputs["latitude"] = city_coords[city]["lat"]
        user_inputs["longitude"] = city_coords[city]["lon"]
        user_inputs["distance_to_centre_km"] = 3.0  # Assuming property is ~3km from center

        if city != "Paris":
            user_inputs[f"city_{city}"] = 1

        # OHE mappings for Room Type
        if room_type != "Entire home/apt":
            user_inputs[f"room_type_{room_type}"] = 1

with col2:
    st.subheader("Estimated Nightly Rate")

    with st.spinner("Running Inference..."):
        pred, low, high = predict_price(user_inputs)

    st.metric(label="Recommended Base Price (USD)", value=f"${pred:,.0f} / night")

    st.markdown(
        f"""
    ### Expected Pricing Range
    Based on market quantiles (10th - 90th percentile), this property should command between:
    
    <div style='padding: 20px; background-color: #2e2e2e; border-radius: 10px; border-left: 5px solid #FF5A5F; font-size: 24px;'>
    <b>${low:,.0f} &nbsp; — &nbsp; ${high:,.0f}</b>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.info(
        "⚠️ **Note:** This is a *Baseline Estimator*. A true Dynamic Pricing Engine would apply temporal multipliers based on calendar events, weekend premiums, and lead-time factors."
    )
