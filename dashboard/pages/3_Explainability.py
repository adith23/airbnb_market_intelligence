import streamlit as st

from dashboard.backend.shap_explainer import render_shap_waterfall
from src.platform.mlops.serving.ml_client import build_feature_vector, load_models

st.set_page_config(page_title="Explainability & ROI", page_icon="🔍", layout="wide")
st.title("🔍 Explainability & Amenity ROI")

st.markdown("""
This module uses **SHAP (SHapley Additive exPlanations)** to deconstruct the ML model's price predictions. 
It shows *exactly* how many dollars each feature adds or subtracts from the base price.
""")

st.divider()

if "user_inputs" not in st.session_state:
    st.warning(
        "⚠️ No property configured! Please go to the **Price Estimator** tab and configure a property first to see its SHAP explanation."
    )
else:
    st.subheader("Local Explanation: Your Custom Property")

    with st.spinner("Calculating SHAP values (this uses the TreeExplainer)..."):
        user_inputs = st.session_state["user_inputs"]
        _, _, _, features = load_models()
        df_vector = build_feature_vector(user_inputs, features)

        st.markdown(
            "**SHAP Waterfall Chart:** How the final price was derived from the base expected value."
        )
        render_shap_waterfall(df_vector)

    st.info(
        "💡 **How to read this:** The `E[f(x)]` at the bottom is the average price across the entire dataset. The red bars push the price up (premium features), and the blue bars push the price down (penalties or missing amenities). The top value `f(x)` is the final predicted price."
    )
