"""SHAP Explainer module for interpreting ML predictions."""

import matplotlib.pyplot as plt
import shap
import streamlit as st

from dashboard.backend.ml_service import load_models


@st.cache_resource
def get_shap_explainer():
    """Load and cache the TreeExplainer."""
    main_model, _, _, _ = load_models()
    return shap.TreeExplainer(main_model)


def render_shap_waterfall(df):
    """Render the SHAP waterfall chart for a given feature vector."""
    explainer = get_shap_explainer()
    shap_values = explainer(df)

    # Matplotlib config for dark mode
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 6))

    # Generate the plot
    shap.plots.waterfall(shap_values[0], show=False)

    # Clean up background
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    plt.tight_layout()

    st.pyplot(fig)
