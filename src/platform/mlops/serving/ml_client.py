"""ML Client to load models and perform inference."""

import json

import joblib
import pandas as pd
import streamlit as st

from dashboard.config import MODELS_DIR

# Dynamic resolution of the latest model run directory
def get_latest_run() -> str:
    """Retrieve the run ID dynamically. Checks environment override first, then does auto-discovery."""
    import os

    # Allow manual override via environment variable
    env_override = os.environ.get("LATEST_RUN")
    if env_override:
        return env_override

    # Auto-discover the latest model folder by timestamp name
    try:
        if MODELS_DIR.exists():
            subdirs = [
                d for d in MODELS_DIR.iterdir()
                if d.is_dir() and (d / "xgboost.joblib").exists()
            ]
            if subdirs:
                # Alphanumeric sort works chronologically since names start with YYYYMMDD_HHMMSS
                latest_dir = max(subdirs, key=lambda x: x.name)
                return latest_dir.name
    except Exception:
        pass

    # Safe fallback default ID
    return "20260623_025238_9d636ad3"


LATEST_RUN = get_latest_run()



@st.cache_resource
def load_models():
    """Load the XGBoost and Quantile Regression models into memory once."""
    model_path = MODELS_DIR / LATEST_RUN
    main_model = joblib.load(model_path / "xgboost.joblib")

    try:
        q_low = joblib.load(model_path / "q_0.10.joblib")
        q_high = joblib.load(model_path / "q_0.90.joblib")
    except FileNotFoundError:
        q_low, q_high = None, None

    with open(model_path / "feature_columns.json") as f:
        features = json.load(f)

    return main_model, q_low, q_high, features


def build_feature_vector(user_inputs: dict, features: list) -> pd.DataFrame:
    """Construct the dataframe vector with defaults."""
    df = pd.DataFrame(0.0, index=[0], columns=features)

    defaults = {
        "accommodates": 2.0,
        "bedrooms": 1.0,
        "beds": 1.0,
        "bathrooms": 1.0,
        "amenity_count": 15.0,
        "number_of_reviews": 10.0,
        "review_scores_rating": 4.8,
        "availability_365": 100.0,
        "minimum_nights": 2.0,
        "host_tenure_years": 5.0,
        "latitude": 48.85,
        "longitude": 2.35,
        "distance_to_centre_km": 3.0,
        "occupancy_rate_pct": 50.0,
    }

    for k, v in defaults.items():
        if k in df.columns:
            df.loc[0, k] = v

    for k, v in user_inputs.items():
        if k in df.columns:
            df.loc[0, k] = v

    if "accommodates_x_bedrooms" in df.columns:
        df.loc[0, "accommodates_x_bedrooms"] = df.loc[0, "accommodates"] * df.loc[0, "bedrooms"]
    if "bedrooms_x_bathrooms" in df.columns:
        df.loc[0, "bedrooms_x_bathrooms"] = df.loc[0, "bedrooms"] * df.loc[0, "bathrooms"]

    return df


def predict_price(user_inputs: dict) -> tuple[float, float, float]:
    """Generate price prediction and bounds."""
    main_model, q_low, q_high, features = load_models()

    df = build_feature_vector(user_inputs, features)

    pred = float(main_model.predict(df)[0])

    # Handle bounds if quantile models exist
    if q_low is not None and q_high is not None:
        low = float(q_low.predict(df)[0])
        high = float(q_high.predict(df)[0])
    else:
        # Fallback to rough 20% margin if quantile models are missing
        low = pred * 0.8
        high = pred * 1.2

    # The models are trained on log1p(price) to handle skewness.
    # We must inverse transform (expm1) to get back to raw dollars!
    import numpy as np

    pred_dollar = np.expm1(pred)
    low_dollar = np.expm1(low)
    high_dollar = np.expm1(high)

    # Ensure logical bounds
    low_dollar = min(pred_dollar, max(10.0, low_dollar))
    high_dollar = max(pred_dollar, high_dollar)

    return pred_dollar, low_dollar, high_dollar
