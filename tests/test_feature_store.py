import pandas as pd
import numpy as np
import pytest

from pipeline.ml.feature_store import (
    extract_amenity_flags,
    compute_distance_to_centre_vectorised,
    create_interaction_terms,
    handle_missing_values,
    encode_categoricals,
    FeatureSet
)

def test_extract_amenity_flags():
    amenities = pd.Series([
        '["Wifi", "Kitchen", "Air conditioning"]',
        '["Pool", "TV"]',
        '[]',
        None
    ])
    
    keyword_map = {
        "has_wifi": ["wifi", "wi-fi", "internet"],
        "has_kitchen": ["kitchen"],
        "has_ac": ["air conditioning", "ac", "a/c"]
    }
    
    df = extract_amenity_flags(amenities, keyword_map)
    
    assert len(df) == 4
    assert list(df.columns) == ["has_wifi", "has_kitchen", "has_ac"]
    
    # First row has all 3
    assert df.loc[0, "has_wifi"] == 1
    assert df.loc[0, "has_kitchen"] == 1
    assert df.loc[0, "has_ac"] == 1
    
    # Second row has none of the keywords
    assert df.loc[1].sum() == 0
    
    # Empty and None handle correctly
    assert df.loc[2].sum() == 0
    assert df.loc[3].sum() == 0


def test_compute_distance_to_centre():
    df = pd.DataFrame({
        "latitude": [48.8566, 48.8600],
        "longitude": [2.3522, 2.3600],
        "city_key": ["paris", "paris"]
    })
    
    city_centres = {
        "paris": {"lat": 48.8566, "lon": 2.3522}
    }
    
    distances = compute_distance_to_centre_vectorised(df, city_centres)
    
    # First row is exactly at center, distance should be 0
    assert abs(distances[0] - 0.0) < 1e-5
    
    # Second row is slightly away
    assert distances[1] > 0.0


def test_create_interaction_terms():
    df = pd.DataFrame({
        "accommodates": [2, 4],
        "bedrooms": [1, 2]
    })
    
    interactions = [["accommodates", "bedrooms"]]
    
    result = create_interaction_terms(df, interactions)
    
    assert list(result.columns) == ["accommodates_x_bedrooms"]
    assert result.loc[0, "accommodates_x_bedrooms"] == 2 * 1
    assert result.loc[1, "accommodates_x_bedrooms"] == 4 * 2


def test_handle_missing_values():
    df = pd.DataFrame({
        "num": [1.0, np.nan, 3.0],
        "bool_val": [1.0, np.nan, 0.0],
        "cat": ["A", np.nan, "B"]
    })
    
    config = {
        "add_indicators": True,
        "categorical": "__MISSING__"
    }
    
    result, indicators = handle_missing_values(
        df, config, 
        numeric_cols=["num"], 
        boolean_cols=["bool_val"], 
        categorical_cols=["cat"]
    )
    
    # Numeric median imputation
    assert result.loc[1, "num"] == 2.0
    
    # Boolean imputation
    assert result.loc[1, "bool_val"] == 0
    
    # Categorical imputation
    assert result.loc[1, "cat"] == "__MISSING__"
    
    # Indicators
    assert "is_missing_num" in indicators
    assert "is_missing_num" in result.columns
    assert result.loc[0, "is_missing_num"] == 0
    assert result.loc[1, "is_missing_num"] == 1


def test_encode_categoricals():
    df = pd.DataFrame({
        "color": ["red", "blue", "green", "red", "yellow", "purple", "orange"]
    })
    
    # Max cardinality 2 -> Top 2 are 'red', and all others are 1 each.
    # Since red=2, and rest=1, top 2 categories will be kept, rest grouped.
    result = encode_categoricals(df, ["color"], max_cardinality=2)
    
    assert "color" not in result.columns
    # With drop_first=True, we get n-1 columns. We have 3 categories: red, another color(depends on ranking), __OTHER__
    # Because there are many colors with count 1, which one is chosen as 2nd is arbitrary but there will be __OTHER__
    assert any(col.startswith("color_") for col in result.columns)
