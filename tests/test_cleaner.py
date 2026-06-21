"""Unit tests for pipeline.cleaner — data cleaning and standardization.

Tests each cleaning transformation with:
  - Normal/expected inputs
  - Edge cases (nulls, empty strings, malformed values)
  - Validation flag computation
  - Output partitioning logic
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from pipeline.cleaner import (
    CleaningResult,
    _cast_boolean_columns,
    _clean_percentage_columns,
    _clean_price_columns,
    _compute_validation_flags,
    _count_amenities,
    _normalize_text_columns,
    _parse_bathrooms_column,
    _parse_date_columns,
    _strip_html_column,
    _apply_missing_strategies,
)


# ===================================================================
# Price cleaning
# ===================================================================

class TestCleanPriceColumns:
    def test_usd_prices(self):
        df = pl.DataFrame({"price": ["$100.00", "$1,250.00", "$0.00"]})
        result = _clean_price_columns(df, ["price"])
        assert result["price"].to_list() == [100.0, 1250.0, 0.0]

    def test_euro_prices(self):
        df = pl.DataFrame({"price": ["€99.00", "€1.500,00"]})
        result = _clean_price_columns(df, ["price"])
        # €1.500,00 → after stripping €, → "1.500,00" → cast fails → null
        # (European format with comma-decimal is not handled — by design)
        assert result["price"][0] == 99.0

    def test_null_handling(self):
        df = pl.DataFrame({"price": ["$100.00", None, "$50.00"]})
        result = _clean_price_columns(df, ["price"])
        assert result["price"][0] == 100.0
        assert result["price"][1] is None
        assert result["price"][2] == 50.0

    def test_empty_column_list(self):
        df = pl.DataFrame({"price": ["$100.00"]})
        result = _clean_price_columns(df, [])
        assert result["price"].to_list() == ["$100.00"]

    def test_missing_column(self):
        df = pl.DataFrame({"other": [1]})
        result = _clean_price_columns(df, ["price"])
        assert "other" in result.columns


# ===================================================================
# Boolean casting
# ===================================================================

class TestCastBooleanColumns:
    def test_standard_values(self):
        df = pl.DataFrame({"flag": ["t", "f", "t"]})
        result = _cast_boolean_columns(df, ["flag"])
        assert result["flag"].to_list() == [True, False, True]

    def test_null_handling(self):
        df = pl.DataFrame({"flag": ["t", None, "f"]})
        result = _cast_boolean_columns(df, ["flag"])
        assert result["flag"][0] is True
        assert result["flag"][1] is None
        assert result["flag"][2] is False

    def test_invalid_values_become_null(self):
        df = pl.DataFrame({"flag": ["t", "yes", "f"]})
        result = _cast_boolean_columns(df, ["flag"])
        assert result["flag"][0] is True
        assert result["flag"][1] is None
        assert result["flag"][2] is False

    def test_multiple_columns(self):
        df = pl.DataFrame({"a": ["t", "f"], "b": ["f", "t"]})
        result = _cast_boolean_columns(df, ["a", "b"])
        assert result["a"].to_list() == [True, False]
        assert result["b"].to_list() == [False, True]


# ===================================================================
# Date parsing
# ===================================================================

class TestParseDateColumns:
    def test_valid_dates(self):
        df = pl.DataFrame({"date": ["2024-01-15", "2024-12-31"]})
        result = _parse_date_columns(df, ["date"])
        assert result["date"].dtype == pl.Date

    def test_null_dates(self):
        df = pl.DataFrame({"date": ["2024-01-15", None]})
        result = _parse_date_columns(df, ["date"])
        assert result["date"][0] is not None
        assert result["date"][1] is None

    def test_invalid_date_becomes_null(self):
        df = pl.DataFrame({"date": ["2024-01-15", "not-a-date"]})
        result = _parse_date_columns(df, ["date"])
        assert result["date"][0] is not None
        assert result["date"][1] is None


# ===================================================================
# Percentage cleaning
# ===================================================================

class TestCleanPercentageColumns:
    def test_standard_pct(self):
        df = pl.DataFrame({"rate": ["95%", "100%", "0%"]})
        result = _clean_percentage_columns(df, ["rate"])
        values = result["rate"].to_list()
        assert abs(values[0] - 0.95) < 1e-6
        assert abs(values[1] - 1.0) < 1e-6
        assert abs(values[2] - 0.0) < 1e-6

    def test_null_na(self):
        df = pl.DataFrame({"rate": ["95%", None]})
        result = _clean_percentage_columns(df, ["rate"])
        assert result["rate"][1] is None


# ===================================================================
# Bathrooms parsing
# ===================================================================

class TestParseBathroomsColumn:
    def test_numeric_baths(self):
        df = pl.DataFrame({"bathrooms_text": ["1.5 baths", "2 baths"]})
        result = _parse_bathrooms_column(df)
        assert result["bathrooms"].to_list() == [1.5, 2.0]
        assert result["bathrooms_shared"].to_list() == [False, False]

    def test_half_bath(self):
        df = pl.DataFrame({"bathrooms_text": ["Half-bath"]})
        result = _parse_bathrooms_column(df)
        assert result["bathrooms"][0] == 0.5

    def test_shared(self):
        df = pl.DataFrame({"bathrooms_text": ["Shared half-bath"]})
        result = _parse_bathrooms_column(df)
        assert result["bathrooms"][0] == 0.5
        assert result["bathrooms_shared"][0] is True

    def test_null_input(self):
        df = pl.DataFrame({"bathrooms_text": [None, "1 bath"]})
        result = _parse_bathrooms_column(df)
        assert result["bathrooms"][0] is None

    def test_no_column(self):
        df = pl.DataFrame({"other": [1]})
        result = _parse_bathrooms_column(df)
        assert "bathrooms" not in result.columns


# ===================================================================
# Amenity counting
# ===================================================================

class TestCountAmenities:
    def test_json_array(self):
        df = pl.DataFrame({"amenities": ['["Wifi","Kitchen","Pool"]']})
        result = _count_amenities(df)
        assert result["amenity_count"][0] == 3

    def test_empty_array(self):
        df = pl.DataFrame({"amenities": ["[]"]})
        result = _count_amenities(df)
        assert result["amenity_count"][0] == 0

    def test_null(self):
        df = pl.DataFrame({"amenities": [None]})
        result = _count_amenities(df)
        assert result["amenity_count"][0] == 0


# ===================================================================
# HTML stripping
# ===================================================================

class TestStripHtmlColumn:
    def test_basic_html(self):
        df = pl.DataFrame({"text": ["<b>Hello</b> world"]})
        result = _strip_html_column(df, "text")
        assert result["text"][0] == "Hello world"

    def test_entities(self):
        df = pl.DataFrame({"text": ["A &amp; B"]})
        result = _strip_html_column(df, "text")
        assert result["text"][0] == "A & B"

    def test_no_column(self):
        df = pl.DataFrame({"other": [1]})
        result = _strip_html_column(df, "text")
        assert "other" in result.columns


# ===================================================================
# Text normalization
# ===================================================================

class TestNormalizeTextColumns:
    def test_whitespace_stripping(self):
        df = pl.DataFrame({"type": ["  Studio  ", "Apartment"]})
        result = _normalize_text_columns(df, ["type"])
        assert result["type"].to_list() == ["Studio", "Apartment"]

    def test_multi_space_collapse(self):
        df = pl.DataFrame({"type": ["Entire   home", "Private  room"]})
        result = _normalize_text_columns(df, ["type"])
        assert result["type"][0] == "Entire home"


# ===================================================================
# Missing value strategies
# ===================================================================

class TestApplyMissingStrategies:
    def test_sentinel_fill(self):
        df = pl.DataFrame({"name": ["Alice", None, "Bob"]})
        strategies = {"name": {"strategy": "sentinel", "fill_value": "Unknown"}}
        result, imputed = _apply_missing_strategies(df, strategies)
        assert result["name"].to_list() == ["Alice", "Unknown", "Bob"]
        assert imputed == []

    def test_explicit_null_no_change(self):
        df = pl.DataFrame({"score": [4.5, None, 3.2]})
        strategies = {"score": {"strategy": "explicit_null"}}
        result, imputed = _apply_missing_strategies(df, strategies)
        assert result["score"][1] is None

    def test_reject_no_change(self):
        df = pl.DataFrame({"id": [1, None, 3]})
        strategies = {"id": {"strategy": "reject"}}
        result, imputed = _apply_missing_strategies(df, strategies)
        assert result["id"][1] is None

    def test_impute_zero(self):
        df = pl.DataFrame({"count": [5, None, 3]})
        strategies = {"count": {"strategy": "impute_zero"}}
        result, imputed = _apply_missing_strategies(df, strategies)
        assert result["count"].to_list() == [5, 0, 3]


# ===================================================================
# Validation flags
# ===================================================================

class TestComputeValidationFlags:
    def test_all_valid(self):
        df = pl.DataFrame({"price": [100.0, 50.0]})
        rules = {"price": [{"check": "not_null"}, {"check": "positive"}]}
        result = _compute_validation_flags(df, rules)
        assert result["_is_valid"].to_list() == [True, True]
        assert all(f == "" for f in result["_validation_flags"].to_list())

    def test_null_flagged(self):
        df = pl.DataFrame({"price": [100.0, None]})
        rules = {"price": [{"check": "not_null"}]}
        result = _compute_validation_flags(df, rules)
        assert result["_is_valid"][0] is True
        assert result["_is_valid"][1] is False
        assert "price_not_null" in result["_validation_flags"][1]

    def test_positive_flagged(self):
        df = pl.DataFrame({"price": [100.0, -5.0, 0.0]})
        rules = {"price": [{"check": "positive"}]}
        result = _compute_validation_flags(df, rules)
        assert result["_is_valid"][0] is True
        assert result["_is_valid"][1] is False
        assert result["_is_valid"][2] is False  # 0 is not positive

    def test_range_flagged(self):
        df = pl.DataFrame({"lat": [48.8, -100.0, 48.9]})
        rules = {"lat": [{"check": "range", "min": -90.0, "max": 90.0}]}
        result = _compute_validation_flags(df, rules)
        assert result["_is_valid"][0] is True
        assert result["_is_valid"][1] is False
        assert result["_is_valid"][2] is True

    def test_enum_flagged(self):
        df = pl.DataFrame({"room": ["Private room", "Unknown type"]})
        rules = {"room": [{"check": "enum", "values": ["Private room", "Entire home/apt"]}]}
        result = _compute_validation_flags(df, rules)
        assert result["_is_valid"][0] is True
        assert result["_is_valid"][1] is False

    def test_multiple_flags_combined(self):
        df = pl.DataFrame({"price": [None], "lat": [-200.0]})
        rules = {
            "price": [{"check": "not_null"}],
            "lat": [{"check": "range", "min": -90.0, "max": 90.0}],
        }
        result = _compute_validation_flags(df, rules)
        assert result["_is_valid"][0] is False
        flags = result["_validation_flags"][0]
        assert "price_not_null" in flags
        assert "lat_range" in flags

    def test_no_rules(self):
        df = pl.DataFrame({"x": [1, 2, 3]})
        result = _compute_validation_flags(df, {})
        assert all(result["_is_valid"].to_list())

    def test_missing_column_skipped(self):
        df = pl.DataFrame({"x": [1, 2]})
        rules = {"nonexistent": [{"check": "not_null"}]}
        result = _compute_validation_flags(df, rules)
        assert all(result["_is_valid"].to_list())
