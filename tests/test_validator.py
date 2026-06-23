"""Unit tests for pipeline.validator — data quality validation."""

from __future__ import annotations

import polars as pl

from src.platform.data_engineering.ingestion.validator import (
    detect_duplicates,
    detect_scraping_artifacts,
    validate_column,
)

# ===================================================================
# Column validation
# ===================================================================


class TestValidateColumn:
    def test_not_null_pass(self):
        series = pl.Series("test", [1, 2, 3])
        result = validate_column(series, "test", {"not_null": True})
        assert result["passed"] is True

    def test_not_null_fail(self):
        series = pl.Series("test", [1, None, 3])
        result = validate_column(series, "test", {"not_null": True})
        assert result["passed"] is False
        assert any(v["rule"] == "not_null" for v in result["violations"])

    def test_unique_pass(self):
        series = pl.Series("test", [1, 2, 3])
        result = validate_column(series, "test", {"unique": True})
        assert result["passed"] is True

    def test_unique_fail(self):
        series = pl.Series("test", [1, 2, 2])
        result = validate_column(series, "test", {"unique": True})
        assert result["passed"] is False

    def test_positive_pass(self):
        series = pl.Series("test", [1, 2, 3])
        result = validate_column(series, "test", {"positive": True})
        assert result["passed"] is True

    def test_positive_fail(self):
        series = pl.Series("test", [1, -1, 3])
        result = validate_column(series, "test", {"positive": True})
        assert result["passed"] is False

    def test_range_pass(self):
        series = pl.Series("test", [1.0, 2.0, 3.0])
        result = validate_column(series, "test", {"range": {"min": 0, "max": 5}})
        assert result["passed"] is True

    def test_range_fail(self):
        series = pl.Series("test", [1.0, 10.0, 3.0])
        result = validate_column(series, "test", {"range": {"min": 0, "max": 5}})
        assert result["passed"] is False

    def test_enum_pass(self):
        series = pl.Series("test", ["a", "b", "a"])
        result = validate_column(series, "test", {"enum": ["a", "b", "c"]})
        assert result["passed"] is True

    def test_enum_fail(self):
        series = pl.Series("test", ["a", "b", "x"])
        result = validate_column(series, "test", {"enum": ["a", "b", "c"]})
        assert result["passed"] is False

    def test_multiple_rules(self):
        series = pl.Series("test", [1, 2, None])
        result = validate_column(series, "test", {"not_null": True, "positive": True})
        assert result["passed"] is False
        assert len(result["violations"]) >= 1


# ===================================================================
# Duplicate detection
# ===================================================================


class TestDetectDuplicates:
    def test_no_duplicates(self):
        df = pl.DataFrame({"id": [1, 2, 3]})
        result = detect_duplicates(df, ["id"])
        assert result["duplicate_keys"] == 0

    def test_with_duplicates(self):
        df = pl.DataFrame({"id": [1, 2, 2, 3]})
        result = detect_duplicates(df, ["id"])
        assert result["duplicate_keys"] == 1

    def test_composite_key(self):
        df = pl.DataFrame(
            {
                "listing_id": [1, 1, 2],
                "date": ["2024-01-01", "2024-01-02", "2024-01-01"],
            }
        )
        result = detect_duplicates(df, ["listing_id", "date"])
        assert result["duplicate_keys"] == 0

    def test_missing_key_column(self):
        df = pl.DataFrame({"id": [1, 2, 3]})
        result = detect_duplicates(df, ["nonexistent"])
        assert result["checked"] is False


# ===================================================================
# Scraping artifact detection
# ===================================================================


class TestDetectScrapingArtifacts:
    def test_no_artifacts(self):
        df = pl.DataFrame({"text": ["clean", "data", "here"]})
        result = detect_scraping_artifacts(df)
        assert result["columns_with_artifacts"] == 0

    def test_html_detected(self):
        df = pl.DataFrame({"text": ["clean", "<br>html tag", "here"]})
        result = detect_scraping_artifacts(df)
        assert result["columns_with_artifacts"] > 0

    def test_encoding_artifacts(self):
        df = pl.DataFrame({"text": ["normal", "Ã©tranger", "data"]})
        result = detect_scraping_artifacts(df)
        assert result["columns_with_artifacts"] > 0
