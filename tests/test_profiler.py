"""Unit tests for pipeline.profiler — schema inference and statistical profiling.

Uses temporary CSV files to validate profiling logic without
requiring actual Inside Airbnb data.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from src.platform.data_science.evaluation.profiler import (
    compute_column_stats,
    infer_schema,
    profile_file,
)

# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    """Create a minimal sample CSV for testing."""
    filepath = tmp_path / "listings.csv"
    df = pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["Studio A", "Loft B", "Apt C", None, "Room E"],
            "price": ["$100.00", "$200.00", "$150.00", "$0.00", "$300.00"],
            "latitude": [48.856, 48.857, 48.858, 48.859, 48.860],
            "host_is_superhost": ["t", "f", "t", "f", None],
            "bedrooms": [1, 2, None, 3, 1],
        }
    )
    df.write_csv(filepath)
    return filepath


@pytest.fixture
def sample_df() -> pl.DataFrame:
    """Create a sample DataFrame for column stats tests."""
    return pl.DataFrame(
        {
            "numeric_col": [1, 2, 3, 4, 5, None],
            "string_col": ["a", "b", "a", "c", "b", None],
            "date_col": [
                "2024-01-01",
                "2024-06-15",
                "2024-12-31",
                None,
                "2024-03-01",
                "2024-09-01",
            ],
        }
    )


# ===================================================================
# Schema inference
# ===================================================================


class TestInferSchema:
    def test_basic_schema(self, sample_csv: Path):
        schema = infer_schema(sample_csv)

        assert schema["file"] == "listings.csv"
        assert schema["file_type"] == "listings"
        assert schema["row_count"] == 5
        assert schema["column_count"] == 6
        assert "id" in schema["columns"]
        assert "name" in schema["columns"]

    def test_null_counts(self, sample_csv: Path):
        schema = infer_schema(sample_csv)

        # 'name' has 1 null
        assert schema["columns"]["name"]["null_count"] == 1
        # 'id' has 0 nulls
        assert schema["columns"]["id"]["null_count"] == 0

    def test_sample_values(self, sample_csv: Path):
        schema = infer_schema(sample_csv)

        samples = schema["columns"]["id"]["sample_values"]
        assert len(samples) > 0
        assert len(samples) <= 5

    def test_schema_hash(self, sample_csv: Path):
        schema = infer_schema(sample_csv)
        assert isinstance(schema["schema_hash"], str)
        assert len(schema["schema_hash"]) == 32  # MD5 hex


# ===================================================================
# Column statistics
# ===================================================================


class TestComputeColumnStats:
    def test_numeric_stats(self, sample_df: pl.DataFrame):
        stats = compute_column_stats(sample_df, "numeric_col")

        assert stats["unique_count"] == 5
        assert "min" in stats
        assert "max" in stats
        assert "mean" in stats
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0

    def test_string_stats(self, sample_df: pl.DataFrame):
        stats = compute_column_stats(sample_df, "string_col")

        assert stats["unique_count"] == 3
        assert "top_values" in stats
        assert "min_length" in stats


# ===================================================================
# Full file profiling
# ===================================================================


class TestProfileFile:
    def test_profile_output(self, sample_csv: Path):
        profile = profile_file(sample_csv)

        assert profile["file"] == "listings.csv"
        assert profile["row_count"] == 5
        assert "columns" in profile

        # Each column should have both schema and stats
        col = profile["columns"]["id"]
        assert "dtype" in col
        assert "null_count" in col
        assert "unique_count" in col
