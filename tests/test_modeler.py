"""Tests for Section 3.4 DuckDB star-schema modeling."""

from __future__ import annotations

from datetime import date

import duckdb
import polars as pl

from pipeline import modeler


def test_build_star_schema_creates_dimensions_and_facts(monkeypatch, tmp_path):
    enriched_dir = tmp_path / "enriched"
    staging_root = tmp_path / "staging"
    city_dir = staging_root / "test_city"
    enriched_dir.mkdir(parents=True)
    city_dir.mkdir(parents=True)
    db_path = tmp_path / "airbnb.duckdb"

    monkeypatch.setattr(modeler, "get_enriched_dir", lambda: enriched_dir)
    monkeypatch.setattr(modeler, "get_staging_dir", lambda city: staging_root / city)
    monkeypatch.setattr(modeler, "get_db_path", lambda: db_path)

    def fake_city_config(city_name: str | None = None):
        configs = {
            "test_city": {
                "display_name": "Test City",
                "country": "testland",
                "currency_code": "USD",
                "currency_symbol": "$",
                "timezone": "UTC",
            }
        }
        return configs if city_name is None else configs[city_name]

    monkeypatch.setattr(modeler, "load_city_config", fake_city_config)

    pl.DataFrame(
        {
            "listing_id": [1, 2],
            "city": ["test_city", "test_city"],
            "scrape_date": [date(2026, 1, 31), date(2026, 1, 31)],
            "host_id": [10, 20],
            "host_name": ["Ada", "Linus"],
            "host_since": [date(2020, 1, 1), date(2021, 1, 1)],
            "is_professional_host": [True, False],
            "host_tenure_years": [6.08, 5.08],
            "name": ["Central flat", "West room"],
            "property_type": ["Entire rental unit", "Private room"],
            "room_type": ["Entire home/apt", "Private room"],
            "accommodates": [2, 1],
            "bathrooms": [1.0, 1.0],
            "bathrooms_shared": [False, True],
            "bedrooms": [1, 1],
            "beds": [1, 1],
            "amenity_count": [5, 3],
            "instant_bookable": [True, False],
            "license": ["ABC", None],
            "neighbourhood_cleansed": ["Central", "West"],
            "neighbourhood_listing_count": [1, 1],
            "neighbourhood_median_price": [100.0, 80.0],
            "price_local": [100.0, 80.0],
            "price_usd": [100.0, 80.0],
            "latitude": [1.0, 2.0],
            "longitude": [3.0, 4.0],
            "minimum_nights": [1, 2],
            "maximum_nights": [30, 60],
            "number_of_reviews": [2, 1],
            "number_of_reviews_ltm": [2, 1],
            "review_scores_rating": [4.8, 4.2],
            "reviews_per_month": [1.0, 0.5],
            "availability_365": [100, 50],
            "occupancy_rate_pct": [50.0, 25.0],
            "estimated_annual_revenue": [1200.0, 600.0],
            "estimated_monthly_revenue": [100.0, 50.0],
            "avg_booked_price": [100.0, 80.0],
            "price_per_bedroom": [100.0, 80.0],
            "price_per_person": [50.0, 80.0],
            "price_vs_neighbourhood": [1.0, 1.0],
            "is_active": [True, True],
        }
    ).write_parquet(enriched_dir / "test_city_master_listings.parquet")

    pl.DataFrame(
        {
            "listing_id": [1, 2],
            "date": [date(2026, 2, 1), date(2026, 2, 1)],
            "available": [False, True],
            "price": [100.0, 80.0],
            "minimum_nights": [1, 2],
            "maximum_nights": [30, 60],
        }
    ).write_parquet(city_dir / "calendar.parquet")

    pl.DataFrame(
        {
            "review_id": [1000, 1001],
            "listing_id": [1, 2],
            "date": [date(2026, 1, 1), date(2026, 1, 2)],
            "reviewer_id": [500, 501],
            "reviewer_name": ["R1", "R2"],
            "comments": ["great", "fine"],
        }
    ).write_parquet(city_dir / "reviews.parquet")

    result = modeler.build_star_schema(["test_city"], db_path=db_path)

    assert result.table_counts["dim_city"] == 1
    assert result.table_counts["dim_host"] == 2
    assert result.table_counts["dim_property"] == 2
    assert result.table_counts["dim_neighbourhood"] == 2
    assert result.table_counts["dim_reviewer"] == 2
    assert result.table_counts["fact_listing_snapshot"] == 2
    assert result.table_counts["fact_calendar"] == 2
    assert result.table_counts["fact_review"] == 2

    con = duckdb.connect(str(db_path))
    try:
        city = con.execute(
            """
            SELECT c.display_name
            FROM fact_listing_snapshot f
            JOIN dim_city c ON f.city_key = c.city_key
            LIMIT 1
            """
        ).fetchone()[0]
        assert city == "Test City"
    finally:
        con.close()
