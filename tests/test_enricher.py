"""Tests for Section 3.3 enrichment and joining."""

from __future__ import annotations

from datetime import date

import polars as pl

from src.platform.feature_engineering import enricher


def _patch_enricher_paths(monkeypatch, tmp_path):
    staging_root = tmp_path / "staging"
    enriched_dir = tmp_path / "enriched"
    outputs_dir = tmp_path / "outputs"
    enriched_dir.mkdir(parents=True)

    monkeypatch.setattr(enricher, "get_staging_dir", lambda city: staging_root / city)
    monkeypatch.setattr(enricher, "get_enriched_dir", lambda: enriched_dir)

    def fake_output_dir(subdir: str):
        path = outputs_dir / subdir
        path.mkdir(parents=True, exist_ok=True)
        return path

    monkeypatch.setattr(enricher, "get_output_dir", fake_output_dir)
    return staging_root, enriched_dir


def test_enrich_city_builds_master_with_calendar_reviews_and_neighbourhoods(monkeypatch, tmp_path):
    staging_root, enriched_dir = _patch_enricher_paths(monkeypatch, tmp_path)
    city_dir = staging_root / "test_city"
    city_dir.mkdir(parents=True)

    monkeypatch.setattr(
        enricher,
        "load_city_config",
        lambda city: {
            "display_name": "Test City",
            "country": "testland",
            "currency_code": "USD",
            "currency_symbol": "$",
            "timezone": "UTC",
            "scrape_date": "2026-01-31",
        },
    )

    pl.DataFrame(
        {
            "id": [1, 2],
            "host_id": [10, 20],
            "host_since": [date(2020, 1, 1), date(2024, 1, 1)],
            "host_listings_count": [3, 1],
            "neighbourhood_cleansed": ["Central", "West"],
            "price": [100.0, 200.0],
            "bedrooms": [1, 2],
            "accommodates": [2, 4],
            "availability_365": [100, 0],
            "first_review": [date(2025, 1, 31), None],
            "last_review": [date(2026, 1, 1), None],
            "number_of_reviews": [12, 0],
            "review_scores_rating": [4.8, None],
        }
    ).write_parquet(city_dir / "listings.parquet")

    pl.DataFrame(
        {
            "listing_id": [1, 1, 2, 2],
            "date": [
                date(2026, 2, 1),
                date(2026, 2, 2),
                date(2026, 2, 1),
                date(2026, 2, 2),
            ],
            "available": [False, True, False, False],
            "price": [100.0, 120.0, 200.0, 220.0],
            "minimum_nights": [1, 1, 2, 2],
            "maximum_nights": [30, 30, 60, 60],
        }
    ).write_parquet(city_dir / "calendar.parquet")

    pl.DataFrame(
        {
            "listing_id": [1, 1, 2],
            "date": [date(2025, 2, 1), date(2025, 3, 1), date(2025, 4, 1)],
            "reviewer_id": [100, 101, 100],
            "comments": ["great", "nice stay", "ok"],
        }
    ).write_parquet(city_dir / "reviews.parquet")

    pl.DataFrame(
        {
            "neighbourhood": ["Central", "West"],
            "neighbourhood_group": ["Core", "Outer"],
        }
    ).write_parquet(city_dir / "neighbourhoods.parquet")

    result = enricher.enrich_city("test_city")
    master = pl.read_parquet(result.output_path).sort("id")

    assert result.listings_count == 2
    assert result.join_coverage == {
        "calendar": 100.0,
        "reviews": 100.0,
        "neighbourhood": 100.0,
    }
    assert (enriched_dir / "test_city_master_listings.parquet").exists()
    assert master["days_booked"].to_list() == [1, 2]
    assert master["occupancy_rate_pct"].to_list() == [50.0, 100.0]
    assert master["price_local"].to_list() == [100.0, 200.0]
    assert master["price_usd"].to_list() == [100.0, 200.0]
    assert master["is_professional_host"].to_list() == [True, False]
    assert "price_vs_neighbourhood" in master.columns


def test_build_unified_master_aligns_city_schemas(monkeypatch, tmp_path):
    _, enriched_dir = _patch_enricher_paths(monkeypatch, tmp_path)

    pl.DataFrame({"listing_id": [1], "city": ["a"], "price_usd": [10.0]}).write_parquet(
        enriched_dir / "a_master_listings.parquet"
    )
    pl.DataFrame({"listing_id": [2], "city": ["b"], "extra": ["x"]}).write_parquet(
        enriched_dir / "b_master_listings.parquet"
    )

    output = enricher.build_unified_master(["a", "b"])
    unified = pl.read_parquet(output).sort("listing_id")

    assert unified.height == 2
    assert set(unified.columns) == {"city", "extra", "listing_id", "price_usd"}
    assert unified["price_usd"].to_list() == [10.0, None]
