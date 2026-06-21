"""Tests for Section 3.5 metadata and lineage support."""

from __future__ import annotations

from pathlib import Path

import duckdb

from pipeline import metadata


def test_metadata_run_lifecycle_and_lineage(tmp_path):
    db_path = tmp_path / "metadata.duckdb"
    source = tmp_path / "source.txt"
    source.write_text("hello", encoding="utf-8")

    source_hash = metadata.compute_file_hash(source)
    run_id = metadata.start_run(
        city="test_city",
        stage="clean",
        source_file=str(source),
        source_hash=source_hash,
        db_path=db_path,
    )
    metadata.complete_run(
        run_id,
        rows_in=10,
        rows_out=9,
        rows_rejected=1,
        db_path=db_path,
    )
    lineage_id = metadata.record_lineage(
        run_id=run_id,
        output_table="staging.test_city",
        sources=[source],
        transforms=["clean_price", "validate_records"],
        db_path=db_path,
    )

    assert lineage_id
    assert metadata.check_already_processed(
        source_file=str(source),
        source_hash=source_hash,
        stage="clean",
        city="test_city",
        db_path=db_path,
    )

    lineage = metadata.get_lineage("staging.test_city", db_path=db_path)
    assert len(lineage) == 1
    assert lineage[0]["run_id"] == run_id

    runs = metadata.get_recent_runs(city="test_city", db_path=db_path)
    assert runs[0]["status"] == "SUCCESS"
    assert runs[0]["rows_output"] == 9


def test_schema_snapshot_records_hash(tmp_path):
    db_path = tmp_path / "metadata.duckdb"
    schema_id = metadata.record_schema_snapshot(
        object_name="fact_listing_snapshot",
        columns=["listing_id", "city_key", "price_local"],
        db_path=db_path,
    )

    con = duckdb.connect(str(db_path))
    try:
        row = con.execute(
            "SELECT schema_id, object_name, schema_hash FROM schema_history"
        ).fetchone()
    finally:
        con.close()

    assert row[0] == schema_id
    assert row[1] == "fact_listing_snapshot"
    assert len(row[2]) == 32
