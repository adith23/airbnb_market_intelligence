"""Pipeline orchestration for Section 3.5 automation.

This module coordinates stages 1-4 while recording metadata and lineage.
It intentionally delegates domain work to the existing stage modules:
downloader/profiler/cleaner/enricher/modeler.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from pipeline.metadata import (
    check_already_processed,
    complete_run,
    compute_files_hash,
    configure_file_logging,
    fail_run,
    init_metadata_store,
    record_lineage,
    start_run,
)
from pipeline.utils import get_enriched_dir, get_raw_data_dir, get_staging_dir, load_city_config, filter_raw_files

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StageExecutionResult:
    """Execution summary for a single pipeline stage."""

    stage: str
    status: str
    run_id: str | None = None
    rows_input: int | None = None
    rows_output: int | None = None
    rows_rejected: int | None = None
    output: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class PipelineExecutionResult:
    """Execution summary for a city-level or multi-city pipeline run."""

    city: str
    stages: list[StageExecutionResult] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return all(stage.status in {"SUCCESS", "SKIPPED"} for stage in self.stages)


def _existing(paths: list[Path]) -> list[Path]:
    return [path for path in paths if path.exists()]


def _raw_sources(city: str) -> list[Path]:
    raw_dir = get_raw_data_dir(city)
    if not raw_dir.exists():
        return []
    return filter_raw_files(raw_dir)


def _staging_sources(city: str) -> list[Path]:
    staging_dir = get_staging_dir(city)
    return _existing(
        [
            staging_dir / "listings.parquet",
            staging_dir / "calendar.parquet",
            staging_dir / "reviews.parquet",
            staging_dir / "neighbourhoods.parquet",
        ]
    )


def _enriched_sources(city: str) -> list[Path]:
    return _existing([get_enriched_dir() / f"{city}_master_listings.parquet"])


def _source_label(paths: list[Path], fallback: str) -> str:
    if not paths:
        return fallback
    return ";".join(str(path).replace("\\", "/") for path in paths)


def _tracked_stage(
    city: str,
    stage: str,
    source_paths: list[Path],
    work: Callable[[], tuple[int | None, int | None, int | None, str | None]],
    lineage_output: str | None,
    transforms: list[str],
    force: bool,
) -> StageExecutionResult:
    """Run a stage with metadata tracking and optional incremental skip."""
    source_file = _source_label(source_paths, f"{city}:{stage}")
    source_hash = compute_files_hash(source_paths)

    if (
        not force
        and source_hash is not None
        and check_already_processed(source_file, source_hash, stage=stage, city=city)
    ):
        logger.info("Skipping already processed stage: city=%s stage=%s", city, stage)
        return StageExecutionResult(stage=stage, status="SKIPPED")

    run_id = start_run(city=city, stage=stage, source_file=source_file, source_hash=source_hash)
    configure_file_logging(run_id)

    try:
        rows_in, rows_out, rows_rejected, output = work()
        complete_run(run_id, rows_in=rows_in, rows_out=rows_out, rows_rejected=rows_rejected)
        if lineage_output:
            record_lineage(
                run_id=run_id,
                output_table=lineage_output,
                sources=source_paths,
                transforms=transforms,
            )
        return StageExecutionResult(
            stage=stage,
            status="SUCCESS",
            run_id=run_id,
            rows_input=rows_in,
            rows_output=rows_out,
            rows_rejected=rows_rejected,
            output=output,
        )
    except Exception as exc:
        fail_run(run_id, str(exc))
        return StageExecutionResult(
            stage=stage,
            status="FAILED",
            run_id=run_id,
            error=str(exc),
        )


def run_city_pipeline(
    city: str,
    skip_download: bool = False,
    force: bool = False,
    build_model: bool = True,
) -> PipelineExecutionResult:
    """Run stages 1-4 for one city."""
    init_metadata_store()
    stages: list[StageExecutionResult] = []

    def ingest_work() -> tuple[int | None, int | None, int | None, str | None]:
        from pipeline.downloader import download_city, verify_downloads
        from pipeline.profiler import generate_data_quality_report, profile_city

        if not skip_download:
            download_city(city, force=force)
            verify_downloads(city)
        profiles = profile_city(city)
        quality_report = generate_data_quality_report(city)
        total_rows = sum(profile.get("row_count", 0) for profile in profiles.values())
        return total_rows, quality_report.get("executive_summary", {}).get("total_rows", total_rows), 0, "outputs/quality"

    ingest_sources = _raw_sources(city)
    ingest_result = _tracked_stage(
        city=city,
        stage="ingest",
        source_paths=ingest_sources,
        work=ingest_work,
        lineage_output="raw_profiles",
        transforms=["download", "profile", "quality_report"],
        force=force or not skip_download,
    )
    stages.append(ingest_result)
    if ingest_result.status == "FAILED":
        return PipelineExecutionResult(city=city, stages=stages)

    def clean_work() -> tuple[int | None, int | None, int | None, str | None]:
        from pipeline.cleaner import clean_city

        results = clean_city(city)
        rows_in = sum(result.input_rows for result in results.values())
        rows_out = sum(result.output_rows for result in results.values())
        rows_rejected = sum(result.rejected_rows for result in results.values())
        return rows_in, rows_out, rows_rejected, str(get_staging_dir(city))

    raw_sources = _raw_sources(city)
    clean_result = _tracked_stage(
        city=city,
        stage="clean",
        source_paths=raw_sources,
        work=clean_work,
        lineage_output=f"staging.{city}",
        transforms=["clean_price", "parse_dates", "validate_records", "write_parquet"],
        force=force,
    )
    stages.append(clean_result)
    if clean_result.status == "FAILED":
        return PipelineExecutionResult(city=city, stages=stages)

    def enrich_work() -> tuple[int | None, int | None, int | None, str | None]:
        from pipeline.enricher import enrich_city

        result = enrich_city(city)
        rows_in = result.listings_count
        rows_out = result.listings_count
        return rows_in, rows_out, 0, result.output_path

    staging_sources = _staging_sources(city)
    enrich_result = _tracked_stage(
        city=city,
        stage="enrich",
        source_paths=staging_sources,
        work=enrich_work,
        lineage_output=f"enriched.{city}_master_listings",
        transforms=["aggregate_calendar", "aggregate_reviews", "join_master", "derive_fields"],
        force=force,
    )
    stages.append(enrich_result)
    if enrich_result.status == "FAILED" or not build_model:
        return PipelineExecutionResult(city=city, stages=stages)

    model_result = run_model_pipeline([city], force=force)
    stages.extend(model_result.stages)
    return PipelineExecutionResult(city=city, stages=stages)


def run_model_pipeline(city_names: list[str], force: bool = False) -> PipelineExecutionResult:
    """Build the DuckDB star schema for one or more already-enriched cities."""
    from pipeline.modeler import build_star_schema

    pipeline_city = ",".join(city_names)
    source_paths: list[Path] = []
    for city in city_names:
        source_paths.extend(_enriched_sources(city))
        source_paths.extend(_staging_sources(city))

    def model_work() -> tuple[int | None, int | None, int | None, str | None]:
        result = build_star_schema(city_names)
        rows_out = sum(result.table_counts.values())
        return None, rows_out, 0, result.db_path

    result = _tracked_stage(
        city=pipeline_city,
        stage="model",
        source_paths=source_paths,
        work=model_work,
        lineage_output="duckdb.star_schema",
        transforms=["build_dimensions", "build_facts", "load_duckdb"],
        force=force,
    )
    return PipelineExecutionResult(city=pipeline_city, stages=[result])


def run_all_pipelines(
    city_names: list[str] | None = None,
    skip_download: bool = False,
    force: bool = False,
) -> list[PipelineExecutionResult]:
    """Run city pipelines, then build unified master and multi-city model."""
    if city_names is None:
        city_names = list(load_city_config().keys())
    results: list[PipelineExecutionResult] = []
    for city in city_names:
        results.append(
            run_city_pipeline(
                city=city,
                skip_download=skip_download,
                force=force,
                build_model=False,
            )
        )

    successful_cities = [result.city for result in results if result.success]
    if len(successful_cities) >= 2:
        from pipeline.enricher import build_unified_master

        def unify_work() -> tuple[int | None, int | None, int | None, str | None]:
            output = build_unified_master(successful_cities)
            return None, None, 0, str(output)

        source_paths = [get_enriched_dir() / f"{city}_master_listings.parquet" for city in successful_cities]
        unify_result = _tracked_stage(
            city=",".join(successful_cities),
            stage="unify",
            source_paths=source_paths,
            work=unify_work,
            lineage_output="enriched.unified_master_listings",
            transforms=["align_schemas", "concat_city_masters"],
            force=force,
        )
        results.append(PipelineExecutionResult(city=",".join(successful_cities), stages=[unify_result]))

    if successful_cities:
        results.append(run_model_pipeline(successful_cities, force=force))

    return results
