"""CLI entry point for the Airbnb Market Intelligence pipeline.

Orchestrates all exploration tasks from Section 2.3:
  - download:  Fetch datasets from Inside Airbnb
  - profile:   Schema discovery and statistical profiling
  - validate:  Data quality checks and constraint validation
  - map-rels:  Relationship mapping and referential integrity
  - harmonize: Cross-city schema comparison (multi-city)
  - explore:   Run all of the above for a single city
  - explore-all: Run all steps for all configured cities + harmonization

Usage:
    python main.py download --city paris
    python main.py profile --city paris
    python main.py validate --city paris
    python main.py map-rels --city paris
    python main.py harmonize --cities paris,new_york_city
    python main.py explore --city paris
    python main.py explore-all
"""

from __future__ import annotations

import json
import sys

import click

from src.platform.common.utils import load_city_config, setup_logging

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


# ===================================================================
# CLI group
# ===================================================================


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
def cli(verbose: bool) -> None:
    """Airbnb Market Intelligence — Exploration Pipeline CLI."""
    import logging

    setup_logging(level=logging.DEBUG if verbose else logging.INFO)


# ===================================================================
# Download command
# ===================================================================


@cli.command()
@click.option("--city", required=True, help="City key from cities.yaml (e.g., paris).")
@click.option("--force", is_flag=True, help="Re-download existing files.")
def download(city: str, force: bool) -> None:
    city = city.replace("-", "_")
    """Download dataset files for a city from Inside Airbnb."""
    from src.platform.data_engineering.ingestion.downloader import (
        download_city,
        verify_downloads,
    )

    click.echo(f"📥 Downloading data for: {city}")
    results = download_city(city, force=force)

    summary = results["summary"]
    click.echo(
        f"✅ Done: {summary['successful']} downloaded, "
        f"{summary['skipped']} skipped, {summary['failed']} failed"
    )

    if summary["failed"] > 0:
        click.echo("⚠️  Some downloads failed. Check logs for details.", err=True)
        sys.exit(1)

    click.echo("\n🔍 Verifying downloads...")
    verification = verify_downloads(city)

    if verification["all_present"] and verification["all_readable"]:
        click.echo("✅ All files verified successfully.")
    else:
        click.echo("⚠️  Verification issues detected:", err=True)
        for f in verification["files"]:
            if not f["exists"]:
                click.echo(f"   ❌ Missing: {f['file']}", err=True)
            elif not f["readable"]:
                click.echo(f"   ❌ Unreadable: {f['file']}", err=True)


# ===================================================================
# Profile command
# ===================================================================


@cli.command()
@click.option("--city", required=True, help="City key from cities.yaml.")
def profile(city: str) -> None:
    city = city.replace("-", "_")
    """Run schema discovery and statistical profiling for a city."""
    from src.platform.data_science.evaluation.profiler import profile_city

    click.echo(f"📊 Profiling data for: {city}")
    profiles = profile_city(city)

    click.echo(f"\n✅ Profiled {len(profiles)} files:")
    for filename, p in profiles.items():
        click.echo(f"   📄 {filename}: {p['row_count']:,} rows × {p['column_count']} columns")

    click.echo("\n📁 Outputs saved to: outputs/schemas/ and outputs/profiles/")


# ===================================================================
# Validate command
# ===================================================================


@cli.command()
@click.option("--city", required=True, help="City key from cities.yaml.")
def validate(city: str) -> None:
    city = city.replace("-", "_")
    """Run data quality validation for a city."""
    from src.platform.data_engineering.ingestion.validator import (
        generate_quality_report,
    )

    click.echo(f"🔎 Validating data quality for: {city}")
    report = generate_quality_report(city)

    click.echo(f"\n✅ Validated {report['files_checked']} files:")
    for filename, fr in report["file_reports"].items():
        if fr.get("status") == "READ_ERROR":
            click.echo(f"   ❌ {filename}: READ ERROR")
            continue

        summary = fr.get("summary", {})
        status_icon = "✅" if summary.get("rules_failed", 0) == 0 else "⚠️"
        click.echo(
            f"   {status_icon} {filename}: "
            f"{summary.get('rules_passed', 0)}/{summary.get('rules_checked', 0)} rules passed"
        )

        if summary.get("has_duplicates"):
            dup_info = fr.get("duplicates", {})
            click.echo(f"      🔁 Duplicates: {dup_info.get('duplicate_keys', '?')} duplicate keys")

        if summary.get("has_artifacts"):
            art_info = fr.get("scraping_artifacts", {})
            click.echo(
                f"      🕷️  Artifacts: {art_info.get('columns_with_artifacts', '?')} columns affected"
            )

    click.echo("\n📁 Report saved to: outputs/quality/")


# ===================================================================
# Relationship mapping command
# ===================================================================


@cli.command(name="map-rels")
@click.option("--city", required=True, help="City key from cities.yaml.")
def map_relationships(city: str) -> None:
    city = city.replace("-", "_")
    """Map PK/FK relationships and validate referential integrity."""
    from src.platform.data_engineering.modeling.relationship_mapper import (
        generate_relationship_report,
    )

    click.echo(f"🔗 Mapping relationships for: {city}")
    report = generate_relationship_report(city)

    # PK summary
    click.echo("\n📌 Primary Key Validation:")
    for file_type, pk_result in report.get("primary_key_validation", {}).items():
        status = "✅" if pk_result.get("valid") else "⚠️"
        click.echo(
            f"   {status} {file_type}: "
            f"{pk_result.get('unique_keys', '?'):,} unique keys "
            f"({pk_result.get('duplicate_keys', 0)} duplicates)"
        )

    # FK summary
    click.echo("\n🔗 Foreign Key Validation:")
    for fk_result in report.get("foreign_key_validation", []):
        if fk_result.get("skipped"):
            click.echo(f"   ⏭️  {fk_result['relationship']}: skipped ({fk_result.get('reason')})")
            continue
        status = "✅" if fk_result.get("is_valid") else "⚠️"
        click.echo(
            f"   {status} {fk_result['relationship']}: "
            f"{fk_result.get('coverage_pct', '?')}% coverage "
            f"({fk_result.get('orphan_values', 0)} orphans)"
        )

    click.echo("\n📁 Outputs saved to: outputs/relationships/")


# ===================================================================
# Harmonization command
# ===================================================================


@cli.command()
@click.option(
    "--cities",
    required=True,
    help="Comma-separated city keys (e.g., paris,new_york_city).",
)
def harmonize(cities: str) -> None:
    cities = cities.replace("-", "_")
    """Compare schemas across cities and generate harmonization strategy."""
    from src.platform.data_engineering.modeling.harmonizer import (
        generate_harmonization_report,
    )

    city_list = [c.strip() for c in cities.split(",") if c.strip()]

    click.echo(f"🌍 Harmonization analysis for {len(city_list)} cities: {city_list}")
    report = generate_harmonization_report(city_list)

    comparisons = report.get("schema_comparison", {}).get("comparisons", {})
    click.echo(f"\n✅ Compared {len(comparisons)} file types:")
    for file_type, comp in comparisons.items():
        click.echo(
            f"   📄 {file_type}: "
            f"{comp.get('common_columns', 0)} common columns, "
            f"{comp.get('total_unique_columns', 0)} total"
        )
        specific = comp.get("city_specific_columns", {})
        if specific:
            for city, cols in specific.items():
                click.echo(f"      └─ {city}-only: {len(cols)} columns")

    click.echo("\n📁 Outputs saved to: outputs/harmonization/")


# ===================================================================
# Explore command (full single-city pipeline)
# ===================================================================


@cli.command()
@click.option("--city", required=True, help="City key from cities.yaml.")
@click.option("--skip-download", is_flag=True, help="Skip download step.")
def explore(city: str, skip_download: bool) -> None:
    city = city.replace("-", "_")
    """Run the full exploration pipeline for a single city.

    Steps: download → profile → validate → map-relationships
    """
    from src.platform.data_engineering.ingestion.downloader import (
        download_city,
        verify_downloads,
    )
    from src.platform.data_engineering.ingestion.validator import (
        generate_quality_report,
    )
    from src.platform.data_engineering.modeling.relationship_mapper import (
        generate_relationship_report,
    )
    from src.platform.data_science.evaluation.profiler import profile_city

    click.echo(f"🚀 Running full exploration pipeline for: {city}")
    click.echo("=" * 60)

    # Step 1: Download
    if not skip_download:
        click.echo("\n📥 Step 1/4: Downloading data...")
        download_city(city)
        verify_downloads(city)
    else:
        click.echo("\n⏭️  Step 1/4: Download skipped.")

    # Step 2: Profile
    click.echo("\n📊 Step 2/4: Profiling data...")
    profiles = profile_city(city)
    click.echo(f"   Profiled {len(profiles)} files.")

    # Step 3: Validate
    click.echo("\n🔎 Step 3/4: Validating data quality...")
    quality_report = generate_quality_report(city)
    click.echo(f"   Validated {quality_report['files_checked']} files.")

    # Step 4: Map relationships
    click.echo("\n🔗 Step 4/4: Mapping relationships...")
    rel_report = generate_relationship_report(city)
    click.echo(f"   Mapped {len(rel_report.get('foreign_key_validation', []))} relationships.")

    click.echo("\n" + "=" * 60)
    click.echo(f"✅ Exploration complete for: {city}")
    click.echo("📁 All outputs saved to: outputs/")


# ===================================================================
# Explore-all command (all cities + harmonization)
# ===================================================================


@cli.command(name="explore-all")
@click.option("--skip-download", is_flag=True, help="Skip download step.")
def explore_all(skip_download: bool) -> None:
    """Run the full exploration pipeline for all configured cities + harmonization."""
    from src.platform.data_engineering.ingestion.downloader import (
        download_city,
        verify_downloads,
    )
    from src.platform.data_engineering.ingestion.validator import (
        generate_quality_report,
    )
    from src.platform.data_engineering.modeling.harmonizer import (
        generate_harmonization_report,
    )
    from src.platform.data_engineering.modeling.relationship_mapper import (
        generate_relationship_report,
    )
    from src.platform.data_science.evaluation.profiler import profile_city

    all_cities = load_city_config()
    city_names = list(all_cities.keys())

    click.echo(f"🚀 Running full exploration for {len(city_names)} cities: {city_names}")
    click.echo("=" * 60)

    for city in city_names:
        click.echo(f"\n{'─' * 40}")
        click.echo(f"🏙️  Processing: {city}")
        click.echo(f"{'─' * 40}")

        try:
            if not skip_download:
                click.echo("  📥 Downloading...")
                download_city(city)
                verify_downloads(city)

            click.echo("  📊 Profiling...")
            profile_city(city)

            click.echo("  🔎 Validating...")
            generate_quality_report(city)

            click.echo("  🔗 Mapping relationships...")
            generate_relationship_report(city)

            click.echo(f"  ✅ {city} complete.")

        except Exception as exc:
            click.echo(f"  ❌ Error processing {city}: {exc}", err=True)

    # Cross-city harmonization
    if len(city_names) >= 2:
        click.echo(f"\n{'─' * 40}")
        click.echo("🌍 Cross-city harmonization analysis...")
        click.echo(f"{'─' * 40}")

        try:
            generate_harmonization_report(city_names)
            click.echo("✅ Harmonization report generated.")
        except Exception as exc:
            click.echo(f"❌ Harmonization failed: {exc}", err=True)

    click.echo("\n" + "=" * 60)
    click.echo("✅ Full exploration pipeline complete!")
    click.echo("📁 All outputs saved to: outputs/")


# ===================================================================
# Quality report command (§3.1)
# ===================================================================


@cli.command(name="quality-report")
@click.option("--city", required=True, help="City key from cities.yaml.")
def quality_report(city: str) -> None:
    city = city.replace("-", "_")
    """Generate a consolidated data quality report with outlier detection.

    Produces a single report covering profiling, completeness analysis,
    and IQR-based outlier detection across all data files for a city.
    """
    from src.platform.data_science.evaluation.profiler import (
        generate_data_quality_report,
    )

    click.echo(f"📋 Generating consolidated quality report for: {city}")
    report = generate_data_quality_report(city)

    summary = report.get("executive_summary", {})
    click.echo("\n✅ Quality report generated:")
    click.echo(f"   📊 Files: {summary.get('total_files', 0)}")
    click.echo(f"   📊 Total rows: {summary.get('total_rows', 0):,}")
    click.echo(f"   📊 Quality score: {summary.get('overall_quality_score', 0)}/100")

    issues = summary.get("critical_issues", [])
    if issues:
        click.echo(f"\n   ⚠️  Critical issues ({len(issues)}):")
        for issue in issues:
            click.echo(f"      • {issue}")
    else:
        click.echo("   ✅ No critical issues detected.")

    # Show outlier summary per file
    for filename, fr in report.get("file_reports", {}).items():
        if fr.get("status") == "ERROR":
            continue
        outliers = fr.get("outliers", [])
        outlier_cols = [o for o in outliers if o.get("outlier_count", 0) > 0]
        if outlier_cols:
            click.echo(f"\n   🔍 Outliers in {filename}:")
            for o in outlier_cols:
                click.echo(
                    f"      {o['column']}: {o['outlier_count']} "
                    f"({o['outlier_pct']}%) outside [{o['lower_bound']:.0f}, {o['upper_bound']:.0f}]"
                )

    click.echo("\n📁 Report saved to: outputs/quality/")


# ===================================================================
# Clean command (§3.2)
# ===================================================================


@cli.command()
@click.option("--city", required=True, help="City key from cities.yaml.")
@click.option(
    "--file-type",
    type=click.Choice(["all", "listings", "calendar", "reviews", "neighbourhoods"]),
    default="all",
    help="File type to clean (default: all).",
)
def clean(city: str, file_type: str) -> None:
    city = city.replace("-", "_")
    """Clean and standardize raw data into typed Parquet files.

    Applies price/boolean/date coercion, text normalization,
    missing value strategies, and validation flagging. Valid records
    go to data/staging/, rejected records to data/staging/_rejected/.
    """
    from src.platform.data_engineering.ingestion.cleaner import (
        clean_city,
    )

    click.echo(f"🧹 Cleaning data for: {city} (scope: {file_type})")

    if file_type == "all":
        results = clean_city(city)
    else:
        # Import the specific cleaner
        from src.platform.data_engineering.ingestion import cleaner as cleaner_mod

        cleaner_fn = getattr(cleaner_mod, f"clean_{file_type}", None)
        if cleaner_fn is None:
            click.echo(f"❌ Unknown file type: {file_type}", err=True)
            return
        result = cleaner_fn(city)
        results = {file_type: result}

    click.echo(f"\n✅ Cleaning complete ({len(results)} files):\n")

    total_input = 0
    total_output = 0
    total_rejected = 0

    for ft, result in results.items():
        total_input += result.input_rows
        total_output += result.output_rows
        total_rejected += result.rejected_rows

        status = "✅" if result.rejected_rows == 0 else "⚠️"
        click.echo(
            f"   {status} {ft}: "
            f"{result.input_rows:,} → {result.output_rows:,} valid"
            + (f" + {result.rejected_rows:,} rejected" if result.rejected_rows else "")
        )

        if result.imputed_columns:
            click.echo(f"      📊 Imputed: {', '.join(result.imputed_columns)}")

    # Summary
    rejection_pct = round(total_rejected / max(total_input, 1) * 100, 2)
    click.echo(
        f"\n   📊 Total: {total_input:,} → {total_output:,} ({rejection_pct}% rejection rate)"
    )
    click.echo(f"\n📁 Staging output: data/staging/{city}/")
    if total_rejected > 0:
        click.echo(f"📁 Rejected records: data/staging/{city}/_rejected/")
    click.echo("📁 Cleaning summary: outputs/quality/")


# ===================================================================
# Ingest command (§3.1 combined: download + profile + quality report)
# ===================================================================


@cli.command()
@click.option("--city", required=True, help="City key from cities.yaml.")
@click.option("--skip-download", is_flag=True, help="Skip download step.")
def ingest(city: str, skip_download: bool) -> None:
    city = city.replace("-", "_")
    """Run the full ingestion pipeline: download → profile → quality report.

    This is the Section 3.1 workflow — prepares raw data and produces
    a comprehensive quality assessment before cleaning.
    """
    from src.platform.data_engineering.ingestion.downloader import (
        download_city,
        verify_downloads,
    )
    from src.platform.data_science.evaluation.profiler import (
        generate_data_quality_report,
        profile_city,
    )

    click.echo(f"📥 Running ingestion pipeline for: {city}")
    click.echo("=" * 60)

    # Step 1: Download
    if not skip_download:
        click.echo("\n📥 Step 1/3: Downloading data...")
        results = download_city(city)
        summary = results["summary"]
        click.echo(f"   Done: {summary['successful']} downloaded, {summary['skipped']} skipped")
        verify_downloads(city)
    else:
        click.echo("\n⏭️  Step 1/3: Download skipped.")

    # Step 2: Profile
    click.echo("\n📊 Step 2/3: Profiling data...")
    profiles = profile_city(city)
    click.echo(f"   Profiled {len(profiles)} files.")

    # Step 3: Quality report (with outlier detection)
    click.echo("\n📋 Step 3/3: Generating quality report...")
    report = generate_data_quality_report(city)
    score = report["executive_summary"]["overall_quality_score"]
    click.echo(f"   Quality score: {score}/100")

    click.echo("\n" + "=" * 60)
    click.echo(f"✅ Ingestion complete for: {city}")
    click.echo("📁 Outputs: outputs/schemas/, outputs/profiles/, outputs/quality/")


# ===================================================================
# Enrich command (Section 3.3)
# ===================================================================


@cli.command()
@click.option("--city", required=True, help="City key from cities.yaml.")
def enrich(city: str) -> None:
    city = city.replace("-", "_")
    """Build the enriched master listings dataset for a city."""
    from src.platform.feature_engineering.enricher import enrich_city

    click.echo(f"Enriching data for: {city}")
    result = enrich_city(city)

    click.echo(f"\nEnrichment complete: {result.listings_count:,} listings")
    if result.join_coverage:
        for source, coverage in result.join_coverage.items():
            click.echo(f"   {source}: {coverage}% join coverage")
    if result.derived_fields_added:
        click.echo(f"   Derived fields: {len(result.derived_fields_added)}")
    if result.warnings:
        click.echo(f"   Warnings: {', '.join(result.warnings)}")
    click.echo(f"\nOutput: {result.output_path}")


@cli.command(name="unify-master")
@click.option(
    "--cities",
    required=True,
    help="Comma-separated city keys (e.g., paris,new_york_city,london).",
)
def unify_master(cities: str) -> None:
    cities = cities.replace("-", "_")
    """Build the cross-city unified enriched master table."""
    from src.platform.feature_engineering.enricher import build_unified_master

    city_list = [city.strip() for city in cities.split(",") if city.strip()]
    output_path = build_unified_master(city_list)
    click.echo(f"Unified master written: {output_path}")


# ===================================================================
# Model command (Section 3.4)
# ===================================================================


@cli.command()
@click.option(
    "--cities",
    required=True,
    help="Comma-separated city keys (e.g., paris,new_york_city,london).",
)
@click.option("--skip-calendar", is_flag=True, help="Do not build fact_calendar.")
@click.option("--skip-reviews", is_flag=True, help="Do not build fact_review.")
def model(cities: str, skip_calendar: bool, skip_reviews: bool) -> None:
    cities = cities.replace("-", "_")
    """Build the DuckDB star schema from enriched and staging Parquet files."""
    from src.platform.data_engineering.modeling.modeler import build_star_schema

    city_list = [city.strip() for city in cities.split(",") if city.strip()]
    click.echo(f"Building star schema for: {city_list}")
    result = build_star_schema(
        city_list,
        include_calendar=not skip_calendar,
        include_reviews=not skip_reviews,
    )

    click.echo(f"\nDuckDB model built: {result.db_path}")
    for table, count in result.table_counts.items():
        click.echo(f"   {table}: {count:,} rows")
    if result.warnings:
        click.echo(f"\nWarnings: {', '.join(result.warnings)}")


@cli.command()
@click.option("--name", "query_name", help="Named query from sql/analytical_queries.sql.")
@click.option("--sql", "sql_text", help="Ad-hoc SQL to run against the DuckDB model.")
def query(query_name: str | None, sql_text: str | None) -> None:
    """Run an analytical query against the DuckDB star schema."""
    from src.platform.data_engineering.modeling.modeler import run_analytical_queries

    if bool(query_name) == bool(sql_text):
        raise click.UsageError("Provide exactly one of --name or --sql.")

    results = run_analytical_queries(query_name=query_name, sql=sql_text)
    for name, rows in results.items():
        click.echo(f"Query: {name}")
        click.echo(json.dumps(rows[:50], indent=2, default=str))
        if len(rows) > 50:
            click.echo(f"... {len(rows) - 50:,} additional rows omitted")


# ===================================================================
# Automated pipeline commands (Section 3.5)
# ===================================================================


def _print_pipeline_result(result) -> None:
    """Print a compact pipeline execution summary."""
    status = "SUCCESS" if result.success else "FAILED"
    click.echo(f"\n{result.city}: {status}")
    for stage in result.stages:
        line = f"   {stage.stage}: {stage.status}"
        if stage.rows_output is not None:
            line += f" | rows_out={stage.rows_output:,}"
        if stage.rows_rejected:
            line += f" | rejected={stage.rows_rejected:,}"
        if stage.output:
            line += f" | output={stage.output}"
        if stage.error:
            line += f" | error={stage.error}"
        click.echo(line)


@cli.command(name="run-pipeline")
@click.option("--city", required=True, help="City key from cities.yaml.")
@click.option("--skip-download", is_flag=True, help="Use existing raw files.")
@click.option("--force", is_flag=True, help="Reprocess even when metadata says unchanged.")
def run_pipeline(city: str, skip_download: bool, force: bool) -> None:
    city = city.replace("-", "_")
    """Run stages 1-4 for a single city with metadata tracking."""
    from pipelines.dags.data_pipeline_local import run_city_pipeline

    result = run_city_pipeline(city=city, skip_download=skip_download, force=force)
    _print_pipeline_result(result)
    if not result.success:
        sys.exit(1)


@cli.command(name="run-pipeline-all")
@click.option(
    "--cities",
    default="",
    help="Optional comma-separated city keys. Defaults to all configured cities.",
)
@click.option("--skip-download", is_flag=True, help="Use existing raw files.")
@click.option("--force", is_flag=True, help="Reprocess even when metadata says unchanged.")
def run_pipeline_all(cities: str, skip_download: bool, force: bool) -> None:
    cities = cities.replace("-", "_")
    """Run stages 1-4 for multiple cities, then unify and model."""
    from pipelines.dags.data_pipeline_local import run_all_pipelines

    city_list = [city.strip() for city in cities.split(",") if city.strip()] or None
    results = run_all_pipelines(city_names=city_list, skip_download=skip_download, force=force)
    for result in results:
        _print_pipeline_result(result)

    if not all(result.success for result in results):
        sys.exit(1)


@cli.command()
@click.option("--table", "output_table", help="Output table or artifact name to filter.")
def lineage(output_table: str | None) -> None:
    """Show recorded data lineage from the DuckDB metadata store."""
    from src.platform.common.metadata import get_lineage

    rows = get_lineage(output_table)
    click.echo(json.dumps(rows, indent=2, default=str))


# ===================================================================
# ML Pipeline commands (Section 6)
# ===================================================================


@cli.command()
@click.option("--config", default="config/ml_config.yaml", help="Path to ml_config.yaml.")
@click.option("--force", is_flag=True, help="Retrain models even if they exist.")
def train(config: str, force: bool) -> None:
    """§6.1: Train price prediction models with cross-validation."""
    from src.platform.common.utils import get_db_path
    from src.platform.data_science.training.trainer import train_experiment
    from src.platform.feature_engineering.feature_store import (
        build_feature_matrix,
        load_ml_config,
        prepare_train_test_split,
    )

    click.echo("🚀 Running ML Training Pipeline...")
    cfg = load_ml_config(config)
    feature_set = build_feature_matrix(cfg, get_db_path())
    split = prepare_train_test_split(feature_set, cfg)
    result = train_experiment(feature_set, split, cfg)

    click.echo(f"\n✅ Training complete! Experiment ID: {result.experiment_id}")
    click.echo(
        f"🏆 Best model: {result.best_model_name} ({result.primary_metric}={result.best_metric_value:.4f})"
    )
    click.echo(f"📁 Output saved to: data/models/{result.experiment_id}/")


@cli.command()
@click.option("--experiment-id", required=True, help="Experiment ID to evaluate.")
@click.option("--config", default="config/ml_config.yaml", help="Path to ml_config.yaml.")
def evaluate(experiment_id: str, config: str) -> None:
    """§6.1: Evaluate models on the held-out test set."""
    from src.platform.common.utils import get_db_path
    from src.platform.data_science.validation.evaluator import evaluate_experiment
    from src.platform.data_science.training.trainer import (
        load_experiment,
    )
    from src.platform.feature_engineering.feature_store import (
        build_feature_matrix,
        load_ml_config,
        prepare_train_test_split,
    )

    click.echo(f"📊 Evaluating experiment {experiment_id}...")
    cfg = load_ml_config(config)

    # We need the test set
    feature_set = build_feature_matrix(cfg, get_db_path())
    split = prepare_train_test_split(feature_set, cfg)

    # Mock an ExperimentResult from saved data to pass into evaluate_experiment
    # evaluate_experiment only needs experiment_id, models dict, and best_model_name
    models, _, exp_dir = load_experiment(experiment_id)

    # Read metrics.json to find the best model
    import json

    with open(exp_dir / "metrics.json") as fh:
        metrics = json.load(fh)
        best_name = metrics["_best_model"]

    class MockExperiment:
        def __init__(self, e_id, mods, best):
            self.experiment_id = e_id
            self.models = mods
            self.best_model_name = best

    mock_exp = MockExperiment(experiment_id, models, best_name)
    report = evaluate_experiment(mock_exp, split, cfg)

    click.echo("\n✅ Evaluation complete!")
    click.echo(f"📁 Reports saved to: outputs/ml/{experiment_id}/")


@cli.command()
@click.option("--experiment-id", required=True, help="Experiment ID to explain.")
@click.option("--config", default="config/ml_config.yaml", help="Path to ml_config.yaml.")
def explain(experiment_id: str, config: str) -> None:
    """§6.1: Generate SHAP explainability report."""
    from src.platform.common.utils import get_db_path
    from src.platform.data_science.explainability.explainer import explain_model
    from src.platform.data_science.training.trainer import load_experiment
    from src.platform.feature_engineering.feature_store import (
        build_feature_matrix,
        load_ml_config,
        prepare_train_test_split,
    )

    click.echo(f"🔍 Generating SHAP explanations for {experiment_id}...")
    cfg = load_ml_config(config)
    feature_set = build_feature_matrix(cfg, get_db_path())
    split = prepare_train_test_split(feature_set, cfg)

    models, _, exp_dir = load_experiment(experiment_id)
    import json

    with open(exp_dir / "metrics.json") as fh:
        best_name = json.load(fh)["_best_model"]

    class MockExperiment:
        def __init__(self, e_id, mods, best):
            self.experiment_id = e_id
            self.models = mods
            self.best_model_name = best

    mock_exp = MockExperiment(experiment_id, models, best_name)
    report = explain_model(mock_exp, split, cfg)

    click.echo("\n✅ Explainability analysis complete!")
    click.echo(f"📁 Reports saved to: outputs/ml/{experiment_id}/")


@cli.command(name="bias-audit")
@click.option("--experiment-id", required=True, help="Experiment ID to audit.")
@click.option("--config", default="config/ml_config.yaml", help="Path to ml_config.yaml.")
def bias_audit(experiment_id: str, config: str) -> None:
    """§6.4: Run model generalization & bias analysis."""
    from src.platform.common.utils import get_db_path
    from src.platform.data_science.validation.bias_auditor import run_bias_audit
    from src.platform.data_science.training.trainer import load_experiment
    from src.platform.feature_engineering.feature_store import (
        build_feature_matrix,
        load_ml_config,
        prepare_train_test_split,
    )

    click.echo(f"⚖️ Running bias audit for {experiment_id}...")
    cfg = load_ml_config(config)
    feature_set = build_feature_matrix(cfg, get_db_path())
    split = prepare_train_test_split(feature_set, cfg)

    models, _, exp_dir = load_experiment(experiment_id)
    import json

    with open(exp_dir / "metrics.json") as fh:
        best_name = json.load(fh)["_best_model"]

    class MockExperiment:
        def __init__(self, e_id, mods, best):
            self.experiment_id = e_id
            self.models = mods
            self.best_model_name = best

    mock_exp = MockExperiment(experiment_id, models, best_name)
    report = run_bias_audit(mock_exp, feature_set, split, cfg)

    click.echo("\n✅ Bias audit complete!")
    click.echo(f"⚠️ Overall fairness risk: {report.fairness_summary.overall_risk}")
    click.echo(f"📁 Reports saved to: outputs/ml/{experiment_id}/")


@cli.command(name="run-ml")
@click.option("--config", default="config/ml_config.yaml", help="Path to ml_config.yaml.")
@click.option("--force", is_flag=True, help="Retrain models even if they exist.")
def run_ml(config: str, force: bool) -> None:
    """Run the complete ML pipeline: train → evaluate → explain → bias-audit."""
    from pipelines.dags.ml_pipeline_local import run_ml_pipeline

    result = run_ml_pipeline(config, force)
    if result.success:
        click.echo("\n✅ ML Pipeline completed successfully!")
        click.echo(f"   Experiment ID: {result.experiment_id}")
        click.echo(f"   Best Model: {result.best_model}")
        click.echo(f"   Test MAE: ${result.mae:.2f}")
        click.echo(f"   Bias Risk: {result.bias_risk}")
        click.echo(f"\n📁 Check outputs/ml/{result.experiment_id}/ for full reports.")
    else:
        click.echo(f"\n❌ ML Pipeline failed: {result.error}", err=True)
        import sys

        sys.exit(1)


# ===================================================================
# Entry point
# ===================================================================

if __name__ == "__main__":
    cli()
