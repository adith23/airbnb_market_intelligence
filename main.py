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

from pipeline.utils import setup_logging, load_city_config

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
    """Download dataset files for a city from Inside Airbnb."""
    from pipeline.downloader import download_city, verify_downloads

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
    """Run schema discovery and statistical profiling for a city."""
    from pipeline.profiler import profile_city

    click.echo(f"📊 Profiling data for: {city}")
    profiles = profile_city(city)

    click.echo(f"\n✅ Profiled {len(profiles)} files:")
    for filename, p in profiles.items():
        click.echo(
            f"   📄 {filename}: {p['row_count']:,} rows × {p['column_count']} columns"
        )

    click.echo("\n📁 Outputs saved to: outputs/schemas/ and outputs/profiles/")


# ===================================================================
# Validate command
# ===================================================================


@cli.command()
@click.option("--city", required=True, help="City key from cities.yaml.")
def validate(city: str) -> None:
    """Run data quality validation for a city."""
    from pipeline.validator import generate_quality_report

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
            click.echo(
                f"      🔁 Duplicates: {dup_info.get('duplicate_keys', '?')} duplicate keys"
            )

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
    """Map PK/FK relationships and validate referential integrity."""
    from pipeline.relationship_mapper import generate_relationship_report

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
            click.echo(
                f"   ⏭️  {fk_result['relationship']}: skipped ({fk_result.get('reason')})"
            )
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
    """Compare schemas across cities and generate harmonization strategy."""
    from pipeline.harmonizer import generate_harmonization_report

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
    """Run the full exploration pipeline for a single city.

    Steps: download → profile → validate → map-relationships
    """
    from pipeline.downloader import download_city, verify_downloads
    from pipeline.profiler import profile_city
    from pipeline.relationship_mapper import generate_relationship_report
    from pipeline.validator import generate_quality_report

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
    click.echo(
        f"   Mapped {len(rel_report.get('foreign_key_validation', []))} relationships."
    )

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
    from pipeline.downloader import download_city, verify_downloads
    from pipeline.harmonizer import generate_harmonization_report
    from pipeline.profiler import profile_city
    from pipeline.relationship_mapper import generate_relationship_report
    from pipeline.validator import generate_quality_report

    all_cities = load_city_config()
    city_names = list(all_cities.keys())

    click.echo(
        f"🚀 Running full exploration for {len(city_names)} cities: {city_names}"
    )
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
    """Generate a consolidated data quality report with outlier detection.

    Produces a single report covering profiling, completeness analysis,
    and IQR-based outlier detection across all data files for a city.
    """
    from pipeline.profiler import generate_data_quality_report

    click.echo(f"📋 Generating consolidated quality report for: {city}")
    report = generate_data_quality_report(city)

    summary = report.get("executive_summary", {})
    click.echo(f"\n✅ Quality report generated:")
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
    """Clean and standardize raw data into typed Parquet files.

    Applies price/boolean/date coercion, text normalization,
    missing value strategies, and validation flagging. Valid records
    go to data/staging/, rejected records to data/staging/_rejected/.
    """
    from pipeline.cleaner import clean_city, CleaningResult

    click.echo(f"🧹 Cleaning data for: {city} (scope: {file_type})")

    if file_type == "all":
        results = clean_city(city)
    else:
        # Import the specific cleaner
        from pipeline import cleaner as cleaner_mod

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
    """Run the full ingestion pipeline: download → profile → quality report.

    This is the Section 3.1 workflow — prepares raw data and produces
    a comprehensive quality assessment before cleaning.
    """
    from pipeline.downloader import download_city, verify_downloads
    from pipeline.profiler import generate_data_quality_report, profile_city

    click.echo(f"📥 Running ingestion pipeline for: {city}")
    click.echo("=" * 60)

    # Step 1: Download
    if not skip_download:
        click.echo("\n📥 Step 1/3: Downloading data...")
        results = download_city(city)
        summary = results["summary"]
        click.echo(
            f"   Done: {summary['successful']} downloaded, "
            f"{summary['skipped']} skipped"
        )
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
    """Build the enriched master listings dataset for a city."""
    from pipeline.enricher import enrich_city

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
    """Build the cross-city unified enriched master table."""
    from pipeline.enricher import build_unified_master

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
    """Build the DuckDB star schema from enriched and staging Parquet files."""
    from pipeline.modeler import build_star_schema

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
@click.option(
    "--name", "query_name", help="Named query from sql/analytical_queries.sql."
)
@click.option("--sql", "sql_text", help="Ad-hoc SQL to run against the DuckDB model.")
def query(query_name: str | None, sql_text: str | None) -> None:
    """Run an analytical query against the DuckDB star schema."""
    from pipeline.modeler import run_analytical_queries

    if bool(query_name) == bool(sql_text):
        raise click.UsageError("Provide exactly one of --name or --sql.")

    results = run_analytical_queries(query_name=query_name, sql=sql_text)
    for name, rows in results.items():
        click.echo(f"Query: {name}")
        click.echo(json.dumps(rows[:50], indent=2, default=str))
        if len(rows) > 50:
            click.echo(f"... {len(rows) - 50:,} additional rows omitted")


# ===================================================================
# Entry point
# ===================================================================

if __name__ == "__main__":
    cli()
