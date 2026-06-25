"""Automated dataset downloader for Inside Airbnb data.

Downloads compressed CSV files and supporting files (neighbourhoods,
geojson) from Inside Airbnb's public data portal. Each city's files
are stored in a dedicated landing directory: data/raw/{city_name}/

Features:
  - Configurable via config/cities.yaml
  - Progress bars for large file downloads
  - Download verification (file size, existence)
  - Idempotent: skips already-downloaded files (use --force to re-download)
  - Structured metadata logging for audit trail

Usage:
    from src.platform.data_engineering.ingestion.downloader import download_city
    results = download_city("paris")
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

from src.platform.common.utils import (
    ensure_dirs,
    get_output_dir,
    get_raw_data_dir,
    load_city_config,
)

logger = logging.getLogger(__name__)

# Download settings
CHUNK_SIZE = 8192  # 8 KB chunks for streaming downloads
REQUEST_TIMEOUT = 60  # seconds
MAX_RETRIES = 3


# ===================================================================
# URL construction
# ===================================================================


def _build_base_url(city_config: dict) -> str:
    """Construct the base URL for a city's dataset on Inside Airbnb.

    Args:
        city_config: City configuration from cities.yaml.

    Returns:
        Base URL string up to the scrape date directory.

    Example:
        >>> _build_base_url({"country": "france", "state": "ile-de-france",
        ...                   "city_slug": "paris", "scrape_date": "2024-09-13"})
        'https://data.insideairbnb.com/france/ile-de-france/paris/2024-09-13'
    """
    return (
        f"https://data.insideairbnb.com"
        f"/{city_config['country']}"
        f"/{city_config['state']}"
        f"/{city_config['city_slug']}"
        f"/{city_config['scrape_date']}"
    )


def _build_file_url(base_url: str, filename: str) -> str:
    """Build the full download URL for a specific file.

    Inside Airbnb organizes files into two subdirectories:
      - data/           → detailed files (*.csv.gz)
      - visualisations/ → summary files (*.csv, *.geojson)

    Args:
        base_url: Base URL from _build_base_url().
        filename: File name to download.

    Returns:
        Full download URL.
    """
    if filename.endswith(".gz"):
        return f"{base_url}/data/{filename}"
    return f"{base_url}/visualisations/{filename}"


# ===================================================================
# Single file download
# ===================================================================


def _download_file(
    url: str,
    dest_path: Path,
    force: bool = False,
) -> dict[str, Any]:
    """Download a single file from a URL to a local path.

    Args:
        url: Source URL.
        dest_path: Local file path to save to.
        force: If True, re-download even if file exists.

    Returns:
        Download result metadata dict with keys:
          file, url, size_bytes, status, error

    Raises:
        Does not raise — errors are captured in the returned dict.
    """
    result: dict[str, Any] = {
        "file": dest_path.name,
        "url": url,
        "dest_path": str(dest_path),
        "size_bytes": 0,
        "status": "SKIPPED",
        "error": None,
    }

    # Skip if already downloaded (idempotent)
    if dest_path.exists() and not force:
        result["size_bytes"] = dest_path.stat().st_size
        result["status"] = "EXISTS"
        logger.info(
            "Already exists, skipping: %s (%s bytes)",
            dest_path.name,
            result["size_bytes"],
        )
        return result

    logger.info("Downloading: %s", url)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, stream=True, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))

            with (
                open(dest_path, "wb") as fh,
                tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    desc=dest_path.name,
                    disable=total_size == 0,
                ) as pbar,
            ):
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    fh.write(chunk)
                    pbar.update(len(chunk))

            result["size_bytes"] = dest_path.stat().st_size
            result["status"] = "SUCCESS"
            logger.info("Downloaded: %s (%s bytes)", dest_path.name, result["size_bytes"])
            return result

        except requests.RequestException as exc:
            logger.warning(
                "Attempt %d/%d failed for %s: %s",
                attempt,
                MAX_RETRIES,
                dest_path.name,
                exc,
            )
            result["error"] = str(exc)
            if attempt == MAX_RETRIES:
                result["status"] = "FAILED"
                logger.error("All %d attempts failed for: %s", MAX_RETRIES, url)

    return result


# ===================================================================
# City-level download
# ===================================================================


def download_city(
    city_name: str,
    force: bool = False,
) -> dict[str, Any]:
    """Download all dataset files for a single city.

    Reads the city configuration from config/cities.yaml, constructs
    download URLs, and fetches all files into data/raw/{city_name}/.

    Args:
        city_name: City key as defined in cities.yaml (e.g., 'paris').
        force: If True, re-download existing files.

    Returns:
        Ingestion metadata dict with download results for each file
        and summary statistics.

    Example:
        >>> results = download_city("paris")
        >>> print(results["summary"]["total_files"])
        7
    """
    city_config = load_city_config(city_name)
    base_url = _build_base_url(city_config)
    dest_dir = get_raw_data_dir(city_name)
    ensure_dirs(dest_dir)

    logger.info(
        "Starting download for %s (scrape: %s)",
        city_config["display_name"],
        city_config["scrape_date"],
    )

    # Collect all files to download
    all_files: list[str] = []
    file_groups = city_config.get("files", {})
    for _group_name, filenames in file_groups.items():
        all_files.extend(filenames)

    # Download each file
    download_results: list[dict[str, Any]] = []
    for filename in all_files:
        url = _build_file_url(base_url, filename)
        dest_path = dest_dir / filename
        result = _download_file(url, dest_path, force=force)
        download_results.append(result)

    # Build summary
    summary = {
        "total_files": len(download_results),
        "successful": sum(1 for r in download_results if r["status"] == "SUCCESS"),
        "skipped": sum(1 for r in download_results if r["status"] in ("EXISTS", "SKIPPED")),
        "failed": sum(1 for r in download_results if r["status"] == "FAILED"),
        "total_bytes": sum(r["size_bytes"] for r in download_results),
    }

    # Build full metadata record
    metadata = {
        "city": city_name,
        "display_name": city_config["display_name"],
        "scrape_date": city_config["scrape_date"],
        "base_url": base_url,
        "downloaded_at": datetime.now(UTC).isoformat(),
        "files": download_results,
        "summary": summary,
    }

    # Persist ingestion metadata
    _save_ingestion_log(city_name, metadata)

    status_msg = (
        f"Download complete: {summary['successful']} succeeded, "
        f"{summary['skipped']} skipped, {summary['failed']} failed"
    )
    logger.info(status_msg)

    return metadata


# ===================================================================
# Verification
# ===================================================================


def verify_downloads(city_name: str) -> dict[str, Any]:
    """Verify downloaded files for a city: existence, size, readability.

    Args:
        city_name: City key from cities.yaml.

    Returns:
        Verification report dict.
    """
    city_config = load_city_config(city_name)
    raw_dir = get_raw_data_dir(city_name)

    results: list[dict[str, Any]] = []
    all_files: list[str] = []
    for filenames in city_config.get("files", {}).values():
        all_files.extend(filenames)

    for filename in all_files:
        filepath = raw_dir / filename
        file_result = {
            "file": filename,
            "exists": filepath.exists(),
            "size_bytes": filepath.stat().st_size if filepath.exists() else 0,
            "readable": False,
        }

        if filepath.exists():
            try:
                # Quick readability check: read first 1KB
                with open(filepath, "rb") as fh:
                    fh.read(1024)
                file_result["readable"] = True
            except OSError:
                file_result["readable"] = False

        results.append(file_result)

    report = {
        "city": city_name,
        "raw_dir": str(raw_dir),
        "verified_at": datetime.now(UTC).isoformat(),
        "files": results,
        "all_present": all(r["exists"] for r in results),
        "all_readable": all(r["readable"] for r in results),
    }

    logger.info(
        "Verification for %s: present=%s, readable=%s",
        city_name,
        report["all_present"],
        report["all_readable"],
    )
    return report


# ===================================================================
# Metadata persistence
# ===================================================================


def _save_ingestion_log(city_name: str, metadata: dict) -> Path:
    """Save ingestion metadata to the outputs directory.

    Args:
        city_name: City key.
        metadata: Ingestion metadata dict.

    Returns:
        Path to the saved metadata file.
    """
    output_dir = get_output_dir("ingestion")
    filepath = output_dir / f"{city_name}_ingestion_log.json"

    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, default=str)

    logger.info("Ingestion log saved: %s", filepath)
    return filepath
