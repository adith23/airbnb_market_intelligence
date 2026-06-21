"""Pipeline metadata, lineage, and incremental-processing support.

Section 3.5 stores operational metadata in the same DuckDB database used
for the analytical model. The metadata tables are intentionally small and
append-friendly:

  - pipeline_runs: execution status and row counts per stage
  - data_lineage: source-to-output transformation provenance
  - schema_history: lightweight schema drift snapshots
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb

from pipeline.utils import get_db_path, get_output_dir

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineRun:
    """A pipeline run record returned from metadata queries."""

    run_id: str
    city: str
    stage: str
    status: str
    source_file: str | None
    source_hash: str | None
    started_at: str
    completed_at: str | None
    rows_input: int | None
    rows_output: int | None
    rows_rejected: int | None
    error_message: str | None


def _connect(db_path: str | Path | None = None) -> duckdb.DuckDBPyConnection:
    path = Path(db_path) if db_path is not None else get_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _fetch_dicts(con: duckdb.DuckDBPyConnection, sql: str, params: list[Any] | None = None) -> list[dict[str, Any]]:
    cursor = con.execute(sql, params or [])
    columns = [desc[0] for desc in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def init_metadata_store(db_path: str | Path | None = None) -> None:
    """Create metadata tables if they do not already exist."""
    con = _connect(db_path)
    try:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                run_id VARCHAR PRIMARY KEY,
                city VARCHAR NOT NULL,
                stage VARCHAR NOT NULL,
                source_file VARCHAR,
                source_hash VARCHAR,
                started_at TIMESTAMP NOT NULL,
                completed_at TIMESTAMP,
                status VARCHAR NOT NULL,
                rows_input BIGINT,
                rows_output BIGINT,
                rows_rejected BIGINT,
                error_message VARCHAR
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS data_lineage (
                lineage_id VARCHAR PRIMARY KEY,
                run_id VARCHAR NOT NULL,
                output_table VARCHAR NOT NULL,
                source_files VARCHAR NOT NULL,
                transformations VARCHAR NOT NULL,
                created_at TIMESTAMP NOT NULL
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_history (
                schema_id VARCHAR PRIMARY KEY,
                run_id VARCHAR,
                object_name VARCHAR NOT NULL,
                source_file VARCHAR,
                schema_hash VARCHAR NOT NULL,
                columns_json VARCHAR NOT NULL,
                captured_at TIMESTAMP NOT NULL
            )
            """
        )
    finally:
        con.close()


def compute_file_hash(filepath: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute an MD5 hash for a file using bounded memory."""
    path = Path(filepath)
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_files_hash(paths: list[str | Path]) -> str | None:
    """Compute a deterministic composite hash for multiple files."""
    existing = [Path(path) for path in paths if Path(path).exists()]
    if not existing:
        return None

    digest = hashlib.md5()
    for path in sorted(existing, key=lambda p: str(p).lower()):
        digest.update(str(path).replace("\\", "/").encode("utf-8"))
        digest.update(compute_file_hash(path).encode("utf-8"))
    return digest.hexdigest()


def configure_file_logging(run_id: str | None = None, level: int = logging.INFO) -> Path:
    """Attach a per-run file log handler under outputs/logs/."""
    log_dir = get_output_dir("logs")
    stamp = _utc_now().strftime("%Y%m%d_%H%M%S")
    filename = f"pipeline_{stamp}" + (f"_{run_id[:8]}" if run_id else "") + ".log"
    log_path = log_dir / filename

    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == log_path:
            return log_path

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root_logger.addHandler(handler)
    logger.info("Pipeline file logging enabled: %s", log_path)
    return log_path


def start_run(
    city: str,
    stage: str,
    source_file: str | None = None,
    source_hash: str | None = None,
    db_path: str | Path | None = None,
) -> str:
    """Log the start of a pipeline stage and return its run_id."""
    init_metadata_store(db_path)
    run_id = str(uuid.uuid4())
    con = _connect(db_path)
    try:
        con.execute(
            """
            INSERT INTO pipeline_runs (
                run_id, city, stage, source_file, source_hash,
                started_at, status
            )
            VALUES (?, ?, ?, ?, ?, ?, 'RUNNING')
            """,
            [run_id, city, stage, source_file, source_hash, _utc_now()],
        )
    finally:
        con.close()

    logger.info("Started pipeline run: city=%s stage=%s run_id=%s", city, stage, run_id)
    return run_id


def complete_run(
    run_id: str,
    rows_in: int | None = None,
    rows_out: int | None = None,
    rows_rejected: int | None = None,
    db_path: str | Path | None = None,
) -> None:
    """Mark a pipeline run as successful."""
    init_metadata_store(db_path)
    con = _connect(db_path)
    try:
        con.execute(
            """
            UPDATE pipeline_runs
            SET completed_at = ?,
                status = 'SUCCESS',
                rows_input = ?,
                rows_output = ?,
                rows_rejected = ?,
                error_message = NULL
            WHERE run_id = ?
            """,
            [_utc_now(), rows_in, rows_out, rows_rejected, run_id],
        )
    finally:
        con.close()
    logger.info("Completed pipeline run: run_id=%s", run_id)


def fail_run(
    run_id: str,
    error_message: str,
    db_path: str | Path | None = None,
) -> None:
    """Mark a pipeline run as failed."""
    init_metadata_store(db_path)
    con = _connect(db_path)
    try:
        con.execute(
            """
            UPDATE pipeline_runs
            SET completed_at = ?,
                status = 'FAILED',
                error_message = ?
            WHERE run_id = ?
            """,
            [_utc_now(), error_message[:4000], run_id],
        )
    finally:
        con.close()
    logger.error("Pipeline run failed: run_id=%s error=%s", run_id, error_message)


def check_already_processed(
    source_file: str,
    source_hash: str | None = None,
    stage: str | None = None,
    city: str | None = None,
    db_path: str | Path | None = None,
) -> bool:
    """Return True when a successful run already processed this source hash."""
    init_metadata_store(db_path)
    clauses = ["source_file = ?", "status = 'SUCCESS'"]
    params: list[Any] = [source_file]

    if source_hash is not None:
        clauses.append("source_hash = ?")
        params.append(source_hash)
    if stage is not None:
        clauses.append("stage = ?")
        params.append(stage)
    if city is not None:
        clauses.append("city = ?")
        params.append(city)

    con = _connect(db_path)
    try:
        count = con.execute(
            f"SELECT count(*) FROM pipeline_runs WHERE {' AND '.join(clauses)}",
            params,
        ).fetchone()[0]
    finally:
        con.close()
    return count > 0


def record_lineage(
    run_id: str,
    output_table: str,
    sources: list[str | Path],
    transforms: list[str],
    db_path: str | Path | None = None,
) -> str:
    """Write a lineage record and return its lineage_id."""
    init_metadata_store(db_path)
    lineage_id = str(uuid.uuid4())
    source_files = ",".join(str(source).replace("\\", "/") for source in sources)
    transformations = ",".join(transforms)

    con = _connect(db_path)
    try:
        con.execute(
            """
            INSERT INTO data_lineage (
                lineage_id, run_id, output_table, source_files,
                transformations, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [lineage_id, run_id, output_table, source_files, transformations, _utc_now()],
        )
    finally:
        con.close()

    logger.info("Recorded lineage: output=%s run_id=%s", output_table, run_id)
    return lineage_id


def get_lineage(
    output_table: str | None = None,
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Fetch lineage records, optionally filtered by output table."""
    init_metadata_store(db_path)
    con = _connect(db_path)
    try:
        if output_table:
            return _fetch_dicts(
                con,
                """
                SELECT *
                FROM data_lineage
                WHERE output_table = ?
                ORDER BY created_at DESC
                """,
                [output_table],
            )
        return _fetch_dicts(con, "SELECT * FROM data_lineage ORDER BY created_at DESC")
    finally:
        con.close()


def record_schema_snapshot(
    object_name: str,
    columns: list[str],
    run_id: str | None = None,
    source_file: str | Path | None = None,
    db_path: str | Path | None = None,
) -> str:
    """Persist a lightweight schema-history snapshot."""
    init_metadata_store(db_path)
    schema_id = str(uuid.uuid4())
    schema_hash = hashlib.md5("|".join(sorted(columns)).encode("utf-8")).hexdigest()
    columns_json = json.dumps(columns)

    con = _connect(db_path)
    try:
        con.execute(
            """
            INSERT INTO schema_history (
                schema_id, run_id, object_name, source_file, schema_hash,
                columns_json, captured_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                schema_id,
                run_id,
                object_name,
                str(source_file).replace("\\", "/") if source_file else None,
                schema_hash,
                columns_json,
                _utc_now(),
            ],
        )
    finally:
        con.close()
    return schema_id


def get_recent_runs(
    city: str | None = None,
    stage: str | None = None,
    limit: int = 20,
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Fetch recent pipeline run records."""
    init_metadata_store(db_path)
    clauses: list[str] = []
    params: list[Any] = []
    if city:
        clauses.append("city = ?")
        params.append(city)
    if stage:
        clauses.append("stage = ?")
        params.append(stage)

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    con = _connect(db_path)
    try:
        return _fetch_dicts(
            con,
            f"""
            SELECT *
            FROM pipeline_runs
            {where}
            ORDER BY started_at DESC
            LIMIT ?
            """,
            [*params, limit],
        )
    finally:
        con.close()
