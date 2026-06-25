# Airbnb Market Intelligence

> End-to-end market intelligence platform for Inside Airbnb data — from raw CSV ingestion
> through a Medallion data architecture, a DuckDB star-schema warehouse, XGBoost price
> prediction with SHAP explainability, automated bias auditing, and a seven-page Streamlit
> analytics command centre with an AI-powered natural-language SQL assistant.

[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![Polars](https://img.shields.io/badge/polars-1.0%2B-orange)](https://pola.rs/)
[![DuckDB](https://img.shields.io/badge/duckdb-1.0%2B-yellow)](https://duckdb.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.30%2B-red)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/tests-pytest-informational)](tests/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## Table of Contents

- [Overview](#overview)
- [Quick Start (Docker & Manual)](#quick-start-docker--manual)
- [System Architecture](#system-architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Execution Order — Full Reproduction Guide](#execution-order-full-reproduction-guide)
- [CLI Reference](#cli-reference)
- [Streamlit Dashboard](#streamlit-dashboard)
- [Project Structure](#project-structure)
- [Data Architecture](#data-architecture)
- [ML Pipeline](#ml-pipeline)
- [Engineering Decisions](#engineering-decisions)
- [Tests](#tests)
- [Artifact Review Order](#artifact-review-order)
- [Known Limitations](#known-limitations)
- [Assessment Section Coverage](#assessment-section-coverage)

---

## Overview

This project implements a **production-quality market intelligence system** on top of the
[Inside Airbnb](https://insideairbnb.com/) public dataset. It is built around three
coordinated components:

**Data Engineering Pipeline** — downloads raw city snapshots, profiles and validates data
quality, applies config-driven cleaning rules, enriches listings with calendar occupancy
and neighbourhood aggregates, then loads a star-schema dimensional model into DuckDB.
Covers Amsterdam, New York City, and Barcelona out of the box.

**ML Pipeline** — builds a reproducible feature matrix from the star schema, trains and
cross-validates Ridge, Random Forest, XGBoost, and LightGBM price predictors, evaluates
residual errors by neighbourhood and room type, generates global and local SHAP
explanations, and audits the best model for geographic and cross-city bias.

**Streamlit Dashboard** — a seven-page analytics command centre that reads directly from
the DuckDB warehouse, featuring interactive WebGL pricing maps (PyDeck), live price
prediction with SHAP waterfall charts, MLOps governance views, valuation arbitrage
scanners, supply/demand seasonal yield curves, listing health scoring, and an AI-powered
natural-language SQL assistant backed by Gemini 2.5 Flash.

Everything is **config-driven**: adding a new city requires only a YAML entry in
`config/cities.yaml`. No source code changes are needed.

---

## Quick Start (Docker & Manual)

### 1. Docker Compose Setup (Recommended)
You can run the entire system end-to-end (data pipeline, ML training, and Streamlit dashboard) with a single command:
```bash
docker compose up --build
```
*This command builds the Docker containers, runs the full data pipeline (`run-pipeline-all --skip-download`), trains the ML pipeline (`run-ml`), and starts the Streamlit dashboard on port `8080`. Access the dashboard in your browser at **http://localhost:8080**.*

- **To run only the Streamlit dashboard** (if database and ML models are already generated):
  ```bash
  docker compose up dashboard
  ```
- **To run specific CLI commands inside Docker**:
  ```bash
  docker compose run --rm pipeline python main.py <command>
  ```

### 2. Manual Commands
If running locally, ensure you have completed the [Prerequisites](#prerequisites) and [Installation](#installation) sections first.

- **Step 1 — Run the full Data Engineering Pipeline** (Download, Profile, Validate, Clean, Enrich, Unify, Model):
  ```bash
  python main.py run-pipeline-all
  ```
- **Step 2 — Run the full Machine Learning Pipeline** (Train, Evaluate, Explain, Bias Audit):
  ```bash
  python main.py run-ml
  ```
- **Step 3 — Launch the Dashboard**:
  ```bash
  cd dashboard && streamlit run app.py
  ```
  *Access the local dashboard at **http://localhost:8501**.*

---

## System Architecture

```
┌───────────────────────────────────────────────────────────────────────┐
│                    DATA ENGINEERING PIPELINE (§2–3)                   │
│                                                                       │
│  Inside Airbnb ──► Download  ──► Profile & QA ──► Clean  ──► Enrich  │
│  (CSV / GZ)        §2.3/§3.1     §3.1            §3.2      §3.3      │
│                                                                       │
│  data/raw/{city}/     outputs/profiles/     data/staging/{city}/      │
│    [Bronze]            outputs/quality/       [Silver]                │
│                                                    │                  │
│                                             data/enriched/            │
│                                               [Gold]                  │
│                                                    │                  │
│                                     ┌──────────────▼──────────────┐  │
│                                     │  DuckDB Star Schema         │  │
│                                     │  data/airbnb.duckdb         │  │
│                                     │  [Platinum]                 │  │
│                                     └──────────┬──────────────────┘  │
└────────────────────────────────────────────────┼────────────────────-┘
                                                 │
           ┌─────────────────────────────────────┤
           │  feature_store.py                   │  data_client.py
           │  (ML interface)                     │  (Dashboard interface)
           ▼                                     ▼
┌──────────────────────────┐   ┌──────────────────────────────────────┐
│   ML PIPELINE (§6)       │   │   STREAMLIT DASHBOARD (§8)           │
│                          │   │                                      │
│  Feature Matrix          │   │  1. Market Overview  (KPIs + maps)   │
│  ──► Train + CV          │   │  2. Price Estimator  (ML inference)  │
│  ──► Evaluate            │   │  3. Explainability   (SHAP)          │
│  ──► SHAP Explain        │   │  4. MLOps Governance (bias audit)    │
│  ──► Bias Audit          │   │  5. Valuation & Arbitrage            │
│                          │   │  6. Supply & Demand  (calendar)      │
│  data/models/{exp_id}/   │   │  7. Intervention Radar               │
│  outputs/ml/{exp_id}/    │   │  + AI SQL Assistant (Gemini)         │
└──────────────────────────┘   └──────────────────────────────────────┘
```

### Medallion Layers

| Layer | Path | Format | Purpose |
|---|---|---|---|
| Bronze | `data/raw/{city}/` | CSV / GZ | Unmodified downloads — never overwritten |
| Silver | `data/staging/{city}/` | Parquet | Type-coerced, validated; rejected records in `_rejected/` |
| Gold | `data/enriched/` | Parquet | Joined master listings with all derived metrics |
| Platinum | `data/airbnb.duckdb` | DuckDB | Star-schema facts, dimensions, and pipeline metadata |

---

## Prerequisites

| Requirement | Version | Check |
|---|---|---|
| Python | 3.11 or later | `python --version` |
| pip | 23 or later | `pip --version` |
| Git | any | `git --version` |
| Disk space | ~5 GB per city | — |
| RAM | 4 GB minimum | — |

> **Cloud / Airflow users** — set `AIRFLOW_DATA_DIR` to a shared volume path and all
> data reads/writes are redirected automatically (see [Configuration](#configuration)).

> **AI Assistant** — set `GEMINI_API_KEY` to enable the live Gemini 2.5 Flash SQL agent.
> Without the key the dashboard falls back to a deterministic mock LLM automatically.

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/adith23/airbnb_market_intelligence.git
cd airbnb_market_intelligence

# 2. Create and activate a virtual environment
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# 3. Install core pipeline dependencies
pip install -r requirements.txt

# 4. Install dashboard dependencies (required to run the Streamlit app)
pip install -r requirements-dashboard.txt

# 5. Verify the CLI is functional
python main.py --help
```

The `python main.py --help` command should print the full command listing.
If imports fail, confirm that the virtual environment is active and Python 3.11+ is in use.

### Optional: install as a package

```bash
pip install -e .
# The CLI is then also available as:
airbnb-pipeline --help
```

### Environment variables

```bash
# Optional: enable the Gemini 2.5 Flash AI SQL assistant in the dashboard
export GEMINI_API_KEY="your-api-key-here"

# Optional: redirect all data paths to a shared cloud volume (Airflow / Composer)
export AIRFLOW_DATA_DIR="/mnt/gcs/airbnb-pipeline"
```

---

## Configuration

### Adding a city

Open `config/cities.yaml` and add an entry following the existing pattern:

```yaml
cities:
  london:
    display_name: "London"
    country: united-kingdom
    state: england
    city_slug: london
    currency_code: GBP
    currency_symbol: "£"
    timezone: "Europe/London"
    scrape_date: "2025-09-11"       # update to latest from insideairbnb.com/get-the-data
    admin_unit_name: "Borough"
    files:
      detailed:
        - listings.csv.gz
        - calendar.csv.gz
        - reviews.csv.gz
      summary:
        - listings.csv
        - reviews.csv
        - neighbourhoods.csv
        - neighbourhoods.geojson
```

Then run `python main.py run-pipeline --city london`. No code changes needed.

> **Scrape dates** — Inside Airbnb publishes new snapshots monthly. Update `scrape_date`
> before each fresh download. Current configured dates:
>
> | City | Scrape date |
> |---|---|
> | Amsterdam | 2025-09-11 |
> | New York City | 2026-04-14 |
> | Barcelona | 2026-03-21 |

### Cleaning rules (`config/cleaning_rules.yaml`)

Controls per-file-type transformations applied during `clean`:

- `price_columns` — currency strings (`"$1,250.00"`) → `Float64`
- `boolean_columns` — Airbnb `"t"` / `"f"` encoding → `Boolean`
- `date_columns` — `"YYYY-MM-DD"` strings → `Date`
- `percentage_columns` — `"95%"` strings → proportions (`0.0–1.0`)
- `missing_value_strategies` — per-column: `reject` | `sentinel` | `impute_median` | `null`

### Validation rules (`config/validation_rules.yaml`)

Constraint checks applied before cleaning. Supported rule types: `not_null`, `unique`,
`range` (`min`/`max`), `enum`, `positive`, `regex`.
Records failing constraints are written to `data/staging/{city}/_rejected/`, not silently dropped.

### Schema harmonisation (`config/schema_map.yaml`)

Maps raw column name variants across scrape vintages and cities to a canonical schema,
used by `harmonize` and `unify-master` for multi-city analysis.

### ML configuration (`config/ml_config.yaml`)

```yaml
target:
  column: price_usd
  transform: log1p          # applied during training
  inverse_transform: expm1  # applied to recover dollar predictions

split:
  test_size: 0.20
  stratify_by: [city, price_quintile]
  random_state: 42

cross_validation:
  n_folds: 5
  strategy: stratified

features:
  numeric: [accommodates, bedrooms, beds, bathrooms, amenity_count, ...]
```

All ML behaviour — features, target transform, CV strategy, model selection — is
controlled through this file. No code changes are needed to experiment.

---

## Execution Order — Full Reproduction Guide

Follow this sequence exactly to reproduce all artifacts from scratch.

### Phase 1 — Environment setup

```bash
git clone https://github.com/adith23/airbnb_market_intelligence.git
cd airbnb_market_intelligence
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-dashboard.txt
```

### Phase 2 — Data exploration (§2.3)

These commands run schema discovery, profiling, and relationship mapping before any
cleaning is applied. Outputs are written to `outputs/`.

```bash
# Option A: run all exploration steps for a single city in one command
python main.py explore --city amsterdam

# Option B: run each step individually for inspection
python main.py download  --city amsterdam
python main.py profile   --city amsterdam
python main.py validate  --city amsterdam
python main.py map-rels  --city amsterdam

# Cross-city schema comparison (requires ≥2 cities downloaded)
python main.py harmonize --cities amsterdam,new_york_city,barcelona
```

### Phase 3 — Data engineering pipeline (§3.1–3.5)

```bash
# Recommended: run all four stages for a city in one command with metadata tracking
python main.py run-pipeline --city amsterdam
python main.py run-pipeline --city new-york-city
python main.py run-pipeline --city barcelona

# Or run every city defined in cities.yaml in one command
python main.py run-pipeline-all

# Inspect recorded lineage after any pipeline run
python main.py lineage
```

To run each stage individually:

```bash
# Stage 3.1 — Download raw files, profile, and generate quality report
python main.py ingest --city amsterdam

# Stage 3.2 — Clean raw CSV/GZ → typed Parquet
python main.py clean --city amsterdam

# Stage 3.3 — Build enriched master listings (joins + derived metrics)
python main.py enrich --city amsterdam

# Stage 3.3 — Build unified multi-city master (run after enriching all cities)
python main.py unify-master --cities amsterdam,new_york_city,barcelona

# Stage 3.4 — Build DuckDB star schema
python main.py model --cities amsterdam,new_york_city,barcelona

# Stage 3.4 — Run named analytical SQL queries
python main.py query --name market_overview
python main.py query --name host_segmentation
python main.py query --sql "SELECT city_key, COUNT(*) FROM fact_listing_snapshot GROUP BY 1"
```

### Phase 4 — ML pipeline (§6)

The DE pipeline must complete for at least one city before running the ML pipeline,
as the ML pipeline reads from `data/airbnb.duckdb`.

```bash
# Recommended: run the full ML pipeline in one command
python main.py run-ml

# Or run each stage individually
python main.py train

# Note the experiment ID printed after training, then pass it to subsequent commands
python main.py evaluate  --experiment-id <experiment_id>
python main.py explain   --experiment-id <experiment_id>
python main.py bias-audit --experiment-id <experiment_id>
```

### Phase 5 — Dashboard

```bash
cd dashboard
streamlit run app.py
```

The app opens at `http://localhost:8501`. It reads `data/airbnb.duckdb` and
`data/models/` directly via a cached read-only DuckDB connection.

To enable the AI SQL assistant, set your Gemini API key before launching:

```bash
export GEMINI_API_KEY="your-api-key-here"
streamlit run app.py
```

Without the key, the assistant falls back to a deterministic mock response automatically.

---

## CLI Reference

Global flag: add `--verbose` / `-v` to any command for `DEBUG`-level logging.

### Exploration commands (§2.3)

| Command | Description |
|---|---|
| `download --city <name>` | Download all configured files for a city from Inside Airbnb |
| `profile --city <name>` | Schema discovery and statistical profiling |
| `validate --city <name>` | Constraint validation against `validation_rules.yaml` |
| `map-rels --city <name>` | PK/FK mapping and referential integrity analysis |
| `harmonize --cities <a,b,...>` | Cross-city schema comparison and harmonisation report |
| `explore --city <name>` | Run `download → profile → validate → map-rels` for one city |
| `explore-all` | Run `explore` for all configured cities, then `harmonize` |

### Data engineering commands (§3)

| Command | Key flags | Description |
|---|---|---|
| `ingest --city <name>` | `--skip-download` | Download + profile + quality report (§3.1) |
| `quality-report --city <name>` | — | IQR-based outlier detection across all files |
| `clean --city <name>` | `--file-type [all\|listings\|calendar\|reviews\|neighbourhoods]` | Raw CSV/GZ → typed Parquet (§3.2) |
| `enrich --city <name>` | — | Enriched master listings with joins and derived fields (§3.3) |
| `unify-master --cities <a,b,...>` | — | Cross-city unified master Parquet (§3.3) |
| `model --cities <a,b,...>` | `--skip-calendar` `--skip-reviews` | DuckDB star schema build (§3.4) |
| `query` | `--name <name>` or `--sql <text>` | Run analytical SQL against the star schema |
| `run-pipeline --city <name>` | `--skip-download` `--force` | Orchestrated §3.1–3.4 with metadata tracking (§3.5) |
| `run-pipeline-all` | `--cities <a,b,...>` `--skip-download` `--force` | All configured cities, then unify and model |
| `lineage` | `--table <name>` | Print source-to-output lineage from the metadata store |

The `--force` flag bypasses MD5 hash-based incremental skip logic and reprocesses a
stage even when source files have not changed since the last successful run.

### ML commands (§6)

| Command | Key flags | Description |
|---|---|---|
| `train` | `--config <path>` `--force` | Train all model families with 5-fold stratified CV |
| `evaluate --experiment-id <id>` | `--config <path>` | Test-set metrics and residual analysis |
| `explain --experiment-id <id>` | `--config <path>` | Global and local SHAP explanations |
| `bias-audit --experiment-id <id>` | `--config <path>` | LONO-CV neighbourhood bias + cross-city transfer |
| `run-ml` | `--config <path>` `--force` | Orchestrated `train → evaluate → explain → bias-audit` |

---

## Streamlit Dashboard

Launch with:

```bash
cd dashboard
streamlit run app.py
# Opens at http://localhost:8501
```

The dashboard reads directly from `data/airbnb.duckdb` (read-only) via a Streamlit-cached
DuckDB connection and from `data/models/` for ML inference. All queries are cached for
one hour (`@st.cache_data(ttl=3600)`).

### Pages

| Page | What it shows |
|---|---|
| **1 — Market Overview** | City-level KPIs (total listings, average daily rate, occupancy, professional host %) with WebGL pricing density heatmap via PyDeck |
| **2 — Price Estimator** | Interactive listing configurator (room type, bedrooms, amenities) → XGBoost price prediction with confidence context |
| **3 — Explainability** | SHAP waterfall chart for individual predictions; global feature importance ranked by mean absolute SHAP value; Amenity ROI ladder |
| **4 — MLOps Governance** | Cross-city model generalisability matrix; neighbourhood-level LONO-CV error distribution; bias risk summary from `bias_audit_report.json` |
| **5 — Valuation & Arbitrage** | Revenue leakage scanner: listings with `rating ≥ 4.8`, `reviews > 30`, and `price < city median` — sorted by estimated monthly revenue gap |
| **6 — Supply & Demand** | Professional vs casual host concentration comparison; 365-day seasonal yield curves from `fact_calendar` (`avg_price` and `booked_occupancy_rate` by date) |
| **7 — Intervention Radar** | Composite listing health score (price position 25%, review quality 40%, occupancy 35%); priority action queue flagging underpriced, overpriced, low-quality, and weak-conversion listings; neighbourhood opportunity zone map |

### AI SQL Assistant

A collapsible panel available on every page. Enter a plain-English question and receive:

1. Generated SQL (shown for transparency)
2. Query results as a Pandas DataFrame
3. A 2–3 sentence executive summary synthesised by the LLM

The assistant uses `gemini-2.5-flash` via the Google GenAI SDK when `GEMINI_API_KEY` is
set, and falls back to a deterministic `MockLLM` otherwise. SQL is validated through
a security guardrail that blocks `DROP`, `DELETE`, `UPDATE`, `INSERT`, `ALTER`, `GRANT`,
and `TRUNCATE` before execution. All queries are capped at `LIMIT 50`.

---

## Project Structure

```
airbnb_market_intelligence/
│
├── config/
│   ├── cities.yaml              # City definitions, scrape dates, currency metadata
│   ├── cleaning_rules.yaml      # Per-column type coercions and missing value strategies
│   ├── schema_map.yaml          # Canonical column names for multi-city harmonisation
│   ├── validation_rules.yaml    # Pre-cleaning constraint checks
│   └── ml_config.yaml           # ML target, features, CV strategy, model hyperparameters
│
├── src/
│   └── platform/
│       ├── common/
│       │   ├── utils.py         # Pure parsing utilities: price, boolean, date, amenities
│       │   └── metadata.py      # Pipeline run tracking, lineage, schema history in DuckDB
│       │
│       ├── data_engineering/
│       │   ├── ingestion/
│       │   │   ├── downloader.py        # HTTP download + GZ extraction + verification
│       │   │   ├── profiler.py          # Lazy Polars profiling; schema + statistical JSON
│       │   │   ├── validator.py         # Constraint checks → quality reports
│       │   │   └── cleaner.py           # Raw CSV/GZ → validated staging Parquet (vectorised)
│       │   ├── modeling/
│       │   │   ├── modeler.py           # DuckDB star schema builder
│       │   │   ├── relationship_mapper.py  # PK/FK and referential integrity
│       │   │   └── harmonizer.py        # Cross-city schema comparison
│       │   └── storage/
│       │       └── data_client.py       # Cached DuckDB queries for the dashboard
│       │
│       ├── data_science/
│       │   ├── training/
│       │   │   └── trainer.py           # Ridge/RF/XGBoost/LightGBM with k-fold CV
│       │   ├── validation/
│       │   │   ├── evaluator.py         # Test-set metrics + residual segment analysis
│       │   │   └── bias_auditor.py      # LONO-CV + cross-city transfer + fairness
│       │   └── explainability/
│       │       └── explainer.py         # SHAP TreeExplainer + KernelExplainer fallback
│       │
│       ├── feature_engineering/
│       │   ├── enricher.py              # Gold layer: calendar/review joins + derived fields
│       │   └── feature_store.py         # ML feature matrix from DuckDB
│       │
│       └── agentic_ai/
│           └── sql_agent.py             # Gemini 2.5 Flash Text-to-SQL with security guardrails
│
├── pipelines/
│   └── dags/
│       ├── data_pipeline_local.py       # DE pipeline orchestrator with metadata tracking
│       └── ml_pipeline_local.py         # ML pipeline orchestrator with metadata tracking
│
├── dashboard/
│   ├── app.py                           # Streamlit entry point (multi-page app)
│   ├── config.py                        # DB_PATH, MODELS_DIR, UI settings
│   └── components/
│       └── ai_chat.py                   # AI SQL Assistant UI component
│
├── sql/
│   └── analytical_queries.sql           # Named queries run via `python main.py query`
│
├── tests/
│   ├── test_utils.py                    # Unit tests for all parsing utilities
│   ├── test_cleaner.py                  # Unit tests for Polars cleaning transformations
│   └── test_enricher.py                 # Unit tests for enrichment logic
│
├── data/                                # Gitignored — created at runtime
│   ├── raw/{city}/                      # Bronze: unmodified downloads
│   ├── staging/{city}/                  # Silver: clean Parquet
│   │   └── _rejected/                   # Records that failed validation
│   ├── enriched/                        # Gold: joined master Parquet files
│   ├── models/{experiment_id}/          # Trained model artefacts + metrics.json
│   └── airbnb.duckdb                    # Platinum: star schema + pipeline metadata
│
├── outputs/                             # Gitignored — created at runtime
│   ├── schemas/                         # JSON schema per file per city
│   ├── profiles/                        # Statistical profiles
│   ├── quality/                         # Quality reports + cleaning summaries
│   ├── relationships/                   # PK/FK analysis reports
│   ├── harmonization/                   # Cross-city comparison reports
│   ├── ml/{experiment_id}/              # Evaluation, SHAP, and bias audit reports
│   └── logs/                            # Per-run structured log files
│
├── main.py                              # Click CLI entry point
├── requirements.txt                     # Core pipeline dependencies
├── requirements-dashboard.txt           # Dashboard-specific dependencies (Streamlit, PyDeck, Altair)
└── pyproject.toml                       # Package metadata; optional entrypoint: airbnb-pipeline
```

---

## Data Architecture

### Star schema

```
              ┌──────────────┐
              │   dim_date   │
              └──────┬───────┘
                     │
 ┌──────────┐  ┌─────▼───────────────────┐  ┌──────────────┐
 │ dim_host │  │  fact_listing_snapshot  │  │ dim_property │
 └─────┬────┘  └─────────────┬───────────┘  └──────┬───────┘
       │                     │                      │
       └─────────────────────┼──────────────────────┘
                             │
          ┌──────────────────┼───────────────────┐
          │                  │                   │
 ┌────────▼──────┐  ┌────────▼──────┐  ┌────────▼──────┐
 │ fact_calendar │  │   dim_city    │  │  fact_review  │
 └───────────────┘  └───────────────┘  └───────────────┘
                             │
                    ┌────────▼──────────┐
                    │ dim_neighbourhood │
                    └───────────────────┘
```

**Fact tables** — grain is one row per `(listing_id, scrape_date)` for snapshots, one
row per `(listing_id, date)` for calendar, and one row per review.

### Derived metrics in the Gold layer

| Field | Formula | Business use |
|---|---|---|
| `host_tenure_years` | `(scrape_date − host_since).days / 365` | Experience signal |
| `price_per_bedroom` | `price_usd / max(bedrooms, 1)` | Cross-listing normalisation |
| `review_frequency_per_month` | `reviews / max(tenure_months, 1)` | Demand proxy |
| `estimated_occupancy_rate` | `1 − (availability_365 / 365)` | Booking estimate |
| `estimated_monthly_revenue` | `price_usd × occupancy_rate × 30` | Revenue potential |
| `estimated_annual_revenue` | `price_usd × occupancy_rate × 365` | Investment view |
| `is_professional_host` | `host_listings_count ≥ 2` | Market structure flag |
| `neighbourhood_median_price` | `MEDIAN(price_usd)` per neighbourhood | Peer benchmark |

### Pipeline metadata

Three metadata tables live in `data/airbnb.duckdb` alongside the analytical model:

| Table | Purpose |
|---|---|
| `pipeline_runs` | Stage status, row counts, duration, error messages |
| `data_lineage` | Source file → transformation → output artifact provenance |
| `schema_history` | Schema hash snapshots for drift detection |

Incremental processing uses MD5 hashes of source files. A stage is skipped when a
successful run with the same source hash already exists in `pipeline_runs`, unless
`--force` is passed.

---

## ML Pipeline

### Flow

```
DuckDB Star Schema (data/airbnb.duckdb)
         │
         ▼
feature_store.py           Queries dim_listing, dim_property, dim_city;
                           extracts amenity flags; computes Haversine
                           distance to city centroid; adds interaction
                           terms; median imputation + _missing indicators;
                           one-hot encoding with cardinality cap.
         │
         ▼
trainer.py                 Trains Ridge / Random Forest / XGBoost /
                           LightGBM against log1p(price_usd); 5-fold
                           stratified CV; persists best model + all
                           metrics to data/models/{experiment_id}/
         │
         ▼
evaluator.py               Test-set MAE, RMSE, MAPE, R²; residual
                           breakdown by neighbourhood, room type, and
                           price quartile; Markdown + JSON reports.
         │
         ▼
explainer.py               SHAP TreeExplainer for tree models;
                           KernelExplainer fallback for Ridge;
                           global importance JSON + summary plot;
                           per-listing waterfall data.
         │
         ▼
bias_auditor.py            Leave-One-Neighbourhood-Group-Out CV;
                           cross-city train-A / evaluate-B transfer;
                           group-level fairness metrics; overall
                           risk rating (Low / Medium / High).
```

### Experiment artefacts

```
data/models/{experiment_id}/
├── {model_name}.joblib       # Serialised model (scikit-learn / XGBoost / LightGBM)
├── metrics.json              # CV results + test-set metrics for all models
└── feature_names.json        # Ordered feature list

outputs/ml/{experiment_id}/
├── evaluation_report.md      # Human-readable model comparison
├── shap/
│   ├── feature_importance.json
│   └── summary_plot.png
└── bias_audit_report.json    # Per-neighbourhood + per-city fairness metrics
```

---

## Engineering Decisions

| Decision | Choice | Alternatives considered | Trade-offs accepted |
|---|---|---|---|
| Processing engine | **Polars** | pandas, PySpark | Vectorised columnar API handles 24 M-row calendar files without a cluster. No `apply` or `map_elements` anywhere in the pipeline. PySpark would be correct at 50+ cities; Polars is sufficient here. |
| Analytical store | **DuckDB** | SQLite, PostgreSQL, Snowflake | Zero-config embedded OLAP with native Parquet reads, window functions, and LATERAL joins. No server to manage. Not horizontally scalable, but no server needed for single-node assessment workloads. |
| Intermediate format | **Parquet** | CSV, Feather | Schema-preserving, column-pruning, ~10× compression vs raw CSV. Requires Parquet-aware tools but that is standard in modern DE stacks. |
| Analytical model | **Star schema** | Wide flat table (OBT) | Predictable BI join semantics; clean separation of slowly-changing attributes from event-grain facts. OBT would be simpler but loses dimensional clarity. |
| SCD strategy | **Type 1** | Type 2, Type 4 | Inside Airbnb provides point-in-time scrape snapshots; full CDC history is not available from the source. Type 2 would require a reliable change feed. |
| Incremental processing | **MD5 source hashes** | Watermark timestamps, file mtimes | Deterministic: same source file always produces the same hash regardless of filesystem metadata. Safe to re-run on any machine. |
| ML target transform | **log1p / expm1** | Raw price, Box-Cox | Price distributions are strongly right-skewed. `log1p` linearises the relationship, is numerically stable at zero, and the inverse `expm1` is exact. |
| Bias analysis strategy | **LONO-CV** | Random holdout | Leave-One-Neighbourhood-Group-Out isolates exactly where the model fails to generalise: it must predict unseen listings in a withheld neighbourhood using only the other neighbourhoods for training. |
| AI assistant | **Gemini 2.5 Flash** | GPT-4o, local LLM | Good balance of speed and reasoning for Text-to-SQL. MockLLM fallback ensures the dashboard works without any API key. |
| Dashboard data access | **Read-only DuckDB + Streamlit cache** | REST API layer | Minimal infrastructure for a single-node deployment. `@st.cache_resource` holds a single connection; `@st.cache_data(ttl=3600)` caches query results. |
| Pipeline orchestration | **Custom DAG in `pipelines/dags/`** | Airflow, Prefect | Structure mirrors an Airflow DAG (named files in `dags/`) for easy migration. The `AIRFLOW_DATA_DIR` env var enables drop-in replacement. At production scale, these would be Prefect flows or Airflow DAGs without code changes to the core modules. |

---

## Tests

```bash
# Run the full test suite
pytest tests/ -v

# Run a specific file
pytest tests/test_utils.py -v
pytest tests/test_cleaner.py -v
pytest tests/test_enricher.py -v

# Run a specific test class
pytest tests/test_cleaner.py::TestCleanPriceColumns -v
```

| File | What is tested |
|---|---|
| `tests/test_utils.py` | All pure parsing functions in `common/utils.py`: `clean_price`, `parse_boolean`, `parse_bathrooms_text`, `parse_amenities`, `parse_host_verifications`, `parse_rate_pct`, `strip_html`, `detect_price_currency`, `compute_schema_hash`, `infer_file_type` |
| `tests/test_cleaner.py` | All Polars cleaning transformations: `_clean_price_columns`, `_cast_boolean_columns`, `_clean_percentage_columns`, `_apply_missing_strategies`, `_compute_validation_flags`, `_count_amenities`, `_normalize_text_columns`, `_parse_bathrooms_column`, `_parse_date_columns`, `_strip_html_column` |
| `tests/test_enricher.py` | Section 3.3 enrichment and join logic using monkeypatched temporary paths |

All test functions are pure: no network calls, no filesystem reads from the real data
directory. Edge cases covered include nulls, empty strings, malformed values, and type
boundary conditions.

---

## Artifact Review Order

For anyone reviewing this submission, the recommended reading order is:

**1. Configuration** — start with `config/cities.yaml` and `config/ml_config.yaml` to
understand scope and design intent before looking at any code.

**2. Data quality outputs** — `outputs/quality/{city}_quality_report.json` and
`outputs/profiles/` — shows what was found in the raw data before any transformations.

**3. Relationship maps** — `outputs/relationships/{city}_relationships.json` —
documents PK/FK structure and referential integrity findings.

**4. Star schema** — `data/airbnb.duckdb` — open with any DuckDB client and run
`SHOW TABLES;` followed by `DESCRIBE fact_listing_snapshot;` to confirm the dimensional
model is correctly structured.

**5. Lineage** — `python main.py lineage` — prints the full source-to-output provenance
chain for all pipeline stages that have been run.

**6. ML evaluation report** — `outputs/ml/{experiment_id}/evaluation_report.md` —
human-readable model comparison table with MAE, RMSE, MAPE, and R² across all model
families on the held-out test set.

**7. SHAP outputs** — `outputs/ml/{experiment_id}/shap/feature_importance.json` and
`summary_plot.png` — global feature importance for price prediction.

**8. Bias audit** — `outputs/ml/{experiment_id}/bias_audit_report.json` — per-
neighbourhood LONO-CV gaps, cross-city transfer results, and overall fairness risk rating.

**9. Dashboard** — `cd dashboard && streamlit run app.py` — interactive exploration of
all findings, with the Intervention Radar and Valuation Arbitrage pages being the most
novel outputs not surfaced elsewhere.

**10. Source code** — suggested module reading order:
`common/utils.py` → `ingestion/cleaner.py` → `feature_engineering/enricher.py` →
`data_engineering/modeling/modeler.py` → `feature_engineering/feature_store.py` →
`data_science/training/trainer.py` → `agentic_ai/sql_agent.py`

---

## Known Limitations

**Metadata co-location.** `pipeline_runs`, `data_lineage`, and `schema_history` live
in `data/airbnb.duckdb` alongside the analytical star schema. A corrupted database file
would lose both simultaneously. In production these would live in a separate operational
store.

**ML pipeline reads DE output via filesystem path.** `feature_store.py` resolves
`data/airbnb.duckdb` through a shared utility. This creates an implicit dependency:
the ML pipeline silently fails if the DE pipeline has not yet populated the database.
In a production Airflow deployment, the ML DAG would declare an `ExternalTaskSensor`
on the DE DAG and the connection would be injected via config, not a shared path utility.

**`evaluate`, `explain`, and `bias-audit` CLI commands reconstruct an experiment
context inline.** These commands load a saved model and rebuild a local object to satisfy
the function signatures of `evaluate_experiment`, `explain_model`, and `run_bias_audit`.
This is a known coupling gap: `load_experiment()` returns raw artefacts rather than a
typed `ExperimentResult`. The `run-ml` orchestration command is unaffected and is the
recommended entrypoint.

**Occupancy estimation is approximate.** `estimated_occupancy_rate` is derived from
`availability_365`, which reflects host-configured availability rather than confirmed
bookings. Inside Airbnb's own documentation notes this limitation. Treat occupancy and
revenue figures as directional market signals, not precise measurements.

**Dashboard SQL injection surface.** The AI assistant applies keyword-based guardrails
(`DROP`, `DELETE`, etc.) and enforces a `LIMIT 50` cap. The DuckDB connection is
read-only, providing a second line of defence. For a public deployment, parameterised
queries and stricter input validation would be required.

**No model serving layer.** The pipeline trains, evaluates, and explains models but
provides no `score` command or REST endpoint for batch or real-time inference against
new listings. The dashboard's Price Estimator serves as a lightweight local inference
demonstration.

---

## Assessment Section Coverage

| Section | Task | Implementing code |
|---|---|---|
| §2.3 | Dataset download and schema discovery | `ingestion/downloader.py`, `ingestion/profiler.py` |
| §2.3 | PK/FK relationship mapping | `modeling/relationship_mapper.py` |
| §2.3 | Cross-city schema harmonisation | `modeling/harmonizer.py` |
| §3.1 | Ingestion with quality reporting | `ingestion/downloader.py` + `ingestion/profiler.py` → `ingest` command |
| §3.1 | Duplicate detection and IQR outlier analysis | `ingestion/profiler.py` → `generate_data_quality_report` |
| §3.1 | Constraint-based validation | `ingestion/validator.py`, `config/validation_rules.yaml` |
| §3.2 | Price, boolean, date, percentage coercion | `ingestion/cleaner.py`, `config/cleaning_rules.yaml` |
| §3.2 | Missing value strategies | `config/cleaning_rules.yaml` → `cleaner.py` |
| §3.2 | Rejected record partitioning | `data/staging/{city}/_rejected/` |
| §3.3 | Enriched listing master | `feature_engineering/enricher.py` |
| §3.3 | Derived metrics | `enricher.py` (occupancy, revenue, tenure, price_per_bedroom…) |
| §3.3 | Unified cross-city master | `enricher.py` → `unify-master` command |
| §3.4 | Star schema dimensional model | `data_engineering/modeling/modeler.py` |
| §3.4 | Analytical SQL | `sql/analytical_queries.sql` → `query` command |
| §3.5 | Orchestrated pipeline with metadata | `pipelines/dags/data_pipeline_local.py` |
| §3.5 | Incremental hash-based processing | `common/metadata.py` → `check_already_processed` |
| §3.5 | Data lineage recording | `common/metadata.py` → `record_lineage` |
| §3.5 | Schema drift history | `common/metadata.py` → `schema_history` table |
| §6.1 | Feature engineering and matrix | `feature_engineering/feature_store.py` |
| §6.1 | Multi-model training with CV | `data_science/training/trainer.py` |
| §6.1 | Residual and segment error analysis | `data_science/validation/evaluator.py` |
| §6.1 | SHAP explainability | `data_science/explainability/explainer.py` |
| §6.4 | LONO-CV neighbourhood bias audit | `data_science/validation/bias_auditor.py` |
| §6.4 | Cross-city transfer evaluation | `data_science/validation/bias_auditor.py` |
| §7.2 | LLM-powered SQL assistant | `agentic_ai/sql_agent.py` (Gemini 2.5 Flash) |
| §8 | Interactive market intelligence dashboard | `dashboard/` (7 pages + AI chat) |

---

## Acknowledgements

Dataset sourced from [Inside Airbnb](https://insideairbnb.com/), an independent,
non-commercial project that provides publicly available Airbnb listing data for
community research and advocacy.

This project was developed as a technical assessment submission for
[Expernetic (Pvt) Ltd](https://expernetic.com/) — Talent Assessment Program,
Data Engineer Intern.
