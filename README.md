# Airbnb Market Intelligence Pipeline

Production-grade data engineering pipeline for Inside Airbnb datasets.
Covers ingestion, profiling, cleaning, enrichment, and dimensional modeling.

## Quick Start

```bash
# 1. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Update scrape dates in config/cities.yaml to latest available
#    Check: http://insideairbnb.com/get-the-data

# 4. Ingest & profile a city (§3.1: download + profile + quality report)
python main.py ingest --city new-york-city

# 5. Clean & standardize (§3.2: raw CSV → typed Parquet)
python main.py clean --city new-york-city

# 6. Enrich & join (Section 3.3)
python main.py enrich --city new-york-city

# 7. Model as a DuckDB star schema (Section 3.4)
python main.py model --cities new-york-city

# Or run exploration (§2.3) and engineering steps individually:
python main.py download --city paris
python main.py profile --city paris
python main.py quality-report --city paris
python main.py validate --city paris
python main.py map-rels --city paris
python main.py clean --city paris --file-type listings
python main.py enrich --city paris
python main.py model --cities paris,new_york_city,london
python main.py query --name market_overview
```

## Pipeline Stages

```
Raw Data (CSV/GZ) ──→ Profile & QA ──→ Clean (Parquet) ──→ Enrich ──→ Model (DuckDB)
     §3.1                §3.1              §3.2              §3.3        §3.4
  data/raw/           outputs/quality/   data/staging/     data/enriched/  data/airbnb.duckdb
```

## CLI Commands

| Command | Section | Description |
|:--------|:--------|:------------|
| `download --city <name>` | §2.3 | Download dataset files from Inside Airbnb |
| `profile --city <name>` | §2.3 | Schema discovery + statistical profiling |
| `validate --city <name>` | §2.3 | Data quality checks + constraint validation |
| `map-rels --city <name>` | §2.3 | PK/FK relationship mapping + integrity checks |
| `harmonize --cities <a,b>` | §2.3 | Cross-city schema comparison |
| `explore --city <name>` | §2.3 | Run all exploration steps for one city |
| `explore-all` | §2.3 | Run all steps for all configured cities |
| **`ingest --city <name>`** | **§3.1** | **Download → profile → quality report** |
| **`quality-report --city <name>`** | **§3.1** | **Consolidated quality report with outlier detection** |
| **`clean --city <name>`** | **§3.2** | **Clean raw data → typed Parquet (staging)** |
| **`enrich --city <name>`** | **§3.3** | **Build enriched master listings with joins and derived fields** |
| **`unify-master --cities <a,b>`** | **§3.3** | **Build cross-city unified master listings Parquet** |
| **`model --cities <a,b>`** | **§3.4** | **Build DuckDB star schema dimensions and facts** |
| **`query --name <query>`** | **§3.4** | **Run named analytical queries against DuckDB** |

Add `--verbose` / `-v` for debug logging.

## Project Structure

```
├── config/
│   ├── cities.yaml           # City definitions + URLs
│   ├── cleaning_rules.yaml   # Missing value strategies, validation rules
│   ├── schema_map.yaml       # Canonical column mapping
│   └── validation_rules.yaml # Pre-cleaning validation constraints
├── data/                     # All data (gitignored)
│   ├── raw/{city}/           # Bronze: untouched downloads
│   ├── staging/{city}/       # Silver: cleaned Parquet files
│   │   └── _rejected/       # Records that failed validation
│   ├── enriched/             # Gold: joined, enriched tables
│   └── airbnb.duckdb         # Platinum: star schema database
├── outputs/                  # Reports & artifacts
│   ├── schemas/              # JSON schemas per file
│   ├── profiles/             # Statistical profiles
│   ├── quality/              # Quality + cleaning reports
│   ├── relationships/        # ERD + integrity reports
│   └── harmonization/        # Cross-city comparisons
├── pipeline/                 # Core modules
│   ├── utils.py              # Shared parsing utilities
│   ├── downloader.py         # Stage 1: dataset acquisition
│   ├── profiler.py           # Stage 1: profiling + outlier detection
│   ├── validator.py          # Stage 1: quality validation
│   ├── cleaner.py            # Stage 2: cleaning & standardization
│   ├── enricher.py           # Stage 3: enrichment & joining
│   ├── modeler.py            # Stage 4: DuckDB star schema
│   ├── relationship_mapper.py  # PK/FK analysis
│   └── harmonizer.py         # Multi-city comparison
├── tests/                    # Unit tests
└── main.py                   # CLI entry point
```

## Running Tests

```bash
pytest tests/ -v
```
