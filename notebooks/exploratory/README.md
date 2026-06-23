# Exploratory Data Analysis & Statistical Analysis Notebooks

## Prerequisites

1. **Pipeline must be run first** — notebooks read from the DuckDB star schema:
   ```bash
   python main.py ingest --city paris
   python main.py clean --city paris
   python main.py enrich --city paris
   # Repeat for london, new_york_city

   python main.py model --cities paris,london,new_york_city
   ```

2. **Install notebook dependencies**:
   ```bash
   pip install -r notebooks/requirements-notebooks.txt
   ```

3. **Launch Jupyter**:
   ```bash
   jupyter notebook notebooks/
   ```

## Notebook Index

### Section 4 — Exploratory Data Analysis

| # | Notebook | Section | Description |
|:--|:---------|:--------|:------------|
| 01 | `01_summary_statistics.ipynb` | §4.1 | Descriptive stats, price distributions, power-law analysis, rating inflation |
| 02 | `02_geographic_analysis.ipynb` | §4.2 | Folium maps, density choropleth, pricing gradient, spatial clustering |
| 03 | `03_temporal_trends.ipynb` | §4.3 | Seasonal pricing, review velocity, host tenure vs price |
| 04 | `04_host_supply_analysis.ipynb` | §4.4 | Host segmentation, professional vs casual, market concentration |
| 05 | `05_review_demand_analysis.ipynb` | §4.5 | Review-price-score relationships, sub-score analysis |

### Section 5 — Statistical Analysis (in `statistics/` subdirectory)

| # | Notebook | Section | Description |
|:--|:---------|:--------|:------------|
| 01 | `statistics/01_hypothesis_testing.ipynb` | §5.1-5.2 | 5 formal hypothesis tests + confidence intervals + effect sizes |
| 02 | `statistics/02_correlation_regression.ipynb` | §5.3 | Correlation matrix, OLS regression, VIF, LOWESS |
| 03 | `statistics/03_multi_city_statistics.ipynb` | §5.4 | Cross-city comparisons with multiple testing corrections |

## Shared Modules

- **`helpers.py`** — DuckDB connection, plot styling, business insight formatter
- **`statistics/stats_utils.py`** — Reusable statistical functions (hypothesis tests, effect sizes, CIs)

## Data Sources

All notebooks query `data/airbnb.duckdb` (read-only). The star schema contains:

| Table | Description | Typical Size |
|:------|:------------|:-------------|
| `fact_listing_snapshot` | One row per listing with metrics | ~70K rows |
| `fact_calendar` | Daily availability grain | ~68M rows |
| `fact_review` | Individual review events | ~2.5M rows |
| `dim_host` | Host attributes | ~50K rows |
| `dim_property` | Property attributes | ~70K rows |
| `dim_neighbourhood` | Area-level stats | ~300 rows |
| `dim_city` | City metadata | 3 rows |
| `dim_date` | Calendar dimension | ~800 rows |
