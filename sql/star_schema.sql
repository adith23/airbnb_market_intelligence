-- =====================================================================
-- Airbnb Market Intelligence — Star Schema DDL
-- Target: DuckDB (columnar analytical engine)
--
-- This file documents the dimensional model structure.
-- Tables are created programmatically by pipeline/modeler.py using
-- CREATE OR REPLACE TABLE ... AS SELECT patterns from Parquet sources.
--
-- Schema: 6 dimensions + 3 facts
-- =====================================================================

-- -------------------------------------------------------------------
-- DIMENSION: dim_date
-- Generated date dimension covering the full calendar range.
-- Grain: one row per calendar date.
-- -------------------------------------------------------------------
CREATE OR REPLACE TABLE dim_date (
    date_key        INTEGER PRIMARY KEY,  -- YYYYMMDD integer
    full_date       DATE NOT NULL,
    year            SMALLINT NOT NULL,
    quarter         TINYINT NOT NULL,
    month           TINYINT NOT NULL,
    day_of_month    TINYINT NOT NULL,
    day_of_week     TINYINT NOT NULL,     -- 0=Sunday, 6=Saturday
    day_name        VARCHAR NOT NULL,
    month_name      VARCHAR NOT NULL,
    is_weekend      BOOLEAN NOT NULL,
    week_of_year    TINYINT NOT NULL
);

-- -------------------------------------------------------------------
-- DIMENSION: dim_city
-- One row per city in the analysis. Sourced from config/cities.yaml.
-- -------------------------------------------------------------------
CREATE OR REPLACE TABLE dim_city (
    city_key        INTEGER PRIMARY KEY,  -- Surrogate key
    city_name       VARCHAR NOT NULL,     -- Natural key (e.g., 'paris')
    display_name    VARCHAR NOT NULL,
    country         VARCHAR NOT NULL,
    currency_code   VARCHAR(3) NOT NULL,
    currency_symbol VARCHAR(4),
    timezone        VARCHAR
);

-- -------------------------------------------------------------------
-- DIMENSION: dim_host
-- One row per unique host across all cities.
-- Deduped on host_id (natural key).
-- -------------------------------------------------------------------
CREATE OR REPLACE TABLE dim_host (
    host_key                INTEGER PRIMARY KEY,    -- Surrogate key
    host_id                 BIGINT NOT NULL UNIQUE, -- Natural key
    host_name               VARCHAR,
    host_since              DATE,
    host_location           VARCHAR,
    host_response_time      VARCHAR,
    host_response_rate      DOUBLE,
    host_acceptance_rate    DOUBLE,
    host_is_superhost       BOOLEAN,
    host_listings_count     INTEGER,
    host_total_listings_count INTEGER,
    host_has_profile_pic    BOOLEAN,
    host_identity_verified  BOOLEAN,
    host_verification_count INTEGER,
    is_professional_host    BOOLEAN,
    host_tenure_years       DOUBLE
);

-- -------------------------------------------------------------------
-- DIMENSION: dim_property
-- One row per listing. Contains static property attributes.
-- -------------------------------------------------------------------
CREATE OR REPLACE TABLE dim_property (
    property_key        INTEGER PRIMARY KEY,      -- Surrogate key
    listing_id          BIGINT NOT NULL UNIQUE,   -- Natural key
    listing_url         VARCHAR,
    name                VARCHAR,
    description         VARCHAR,
    property_type       VARCHAR,
    room_type           VARCHAR,
    accommodates        INTEGER,
    bathrooms           DOUBLE,
    bathrooms_shared    BOOLEAN,
    bedrooms            INTEGER,
    beds                INTEGER,
    amenities           VARCHAR,   -- Raw JSON string preserved
    amenity_count       INTEGER,
    instant_bookable    BOOLEAN,
    license             VARCHAR
);

-- -------------------------------------------------------------------
-- DIMENSION: dim_neighbourhood
-- One row per neighbourhood per city.
-- Contains pre-aggregated area-level statistics.
-- -------------------------------------------------------------------
CREATE OR REPLACE TABLE dim_neighbourhood (
    neighbourhood_key       INTEGER PRIMARY KEY,  -- Surrogate key
    neighbourhood_name      VARCHAR NOT NULL,
    neighbourhood_group     VARCHAR,
    city                    VARCHAR NOT NULL,
    listing_count           INTEGER,
    median_price            DOUBLE,
    mean_price              DOUBLE,
    avg_rating              DOUBLE,
    avg_availability        DOUBLE
);

-- -------------------------------------------------------------------
-- DIMENSION: dim_reviewer
-- One row per unique reviewer across all reviews.
-- -------------------------------------------------------------------
CREATE OR REPLACE TABLE dim_reviewer (
    reviewer_key    INTEGER PRIMARY KEY,      -- Surrogate key
    reviewer_id     BIGINT NOT NULL UNIQUE,   -- Natural key
    reviewer_name   VARCHAR
);

-- -------------------------------------------------------------------
-- FACT: fact_listing_snapshot
-- One row per listing per scrape snapshot.
-- Contains measurable metrics + FK references to all dimensions.
-- Grain: listing × snapshot_date
-- -------------------------------------------------------------------
CREATE OR REPLACE TABLE fact_listing_snapshot (
    listing_key             INTEGER PRIMARY KEY,  -- Surrogate key
    listing_id              BIGINT NOT NULL,       -- Degenerate dimension
    host_key                INTEGER REFERENCES dim_host(host_key),
    property_key            INTEGER REFERENCES dim_property(property_key),
    neighbourhood_key       INTEGER REFERENCES dim_neighbourhood(neighbourhood_key),
    city_key                INTEGER REFERENCES dim_city(city_key),
    snapshot_date_key       INTEGER REFERENCES dim_date(date_key),
    -- Price metrics
    price_local             DOUBLE,
    price_usd               DOUBLE,
    -- Location
    latitude                DOUBLE,
    longitude               DOUBLE,
    -- Stay constraints
    minimum_nights          INTEGER,
    maximum_nights          INTEGER,
    -- Review metrics
    number_of_reviews       INTEGER,
    number_of_reviews_ltm   INTEGER,
    review_scores_rating    DOUBLE,
    review_scores_accuracy  DOUBLE,
    review_scores_cleanliness DOUBLE,
    review_scores_checkin   DOUBLE,
    review_scores_communication DOUBLE,
    review_scores_location  DOUBLE,
    review_scores_value     DOUBLE,
    reviews_per_month       DOUBLE,
    -- Availability
    availability_30         INTEGER,
    availability_60         INTEGER,
    availability_90         INTEGER,
    availability_365        INTEGER,
    -- Occupancy & Revenue (from calendar aggregation)
    occupancy_rate_pct      DOUBLE,
    estimated_annual_revenue DOUBLE,
    estimated_monthly_revenue DOUBLE,
    avg_booked_price        DOUBLE,
    -- Derived fields
    price_per_bedroom       DOUBLE,
    price_per_person        DOUBLE,
    price_vs_neighbourhood  DOUBLE,
    host_tenure_years       DOUBLE,
    is_active               BOOLEAN,
    is_professional_host    BOOLEAN
);

-- -------------------------------------------------------------------
-- FACT: fact_calendar
-- One row per listing × date (daily availability grain).
-- Largest table (~68M rows for 3 cities).
-- -------------------------------------------------------------------
CREATE OR REPLACE TABLE fact_calendar (
    listing_key         INTEGER REFERENCES fact_listing_snapshot(listing_key),
    date_key            INTEGER REFERENCES dim_date(date_key),
    is_available        BOOLEAN,
    price_local         DOUBLE,
    price_usd           DOUBLE,
    adjusted_price      DOUBLE,
    minimum_nights      INTEGER,
    maximum_nights      INTEGER
);

-- -------------------------------------------------------------------
-- FACT: fact_review
-- One row per review event.
-- -------------------------------------------------------------------
CREATE OR REPLACE TABLE fact_review (
    review_id           BIGINT PRIMARY KEY,       -- Degenerate dimension
    listing_key         INTEGER REFERENCES fact_listing_snapshot(listing_key),
    reviewer_key        INTEGER REFERENCES dim_reviewer(reviewer_key),
    review_date_key     INTEGER REFERENCES dim_date(date_key),
    comment_length      INTEGER
);
