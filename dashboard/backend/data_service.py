"""Data access layer using DuckDB for high-performance querying."""

import duckdb
import pandas as pd
import streamlit as st

from dashboard.config import DB_PATH


def get_db_connection() -> duckdb.DuckDBPyConnection:
    """Initialize a read-only DuckDB connection."""
    return duckdb.connect(DB_PATH, read_only=True)


@st.cache_data(ttl=3600)
def fetch_executive_kpis(city_key: str | None = None) -> dict:
    """Fetch top-level KPIs for the executive summary."""
    conn = get_db_connection()

    where_clause = f"WHERE city_key = '{city_key}'" if city_key else ""

    query = f"""
        SELECT 
            COUNT(*) AS total_listings,
            AVG(price_usd) AS avg_daily_rate_usd,
            AVG(occupancy_rate_pct) AS avg_occupancy_pct,
            SUM(CASE WHEN is_professional_host THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS professional_host_pct
        FROM fact_listing_snapshot
        {where_clause}
    """
    df = conn.execute(query).df()
    if df.empty:
        return {}

    return df.iloc[0].to_dict()


@st.cache_data(ttl=3600)
def fetch_geospatial_data(
    city_key: str | None = None, limit: int = 50000
) -> pd.DataFrame:
    """Fetch coordinates and prices for PyDeck map rendering."""
    conn = get_db_connection()
    where_clause = f"WHERE city_key = '{city_key}'" if city_key else ""

    query = f"""
        SELECT 
            latitude, 
            longitude, 
            price_usd,
            room_type
        FROM fact_listing_snapshot
        JOIN dim_property ON fact_listing_snapshot.property_key = dim_property.property_key
        {where_clause}
        LIMIT {limit}
    """
    return conn.execute(query).df()


@st.cache_data(ttl=3600)
def fetch_neighbourhood_metrics(city_key: str | None = None) -> pd.DataFrame:
    """Fetch neighbourhood level aggregates."""
    conn = get_db_connection()
    where_clause = f"WHERE city_key = '{city_key}'" if city_key else ""
    query = f"""
        SELECT 
            dim_neighbourhood.neighbourhood_name,
            COUNT(*) as listing_count,
            MEDIAN(price_usd) as median_price_usd,
            AVG(occupancy_rate_pct) as avg_occupancy_pct,
            MEDIAN(estimated_monthly_revenue) as median_monthly_revenue
        FROM fact_listing_snapshot
        JOIN dim_neighbourhood ON fact_listing_snapshot.neighbourhood_key = dim_neighbourhood.neighbourhood_key
        {where_clause}
        GROUP BY dim_neighbourhood.neighbourhood_name
        HAVING COUNT(*) > 50
        ORDER BY median_price_usd DESC
        LIMIT 20
    """
    return conn.execute(query).df()


@st.cache_data(ttl=3600)
def fetch_room_type_metrics(city_key: str | None = None) -> pd.DataFrame:
    """Fetch room type aggregates."""
    conn = get_db_connection()
    where_clause = f"WHERE city_key = '{city_key}'" if city_key else ""
    query = f"""
        SELECT 
            dim_property.room_type,
            COUNT(*) as listing_count,
            MEDIAN(price_usd) as median_price_usd
        FROM fact_listing_snapshot
        JOIN dim_property ON fact_listing_snapshot.property_key = dim_property.property_key
        {where_clause}
        GROUP BY dim_property.room_type
        ORDER BY listing_count DESC
    """
    return conn.execute(query).df()


@st.cache_data(ttl=3600)
def fetch_valuation_arbitrage(city_key: str | None = None) -> pd.DataFrame:
    """Identify undervalued listings based on high rating but below median price."""
    conn = get_db_connection()
    where_clause = f"WHERE city_key = '{city_key}'" if city_key else "WHERE 1=1"

    query = f"""
        WITH CityMedian AS (
            SELECT MEDIAN(price_usd) as med_price FROM fact_listing_snapshot {where_clause}
        )
        SELECT 
            l.listing_id,
            p.name,
            l.price_usd,
            p.bedrooms,
            ROUND(l.price_usd / NULLIF(p.bedrooms, 0), 2) AS price_per_bedroom,
            l.review_scores_rating,
            l.number_of_reviews,
            l.estimated_monthly_revenue,
            n.neighbourhood_name
        FROM fact_listing_snapshot l
        JOIN dim_property p ON l.property_key = p.property_key
        JOIN dim_neighbourhood n ON l.neighbourhood_key = n.neighbourhood_key
        {where_clause.replace("WHERE", "AND")}
        AND l.review_scores_rating >= 4.8 
        AND l.number_of_reviews > 30
        AND l.price_usd < (SELECT med_price FROM CityMedian)
        ORDER BY l.estimated_monthly_revenue DESC
        LIMIT 50
    """
    return conn.execute(query).df()


@st.cache_data(ttl=3600)
def fetch_host_concentration(city_key: str | None = None) -> pd.DataFrame:
    """Analyze pricing divergence between casual and professional hosts."""
    conn = get_db_connection()
    where_clause = f"WHERE city_key = '{city_key}'" if city_key else ""

    query = f"""
        SELECT 
            CASE WHEN is_professional_host THEN 'Professional (2+)' ELSE 'Casual (1)' END as host_segment,
            COUNT(*) as listing_count,
            MEDIAN(price_usd) as median_price_usd,
            AVG(occupancy_rate_pct) as avg_occupancy_pct,
            MEDIAN(estimated_monthly_revenue) as median_revenue
        FROM fact_listing_snapshot
        {where_clause}
        GROUP BY 1
        ORDER BY listing_count DESC
    """
    return conn.execute(query).df()


@st.cache_data(ttl=3600)
def fetch_temporal_trends(city_key: str | None = None) -> pd.DataFrame:
    """Analyze seasonal yield curves from calendar data."""
    conn = get_db_connection()
    # If city_key is provided, we need to join back to listings. For performance,
    # we'll use a direct join if needed, but calendar queries can be heavy.

    city_join = ""
    where_clause = ""
    if city_key:
        city_join = "JOIN fact_listing_snapshot l ON c.listing_key = l.listing_key"
        where_clause = f"WHERE l.city_key = '{city_key}'"

    query = f"""
        SELECT 
            d.full_date as date,
            AVG(c.price_usd) as avg_price,
            100.0 - (SUM(CASE WHEN c.is_available THEN 1 ELSE 0 END) * 100.0 / COUNT(*)) as booked_occupancy_rate
        FROM fact_calendar c
        JOIN dim_date d ON c.date_key = d.date_key
        {city_join}
        {where_clause}
        GROUP BY d.full_date
        ORDER BY d.full_date
    """
    # DuckDB is fast, but we'll aggregate to monthly if returning to pandas takes too long
    # Actually, daily is fine for a year (365 rows).
    df = conn.execute(query).df()
    if not df.empty:
        if pd.api.types.is_integer_dtype(df["date"]):
            df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
        else:
            df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=3600)
def fetch_intervention_radar(city_key: str | None = None) -> pd.DataFrame:
    """Score listings on health and generate actionable interventions."""
    conn = get_db_connection()
    where_clause = f"WHERE city_key = '{city_key}'" if city_key else "WHERE 1=1"

    query = f"""
        WITH ListingMetrics AS (
            SELECT 
                l.listing_id,
                p.name,
                l.price_usd,
                l.review_scores_rating,
                l.number_of_reviews,
                l.occupancy_rate_pct,
                l.is_professional_host,
                l.availability_365,
                n.neighbourhood_name,
                AVG(l.price_usd) OVER(PARTITION BY n.neighbourhood_name) as hood_avg_price,
                AVG(l.occupancy_rate_pct) OVER(PARTITION BY n.neighbourhood_name) as hood_avg_occ,
                AVG(l.review_scores_rating) OVER(PARTITION BY n.neighbourhood_name) as hood_avg_rating
            FROM fact_listing_snapshot l
            JOIN dim_property p ON l.property_key = p.property_key
            JOIN dim_neighbourhood n ON l.neighbourhood_key = n.neighbourhood_key
            {where_clause.replace("WHERE", "AND l.number_of_reviews > 10 AND l.price_usd > 10 AND")}
        ),
        ScoredListings AS (
            SELECT 
                listing_id,
                name,
                neighbourhood_name,
                price_usd,
                ROUND(review_scores_rating, 2) as rating,
                ROUND(occupancy_rate_pct, 1) as occupancy_pct,
                ROUND((
                    (LEAST(GREATEST(10 - ((price_usd / NULLIF(hood_avg_price, 0)) * 5), 0), 10) * 0.25) + 
                    (LEAST(GREATEST((review_scores_rating - 3.5) * 6.66, 0), 10) * 0.40) + 
                    (LEAST(GREATEST((occupancy_rate_pct / 10), 0), 10) * 0.35)
                ) * 10, 1) as health_score,
                CASE 
                    WHEN price_usd < (hood_avg_price * 0.8) AND occupancy_rate_pct > 75 AND review_scores_rating >= 4.8 THEN '🔥 Underpriced relative to demand'
                    WHEN price_usd > (hood_avg_price * 1.3) AND occupancy_rate_pct < 30 THEN '📉 Price too high for market position'
                    WHEN review_scores_rating < 4.4 AND number_of_reviews > 20 THEN '⚠️ Low review quality risk'
                    WHEN availability_365 > 300 AND occupancy_rate_pct < 15 THEN '🛑 High availability but weak conversion'
                    WHEN is_professional_host AND review_scores_rating < 4.6 THEN '🏢 Host portfolio quality risk'
                    ELSE '✅ Healthy Market Position'
                END as priority_action
            FROM ListingMetrics
        )
        SELECT * FROM ScoredListings
        WHERE priority_action != '✅ Healthy Market Position'
        ORDER BY health_score ASC
        LIMIT 300
    """
    return conn.execute(query).df()


@st.cache_data(ttl=3600)
def fetch_neighbourhood_interventions(city_key: str | None = None) -> pd.DataFrame:
    """Determine neighborhood level zones of opportunity or risk."""
    conn = get_db_connection()
    where_clause = f"WHERE city_key = '{city_key}'" if city_key else ""

    query = f"""
        WITH CityAvg AS (
            SELECT AVG(price_usd) as c_price FROM fact_listing_snapshot {where_clause}
        )
        SELECT 
            n.neighbourhood_name,
            COUNT(l.listing_id) as total_listings,
            ROUND(AVG(l.price_usd), 0) as avg_price,
            ROUND(AVG(l.occupancy_rate_pct), 1) as avg_occupancy_pct,
            ROUND(AVG(l.review_scores_rating), 2) as avg_rating,
            CASE
                WHEN AVG(l.occupancy_rate_pct) > 65 AND AVG(l.price_usd) < (SELECT c_price FROM CityAvg) THEN '🚀 High Demand / Value Zone'
                WHEN AVG(l.occupancy_rate_pct) < 35 AND COUNT(l.listing_id) > 200 THEN '⚠️ Over-saturated area'
                WHEN AVG(l.price_usd) > (SELECT c_price * 1.5 FROM CityAvg) THEN '💎 Premium Pocket'
                WHEN AVG(l.review_scores_rating) < 4.5 THEN '🚩 Weak-performing zone'
                ELSE 'Neutral'
            END as opportunity_zone
        FROM fact_listing_snapshot l
        JOIN dim_neighbourhood n ON l.neighbourhood_key = n.neighbourhood_key
        {where_clause}
        GROUP BY 1
        HAVING COUNT(l.listing_id) > 20
        ORDER BY avg_occupancy_pct DESC
    """
    return conn.execute(query).df()


@st.cache_data(ttl=3600)
def fetch_available_cities() -> dict[int, str]:
    """Fetch available cities as a mapping of city_key (int) -> display_name (str)."""
    conn = get_db_connection()
    query = "SELECT city_key, display_name FROM dim_city ORDER BY display_name"
    df = conn.execute(query).df()
    if df.empty:
        return {}
    return dict(zip(df["city_key"], df["display_name"]))
