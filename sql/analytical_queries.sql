-- =====================================================================
-- Airbnb Market Intelligence — Analytical Queries
-- Target: DuckDB star schema (see sql/star_schema.sql)
--
-- These queries demonstrate key analytical use cases against the
-- dimensional model. Each query is named and documented.
--
-- Run via: python main.py query --name <query_name>
--          python main.py query --sql "SELECT ..."
-- =====================================================================


-- -------------------------------------------------------------------
-- Q1: Market Overview
-- Average price, occupancy, and revenue by city and neighbourhood.
-- -------------------------------------------------------------------
-- name: market_overview
SELECT
    c.display_name                          AS city,
    n.neighbourhood_name                    AS neighbourhood,
    COUNT(*)                                AS listing_count,
    ROUND(AVG(f.price_local), 2)            AS avg_price,
    ROUND(MEDIAN(f.price_local), 2)         AS median_price,
    ROUND(AVG(f.occupancy_rate_pct), 1)     AS avg_occupancy_pct,
    ROUND(AVG(f.estimated_monthly_revenue), 0) AS avg_monthly_revenue,
    ROUND(SUM(f.estimated_annual_revenue), 0)  AS total_annual_revenue
FROM fact_listing_snapshot f
JOIN dim_city c            ON f.city_key = c.city_key
JOIN dim_neighbourhood n   ON f.neighbourhood_key = n.neighbourhood_key
GROUP BY c.display_name, n.neighbourhood_name
ORDER BY c.display_name, avg_monthly_revenue DESC;


-- -------------------------------------------------------------------
-- Q2: Host Segmentation
-- Professional (3+ listings) vs individual hosts.
-- -------------------------------------------------------------------
-- name: host_segmentation
SELECT
    c.display_name                                      AS city,
    CASE WHEN h.is_professional_host THEN 'Professional'
         ELSE 'Individual' END                          AS host_type,
    COUNT(DISTINCT h.host_id)                           AS host_count,
    COUNT(*)                                            AS listing_count,
    ROUND(AVG(f.price_local), 2)                        AS avg_price,
    ROUND(AVG(f.review_scores_rating), 2)               AS avg_rating,
    ROUND(AVG(f.occupancy_rate_pct), 1)                 AS avg_occupancy_pct,
    ROUND(AVG(f.estimated_monthly_revenue), 0)          AS avg_monthly_revenue,
    ROUND(SUM(f.estimated_annual_revenue), 0)           AS total_annual_revenue
FROM fact_listing_snapshot f
JOIN dim_host h ON f.host_key = h.host_key
JOIN dim_city c ON f.city_key = c.city_key
GROUP BY c.display_name, h.is_professional_host
ORDER BY c.display_name, host_type;


-- -------------------------------------------------------------------
-- Q3: Price Positioning
-- Listings priced above/below neighbourhood median, by room type.
-- -------------------------------------------------------------------
-- name: price_positioning
SELECT
    c.display_name                                  AS city,
    p.room_type,
    CASE
        WHEN f.price_vs_neighbourhood > 1.2 THEN 'Premium (>120% median)'
        WHEN f.price_vs_neighbourhood BETWEEN 0.8 AND 1.2 THEN 'Mid-range (80-120%)'
        WHEN f.price_vs_neighbourhood < 0.8 THEN 'Budget (<80% median)'
        ELSE 'Unknown'
    END                                             AS price_segment,
    COUNT(*)                                        AS listing_count,
    ROUND(AVG(f.price_local), 2)                    AS avg_price,
    ROUND(AVG(f.occupancy_rate_pct), 1)             AS avg_occupancy_pct,
    ROUND(AVG(f.review_scores_rating), 2)           AS avg_rating
FROM fact_listing_snapshot f
JOIN dim_city c     ON f.city_key = c.city_key
JOIN dim_property p ON f.property_key = p.property_key
WHERE f.price_vs_neighbourhood IS NOT NULL
GROUP BY c.display_name, p.room_type, price_segment
ORDER BY c.display_name, p.room_type, price_segment;


-- -------------------------------------------------------------------
-- Q4: Seasonal Patterns
-- Monthly occupancy and pricing trends from calendar data.
-- -------------------------------------------------------------------
-- name: seasonal_patterns
SELECT
    d.year,
    d.month,
    d.month_name,
    COUNT(*)                                            AS total_days,
    SUM(CASE WHEN fc.is_available THEN 0 ELSE 1 END)   AS booked_days,
    ROUND(
        SUM(CASE WHEN fc.is_available THEN 0 ELSE 1 END)
        * 100.0 / COUNT(*), 1
    )                                                   AS occupancy_pct,
    ROUND(AVG(fc.price_local), 2)                       AS avg_asking_price,
    ROUND(AVG(CASE WHEN NOT fc.is_available
              THEN fc.price_local END), 2)              AS avg_booked_price
FROM fact_calendar fc
JOIN dim_date d ON fc.date_key = d.date_key
GROUP BY d.year, d.month, d.month_name
ORDER BY d.year, d.month;


-- -------------------------------------------------------------------
-- Q5: Supply Analysis
-- Listing density by neighbourhood, property type distribution.
-- -------------------------------------------------------------------
-- name: supply_analysis
SELECT
    c.display_name                              AS city,
    p.property_type,
    p.room_type,
    COUNT(*)                                    AS listing_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*))
          OVER (PARTITION BY c.display_name), 1) AS pct_of_city_supply,
    ROUND(AVG(p.accommodates), 1)               AS avg_capacity,
    ROUND(AVG(f.price_local), 2)                AS avg_price,
    ROUND(AVG(f.availability_365), 0)           AS avg_availability_365
FROM fact_listing_snapshot f
JOIN dim_city c     ON f.city_key = c.city_key
JOIN dim_property p ON f.property_key = p.property_key
GROUP BY c.display_name, p.property_type, p.room_type
HAVING COUNT(*) >= 10
ORDER BY c.display_name, listing_count DESC;


-- -------------------------------------------------------------------
-- Q6: Review Velocity
-- Monthly review trends as a proxy for booking volume.
-- Assumes approximately 50% of guests leave reviews.
-- -------------------------------------------------------------------
-- name: review_velocity
SELECT
    d.year,
    d.month,
    d.month_name,
    COUNT(*)                    AS review_count,
    COUNT(*) * 2                AS estimated_bookings,
    COUNT(DISTINCT fr.listing_key)  AS listings_reviewed,
    ROUND(AVG(fr.comment_length), 0) AS avg_comment_length
FROM fact_review fr
JOIN dim_date d ON fr.review_date_key = d.date_key
GROUP BY d.year, d.month, d.month_name
ORDER BY d.year, d.month;


-- -------------------------------------------------------------------
-- Q7: Regulatory Compliance
-- License coverage by city and neighbourhood.
-- -------------------------------------------------------------------
-- name: regulatory_compliance
SELECT
    c.display_name                                      AS city,
    n.neighbourhood_name                                AS neighbourhood,
    COUNT(*)                                            AS total_listings,
    SUM(CASE WHEN p.license IS NOT NULL
             AND p.license != '' THEN 1 ELSE 0 END)    AS with_license,
    ROUND(
        SUM(CASE WHEN p.license IS NOT NULL
                 AND p.license != '' THEN 1 ELSE 0 END)
        * 100.0 / COUNT(*), 1
    )                                                   AS license_pct
FROM fact_listing_snapshot f
JOIN dim_city c            ON f.city_key = c.city_key
JOIN dim_property p        ON f.property_key = p.property_key
JOIN dim_neighbourhood n   ON f.neighbourhood_key = n.neighbourhood_key
GROUP BY c.display_name, n.neighbourhood_name
ORDER BY c.display_name, license_pct DESC;


-- -------------------------------------------------------------------
-- Q8: Cross-City Comparison
-- Normalized price, occupancy, and revenue across cities.
-- Uses USD-converted prices for fair comparison.
-- -------------------------------------------------------------------
-- name: cross_city_comparison
SELECT
    c.display_name                                  AS city,
    c.currency_code,
    COUNT(*)                                        AS listing_count,
    ROUND(AVG(f.price_local), 2)                    AS avg_price_local,
    ROUND(AVG(f.price_usd), 2)                      AS avg_price_usd,
    ROUND(MEDIAN(f.price_usd), 2)                   AS median_price_usd,
    ROUND(AVG(f.occupancy_rate_pct), 1)             AS avg_occupancy_pct,
    ROUND(AVG(f.estimated_monthly_revenue), 0)      AS avg_monthly_rev_local,
    ROUND(AVG(f.review_scores_rating), 2)           AS avg_rating,
    ROUND(AVG(f.number_of_reviews), 1)              AS avg_review_count,
    SUM(CASE WHEN f.is_active THEN 1 ELSE 0 END)   AS active_listings,
    ROUND(
        SUM(CASE WHEN f.is_active THEN 1 ELSE 0 END)
        * 100.0 / COUNT(*), 1
    )                                               AS active_pct
FROM fact_listing_snapshot f
JOIN dim_city c ON f.city_key = c.city_key
GROUP BY c.display_name, c.currency_code
ORDER BY avg_price_usd DESC;
