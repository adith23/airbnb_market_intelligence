"""Agentic AI module for Text-to-SQL logic."""

import logging
import os
import re

import pandas as pd

from dashboard.backend.data_service import get_db_connection

logger = logging.getLogger(__name__)


# Fallback Mock LLM if no API key is provided
class MockLLM:
    def generate_content(self, prompt: str, **kwargs) -> object:
        class Response:
            def __init__(self, text):
                self.text = text

        if "SQL" in prompt or "schema" in prompt:
            return Response(
                "SELECT n.neighbourhood_name, AVG(l.price_usd) as avg_price FROM fact_listing_snapshot l JOIN dim_neighbourhood n ON l.neighbourhood_key = n.neighbourhood_key GROUP BY n.neighbourhood_name ORDER BY avg_price DESC LIMIT 5"
            )
        return Response(
            "Based on the data, the premium pockets are showing strong pricing power. Let me know if you need to drill down into specific neighbourhoods."
        )


def _get_llm_client():
    """Retrieve the Google GenAI client or fallback to Mock."""
    try:
        from google import genai

        api_key = os.environ.get("GEMINI_API_KEY")
        
        # 1. Fallback: Check local .env file
        if not api_key:
            try:
                from pathlib import Path
                # Check current dir, parent dir, and project root based on file structure
                env_paths = [
                    Path(".env"),
                    Path("../.env"),
                    Path(__file__).resolve().parents[3] / ".env"
                ]
                for p in env_paths:
                    if p.exists():
                        with open(p) as f:
                            for line in f:
                                line = line.strip()
                                if line and not line.startswith("#") and "=" in line:
                                    k, v = line.split("=", 1)
                                    if k.strip() == "GEMINI_API_KEY":
                                        api_key = v.strip().strip('"').strip("'")
                                        break
                        if api_key:
                            break
            except Exception:
                pass

        if api_key:
            return genai.Client(api_key=api_key)
    except ImportError:
        logger.warning("google-genai not installed. Falling back to MockLLM.")
    return MockLLM()


def is_mock_llm() -> bool:
    """Check if the LLM client is a fallback MockLLM."""
    return isinstance(_get_llm_client(), MockLLM)
    

def get_database_schema() -> str:
    """Return a curated DDL for the LLM to understand the star schema."""
    return """
    Table: fact_listing_snapshot
    Columns:
    - listing_id (BIGINT)
    - property_key (BIGINT)
    - neighbourhood_key (BIGINT)
    - city_key (VARCHAR)
    - price_usd (DOUBLE)
    - review_scores_rating (DOUBLE)
    - number_of_reviews (INTEGER)
    - occupancy_rate_pct (DOUBLE)
    - estimated_monthly_revenue (DOUBLE)
    - is_professional_host (BOOLEAN)
    
    Table: dim_property
    Columns:
    - property_key (BIGINT)
    - name (VARCHAR)
    - room_type (VARCHAR)
    - accommodates (INTEGER)
    - bedrooms (INTEGER)
    - bathrooms (DOUBLE)
    
    Table: dim_neighbourhood
    Columns:
    - neighbourhood_key (BIGINT)
    - neighbourhood_name (VARCHAR)
    - city (VARCHAR)
    
    IMPORTANT SQL RULES:
    1. For text comparisons (like city names or neighbourhoods), ALWAYS use case-insensitive matching with ILIKE (e.g., city ILIKE '%barcelona%') because the database keys may be stored as lowercase slugs (e.g., 'barcelona', 'new-york-city').
    2. Join `fact_listing_snapshot` to `dim_neighbourhood` on `neighbourhood_key` to filter by city.
    """


def generate_sql(user_query: str) -> str:
    """Uses LLM to translate natural language to DuckDB SQL."""
    client = _get_llm_client()
    schema = get_database_schema()

    prompt = f"""
    You are an expert Data Architect for Airbnb Market Intelligence.
    Given the following DuckDB Star Schema:
    {schema}
    
    Translate this user question into a valid, optimized DuckDB SQL query.
    Return ONLY the raw SQL query. Do not wrap it in markdown code blocks like ```sql. Do not explain.
    
    User Question: {user_query}
    """

    response = (
        client.models.generate_content(
            contents=prompt,
            model="gemini-2.5-flash",
        )
        if not isinstance(client, MockLLM)
        else client.generate_content(prompt)
    )

    raw_response = response.text

    # Clean up any potential markdown backticks
    clean_sql = re.sub(r"```sql\s*", "", raw_response, flags=re.IGNORECASE)
    clean_sql = re.sub(r"```\s*", "", clean_sql)
    return clean_sql.strip()


def validate_and_secure_sql(sql: str) -> str:
    """Apply strict security guardrails to the generated SQL."""
    forbidden_keywords = [
        "DROP",
        "DELETE",
        "UPDATE",
        "INSERT",
        "ALTER",
        "GRANT",
        "TRUNCATE",
    ]
    upper_sql = sql.upper()

    for kw in forbidden_keywords:
        if re.search(rf"\b{kw}\b", upper_sql):
            raise ValueError(f"Security Guardrail Triggered: Forbidden keyword '{kw}' detected.")

    # Strip trailing semicolon if present to prevent subquery syntax errors
    sql = sql.strip().rstrip(";")

    # Enforce LIMIT 50 to prevent memory blowouts on the dashboard
    if "LIMIT" not in upper_sql:
        sql = f"SELECT * FROM (\n{sql}\n) LIMIT 50"

    return sql


def execute_agent_query(sql: str) -> pd.DataFrame:
    """Execute the safe SQL against DuckDB."""
    conn = get_db_connection()
    try:
        df = conn.execute(sql).df()
        return df
    except Exception as e:
        raise RuntimeError(f"Database Execution Error: {str(e)}")


def synthesize_results(user_query: str, df: pd.DataFrame) -> str:
    """Uses LLM to write a 2-3 sentence executive summary of the dataframe."""
    if df.empty:
        return "The query returned no results for this market context."

    client = _get_llm_client()
    try:
        data_str = df.head(10).to_markdown()
    except ImportError:
        data_str = df.head(10).to_string()

    prompt = f"""
    You are an expert Business Analyst. 
    A stakeholder asked: "{user_query}"
    
    The database returned this data:
    {data_str}
    
    Write a concise, 2-3 sentence executive summary answering their question based strictly on this data.
    Do not use introductory phrases like "Based on the data". Get straight to the point.
    """

    response = (
        client.models.generate_content(
            contents=prompt,
            model="gemini-2.5-flash",
        )
        if not isinstance(client, MockLLM)
        else client.generate_content(prompt)
    )

    return response.text.strip()
