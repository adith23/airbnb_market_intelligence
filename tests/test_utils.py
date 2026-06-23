"""Unit tests for pipeline.utils — shared parsing and cleaning functions.

Tests each utility function with:
  - Normal/expected inputs
  - Edge cases (empty, null, malformed)
  - Type safety validation
"""

from __future__ import annotations

from src.platform.common.utils import (
    clean_price,
    compute_schema_hash,
    detect_price_currency,
    infer_file_type,
    parse_amenities,
    parse_bathrooms_text,
    parse_boolean,
    parse_host_verifications,
    parse_rate_pct,
    strip_html,
)

# ===================================================================
# clean_price
# ===================================================================


class TestCleanPrice:
    def test_usd_format(self):
        assert clean_price("$1,250.00") == 1250.0

    def test_euro_format(self):
        assert clean_price("€99.00") == 99.0

    def test_gbp_format(self):
        assert clean_price("£1,234.56") == 1234.56

    def test_zero_price(self):
        assert clean_price("$0.00") == 0.0

    def test_no_cents(self):
        assert clean_price("$100") == 100.0

    def test_empty_string(self):
        assert clean_price("") is None

    def test_none_input(self):
        assert clean_price(None) is None

    def test_whitespace(self):
        assert clean_price("  $50.00  ") == 50.0

    def test_non_string(self):
        assert clean_price(123) is None


# ===================================================================
# parse_boolean
# ===================================================================


class TestParseBoolean:
    def test_true(self):
        assert parse_boolean("t") is True

    def test_false(self):
        assert parse_boolean("f") is False

    def test_uppercase(self):
        assert parse_boolean("T") is True

    def test_none(self):
        assert parse_boolean(None) is None

    def test_empty(self):
        assert parse_boolean("") is None

    def test_invalid(self):
        assert parse_boolean("yes") is None

    def test_whitespace(self):
        assert parse_boolean(" t ") is True


# ===================================================================
# parse_amenities
# ===================================================================


class TestParseAmenities:
    def test_json_array(self):
        result = parse_amenities('["Wifi", "Kitchen", "Pool"]')
        assert result == ["Kitchen", "Pool", "Wifi"]

    def test_set_notation(self):
        result = parse_amenities('{"Wifi", "Kitchen"}')
        assert result == ["Kitchen", "Wifi"]

    def test_empty_string(self):
        assert parse_amenities("") == []

    def test_none(self):
        assert parse_amenities(None) == []

    def test_empty_array(self):
        assert parse_amenities("[]") == []

    def test_single_item(self):
        result = parse_amenities('["Wifi"]')
        assert result == ["Wifi"]


# ===================================================================
# parse_bathrooms_text
# ===================================================================


class TestParseBathroomsText:
    def test_numeric(self):
        assert parse_bathrooms_text("1.5 baths") == (1.5, False)

    def test_integer(self):
        assert parse_bathrooms_text("2 baths") == (2.0, False)

    def test_half_bath(self):
        assert parse_bathrooms_text("Half-bath") == (0.5, False)

    def test_shared_half(self):
        assert parse_bathrooms_text("Shared half-bath") == (0.5, True)

    def test_zero(self):
        assert parse_bathrooms_text("0 baths") == (0.0, False)

    def test_none(self):
        assert parse_bathrooms_text(None) == (None, False)

    def test_empty(self):
        assert parse_bathrooms_text("") == (None, False)

    def test_private_bath(self):
        count, shared = parse_bathrooms_text("1 private bath")
        assert count == 1.0
        assert shared is False


# ===================================================================
# parse_rate_pct
# ===================================================================


class TestParseRatePct:
    def test_normal(self):
        assert parse_rate_pct("95%") == 0.95

    def test_hundred(self):
        assert parse_rate_pct("100%") == 1.0

    def test_zero(self):
        assert parse_rate_pct("0%") == 0.0

    def test_na(self):
        assert parse_rate_pct("N/A") is None

    def test_empty(self):
        assert parse_rate_pct("") is None

    def test_none(self):
        assert parse_rate_pct(None) is None

    def test_whitespace(self):
        assert parse_rate_pct(" 80% ") == 0.8


# ===================================================================
# parse_host_verifications
# ===================================================================


class TestParseHostVerifications:
    def test_standard_list(self):
        result = parse_host_verifications("['email', 'phone', 'work_email']")
        assert result == ["email", "phone", "work_email"]

    def test_empty(self):
        assert parse_host_verifications("") == []

    def test_none(self):
        assert parse_host_verifications(None) == []

    def test_single(self):
        result = parse_host_verifications("['email']")
        assert result == ["email"]


# ===================================================================
# detect_price_currency
# ===================================================================


class TestDetectPriceCurrency:
    def test_usd(self):
        assert detect_price_currency("$100.00") == "$"

    def test_eur(self):
        assert detect_price_currency("€99.00") == "€"

    def test_gbp(self):
        assert detect_price_currency("£50.00") == "£"

    def test_no_currency(self):
        assert detect_price_currency("100.00") is None

    def test_none(self):
        assert detect_price_currency(None) is None


# ===================================================================
# strip_html
# ===================================================================


class TestStripHtml:
    def test_basic_tags(self):
        assert strip_html("<b>Hello</b> world") == "Hello world"

    def test_entities(self):
        assert strip_html("A &amp; B") == "A & B"

    def test_none(self):
        assert strip_html(None) is None

    def test_no_html(self):
        assert strip_html("plain text") == "plain text"

    def test_empty_after_strip(self):
        assert strip_html("<br>") is None


# ===================================================================
# compute_schema_hash
# ===================================================================


class TestComputeSchemaHash:
    def test_deterministic(self):
        h1 = compute_schema_hash(["id", "name", "price"])
        h2 = compute_schema_hash(["id", "name", "price"])
        assert h1 == h2

    def test_order_independent(self):
        h1 = compute_schema_hash(["id", "name", "price"])
        h2 = compute_schema_hash(["price", "id", "name"])
        assert h1 == h2

    def test_different_columns(self):
        h1 = compute_schema_hash(["id", "name"])
        h2 = compute_schema_hash(["id", "price"])
        assert h1 != h2


# ===================================================================
# infer_file_type
# ===================================================================


class TestInferFileType:
    def test_listings_csv(self):
        assert infer_file_type("listings.csv") == "listings"

    def test_listings_gz(self):
        assert infer_file_type("listings.csv.gz") == "listings"

    def test_calendar(self):
        assert infer_file_type("calendar.csv.gz") == "calendar"

    def test_reviews(self):
        assert infer_file_type("reviews.csv") == "reviews"

    def test_neighbourhoods(self):
        assert infer_file_type("neighbourhoods.csv") == "neighbourhoods"

    def test_full_path(self):
        assert infer_file_type("data/raw/paris/listings.csv.gz") == "listings"

    def test_unknown(self):
        assert infer_file_type("random.csv") == "unknown"
