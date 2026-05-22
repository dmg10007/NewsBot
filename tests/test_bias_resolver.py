"""Tests for bias resolver — runs without network access using mock data."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from bias.resolver import BiasResolver, _merge_bias, _normalize_domain
from bias.source_ratings import BIAS_SCALE


# ---------------------------------------------------------------------------
# Unit tests: helpers
# ---------------------------------------------------------------------------

class TestNormalizeDomain:
    def test_strips_https(self):
        assert _normalize_domain("https://www.foxnews.com/politics") == "foxnews.com"

    def test_strips_www(self):
        assert _normalize_domain("www.reuters.com") == "reuters.com"

    def test_bare_domain(self):
        assert _normalize_domain("apnews.com") == "apnews.com"

    def test_strips_path(self):
        assert _normalize_domain("cnn.com/politics/story") == "cnn.com"


class TestMergeBias:
    def test_both_agree(self):
        lean, conf, notes = _merge_bias("center", "center")
        assert lean == "center"
        assert conf == 1.0
        assert notes == []

    def test_allsides_only(self):
        lean, conf, notes = _merge_bias("right", None)
        assert lean == "right"
        assert conf == 0.75

    def test_mbfc_only(self):
        lean, conf, notes = _merge_bias(None, "left")
        assert lean == "left"
        assert conf == 0.75

    def test_off_by_one(self):
        lean, conf, notes = _merge_bias("center", "center-left")
        assert lean == "center"   # AllSides wins
        assert conf == 0.67
        assert any("minor" in n for n in notes)

    def test_significant_disagreement(self):
        lean, conf, notes = _merge_bias("left", "right")
        assert conf == 0.33
        assert any("significant" in n for n in notes)


# ---------------------------------------------------------------------------
# Integration-style tests: resolver with mocked scrapers
# ---------------------------------------------------------------------------

@pytest.fixture
def resolver_no_scrape():
    """BiasResolver with scraping disabled; uses only fallback table."""
    with patch("bias.resolver.scrape_allsides", return_value={}):
        with patch("bias.resolver.scrape_mbfc_bulk", return_value={}):
            return BiasResolver(auto_scrape=True)


class TestBiasResolverFallback:
    def test_known_domain_fallback(self, resolver_no_scrape):
        rating = resolver_no_scrape.resolve("foxnews.com")
        assert rating.bias_lean == "right"
        assert rating.factuality == "mixed"
        assert rating.confidence == 0.5

    def test_ap_news_fallback(self, resolver_no_scrape):
        rating = resolver_no_scrape.resolve("apnews.com")
        assert rating.bias_lean == "center"
        assert rating.factuality == "very-high"

    def test_unknown_domain_defaults_to_center(self, resolver_no_scrape):
        rating = resolver_no_scrape.resolve("unknownsource.com")
        assert rating.bias_lean == "center"
        assert rating.factuality in ("mostly-factual", "high", "mixed")

    def test_url_input_normalized(self, resolver_no_scrape):
        r1 = resolver_no_scrape.resolve("https://www.reuters.com/world/")
        r2 = resolver_no_scrape.resolve("reuters.com")
        assert r1.bias_lean == r2.bias_lean

    def test_cache_hit(self, resolver_no_scrape):
        # Second call should return cached object
        r1 = resolver_no_scrape.resolve("npr.org")
        r2 = resolver_no_scrape.resolve("npr.org")
        assert r1 is r2


class TestBiasResolverLiveData:
    """Tests that run with mocked live scrape data."""

    @pytest.fixture
    def resolver_with_data(self):
        allsides_mock = {
            "apnews.com": "center",
            "foxnews.com": "right",
            "npr.org": "center-left",
        }
        mbfc_mock = {
            "apnews.com": {"bias": "center", "factuality": "very-high"},
            "foxnews.com": {"bias": "right", "factuality": "mixed"},
            "npr.org": {"bias": "center-left", "factuality": "high"},
        }
        with patch("bias.resolver.scrape_allsides", return_value=allsides_mock):
            with patch("bias.resolver.scrape_mbfc_bulk", return_value=mbfc_mock):
                return BiasResolver(auto_scrape=True)

    def test_full_agreement_high_confidence(self, resolver_with_data):
        rating = resolver_with_data.resolve("apnews.com")
        assert rating.confidence == 1.0
        assert rating.allsides_bias == "center"
        assert rating.mbfc_bias == "center"
        assert rating.mbfc_factuality == "very-high"

    def test_factuality_from_mbfc(self, resolver_with_data):
        rating = resolver_with_data.resolve("foxnews.com")
        assert rating.factuality == "mixed"
