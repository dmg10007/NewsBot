"""Shared ingestion pipeline entry point.

Previously, the source-tier loop (national / state / local) was copy-pasted
between scheduler.py:run_digest() and main.py:_run_test_ingest(). Any change
had to be made in two places. This module is the single authoritative
implementation that both callers import.

Usage::

    from ingestion.pipeline import ingest_all_sources
    from config.loader import get_sources

    articles = ingest_all_sources(get_sources())

Source structure
----------------
sources.yaml organises each tier into two sub-keys::

    national:
      rss:
        - {name: ..., url: ..., bias_lean: ..., ...}
      scrapers:
        - {name: ..., url: ..., scraper_class: ..., ...}

FeedFetcher only handles RSS sources. Scraper sources are collected here
and logged; full scraper execution is a planned extension.

The function handles its own FeedFetcher lifecycle (open/close). Callers
do not need to manage the fetcher directly.
"""

from __future__ import annotations

import logging
from typing import Any

from config.loader import get_settings
from ingestion.deduplicator import Deduplicator
from ingestion.fetcher import FeedFetcher, RawArticle

logger = logging.getLogger(__name__)

_TIERS = ("national", "state", "local")


def _rss_sources(tier_data: Any) -> list[dict]:
    """Extract the RSS source list from a tier entry.

    sources.yaml structures each tier as a dict with 'rss' and optionally
    'scrapers' sub-keys. FeedFetcher only processes RSS sources; scraper
    sources are handled separately.

    Accepts both the nested dict form (normal) and a bare list (fallback
    for simplified configs or tests that pass sources directly).
    """
    if isinstance(tier_data, dict):
        return tier_data.get("rss", [])
    if isinstance(tier_data, list):
        # Simplified config or test fixture: treat whole list as RSS sources.
        return tier_data
    return []


def _scraper_sources(tier_data: Any) -> list[dict]:
    """Extract the scraper source list from a tier entry."""
    if isinstance(tier_data, dict):
        return tier_data.get("scrapers", [])
    return []


def ingest_all_sources(sources: dict) -> list[RawArticle]:
    """Fetch, deduplicate, and return all raw articles across every tier.

    Args:
        sources: The full sources mapping from get_sources(). Expected shape::

            {
              "national": {"rss": [{"name": ..., "url": ..., ...}], "scrapers": [...]},
              "state":    {"rss": [...], "scrapers": [...]},
              "local":    {"rss": [...], "scrapers": [...]},
            }

    Returns:
        Flat, deduplicated list of RawArticle objects from all tiers.
    """
    settings = get_settings()
    fetcher = FeedFetcher()
    deduplicator = Deduplicator()
    all_articles: list[RawArticle] = []

    try:
        for tier in _TIERS:
            tier_data = sources.get(tier)
            if not tier_data:
                logger.debug("No sources configured for tier: %s", tier)
                continue

            rss = _rss_sources(tier_data)
            scrapers = _scraper_sources(tier_data)

            if rss:
                logger.info("Ingesting %d RSS sources from %s tier...", len(rss), tier)
                articles = fetcher.fetch_all(rss)
                logger.info("Fetched %d raw articles from %s tier", len(articles), tier)
                all_articles.extend(articles)

            if scrapers:
                # Scraper execution is not yet implemented in FeedFetcher.
                # Sources are logged so operators know they exist and are
                # intentionally skipped, not silently dropped.
                names = [s.get("name", "<unnamed>") for s in scrapers]
                logger.info(
                    "Skipping %d scraper source(s) from %s tier (not yet implemented): %s",
                    len(scrapers), tier, ", ".join(names),
                )
    finally:
        fetcher.close()

    before = len(all_articles)
    all_articles = deduplicator.deduplicate(all_articles)
    logger.info(
        "Deduplication: %d -> %d articles (%d removed)",
        before, len(all_articles), before - len(all_articles),
    )
    return all_articles
