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

FeedFetcher handles RSS sources one at a time via fetch(source).
Scraper sources are executed via SCRAPER_REGISTRY. Both paths feed
into the same deduplicator.

Lifecycle
---------
FeedFetcher and Deduplicator are both closed in the finally block.
"""

from __future__ import annotations

import logging
from typing import Any

from config.loader import get_settings
from ingestion.deduplicator import Deduplicator
from ingestion.fetcher import FeedFetcher, RawArticle
from ingestion.scraper import SCRAPER_REGISTRY

logger = logging.getLogger(__name__)

_TIERS = ("national", "state", "local")


def _rss_sources(tier_data: Any) -> list[dict]:
    """Extract the RSS source list from a tier entry.

    Accepts both the nested dict form (normal) and a bare list (fallback
    for simplified configs or tests that pass sources directly).
    """
    if isinstance(tier_data, dict):
        return tier_data.get("rss", [])
    if isinstance(tier_data, list):
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
                logger.info(
                    "Ingesting %d RSS sources from %s tier...", len(rss), tier
                )
                tier_articles: list[RawArticle] = []
                for source in rss:
                    fetched = fetcher.fetch(source)
                    logger.debug(
                        "  %s: %d articles", source.get("name", source.get("url")), len(fetched)
                    )
                    tier_articles.extend(fetched)
                logger.info(
                    "Fetched %d raw articles from %s tier", len(tier_articles), tier
                )
                all_articles.extend(tier_articles)

            for scraper_cfg in scrapers:
                scraper_class_name = scraper_cfg.get("scraper_class", "")
                scraper_cls = SCRAPER_REGISTRY.get(scraper_class_name)
                if not scraper_cls:
                    logger.warning(
                        "Unknown scraper_class '%s' for source '%s' — skipping",
                        scraper_class_name, scraper_cfg.get("name", "<unnamed>"),
                    )
                    continue
                scraper = scraper_cls(scraper_cfg)
                try:
                    scraped = scraper.scrape()
                    logger.info(
                        "Scraped %d articles from %s (%s tier)",
                        len(scraped), scraper_cfg.get("name", scraper_class_name), tier,
                    )
                    all_articles.extend(scraped)
                except Exception as exc:
                    logger.error(
                        "Scraper %s failed for source '%s': %s",
                        scraper_class_name, scraper_cfg.get("name", "<unnamed>"), exc,
                    )
                finally:
                    scraper.close()
    finally:
        fetcher.close()
        deduplicator.close()

    before = len(all_articles)
    all_articles = deduplicator.deduplicate(all_articles)
    logger.info(
        "Deduplication: %d -> %d articles (%d removed)",
        before, len(all_articles), before - len(all_articles),
    )
    return all_articles
