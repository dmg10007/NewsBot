"""Shared ingestion pipeline entry point.

Previously, the source-tier loop (national / state / local) was copy-pasted
between scheduler.py:run_digest() and main.py:_run_test_ingest(). Any change
had to be made in two places. This module is the single authoritative
implementation that both callers import.

Usage::

    from ingestion.pipeline import ingest_all_sources
    from config.loader import get_sources

    articles = ingest_all_sources(get_sources())

The function handles its own FeedFetcher lifecycle (open/close). Callers
do not need to manage the fetcher directly.
"""

from __future__ import annotations

import logging

from config.loader import get_settings
from ingestion.deduplicator import Deduplicator
from ingestion.fetcher import FeedFetcher, RawArticle

logger = logging.getLogger(__name__)

_TIERS = ("national", "state", "local")


def ingest_all_sources(sources: dict) -> list[RawArticle]:
    """Fetch, deduplicate, and return all raw articles across every tier.

    Args:
        sources: The full sources mapping from get_sources(). Expected shape::

            {
              "national": [{"name": ..., "url": ..., ...}, ...],
              "state":    [...],
              "local":    [...],
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
            tier_sources = sources.get(tier, [])
            if not tier_sources:
                logger.debug("No sources configured for tier: %s", tier)
                continue
            logger.info("Ingesting %d %s sources...", len(tier_sources), tier)
            articles = fetcher.fetch_all(tier_sources)
            logger.info("Fetched %d raw articles from %s tier", len(articles), tier)
            all_articles.extend(articles)
    finally:
        fetcher.close()

    before = len(all_articles)
    all_articles = deduplicator.deduplicate(all_articles)
    logger.info(
        "Deduplication: %d -> %d articles (%d removed)",
        before, len(all_articles), before - len(all_articles),
    )
    return all_articles
