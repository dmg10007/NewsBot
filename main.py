"""Main entry point for running NewsBot ingestion locally.

This currently fetches and deduplicates articles from configured sources.
Future stages will add parsing, clustering, bias detection, summarization,
and delivery.
"""

from __future__ import annotations

import logging

from config.loader import get_sources, get_settings
from ingestion.deduplicator import Deduplicator
from ingestion.fetcher import FeedFetcher, RawArticle
from ingestion.scraper import SCRAPER_REGISTRY


def configure_logging() -> None:
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings["app"]["log_level"]),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def gather_all_raw_articles() -> list[RawArticle]:
    sources = get_sources()
    fetcher = FeedFetcher()
    raw_articles: list[RawArticle] = []

    for tier_name in ("national", "state"):
        tier = sources.get(tier_name, {})
        rss_sources = tier.get("rss", [])
        if rss_sources:
            raw_articles.extend(fetcher.fetch_all(rss_sources))

        scraper_sources = tier.get("scrapers", [])
        for scraper_source in scraper_sources:
            scraper_class_name = scraper_source.get("scraper_class")
            scraper_cls = SCRAPER_REGISTRY.get(scraper_class_name)
            if scraper_cls:
                scraper = scraper_cls(scraper_source)
                try:
                    raw_articles.extend(scraper.scrape())
                finally:
                    scraper.close()

    local_tier = sources.get("local", {})
    for scraper_source in local_tier.get("scrapers", []):
        scraper_class_name = scraper_source.get("scraper_class")
        scraper_cls = SCRAPER_REGISTRY.get(scraper_class_name)
        if scraper_cls:
            scraper = scraper_cls(scraper_source)
            try:
                raw_articles.extend(scraper.scrape())
            finally:
                scraper.close()

    fetcher.close()
    return raw_articles


def main() -> None:
    configure_logging()
    logger = logging.getLogger(__name__)

    raw_articles = gather_all_raw_articles()
    logger.info("Gathered %d raw articles before deduplication", len(raw_articles))

    deduplicator = Deduplicator()
    deduped = deduplicator.deduplicate(raw_articles)
    logger.info("Final raw article count after deduplication: %d", len(deduped))

    for article in deduped[:10]:
        logger.info("[%s] %s — %s", article.region, article.source_name, article.headline)


if __name__ == "__main__":
    main()
