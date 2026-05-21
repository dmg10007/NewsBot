"""NewsBot main entry point.

Usage:
  python main.py                  # Run a single digest immediately (defaults to 'morning')
  python main.py --period evening  # Run evening digest immediately
  python main.py --schedule        # Start the scheduler (6 AM / 6 PM ET, blocking)
  python main.py --test-ingest     # Ingest + dedup only, print article count (no delivery)
"""

from __future__ import annotations

import argparse
import logging
from dotenv import load_dotenv

load_dotenv()


def configure_logging() -> None:
    from config.loader import get_settings
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings["app"]["log_level"]),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="NewsBot — bias-aware news digest")
    parser.add_argument(
        "--period",
        choices=["morning", "evening"],
        default="morning",
        help="Digest period to run (default: morning)",
    )
    parser.add_argument(
        "--schedule",
        action="store_true",
        help="Start the blocking scheduler (runs forever at 6 AM/6 PM ET)",
    )
    parser.add_argument(
        "--test-ingest",
        action="store_true",
        help="Run ingestion and deduplication only — no delivery",
    )
    args = parser.parse_args()

    if args.schedule:
        from scheduler.scheduler import main as run_scheduler
        run_scheduler()
        return

    if args.test_ingest:
        _run_test_ingest()
        return

    from scheduler.scheduler import run_digest
    run_digest(args.period)


def _run_test_ingest() -> None:
    """Ingest + dedup only. Useful for validating sources without triggering delivery."""
    logger = logging.getLogger(__name__)
    from config.loader import get_sources
    from ingestion.fetcher import FeedFetcher
    from ingestion.scraper import SCRAPER_REGISTRY
    from ingestion.deduplicator import Deduplicator

    sources = get_sources()
    fetcher = FeedFetcher()
    raw_articles = []

    for tier_name in ("national", "state"):
        tier = sources.get(tier_name, {})
        rss_sources = tier.get("rss", [])
        if rss_sources:
            raw_articles.extend(fetcher.fetch_all(rss_sources))
        for scraper_source in tier.get("scrapers", []):
            cls = SCRAPER_REGISTRY.get(scraper_source.get("scraper_class", ""))
            if cls:
                scraper = cls(scraper_source)
                try:
                    raw_articles.extend(scraper.scrape())
                finally:
                    scraper.close()

    local_tier = sources.get("local", {})
    for scraper_source in local_tier.get("scrapers", []):
        cls = SCRAPER_REGISTRY.get(scraper_source.get("scraper_class", ""))
        if cls:
            scraper = cls(scraper_source)
            try:
                raw_articles.extend(scraper.scrape())
            finally:
                scraper.close()

    fetcher.close()
    logger.info("Raw articles fetched: %d", len(raw_articles))

    deduped = Deduplicator().deduplicate(raw_articles)
    logger.info("After deduplication: %d articles", len(deduped))

    for a in deduped[:15]:
        logger.info("[%s][%s] %s — %s", a.region, a.topics, a.source_name, a.headline)


if __name__ == "__main__":
    main()
