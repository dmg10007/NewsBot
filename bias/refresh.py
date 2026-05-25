"""Standalone script to refresh bias lookup data on a weekly cron schedule.

Usage (manual):  python -m bias.refresh
Usage (cron):    0 4 * * 1 /path/to/venv/bin/python -m bias.refresh

What it does:
  1. Scrapes AllSides media bias ratings page via Playwright
  2. Scrapes MBFC detail pages for all known domains via Playwright
  3. Both sources share a single Chromium browser session (one launch)
  4. Writes the merged result to config/bias_ratings_cache.json
  5. The BiasResolver loads from this cache on startup (if present)
     instead of re-scraping every time the bot runs.

Estimated runtime: ~60–90s for 17 domains at 1.5s inter-page delay.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from bias.source_ratings import (
    _BROWSER_HEADERS,
    scrape_all,
)
from bias.resolver import _FALLBACK_RATINGS, _merge_bias, _normalize_domain

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("bias.refresh")

CACHE_PATH = Path("config/bias_ratings_cache.json")


def build_cache() -> dict:
    """Scrape all sources and return a serializable ratings dict."""
    # httpx client retained for API compatibility — scrape_all() uses Playwright.
    client = httpx.Client(timeout=20, headers=_BROWSER_HEADERS, follow_redirects=True)

    known_domains = list(_FALLBACK_RATINGS.keys())

    try:
        # Single Playwright browser session for both AllSides and MBFC.
        allsides, mbfc = scrape_all(client, known_domains, delay=1.5)
    finally:
        client.close()

    merged: dict[str, dict] = {}
    all_domains = set(list(allsides.keys()) + list(mbfc.keys()) + list(_FALLBACK_RATINGS.keys()))
    for domain in all_domains:
        domain = _normalize_domain(domain)
        as_bias = allsides.get(domain)
        mb_data = mbfc.get(domain, {})
        mb_bias = mb_data.get("bias")
        mb_factuality = mb_data.get("factuality")

        fallback = _FALLBACK_RATINGS.get(domain, {})
        if not as_bias and not mb_bias:
            bias_lean = fallback.get("bias_lean", "center")
            confidence = 0.5
            notes = ["fallback only"]
        else:
            bias_lean, confidence, notes = _merge_bias(as_bias, mb_bias)

        factuality = mb_factuality or fallback.get("factuality", "mostly-factual")

        merged[domain] = {
            "bias_lean": bias_lean,
            "factuality": factuality,
            "confidence": confidence,
            "allsides_bias": as_bias,
            "mbfc_bias": mb_bias,
            "mbfc_factuality": mb_factuality,
            "notes": notes,
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_count": len(merged),
        "ratings": merged,
    }


def main() -> None:
    start = time.time()
    logger.info("Starting bias ratings refresh...")

    cache = build_cache()
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2))

    elapsed = time.time() - start
    logger.info(
        "Done. %d sources cached to %s in %.1fs",
        cache["source_count"], CACHE_PATH, elapsed,
    )


if __name__ == "__main__":
    main()
