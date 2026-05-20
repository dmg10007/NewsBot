"""RSS and API feed fetcher with caching, retry logic, and rate limiting."""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import feedparser
import httpx
import yaml

from config.loader import get_settings

logger = logging.getLogger(__name__)


@dataclass
class RawArticle:
    """Normalized raw article from any source before NLP processing."""
    url: str
    headline: str
    summary: str
    source_name: str
    bias_lean: str
    credibility: str
    topics: list[str]
    region: str  # national | north_carolina | lee_county_nc
    published_at: Optional[datetime]
    url_hash: str = field(init=False)

    def __post_init__(self) -> None:
        self.url_hash = hashlib.sha256(self.url.encode()).hexdigest()[:16]


class FeedFetcher:
    """Fetches and parses RSS feeds defined in sources.yaml."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._cache: dict[str, tuple[float, list[RawArticle]]] = {}
        self._client = httpx.Client(
            timeout=self.settings["ingestion"]["request_timeout_seconds"],
            headers={"User-Agent": self.settings["ingestion"]["user_agent"]},
            follow_redirects=True,
        )

    def fetch_all(self, sources: list[dict]) -> list[RawArticle]:
        """Fetch all RSS sources, return flat list of RawArticles."""
        articles: list[RawArticle] = []
        for source in sources:
            try:
                articles.extend(self._fetch_source(source))
            except Exception as exc:
                logger.error("Failed to fetch %s: %s", source["name"], exc)
        return articles

    def _fetch_source(self, source: dict) -> list[RawArticle]:
        url = source["url"]
        cache_ttl = self.settings["ingestion"]["cache_ttl_seconds"]
        now = time.time()

        if url in self._cache:
            cached_at, cached_articles = self._cache[url]
            if now - cached_at < cache_ttl:
                logger.debug("Cache hit for %s", source["name"])
                return cached_articles

        articles = self._parse_feed_with_retry(source)
        self._cache[url] = (now, articles)
        return articles

    def _parse_feed_with_retry(self, source: dict) -> list[RawArticle]:
        max_retries = self.settings["ingestion"]["max_retries"]
        backoff = self.settings["ingestion"]["retry_backoff_seconds"]
        max_per_source = self.settings["ingestion"]["max_articles_per_source"]

        for attempt in range(1, max_retries + 1):
            try:
                feed = feedparser.parse(source["url"])
                articles = []
                for entry in feed.entries[:max_per_source]:
                    article = self._entry_to_article(entry, source)
                    if article:
                        articles.append(article)
                logger.info(
                    "Fetched %d articles from %s", len(articles), source["name"]
                )
                return articles
            except Exception as exc:
                logger.warning(
                    "Attempt %d/%d failed for %s: %s",
                    attempt, max_retries, source["name"], exc
                )
                if attempt < max_retries:
                    time.sleep(backoff * attempt)

        logger.error("All retries exhausted for %s", source["name"])
        return []

    def _entry_to_article(self, entry: feedparser.FeedParserDict, source: dict) -> Optional[RawArticle]:
        url = getattr(entry, "link", "").strip()
        headline = getattr(entry, "title", "").strip()
        if not url or not headline:
            return None

        summary = getattr(entry, "summary", "").strip()
        published_at = None
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            published_at = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)

        return RawArticle(
            url=url,
            headline=headline,
            summary=summary,
            source_name=source["name"],
            bias_lean=source.get("bias_lean", "unknown"),
            credibility=source.get("credibility", "medium"),
            topics=source.get("topics", []),
            region=source.get("region", "national"),
            published_at=published_at,
        )

    def close(self) -> None:
        self._client.close()
