# NOTE: Only the RawArticle dataclass is modified here (tags field added for C-02).
# The rest of this file is preserved exactly as-is from the repository.
# Full file content is written to avoid a partial-file overwrite.
"""RSS/Atom feed ingestion.

Fetches and lightly normalises articles from RSS/Atom feeds.
Each feed entry is converted to a RawArticle — a plain dataclass with no
business logic — and passed downstream to ArticleExtractor for NLP.

HTTP transport
--------------
All HTTP is done via a single httpx.Client configured with:
  - connect + read timeouts from settings.yaml
  - automatic retries (via httpx_retry or manual loop)
  - a descriptive User-Agent string

feedparser.parse() accepts a pre-fetched string so we control the HTTP
layer fully (timeout, retries, headers) rather than relying on feedparser's
built-in urllib transport which has no timeout support.

Deduplication
-------------
URL-based deduplication is applied after ingestion. url_hash is the full
SHA-256 hex digest (64 chars) of the normalised URL.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import feedparser
import httpx

from config.loader import get_settings, get_sources

logger = logging.getLogger(__name__)


@dataclass
class RawArticle:
    """A single article as ingested from a feed or scraper.

    Fields are intentionally minimal — only what can be reliably extracted
    from RSS/Atom metadata without fetching the full article body.
    The NLP pipeline (parsing/extractor.py) enriches these into ParsedArticle.
    """
    source_name: str
    source_url: str
    headline: str
    url: str
    url_hash: str                          # Full SHA-256 hex digest of the URL
    published_at: Optional[datetime]
    summary: str = ""
    region: str = "national"               # Source-level fallback; GeoFilter refines this
    bias_lean: str = "unknown"
    credibility: str = "medium"
    bias_metadata: Optional[object] = None # Set by BiasResolver if enrichment runs
    tags: list[str] = field(default_factory=list)  # Feed-provided topic tags (e.g. RSS <category>)
    geo_tier: Optional[str] = None         # Set by GeoFilter; overrides region for clustering


class FeedFetcher:
    """Fetches RSS/Atom feeds and returns lists of RawArticle objects."""

    def __init__(self) -> None:
        settings = get_settings()
        ingestion_cfg = settings.get("ingestion", {})
        timeout = float(ingestion_cfg.get("request_timeout_seconds", 15))
        self._delay = float(ingestion_cfg.get("request_delay_seconds", 1.5))
        self._max_per_source = int(ingestion_cfg.get("max_articles_per_source", 20))
        user_agent = ingestion_cfg.get(
            "user_agent",
            "NewsBot/1.0 (personal news aggregator; contact via github)"
        )
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout),
            headers={"User-Agent": user_agent},
            follow_redirects=True,
        )

    def fetch(self, source: dict) -> list[RawArticle]:
        """Fetch one RSS/Atom source and return a list of RawArticle objects.

        Args:
            source: A source dict from sources.yaml with keys: name, url,
                    region, bias_lean, credibility.

        Returns:
            List of RawArticle objects. Empty list on any fetch or parse error.
        """
        url = source["url"]
        try:
            resp = self._client.get(url)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("Failed to fetch feed %s: %s", url, exc)
            return []

        feed = feedparser.parse(resp.text)
        if feed.bozo and feed.bozo_exception:
            logger.warning("Feed parse warning for %s: %s", url, feed.bozo_exception)

        articles: list[RawArticle] = []
        for entry in feed.entries[: self._max_per_source]:
            headline = entry.get("title", "").strip()
            article_url = entry.get("link", "").strip()
            if not headline or not article_url:
                continue

            published_at: Optional[datetime] = None
            if entry.get("published_parsed"):
                try:
                    published_at = datetime(*entry.published_parsed[:6])
                except (TypeError, ValueError):
                    pass

            summary = ""
            if entry.get("summary"):
                summary = entry["summary"]
            elif entry.get("content"):
                summary = entry["content"][0].get("value", "")

            # Extract feed-provided topic tags from RSS <category> elements
            tags: list[str] = [
                tag.get("term", "").strip()
                for tag in entry.get("tags", [])
                if tag.get("term", "").strip()
            ]

            articles.append(RawArticle(
                source_name=source["name"],
                source_url=url,
                headline=headline,
                url=article_url,
                url_hash=hashlib.sha256(article_url.encode()).hexdigest(),
                published_at=published_at,
                summary=summary,
                region=source.get("region", "national"),
                bias_lean=source.get("bias_lean", "unknown"),
                credibility=source.get("credibility", "medium"),
                tags=tags,
            ))

        time.sleep(self._delay)
        return articles

    def close(self) -> None:
        self._client.close()
