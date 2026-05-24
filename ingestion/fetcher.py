"""RSS and API feed fetcher with retry logic and rate limiting.

Caching note
------------
The instance-level _cache dict only persists for the lifetime of the
FeedFetcher instance. Because the scheduler creates a new instance on
every run, cache_ttl_seconds in settings.yaml has no effect across runs.
The cache is useful only if fetch_all() is called multiple times on the
same instance within a single run (e.g. during testing). A persistent
cross-run cache (file-based shelve or Redis) is a future improvement.

Timeout note
------------
feedparser.parse(url) uses urllib internally with no socket timeout and
will hang indefinitely on a slow or unresponsive host. All RSS content is
now pre-fetched via httpx (which respects the configured timeout) and the
raw text is passed to feedparser.parse() instead of a URL.

Rate limiting
-------------
A configurable inter-request delay (request_delay_seconds in settings.yaml)
is applied between source fetches to avoid hammering servers and reducing
the risk of IP blocks. Always verify that your usage complies with each
source's robots.txt and Terms of Service.

URL hash
--------
URL hashes use the full 64-char SHA-256 hex digest (256 bits). The previous
16-char truncation created a non-trivial birthday collision probability at
scale and is unsafe for deduplication.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import feedparser
import httpx

from config.loader import get_settings

logger = logging.getLogger(__name__)


@dataclass
class BiasMetadata:
    """Bias and factuality metadata for a source, resolved at ingest time."""
    bias_lean: str          # left | center-left | center | center-right | right
    factuality: str         # very-low | low | mixed | mostly-factual | high | very-high
    confidence: float       # 0.0-1.0 agreement score between AllSides + MBFC
    allsides_bias: Optional[str] = None
    mbfc_bias: Optional[str] = None
    mbfc_factuality: Optional[str] = None
    notes: list[str] = field(default_factory=list)


@dataclass
class RawArticle:
    """Normalized raw article from any source before NLP processing."""
    url: str
    headline: str
    summary: str
    source_name: str
    bias_lean: str          # from sources.yaml (fast path)
    credibility: str
    topics: list[str]
    region: str             # national | north_carolina | lee_county_nc
    published_at: Optional[datetime]
    bias_metadata: Optional[BiasMetadata] = None  # enriched by BiasResolver
    url_hash: str = field(init=False)

    def __post_init__(self) -> None:
        # Full 256-bit SHA-256 digest. The previous 16-char truncation (64 bits)
        # created a non-trivial birthday collision risk at scale.
        self.url_hash = hashlib.sha256(self.url.encode()).hexdigest()


class FeedFetcher:
    """Fetches and parses RSS feeds defined in sources.yaml.

    Uses httpx for all outbound HTTP so that timeouts are enforced uniformly.
    feedparser.parse() is called with the raw response text, never a URL,
    to avoid urllib's lack of a socket timeout.
    """

    def __init__(self, bias_resolver=None) -> None:
        """
        Args:
            bias_resolver: Optional BiasResolver instance. If provided, every
                           RawArticle will have its bias_metadata field populated
                           from AllSides + MBFC data. If None, bias_metadata is
                           left as None and bias_lean from sources.yaml is used.
        """
        self.settings = get_settings()
        self._cache: dict[str, tuple[float, list[RawArticle]]] = {}
        self._bias_resolver = bias_resolver
        timeout = self.settings["ingestion"]["request_timeout_seconds"]
        user_agent = self.settings["ingestion"]["user_agent"]
        self._client = httpx.Client(
            timeout=timeout,
            headers={"User-Agent": user_agent},
            follow_redirects=True,
        )
        self._request_delay: float = self.settings["ingestion"].get(
            "request_delay_seconds", 1.0
        )

    def fetch_all(self, sources: list[dict]) -> list[RawArticle]:
        """Fetch all RSS sources, return flat list of RawArticles."""
        articles: list[RawArticle] = []
        for i, source in enumerate(sources):
            try:
                articles.extend(self._fetch_source(source))
            except Exception as exc:
                logger.error("Failed to fetch %s: %s", source["name"], exc)
            # Apply inter-request delay after every source except the last
            if i < len(sources) - 1 and self._request_delay > 0:
                time.sleep(self._request_delay)
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
                # Pre-fetch via httpx so the configured timeout is enforced.
                # feedparser.parse(url) uses urllib with no socket timeout.
                response = self._client.get(source["url"])
                response.raise_for_status()
                feed = feedparser.parse(response.text)
                articles = []
                for entry in feed.entries[:max_per_source]:
                    article = self._entry_to_article(entry, source)
                    if article:
                        articles.append(article)
                logger.info(
                    "Fetched %d articles from %s", len(articles), source["name"]
                )
                return articles
            except httpx.TimeoutException:
                logger.warning(
                    "Timeout on attempt %d/%d for %s",
                    attempt, max_retries, source["name"],
                )
            except Exception as exc:
                logger.warning(
                    "Attempt %d/%d failed for %s: %s",
                    attempt, max_retries, source["name"], exc
                )
            if attempt < max_retries:
                time.sleep(backoff * attempt)

        logger.error("All retries exhausted for %s", source["name"])
        return []

    def _entry_to_article(
        self, entry: feedparser.FeedParserDict, source: dict
    ) -> Optional[RawArticle]:
        url = getattr(entry, "link", "").strip()
        headline = getattr(entry, "title", "").strip()
        if not url or not headline:
            return None

        summary = getattr(entry, "summary", "").strip()
        published_at = None
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            published_at = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)

        bias_metadata: Optional[BiasMetadata] = None
        if self._bias_resolver is not None:
            try:
                from urllib.parse import urlparse
                domain = urlparse(url).netloc
                rating = self._bias_resolver.resolve(
                    domain,
                    credibility=source.get("credibility", "medium"),
                )
                bias_metadata = BiasMetadata(
                    bias_lean=rating.bias_lean,
                    factuality=rating.factuality,
                    confidence=rating.confidence,
                    allsides_bias=rating.allsides_bias,
                    mbfc_bias=rating.mbfc_bias,
                    mbfc_factuality=rating.mbfc_factuality,
                    notes=rating.notes,
                )
            except Exception as exc:
                logger.debug("Bias resolution failed for %s: %s", url, exc)

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
            bias_metadata=bias_metadata,
        )

    def close(self) -> None:
        """Close the underlying httpx client."""
        self._client.close()
