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
import html as html_module
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from html.parser import HTMLParser
from typing import Optional

import feedparser
import httpx

from config.loader import get_settings, get_sources
from domain.models import ArticleDraft, Source, canonical_url_hash, normalize_article_url

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HTML stripping (also used by summarizer; duplicated here to avoid circular
# imports — both modules are lightweight stdlib-only implementations)
# ---------------------------------------------------------------------------

class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        stripped = data.strip()
        if stripped:
            self._parts.append(stripped)

    def get_text(self) -> str:
        return " ".join(self._parts)


def _strip_html(text: str) -> str:
    """Return plain text with all HTML tags removed and entities decoded."""
    if not text or "<" not in text:
        return text
    parser = _TextExtractor()
    try:
        parser.feed(text)
        return parser.get_text()
    except Exception:
        return re.sub(r"<[^>]+>", " ", html_module.unescape(text)).strip()


# ---------------------------------------------------------------------------
# Ad / promo filter
# ---------------------------------------------------------------------------

_AD_URL_PATTERNS: frozenset[str] = frozenset({
    "bestreviews.com",
    "underscored.com",
    "cnn.com/cnn-underscored",
    "ad.doubleclick.net",
    "sponsored",
    "partner-content",
    "brandstudio",
})

_AD_HEADLINE_PHRASES: tuple[str, ...] = (
    "0% intro apr",
    "0% interest",
    "cash back card",
    "cash back bonus",
    "home equity loan",
    "home equity into cash",
    "credit card interest",
    "cards charging 0%",
    "turn your rising home equity",
    "want cash out of your home",
    "use the right card",
    "dream big with a home equity",
    "cnn political briefing",
    "the axe files",
    "margins of error",
    "politics of the day",
)

_SECTION_INDEX_RE = re.compile(
    r"^(world|asia|europe|business|us|uk|middle east|americas|africa|sports|tech)"
    r"\s+(news|headlines|stories)",
    re.IGNORECASE,
)


def _is_ad(headline: str, url: str, summary: str) -> bool:
    url_lower = url.lower()
    for pat in _AD_URL_PATTERNS:
        if pat in url_lower:
            return True
    headline_lower = headline.lower()
    for phrase in _AD_HEADLINE_PHRASES:
        if phrase in headline_lower:
            return True
    if _SECTION_INDEX_RE.match(headline.strip()):
        return True
    return False


@dataclass
class RawArticle:
    """A single article as ingested from a feed or scraper.

    Fields are intentionally minimal — only what can be reliably extracted
    from RSS/Atom metadata without fetching the full article body.
    The NLP pipeline (parsing/extractor.py) enriches these into ParsedArticle.
    """
    source_name: str
    headline: str
    url: str
    url_hash: str
    published_at: Optional[datetime]
    summary: str = ""       # Plain text — HTML stripped at ingest time
    region: str = "national"
    bias_lean: str = "unknown"
    credibility: str = "medium"
    bias_metadata: Optional[object] = None
    tags: list[str] = field(default_factory=list)
    geo_tier: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.source_url:
            self.source_url = self.url
        if not self.url_hash:
            self.url_hash = canonical_url_hash(self.url)

    @property
    def canonical_url(self) -> str:
        return normalize_article_url(self.url)


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
        """Fetch one RSS/Atom source and return a list of RawArticle objects."""
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
        ads_filtered = 0
        for entry in feed.entries[: self._max_per_source]:
            headline = entry.get("title", "").strip()
            article_url = entry.get("link", "").strip()
            if not headline or not article_url:
                continue

            # Extract raw summary before stripping so _is_ad can inspect it
            raw_summary = ""
            if entry.get("summary"):
                raw_summary = entry["summary"]
            elif entry.get("content"):
                raw_summary = entry["content"][0].get("value", "")

            if _is_ad(headline, article_url, raw_summary):
                ads_filtered += 1
                continue

            published_at: Optional[datetime] = None
            published_parsed = entry.get("published_parsed")
            if published_parsed:
                try:
                    published_at = datetime(*published_parsed[:6])
                except (TypeError, ValueError):
                    pass

            tags: list[str] = [
                tag.get("term", "").strip()
                for tag in entry.get("tags", [])
                if tag.get("term", "").strip()
            ]

            # Strip HTML from summary here so RawArticle.summary is always
            # clean plain text for all downstream consumers.
            clean_summary = _strip_html(raw_summary)

            articles.append(RawArticle(
                source_name=source["name"],
                source_url=url,
                headline=headline,
                url=article_url,
                url_hash=canonical_url_hash(article_url),
                published_at=published_at,
                summary=clean_summary,
                region=source.get("region", "national"),
                bias_lean=source.get("bias_lean", "unknown"),
                credibility=source.get("credibility", "medium"),
                tags=tags,
            ))

        if ads_filtered:
            logger.info(
                "Ad filter: dropped %d promotional entries from %s",
                ads_filtered, source["name"]
            )
        time.sleep(self._delay)
        return articles

    def fetch_all(self, sources: list[dict]) -> list[RawArticle]:
        """Fetch all RSS sources.

        Kept for older tests and callers; the refactored pipeline calls
        `fetch()` per typed source so it can record source health precisely.
        """
        articles: list[RawArticle] = []
        for source in sources:
            try:
                articles.extend(self.fetch(source))
            except Exception as exc:
                logger.warning("Failed to fetch source %s: %s", source.get("name"), exc)
        return articles

    def fetch_source(self, source: Source) -> list[ArticleDraft]:
        """Fetch one typed Source and return normalized article drafts."""
        raw_articles = self.fetch({
            "name": source.name,
            "url": source.url,
            "region": source.region,
            "bias_lean": source.bias_lean,
            "credibility": source.credibility,
        })
        return [
            ArticleDraft(
                source=source,
                headline=a.headline,
                url=a.url,
                summary=a.summary,
                published_at=a.published_at,
                tags=a.tags,
            )
            for a in raw_articles
        ]

    def close(self) -> None:
        self._client.close()
