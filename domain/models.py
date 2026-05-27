"""Project-wide domain models.

These dataclasses are the boundary between pipeline stages. Delivery,
storage, LLM clients, and orchestration should exchange these contracts
rather than stage-specific ad hoc result shapes.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Optional
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

GeoTier = Literal["national", "state", "local", "international"]
SourceType = Literal["rss", "scraper"]

_TRACKING_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "fbclid",
    "gclid",
}


def normalize_article_url(url: str) -> str:
    """Return a stable canonical URL for dedupe and persistence.

    Handles Google News proxy URLs when the original article URL is exposed
    through the common `url` query parameter, lowercases scheme/host, removes
    fragments, and strips common tracking query parameters.
    """
    raw = (url or "").strip()
    if not raw:
        return ""

    parsed = urlparse(raw)
    if parsed.netloc.lower().endswith("news.google.com"):
        qs = parse_qs(parsed.query)
        proxied = qs.get("url") or qs.get("u")
        if proxied and proxied[0]:
            raw = proxied[0]
            parsed = urlparse(raw)

    query = parse_qs(parsed.query, keep_blank_values=False)
    filtered_query = {
        key: values
        for key, values in query.items()
        if key.lower() not in _TRACKING_PARAMS
    }
    encoded_query = urlencode(filtered_query, doseq=True)
    path = re.sub(r"/+$", "", parsed.path or "/")

    return urlunparse((
        parsed.scheme.lower() or "https",
        parsed.netloc.lower(),
        path,
        "",
        encoded_query,
        "",
    ))


def canonical_url_hash(url: str) -> str:
    """Return a SHA-256 hash of the normalized article URL."""
    return hashlib.sha256(normalize_article_url(url).encode()).hexdigest()


def _extract_domain(url: str) -> str:
    """Return the bare hostname of *url*, stripped of 'www.' prefix.

    Used as a fallback publisher identity when Source.publisher is not set.
    For example, 'https://apnews.com/hub/politics' -> 'apnews.com'.
    """
    host = urlparse(url).netloc.lower()
    return host.removeprefix("www.")


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    source_type: SourceType
    tier: GeoTier
    bias_lean: str = "unknown"
    credibility: str = "medium"
    topics: list[str] = field(default_factory=list)
    region: str = "national"
    scraper_class: Optional[str] = None
    rss_url: Optional[str] = None
    selectors: dict = field(default_factory=dict)
    publisher: str = ""  # Outlet identity (e.g. "ap", "reuters"). Distinct from
                         # feed name so multiple category feeds from the same
                         # outlet are counted as one source during clustering.


@dataclass
class ArticleDraft:
    source: Source
    headline: str
    url: str
    summary: str = ""
    published_at: Optional[datetime] = None
    tags: list[str] = field(default_factory=list)
    body_text: str = ""

    @property
    def canonical_url(self) -> str:
        return normalize_article_url(self.url)

    @property
    def url_hash(self) -> str:
        return canonical_url_hash(self.url)


@dataclass
class Article:
    article_id: Optional[int]
    source_name: str
    source_url: str
    headline: str
    url: str
    canonical_url: str
    url_hash: str
    published_at: Optional[datetime]
    summary: str = ""
    body_text: str = ""
    region: str = "national"
    geo_tier: GeoTier = "national"
    geo_profile: str = "default"
    bias_lean: str = "unknown"
    credibility: str = "medium"
    topics: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    fetch_status: str = "ok"
    bias_metadata: Optional[object] = None
    publisher_name: str = ""  # Resolved outlet identity used for source_count
                              # deduplication. Set from Source.publisher, or
                              # falls back to the domain of source_url.

    @classmethod
    def from_draft(
        cls,
        draft: ArticleDraft,
        *,
        geo_tier: GeoTier | None = None,
        geo_profile: str = "default",
    ) -> "Article":
        source = draft.source
        tier = geo_tier or source.tier
        return cls(
            article_id=None,
            source_name=source.name,
            source_url=source.url,
            headline=draft.headline,
            url=draft.url,
            canonical_url=draft.canonical_url,
            url_hash=draft.url_hash,
            published_at=draft.published_at,
            summary=draft.summary,
            body_text=draft.body_text,
            region=source.region,
            geo_tier=tier,
            geo_profile=geo_profile,
            bias_lean=source.bias_lean,
            credibility=source.credibility,
            topics=list(source.topics),
            tags=list(draft.tags),
            publisher_name=source.publisher or _extract_domain(source.url),
        )


@dataclass(frozen=True)
class SourceLink:
    source_name: str
    url: str
    bias_lean: str = "unknown"
    credibility: str = "medium"


@dataclass
class StoryCluster:
    cluster_id: Optional[int]
    articles: list[Article]
    topic: str
    geo_tier: GeoTier
    representative_headline: str
    importance_score: float = 0.0

    @property
    def source_count(self) -> int:
        # Deduplicate by publisher_name (outlet identity) rather than
        # source_name (feed name) so that "AP Top News" and "AP Politics"
        # are counted as a single source, not two.
        return len({a.publisher_name or a.source_name for a in self.articles})

    @property
    def is_single_source(self) -> bool:
        return self.source_count <= 1


@dataclass
class ReportingComparison:
    cluster_id: Optional[int]
    shared_facts: list[str] = field(default_factory=list)
    source_specific_claims: list[str] = field(default_factory=list)
    omissions: list[str] = field(default_factory=list)
    framing_differences: list[str] = field(default_factory=list)
    bias_notes: str = ""
    provider_used: str = "heuristic"
    confidence: float = 0.0
    fallback_used: bool = False


@dataclass
class DigestStory:
    story_id: Optional[int]
    headline: str
    summary: str
    geo_tier: GeoTier
    topic: str
    importance_score: float
    source_links: list[SourceLink]
    comparison: ReportingComparison
    source_count: int
    is_single_source: bool
    summary_provider: str = "extractive"
    fallback_used: bool = False
    article_ids: list[int] = field(default_factory=list)


@dataclass
class DigestRun:
    run_id: Optional[int]
    period: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    delivered_story_ids: list[int] = field(default_factory=list)
    skipped_story_ids: list[int] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
