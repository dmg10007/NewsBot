"""Tests for ingestion.deduplicator."""

from __future__ import annotations

import hashlib

from ingestion.fetcher import RawArticle
from ingestion.deduplicator import Deduplicator


def _make_article(
    url: str,
    headline: str,
    source: str = "Source A",
) -> RawArticle:
    """Build a minimal valid RawArticle for deduplication tests.

    All required fields are provided. Only url, headline, and source
    are varied between test cases — everything else is fixed.
    """
    return RawArticle(
        url=url,
        headline=headline,
        summary="Test summary.",
        source_name=source,
        source_url="https://example.com/feed.xml",
        url_hash=hashlib.sha256(url.encode()).hexdigest(),
        bias_lean="center",
        credibility="high",
        tags=[],
        region="national",
        published_at=None,
    )


def test_dedup_removes_identical_urls():
    articles = [
        _make_article("https://example.com/story-1", "Headline One"),
        _make_article("https://example.com/story-1", "Headline One Again"),  # Same URL
        _make_article("https://example.com/story-2", "Headline Two"),
    ]
    deduped = Deduplicator()._dedup_by_url(articles)
    assert len(deduped) == 2


def test_dedup_same_url_different_sources_collapses():
    """Same URL from two different sources is still a true duplicate.

    URL-hash dedup runs first and removes the second entry regardless of
    source name. This is correct: the same URL is never independent
    corroboration — it's the same content served from the same server.
    Clustering (not dedup) is where multi-source corroboration is measured.
    """
    a1 = _make_article("https://example.com/story-1", "Headline", source="Source A")
    a2 = _make_article("https://example.com/story-1", "Headline", source="Source B")
    deduped = Deduplicator()._dedup_by_url([a1, a2])
    assert len(deduped) == 1


def test_dedup_empty_list():
    assert Deduplicator().deduplicate([]) == []


def test_dedup_single_article():
    articles = [_make_article("https://example.com/only", "Only Article")]
    deduped = Deduplicator().deduplicate(articles)
    assert len(deduped) == 1


def test_dedup_preserves_first_of_duplicate_pair():
    """When two articles share a URL, the first one is always kept."""
    a1 = _make_article("https://example.com/story", "First Version")
    a2 = _make_article("https://example.com/story", "Second Version")
    deduped = Deduplicator()._dedup_by_url([a1, a2])
    assert len(deduped) == 1
    assert deduped[0].headline == "First Version"


def test_dedup_all_unique_urls_passes_through():
    articles = [
        _make_article(f"https://example.com/story-{i}", f"Headline {i}")
        for i in range(5)
    ]
    deduped = Deduplicator()._dedup_by_url(articles)
    assert len(deduped) == 5
