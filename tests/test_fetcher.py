"""Tests for ingestion.fetcher — uses mocked HTTP responses."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from ingestion.fetcher import FeedFetcher, RawArticle


SAMPLE_FEED_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Test Feed</title>
    <item>
      <title>Test Headline One</title>
      <link>https://example.com/story-1</link>
      <description>A summary of story one.</description>
      <pubDate>Wed, 21 May 2026 10:00:00 +0000</pubDate>
    </item>
    <item>
      <title>Test Headline Two</title>
      <link>https://example.com/story-2</link>
      <description>A summary of story two.</description>
      <pubDate>Wed, 21 May 2026 11:00:00 +0000</pubDate>
    </item>
  </channel>
</rss>
"""

SAMPLE_SOURCE = {
    "name": "Test Source",
    "url": "https://example.com/feed.xml",
    "bias_lean": "center",
    "credibility": "high",
    "region": "national",
}


def _make_raw_article(
    url: str = "https://example.com/story",
    headline: str = "Test Headline",
    source_name: str = "Test Source",
    bias_lean: str = "center",
) -> RawArticle:
    """Construct a valid RawArticle for use in tests.

    Centralises all required fields so individual tests only specify
    what they actually care about.
    """
    return RawArticle(
        url=url,
        headline=headline,
        summary="A test summary.",
        source_name=source_name,
        source_url="https://example.com/feed.xml",
        url_hash=hashlib.sha256(url.encode()).hexdigest(),
        bias_lean=bias_lean,
        credibility="high",
        tags=[],
        region="national",
        published_at=None,
    )


@patch("feedparser.parse")
def test_fetch_all_returns_articles(mock_parse):
    mock_entry_1 = {
        "title": "Test Headline One",
        "link": "https://example.com/story-1",
        "summary": "A summary of story one.",
        "published_parsed": (2026, 5, 21, 10, 0, 0, 0, 0, 0),
    }

    mock_entry_2 = {
        "title": "Test Headline Two",
        "link": "https://example.com/story-2",
        "summary": "A summary of story two.",
        "published_parsed": (2026, 5, 21, 11, 0, 0, 0, 0, 0),
    }

    mock_feed = MagicMock()
    mock_feed.entries = [mock_entry_1, mock_entry_2]
    mock_feed.bozo = False
    mock_parse.return_value = mock_feed

    fetcher = FeedFetcher()
    mock_response = MagicMock()
    mock_response.text = SAMPLE_FEED_XML
    mock_response.raise_for_status.return_value = None
    fetcher._client.get = MagicMock(return_value=mock_response)
    articles = fetcher.fetch_all([SAMPLE_SOURCE])

    assert len(articles) == 2
    assert articles[0].headline == "Test Headline One"
    assert articles[0].source_name == "Test Source"
    assert articles[0].bias_lean == "center"
    assert articles[0].region == "national"
    assert isinstance(articles[0].published_at, datetime)
    fetcher.close()


@patch("feedparser.parse")
def test_fetch_skips_entries_without_url(mock_parse):
    mock_entry = MagicMock()
    mock_entry.title = "No Link Article"
    mock_entry.link = ""
    mock_entry.summary = ""
    mock_entry.published_parsed = None

    mock_feed = MagicMock()
    mock_feed.entries = [mock_entry]
    mock_feed.bozo = False
    mock_parse.return_value = mock_feed

    fetcher = FeedFetcher()
    articles = fetcher.fetch_all([SAMPLE_SOURCE])
    assert len(articles) == 0
    fetcher.close()


@patch("feedparser.parse")
def test_fetch_handles_source_error_gracefully(mock_parse):
    mock_parse.side_effect = Exception("Network error")
    fetcher = FeedFetcher()
    articles = fetcher.fetch_all([SAMPLE_SOURCE])
    assert articles == []
    fetcher.close()


def test_raw_article_url_hash_is_deterministic():
    url = "https://example.com/story"
    article = _make_raw_article(url=url)

    # url_hash is the full SHA-256 hex digest: 64 hex characters
    assert len(article.url_hash) == 64
    assert article.url_hash == hashlib.sha256(url.encode()).hexdigest()

    # Same URL always produces the same hash regardless of other fields
    article2 = _make_raw_article(url=url, headline="Different Headline", source_name="Other Source")
    assert article.url_hash == article2.url_hash


def test_raw_article_different_urls_produce_different_hashes():
    a1 = _make_raw_article(url="https://example.com/story-1")
    a2 = _make_raw_article(url="https://example.com/story-2")
    assert a1.url_hash != a2.url_hash
