"""Tests for ingestion.fetcher — uses mocked HTTP responses."""

from __future__ import annotations

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
    "topics": ["politics"],
    "region": "national",
}


@patch("feedparser.parse")
def test_fetch_all_returns_articles(mock_parse):
    mock_entry_1 = MagicMock()
    mock_entry_1.title = "Test Headline One"
    mock_entry_1.link = "https://example.com/story-1"
    mock_entry_1.summary = "A summary of story one."
    mock_entry_1.published_parsed = (2026, 5, 21, 10, 0, 0, 0, 0, 0)

    mock_entry_2 = MagicMock()
    mock_entry_2.title = "Test Headline Two"
    mock_entry_2.link = "https://example.com/story-2"
    mock_entry_2.summary = "A summary of story two."
    mock_entry_2.published_parsed = (2026, 5, 21, 11, 0, 0, 0, 0, 0)

    mock_feed = MagicMock()
    mock_feed.entries = [mock_entry_1, mock_entry_2]
    mock_parse.return_value = mock_feed

    fetcher = FeedFetcher()
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
    article = RawArticle(
        url="https://example.com/story",
        headline="Headline",
        summary="Summary",
        source_name="Source",
        bias_lean="center",
        credibility="high",
        topics=["politics"],
        region="national",
        published_at=None,
    )
    assert len(article.url_hash) == 16
    # Same URL always produces same hash
    article2 = RawArticle(
        url="https://example.com/story",
        headline="Different Headline",
        summary="",
        source_name="Other Source",
        bias_lean="right",
        credibility="medium",
        topics=[],
        region="national",
        published_at=None,
    )
    assert article.url_hash == article2.url_hash
