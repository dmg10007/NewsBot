"""Tests for stable domain model helpers."""

from domain.models import canonical_url_hash, normalize_article_url


def test_normalize_article_url_strips_tracking_and_fragment():
    url = "HTTPS://Example.com/story/?utm_source=x&b=2#section"
    assert normalize_article_url(url) == "https://example.com/story?b=2"


def test_google_news_proxy_url_uses_original_url_parameter():
    url = "https://news.google.com/rss/articles/foo?url=https%3A%2F%2Fexample.com%2Fstory%3Futm_medium%3Drss"
    assert normalize_article_url(url) == "https://example.com/story"


def test_canonical_hash_uses_normalized_url():
    a = canonical_url_hash("https://example.com/story?utm_source=rss")
    b = canonical_url_hash("https://example.com/story")
    assert a == b
