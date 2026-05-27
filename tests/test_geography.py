"""Tests for configurable geography classification."""

from domain.models import Article
from geography import GeographyClassifier, GeographyProfile


def _article(headline: str, region: str = "national") -> Article:
    return Article(
        article_id=None,
        source_name="Source",
        source_url="https://source.test",
        headline=headline,
        url="https://source.test/story",
        canonical_url="https://source.test/story",
        url_hash="hash",
        published_at=None,
        region=region,
        bias_lean="center",
    )


def test_classifies_local_before_state():
    classifier = GeographyClassifier(GeographyProfile(
        name="test",
        labels={},
        local_keywords=["sanford"],
        state_keywords=["north carolina"],
        international_keywords=[],
    ))
    assert classifier.classify_article(_article("Sanford project in North Carolina")) == "local"


def test_filters_international_when_configured():
    classifier = GeographyClassifier(GeographyProfile(
        name="test",
        labels={},
        local_keywords=[],
        state_keywords=[],
        international_keywords=["china"],
        exclude_international=True,
    ))
    assert classifier.classify_all([_article("China announces policy")]) == []
