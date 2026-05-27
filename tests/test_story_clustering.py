"""Tests for domain story clustering and digest selection."""

from __future__ import annotations

from domain.models import Article
from story_clustering import (
    StoryClusterer,
    filter_reportable_clusters,
    score_clusters,
    select_digest_clusters,
)


def _article(headline: str, source: str, geo_tier: str = "national") -> Article:
    return Article(
        article_id=None,
        source_name=source,
        source_url=f"https://{source}.test/feed",
        headline=headline,
        url=f"https://{source}.test/{headline.replace(' ', '-')}",
        canonical_url=f"https://{source}.test/{headline.replace(' ', '-')}",
        url_hash=f"{source}-{headline}",
        published_at=None,
        summary=headline,
        geo_tier=geo_tier,  # type: ignore[arg-type]
        region=geo_tier,
        bias_lean="center",
        topics=["current_events"],
    )


def test_digest_selection_keeps_bounded_singletons():
    articles = [
        _article("Congress passes budget plan", "ap"),
        _article("Senate approves budget plan", "reuters"),
        _article("Court rules on voting case", "npr"),
        _article("Storm damages local school", "local", "local"),
    ]
    settings = {
        "scoring": {"weights": {"source_count": 0.3, "normalization_ceiling": 5.0}},
        "clustering": {"drop_singletons_below_importance": 0.1, "singleton_digest_floor": 0.1},
        "delivery": {"email": {"max_stories_per_category": 7}},
    }

    clusters = StoryClusterer(similarity_threshold=0.5).cluster(articles)
    score_clusters(clusters, settings)
    reportable = filter_reportable_clusters(clusters, settings)
    selected = select_digest_clusters(reportable, settings)

    assert len(selected) >= 3
    assert any(c.is_single_source for c in selected)


def test_geo_compatibility_does_not_merge_national_with_local():
    articles = [
        _article("School board approves budget", "national", "national"),
        _article("School board approves budget", "local", "local"),
    ]

    clusters = StoryClusterer(similarity_threshold=0.5).cluster(articles)

    assert len(clusters) == 2
