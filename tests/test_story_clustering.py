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

    clusters = StoryClusterer(similarity_threshold=0.45).cluster(articles)
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

    clusters = StoryClusterer(similarity_threshold=0.45).cluster(articles)

    assert len(clusters) == 2


def test_cross_outlet_same_story_clusters():
    """Semantic clustering merges same-event articles with different wording.

    This is the failure mode the Jaccard/SequenceMatcher approach produced:
    "Fed raises interest rates" and "Federal Reserve hikes benchmark borrowing
    costs" share zero tokens after stop-word removal, so Jaccard returns 0.0
    and the articles become singletons. Semantic embeddings merge them correctly.
    """
    articles = [
        _article("Federal Reserve raises interest rates again", "ap", "national"),
        _article("Fed hikes benchmark borrowing costs for third time", "reuters", "national"),
        _article("Central bank increases rates amid inflation concerns", "npr", "national"),
    ]
    settings = {
        "scoring": {"weights": {"source_count": 0.3, "normalization_ceiling": 5.0}},
        "clustering": {
            "drop_singletons_below_importance": 0.0,
            "singleton_digest_floor": 0.0,
        },
        "delivery": {"email": {"max_stories_per_category": 7}},
    }

    clusters = StoryClusterer(similarity_threshold=0.45).cluster(articles)
    score_clusters(clusters, settings)

    multi_source = [c for c in clusters if not c.is_single_source]
    assert len(multi_source) >= 1, (
        "Expected at least one multi-source cluster from three semantically "
        "similar articles about Fed rate hikes. If this fails, the clusterer "
        "is using token matching instead of semantic embeddings."
    )
    # The multi-source cluster should contain articles from distinct publishers
    best = max(multi_source, key=lambda c: c.source_count)
    assert best.source_count >= 2
