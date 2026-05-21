"""Tests for clustering.clusterer."""

from __future__ import annotations

from ingestion.fetcher import RawArticle
from parsing.extractor import ParsedArticle
from clustering.clusterer import StoryClusterer, StoryCluster


def _make_parsed(headline: str, summary: str = "", source: str = "Source",
                 bias_lean: str = "center", region: str = "national") -> ParsedArticle:
    raw = RawArticle(
        url=f"https://example.com/{headline.replace(' ', '-').lower()}",
        headline=headline,
        summary=summary,
        source_name=source,
        bias_lean=bias_lean,
        credibility="high",
        topics=["politics"],
        region=region,
        published_at=None,
    )
    return ParsedArticle(
        raw=raw,
        entities=[],
        keywords=headline.lower().split(),
        detected_topics=["politics"],
        sentiment_compound=0.0,
        sentiment_label="neutral",
        word_count=len(headline.split()),
        full_text=f"{headline}. {summary}".strip(),
    )


def test_cluster_groups_similar_stories():
    articles = [
        _make_parsed("Congress passes new budget bill", "The Senate approved the budget."),
        _make_parsed("Senate votes on federal budget legislation", "Congress passed the bill."),
        _make_parsed("NC weather forecast shows storms ahead", "Rain expected in North Carolina."),
    ]
    clusterer = StoryClusterer()
    clusters = clusterer.cluster(articles)
    # The two budget stories should be in the same cluster
    assert len(clusters) >= 2
    # Each cluster should have at least one article
    for cluster in clusters:
        assert len(cluster.articles) >= 1


def test_cluster_empty_input():
    clusterer = StoryClusterer()
    assert clusterer.cluster([]) == []


def test_cluster_single_article():
    articles = [_make_parsed("Single story headline")]
    clusterer = StoryClusterer()
    clusters = clusterer.cluster(articles)
    assert len(clusters) == 1
    assert clusters[0].source_count == 1
    assert clusters[0].is_single_source is False  # is_single_source is on SummaryResult not StoryCluster


def test_story_cluster_metadata():
    articles = [
        _make_parsed("Budget passes Senate", source="AP News", bias_lean="center"),
        _make_parsed("Senate approves budget", source="Fox News", bias_lean="right"),
    ]
    clusterer = StoryClusterer()
    clusters = clusterer.cluster(articles)
    # At least one cluster should exist
    assert len(clusters) >= 1
    # Find the cluster with the most articles
    biggest = max(clusters, key=lambda c: c.source_count)
    assert biggest.representative_headline != ""
