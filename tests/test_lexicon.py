"""Tests for bias.lexicon."""

from __future__ import annotations

from ingestion.fetcher import RawArticle
from parsing.extractor import ParsedArticle
from clustering.clusterer import StoryCluster
from bias.lexicon import LexiconAnalyzer


def _make_cluster(articles_data: list[tuple[str, str, str, float]]) -> StoryCluster:
    """articles_data: list of (headline, source_name, bias_lean, sentiment)"""
    articles = []
    for headline, source, lean, sentiment in articles_data:
        raw = RawArticle(
            url=f"https://example.com/{source.lower().replace(' ', '-')}",
            headline=headline,
            summary=headline,
            source_name=source,
            bias_lean=lean,
            credibility="medium",
            topics=["politics"],
            region="national",
            published_at=None,
        )
        parsed = ParsedArticle(
            raw=raw,
            entities=[],
            keywords=[],
            detected_topics=["politics"],
            sentiment_compound=sentiment,
            sentiment_label="neutral",
            word_count=5,
            full_text=headline,
        )
        articles.append(parsed)
    return StoryCluster(
        cluster_id=1,
        articles=articles,
        topic="politics",
        tiers=["national"],
    )


def test_escalates_on_loaded_language():
    cluster = _make_cluster([
        ("The radical agenda is destroying the economy", "Source A", "right", -0.5),
        ("Policy changes face criticism", "Source B", "center", -0.1),
    ])
    result = LexiconAnalyzer().analyze(cluster)
    assert result.escalate is True
    assert any("radical" in words for words in result.loaded_words_found.values())


def test_escalates_on_high_sentiment_variance():
    cluster = _make_cluster([
        ("Economy is booming wonderfully", "Source A", "right", 0.8),
        ("Economic outlook remains catastrophic", "Source B", "left", -0.7),
    ])
    result = LexiconAnalyzer(escalation_threshold=0.35).analyze(cluster)
    assert result.sentiment_variance > 0.35
    assert result.escalate is True


def test_no_escalation_on_neutral_cluster():
    cluster = _make_cluster([
        ("Senate committee meets on Tuesday", "Source A", "center", 0.0),
        ("Lawmakers hold session this week", "Source B", "center", 0.02),
    ])
    result = LexiconAnalyzer().analyze(cluster)
    # Loaded words unlikely; sentiment variance near zero
    assert result.sentiment_variance < 0.35


def test_single_source_cluster_variance_is_zero():
    cluster = _make_cluster([
        ("Governor signs bill into law", "Source A", "center", 0.1),
    ])
    result = LexiconAnalyzer().analyze(cluster)
    assert result.sentiment_variance == 0.0
