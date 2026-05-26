"""Story clustering for domain Article objects."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher

from domain.models import Article, StoryCluster


class StoryClusterer:
    """Cluster articles that appear to report the same news event.

    This implementation keeps the dependency surface small for the durable
    pipeline. It uses normalized headline/summary similarity and transitive
    union so A-B and B-C matches produce one cluster. The old
    sentence-transformer clusterer can still be tuned separately, but this
    path is deterministic and easy to test.
    """

    def __init__(self, similarity_threshold: float = 0.58) -> None:
        self.similarity_threshold = similarity_threshold

    def cluster(self, articles: list[Article]) -> list[StoryCluster]:
        if not articles:
            return []
        parent = list(range(len(articles)))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i, left in enumerate(articles):
            for j in range(i + 1, len(articles)):
                right = articles[j]
                if _geo_compatible(left, right) and _similarity(left, right) >= self.similarity_threshold:
                    union(i, j)

        grouped: dict[int, list[Article]] = defaultdict(list)
        for index, article in enumerate(articles):
            grouped[find(index)].append(article)

        clusters = []
        for idx, members in enumerate(grouped.values(), start=1):
            clusters.append(_make_cluster(idx, members))
        return clusters


def score_clusters(clusters: list[StoryCluster], settings: dict) -> None:
    weights = settings.get("scoring", {}).get("weights", {})
    source_count_w = float(weights.get("source_count", 0.3))
    ceiling = float(weights.get("normalization_ceiling", 5.0))
    tier_weights = {
        "national": float(weights.get("national_tier", 1.0)),
        "state": float(weights.get("state_tier", 1.2)),
        "local": float(weights.get("local_tier", 1.5)),
    }
    recency_decay = float(weights.get("recency_decay", 0.05))
    now = datetime.now(timezone.utc)

    for cluster in clusters:
        score = max(1, cluster.source_count) * source_count_w
        score *= tier_weights.get(cluster.geo_tier, 1.0)
        published = [
            a.published_at.replace(tzinfo=timezone.utc) if a.published_at and a.published_at.tzinfo is None else a.published_at
            for a in cluster.articles
            if a.published_at is not None
        ]
        if published:
            age_hours = (now - max(published)).total_seconds() / 3600
            score *= max(0.1, 1.0 - age_hours * recency_decay)
        cluster.importance_score = max(min(score / ceiling, 1.0), 0.1)


def filter_reportable_clusters(clusters: list[StoryCluster], settings: dict) -> list[StoryCluster]:
    threshold = float(
        settings.get("clustering", {}).get("drop_singletons_below_importance", 0.4)
    )
    return [
        cluster
        for cluster in clusters
        if not cluster.is_single_source or cluster.importance_score >= threshold
    ]


def _make_cluster(cluster_id: int, articles: list[Article]) -> StoryCluster:
    topic_counts: dict[str, int] = {}
    tier_counts: dict[str, int] = {}
    for article in articles:
        for topic in article.topics or ["current_events"]:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        tier_counts[article.geo_tier] = tier_counts.get(article.geo_tier, 0) + 1
    topic = max(topic_counts, key=topic_counts.get) if topic_counts else "current_events"
    geo_tier = max(tier_counts, key=tier_counts.get) if tier_counts else "national"
    best = sorted(
        articles,
        key=lambda a: ({"high": 0, "medium": 1, "low": 2}.get(a.credibility, 2), a.headline),
    )[0]
    return StoryCluster(
        cluster_id=cluster_id,
        articles=articles,
        topic=topic,
        geo_tier=geo_tier,  # type: ignore[arg-type]
        representative_headline=best.headline,
    )


def _geo_compatible(left: Article, right: Article) -> bool:
    return left.geo_tier == right.geo_tier or "national" in {left.geo_tier, right.geo_tier}


def _similarity(left: Article, right: Article) -> float:
    left_text = f"{left.headline} {left.summary}".lower()
    right_text = f"{right.headline} {right.summary}".lower()
    seq = SequenceMatcher(None, left_text, right_text).ratio()
    left_tokens = _tokens(left_text)
    right_tokens = _tokens(right_text)
    if not left_tokens or not right_tokens:
        return seq
    jaccard = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
    return max(seq, jaccard)


def _tokens(text: str) -> set[str]:
    stop = {"the", "a", "an", "to", "of", "and", "in", "on", "for", "with", "new"}
    return {t for t in text.replace("-", " ").split() if len(t) > 2 and t not in stop}
