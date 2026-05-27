"""Story clustering and digest story selection for domain Article objects."""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher

from domain.models import Article, StoryCluster

logger = logging.getLogger(__name__)


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

        # Deduplicate by canonical URL hash before clustering.
        # The same physical article can be ingested from multiple category
        # feeds (e.g. "AP Top News" and "AP Politics"). Dropping duplicates
        # here ensures each article enters the union-find exactly once and
        # cannot inflate cluster.source_count or article counts.
        seen_hashes: set[str] = set()
        unique_articles: list[Article] = []
        for article in articles:
            if article.url_hash not in seen_hashes:
                seen_hashes.add(article.url_hash)
                unique_articles.append(article)
        dropped = len(articles) - len(unique_articles)
        if dropped:
            logger.debug("url_hash dedup: dropped %d duplicate article(s) before clustering", dropped)
        articles = unique_articles

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
    reportable = [
        cluster
        for cluster in clusters
        if not cluster.is_single_source or cluster.importance_score >= threshold
    ]
    logger.info(
        "Reportable cluster filter: %d -> %d clusters (singleton threshold %.2f)",
        len(clusters), len(reportable), threshold,
    )
    return reportable


def select_digest_clusters(clusters: list[StoryCluster], settings: dict) -> list[StoryCluster]:
    """Select a bounded digest set while preserving useful single-source stories.

    Multi-source coverage is preferred, but local/state/national digests still
    need worthwhile single-source stories. Without this pass, a run with many
    valid singleton clusters can collapse to only the handful of stories that
    multiple configured outlets happened to cover.
    """
    delivery_cfg = settings.get("delivery", {}).get("email", {})
    max_per_tier = int(delivery_cfg.get("max_stories_per_category", 7))
    cluster_cfg = settings.get("clustering", {})
    singleton_floor = float(cluster_cfg.get("singleton_digest_floor", 0.1))
    min_singletons_per_tier = int(cluster_cfg.get("min_singletons_per_tier", 3))

    selected: list[StoryCluster] = []
    for tier in ("national", "state", "local"):
        tier_clusters = [c for c in clusters if c.geo_tier == tier]
        tier_clusters.sort(key=_selection_key, reverse=True)
        multi = [c for c in tier_clusters if not c.is_single_source]
        singletons = [
            c for c in tier_clusters
            if c.is_single_source and c.importance_score >= singleton_floor
        ]

        tier_selected = multi[:max_per_tier]
        remaining = max_per_tier - len(tier_selected)
        if remaining > 0:
            tier_selected.extend(singletons[:remaining])
        elif len(tier_selected) < min_singletons_per_tier:
            tier_selected.extend(singletons[: max(0, min_singletons_per_tier - len(tier_selected))])
            tier_selected = tier_selected[:max_per_tier]

        selected.extend(tier_selected)

    logger.info(
        "Digest cluster selection: %d -> %d clusters (max_per_tier=%d)",
        len(clusters), len(selected), max_per_tier,
    )
    return selected


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
    return left.geo_tier == right.geo_tier


def _similarity(left: Article, right: Article) -> float:
    left_text = f"{left.headline} {left.summary}".lower()
    right_text = f"{right.headline} {right.summary}".lower()
    headline_seq = SequenceMatcher(None, left.headline.lower(), right.headline.lower()).ratio()
    left_tokens = _tokens(left_text)
    right_tokens = _tokens(right_text)
    if not left_tokens or not right_tokens:
        return headline_seq
    shared = left_tokens & right_tokens
    if len(shared) < 2:
        return 0.0
    jaccard = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
    return max(jaccard, headline_seq * 0.8)


def _tokens(text: str) -> set[str]:
    stop = {
        "the", "a", "an", "to", "of", "and", "in", "on", "for", "with", "new",
        "news", "says", "said", "after", "from", "over", "this", "that", "will",
        "has", "have", "are", "was", "were", "into", "about", "latest", "live",
    }
    cleaned = "".join(ch if ch.isalnum() else " " for ch in text)
    return {t for t in cleaned.split() if len(t) > 2 and t not in stop}


def _selection_key(cluster: StoryCluster) -> tuple[float, int, int]:
    corroboration_boost = 1 if not cluster.is_single_source else 0
    return (cluster.importance_score, corroboration_boost, len(cluster.articles))
