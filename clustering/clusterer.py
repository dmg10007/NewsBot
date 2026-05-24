"""Story clustering using semantic similarity.

Groups ParsedArticles covering the same real-world event into StoryCluster
objects, regardless of source or framing. Each cluster becomes the unit of
analysis for bias detection and summarization.

Approach:
  1. Encode all article full_text fields with sentence-transformers
  2. Use util.semantic_search() for batched ANN similarity (O(n) vs O(n²))
  3. Greedily assign articles to clusters using a similarity threshold
     AND a publication-time window gate
  4. Attach corroboration metadata (how many sources, which tiers, bias spread)

Threshold guidance
------------------
  0.65  — recommended default. Same-event stories from different outlets
          often land in the 0.65–0.75 range because each outlet writes the
          same facts with different vocabulary and emphasis.
  0.72+ — too strict: wire rewrites of the same story miss the threshold
          and appear as duplicate entries in the digest.
  0.55- — too loose: topically related but distinct events merge.

Age gate
--------
  max_age_delta_hours (default 24): two articles must have been published
  within this window of each other to be eligible for merging. This
  prevents a new development on the same topic from being merged with
  yesterday’s story (a false merge), while still catching same-day
  wire rewrites regardless of similarity score variance.

Model loading
-------------
StoryClusterer delegates model loading to utils.model_registry.get_model(),
which returns a shared cached instance. This eliminates the double load
that occurred when both Deduplicator and StoryClusterer were instantiated
in the same run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from sentence_transformers import util

from config.loader import get_settings
from parsing.extractor import ParsedArticle
from utils.model_registry import get_model

logger = logging.getLogger(__name__)

_DEFAULT_THRESHOLD = 0.65
_DEFAULT_MAX_AGE_DELTA_HOURS = 24


@dataclass
class StoryCluster:
    """A group of articles covering the same story across sources."""
    cluster_id: int
    articles: list[ParsedArticle]
    topic: str                        # Dominant topic for this cluster
    tiers: list[str]                  # Which geographic tiers are represented
    source_count: int = field(init=False)
    bias_spread: list[str] = field(init=False)  # Unique bias_lean values present
    earliest_published: Optional[datetime] = field(init=False)
    importance_score: float = 0.0     # Computed by scorer — higher = more prominent
    representative_headline: str = field(init=False)

    def __post_init__(self) -> None:
        self.source_count = len(self.articles)
        self.bias_spread = list({
            a.raw.bias_lean for a in self.articles if a.raw.bias_lean != "unknown"
        })
        published_dates = [
            a.raw.published_at for a in self.articles if a.raw.published_at
        ]
        self.earliest_published = min(published_dates) if published_dates else None
        credibility_order = {"high": 0, "medium": 1, "low": 2}
        best = min(
            self.articles,
            key=lambda a: credibility_order.get(a.raw.credibility, 2)
        )
        self.representative_headline = best.raw.headline

    @property
    def has_cross_source_coverage(self) -> bool:
        return len({a.raw.source_name for a in self.articles}) > 1

    @property
    def has_cross_lean_coverage(self) -> bool:
        """True if cluster includes articles from both left and right outlets."""
        leans = {a.raw.bias_lean for a in self.articles}
        has_left = any(l in leans for l in ("left", "center-left"))
        has_right = any(l in leans for l in ("right", "center-right"))
        return has_left and has_right


class StoryClusterer:
    """Clusters ParsedArticles into StoryCluster objects by semantic similarity.

    Uses util.semantic_search() for batched ANN similarity. Two articles
    are eligible to merge only when both:
      - Their embedding cosine similarity >= similarity_threshold
      - Their publication timestamps are within max_age_delta_hours of each other
        (or either timestamp is missing, in which case the age gate is skipped)
    """

    def __init__(self) -> None:
        settings = get_settings()
        c = settings["clustering"]
        self._threshold: float = float(c.get("similarity_threshold", _DEFAULT_THRESHOLD))
        self._max_age_delta: timedelta = timedelta(
            hours=float(c.get("max_age_delta_hours", _DEFAULT_MAX_AGE_DELTA_HOURS))
        )
        self._model_name: str = c["model"]

    def cluster(self, articles: list[ParsedArticle]) -> list[StoryCluster]:
        if not articles:
            return []
        if len(articles) == 1:
            return [self._make_cluster(0, articles)]

        model = get_model(self._model_name)
        texts = [a.full_text for a in articles]
        logger.info("Encoding %d articles for clustering...", len(texts))
        embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=False)

        top_k = min(len(articles), 50)
        hits = util.semantic_search(embeddings, embeddings, top_k=top_k)

        assigned: list[int] = [-1] * len(articles)
        cluster_id = 0

        for i in range(len(articles)):
            if assigned[i] != -1:
                continue
            assigned[i] = cluster_id
            for hit in hits[i]:
                j = hit["corpus_id"]
                if j == i or assigned[j] != -1:
                    continue
                if hit["score"] >= self._threshold and self._within_age_window(articles[i], articles[j]):
                    assigned[j] = cluster_id
            cluster_id += 1

        cluster_map: dict[int, list[ParsedArticle]] = {}
        for article, cid in zip(articles, assigned):
            cluster_map.setdefault(cid, []).append(article)

        clusters = [
            self._make_cluster(cid, members)
            for cid, members in cluster_map.items()
        ]

        logger.info(
            "Clustered %d articles into %d story clusters (threshold=%.2f, age_gate=%dh)",
            len(articles), len(clusters), self._threshold,
            int(self._max_age_delta.total_seconds() / 3600),
        )
        return clusters

    def _within_age_window(self, a: ParsedArticle, b: ParsedArticle) -> bool:
        """Return True if both articles were published within max_age_delta of each other.

        If either timestamp is missing, the gate is skipped (returns True) so
        articles without publication dates are never excluded solely on age.
        """
        ts_a = a.raw.published_at
        ts_b = b.raw.published_at
        if ts_a is None or ts_b is None:
            return True
        # Normalise both to UTC-aware for safe comparison
        if ts_a.tzinfo is None:
            ts_a = ts_a.replace(tzinfo=timezone.utc)
        if ts_b.tzinfo is None:
            ts_b = ts_b.replace(tzinfo=timezone.utc)
        return abs(ts_a - ts_b) <= self._max_age_delta

    def _make_cluster(self, cid: int, members: list[ParsedArticle]) -> StoryCluster:
        topic = self._dominant_topic(members)
        tiers = list({a.raw.geo_tier for a in members})
        return StoryCluster(
            cluster_id=cid,
            articles=members,
            topic=topic,
            tiers=tiers,
        )

    def _dominant_topic(self, articles: list[ParsedArticle]) -> str:
        topic_counts: dict[str, int] = {}
        for a in articles:
            for t in a.detected_topics:
                topic_counts[t] = topic_counts.get(t, 0) + 1
        return max(topic_counts, key=topic_counts.get) if topic_counts else "current_events"
