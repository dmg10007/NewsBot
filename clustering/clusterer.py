"""Story clustering using semantic similarity.

Groups ParsedArticles covering the same real-world event into StoryCluster
objects, regardless of source or framing. Each cluster becomes the unit of
analysis for bias detection and summarization.

Approach:
  1. Encode all article full_text fields with sentence-transformers
  2. Use util.semantic_search() for batched ANN similarity (O(n) vs O(n²))
  3. Greedily assign articles to clusters using a similarity threshold
  4. Attach corroboration metadata (how many sources, which tiers, bias spread)

Model loading
-------------
StoryClusterer no longer loads its own SentenceTransformer instance.
It delegates to utils.model_registry.get_model(), which returns a shared
cached instance. This eliminates the double load that occurred when both
Deduplicator and StoryClusterer were instantiated in the same run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from sentence_transformers import util

from config.loader import get_settings
from parsing.extractor import ParsedArticle
from utils.model_registry import get_model

logger = logging.getLogger(__name__)


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
        # Use the article from the highest-credibility source as representative
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

    Uses util.semantic_search() for batched approximate nearest-neighbor
    similarity instead of an O(n²) pairwise loop. For 200 articles the old
    approach made ~20,000 individual cos_sim() calls; this version computes
    the same matrix in one vectorized shot.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._threshold: float = self.settings["clustering"]["similarity_threshold"]
        cfg = self.settings.get("clustering", {})
        max_age_hours: float = float(cfg.get("max_age_delta_hours", 48))
        from datetime import timedelta
        self._max_age_delta = timedelta(hours=max_age_hours)

    def cluster(self, articles: list[ParsedArticle]) -> list[StoryCluster]:
        if not articles:
            return []

        # Fast path: single article needs no model
        if len(articles) == 1:
            return [self._make_cluster(0, articles)]

        model = get_model(self.settings["clustering"]["model"])
        texts = [a.full_text for a in articles]
        logger.info("Encoding %d articles for clustering...", len(texts))
        embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=False)

        # Batched ANN: returns top_k most similar articles for every article.
        # top_k=len(articles) ensures we see all pairs above threshold.
        top_k = min(len(articles), 50)  # cap at 50 neighbours — sufficient for greedy grouping
        hits = util.semantic_search(embeddings, embeddings, top_k=top_k)

        # Greedy single-linkage clustering using ANN hits
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
                if hit["score"] >= self._threshold and self._within_age_window(
                    articles[i], articles[j]
                ):
                    assigned[j] = cluster_id
            cluster_id += 1

        # Build cluster objects
        cluster_map: dict[int, list[ParsedArticle]] = {}
        for article, cid in zip(articles, assigned):
            cluster_map.setdefault(cid, []).append(article)

        clusters = [
            self._make_cluster(cid, members)
            for cid, members in cluster_map.items()
        ]

        logger.info(
            "Clustered %d articles into %d story clusters",
            len(articles), len(clusters)
        )
        return clusters

    def _within_age_window(self, a: ParsedArticle, b: ParsedArticle) -> bool:
        """Return True if both articles are within the configured age delta.

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
        tiers = list({self._tier(a.raw.region) for a in members})
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

    @staticmethod
    def _tier(region: str) -> str:
        """Map a RawArticle.region value to a display tier string.

        RawArticle.region holds the raw string from sources.yaml
        ('national' | 'north_carolina' | 'lee_county_nc' | ...).
        StoryCluster.tiers expects the human-readable tier labels used
        throughout the pipeline ('national' | 'state' | 'local').
        """
        if region == "national":
            return "national"
        if region == "north_carolina":
            return "state"
        return "local"
