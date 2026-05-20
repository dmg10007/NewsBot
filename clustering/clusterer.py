"""Story clustering using semantic similarity.

Groups ParsedArticles covering the same real-world event into StoryCluster
objects, regardless of source or framing. Each cluster becomes the unit of
analysis for bias detection and summarization.

Approach:
  1. Encode all article full_text fields with sentence-transformers
  2. Build a cosine similarity matrix
  3. Greedily assign articles to clusters using a similarity threshold
  4. Attach corroboration metadata (how many sources, which tiers, bias spread)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer, util

from config.loader import get_settings
from parsing.extractor import ParsedArticle

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
    """Clusters ParsedArticles into StoryCluster objects by semantic similarity."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._threshold: float = self.settings["clustering"]["similarity_threshold"]
        self._model: SentenceTransformer | None = None

    def cluster(self, articles: list[ParsedArticle]) -> list[StoryCluster]:
        if not articles:
            return []

        model = self._get_model()
        texts = [a.full_text for a in articles]
        logger.info("Encoding %d articles for clustering...", len(texts))
        embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=False)

        # Greedy single-linkage clustering
        assigned: list[int] = [-1] * len(articles)
        cluster_id = 0

        for i in range(len(articles)):
            if assigned[i] != -1:
                continue
            assigned[i] = cluster_id
            for j in range(i + 1, len(articles)):
                if assigned[j] != -1:
                    continue
                sim = float(util.cos_sim(embeddings[i], embeddings[j]))
                if sim >= self._threshold:
                    assigned[j] = cluster_id
            cluster_id += 1

        # Build cluster objects
        cluster_map: dict[int, list[ParsedArticle]] = {}
        for article, cid in zip(articles, assigned):
            cluster_map.setdefault(cid, []).append(article)

        clusters = []
        for cid, members in cluster_map.items():
            topic = self._dominant_topic(members)
            tiers = list({self._tier(a.raw.region) for a in members})
            cluster = StoryCluster(
                cluster_id=cid,
                articles=members,
                topic=topic,
                tiers=tiers,
            )
            clusters.append(cluster)

        logger.info(
            "Clustered %d articles into %d story clusters",
            len(articles), len(clusters)
        )
        return clusters

    def _dominant_topic(self, articles: list[ParsedArticle]) -> str:
        topic_counts: dict[str, int] = {}
        for a in articles:
            for t in a.detected_topics:
                topic_counts[t] = topic_counts.get(t, 0) + 1
        return max(topic_counts, key=topic_counts.get) if topic_counts else "current_events"

    def _tier(self, region: str) -> str:
        if region == "national":
            return "national"
        if region == "north_carolina":
            return "state"
        return "local"

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            model_name = self.settings["clustering"]["model"]
            logger.info("Loading sentence-transformer model: %s", model_name)
            self._model = SentenceTransformer(model_name)
        return self._model
