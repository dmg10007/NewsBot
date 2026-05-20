"""Story clustering: groups ParsedArticles covering the same event.

Strategy:
  1. Encode all article headlines with sentence-transformers.
  2. Build a cosine similarity matrix.
  3. Greedy single-pass clustering: each article either joins an existing
     cluster (if sim >= threshold) or seeds a new one.
  4. Score each cluster by source count, tier diversity, and recency.
  5. Return clusters sorted by score descending, ready for bias analysis.

Single-pass greedy is fast and deterministic — good enough for ~200 articles.
If article volume grows significantly, swap for DBSCAN or Agglomerative.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import torch
from sentence_transformers import SentenceTransformer, util

from config.loader import get_settings
from parsing.extractor import ParsedArticle

logger = logging.getLogger(__name__)


@dataclass
class StoryCluster:
    """A group of articles covering the same story across sources."""
    cluster_id: int
    articles: list[ParsedArticle] = field(default_factory=list)
    topic: Optional[str] = None          # Majority-vote topic across articles
    score: float = 0.0                   # Importance score for ranking
    representative_headline: str = ""    # Headline of the highest-credibility article

    @property
    def source_names(self) -> list[str]:
        return [a.raw.source_name for a in self.articles]

    @property
    def regions(self) -> set[str]:
        return {a.raw.region for a in self.articles}

    @property
    def bias_leans(self) -> list[str]:
        return [a.raw.bias_lean for a in self.articles]

    @property
    def latest_published(self) -> Optional[datetime]:
        dates = [a.raw.published_at for a in self.articles if a.raw.published_at]
        return max(dates) if dates else None

    @property
    def source_count(self) -> int:
        return len(set(self.source_names))


class StoryClusterer:
    """Clusters ParsedArticles into StoryCluster objects."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._model: Optional[SentenceTransformer] = None

    def cluster(self, articles: list[ParsedArticle]) -> list[StoryCluster]:
        if not articles:
            return []

        model = self._get_model()
        headlines = [a.raw.headline for a in articles]

        logger.info("Encoding %d headlines for clustering...", len(headlines))
        embeddings = model.encode(
            headlines,
            convert_to_tensor=True,
            show_progress_bar=False,
            batch_size=64,
        )

        # Store embeddings on articles for later bias analysis
        for article, emb in zip(articles, embeddings):
            article.embedding = emb.tolist()

        threshold = self.settings["clustering"]["similarity_threshold"]
        clusters = self._greedy_cluster(articles, embeddings, threshold)

        # Score and assign topics
        for cluster in clusters:
            cluster.topic = self._majority_topic(cluster)
            cluster.score = self._score_cluster(cluster)
            cluster.representative_headline = self._best_headline(cluster)

        # Sort by score descending
        clusters.sort(key=lambda c: c.score, reverse=True)
        logger.info("Produced %d story clusters from %d articles", len(clusters), len(articles))
        return clusters

    def _greedy_cluster(
        self,
        articles: list[ParsedArticle],
        embeddings: torch.Tensor,
        threshold: float,
    ) -> list[StoryCluster]:
        cluster_id = 0
        clusters: list[StoryCluster] = []
        # centroid_embeddings[i] = mean embedding of cluster i
        centroid_embeddings: list[torch.Tensor] = []

        for i, article in enumerate(articles):
            emb = embeddings[i]
            best_cluster_idx: Optional[int] = None
            best_sim = threshold  # Must beat threshold to join

            for j, centroid in enumerate(centroid_embeddings):
                sim = float(util.cos_sim(emb, centroid))
                if sim > best_sim:
                    best_sim = sim
                    best_cluster_idx = j

            if best_cluster_idx is not None:
                # Join existing cluster and update centroid
                clusters[best_cluster_idx].articles.append(article)
                # Recompute centroid as mean of all member embeddings
                member_embs = embeddings[
                    [articles.index(a) for a in clusters[best_cluster_idx].articles]
                ]
                centroid_embeddings[best_cluster_idx] = member_embs.mean(dim=0)
            else:
                # Seed a new cluster
                new_cluster = StoryCluster(cluster_id=cluster_id)
                new_cluster.articles.append(article)
                clusters.append(new_cluster)
                centroid_embeddings.append(emb.clone())
                cluster_id += 1

        return clusters

    def _majority_topic(self, cluster: StoryCluster) -> str:
        from collections import Counter
        topics = [
            a.detected_topic or (a.raw.topics[0] if a.raw.topics else "current_events")
            for a in cluster.articles
        ]
        return Counter(topics).most_common(1)[0][0]

    def _score_cluster(self, cluster: StoryCluster) -> float:
        weights = self.settings["scoring"]["weights"]
        score = 0.0

        # Source credibility
        for article in cluster.articles:
            if article.raw.credibility == "high":
                score += weights["source_credibility_high"]
            else:
                score += weights["source_credibility_medium"]

        # Corroboration bonus (per additional unique source)
        score += (cluster.source_count - 1) * weights["source_count"]

        # Geographic tier boost
        regions = cluster.regions
        if "lee_county_nc" in regions:
            score *= weights["local_tier"]
        elif "north_carolina" in regions:
            score *= weights["state_tier"]
        else:
            score *= weights["national_tier"]

        # Recency decay — reduce score by decay_rate per hour of age
        latest = cluster.latest_published
        if latest:
            now = datetime.now(timezone.utc)
            hours_old = max(0, (now - latest).total_seconds() / 3600)
            score *= max(0.1, 1.0 - weights["recency_decay"] * hours_old)

        return round(score, 4)

    def _best_headline(self, cluster: StoryCluster) -> str:
        """Pick headline from highest-credibility, most-recent article."""
        priority = {"high": 2, "medium": 1}
        best = max(
            cluster.articles,
            key=lambda a: (
                priority.get(a.raw.credibility, 0),
                a.raw.published_at or datetime.min.replace(tzinfo=timezone.utc),
            ),
        )
        return best.raw.headline

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            model_name = self.settings["clustering"]["model"]
            logger.info("Loading sentence-transformer: %s", model_name)
            self._model = SentenceTransformer(model_name)
        return self._model
