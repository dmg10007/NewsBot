"""Story clustering using semantic similarity.

Groups ParsedArticles covering the same real-world event into StoryCluster
objects, regardless of source or framing. Each cluster becomes the unit of
analysis for bias detection and summarization.

Approach
--------
  1. Encode all article cluster_text fields with sentence-transformers
     (headline + lead paragraph — more semantically consistent than full body)
  2. Run full pairwise ANN via util.semantic_search() with top_k=len(articles)
  3. Build an edge list of all pairs above similarity_threshold and within
     the per-tier age window
  4. Merge connected pairs transitively using Union-Find (complete-linkage).
     This fixes the single-linkage chain-break problem: A+B and B+C now
     correctly land in the same cluster even if A→C never appears in top_k.
  5. Drop singleton clusters below drop_singletons_below_importance threshold
  6. Attach corroboration metadata and log quality metrics

Algorithm selection rationale
------------------------------
Single-linkage greedy (old): assigned article i to cluster only if it had
a direct above-threshold hit with the seed article. Transitive membership
was not propagated, so chains broke and most articles became singletons.

Complete-linkage via Union-Find (new): any path of above-threshold pairs
between two articles will merge them into the same cluster. This matches
how news stories actually propagate — wire copy and follow-up pieces don't
always share high cosine similarity with every other cluster member, but
they do share it with at least one.

Model loading
-------------
StoryClusterer delegates to utils.model_registry.get_model(), which returns
a shared cached instance. This eliminates the double load that occurred when
both Deduplicator and StoryClusterer were instantiated in the same run.

Age window
----------
max_age_delta_hours is now a per-tier map in settings.yaml:

  clustering:
    max_age_delta_hours:
      national: 72
      state: 48
      local: 24

Scalar values are still accepted for backward compatibility.

Tier mapping
------------
_tier() maps RawArticle.region values to scorer/clusterer tier strings:
  national       -> 'national'
  north_carolina -> 'state'
  lee_county_nc  -> 'local'   (or any other local region)
  international  -> 'international'  (dropped by GeoFilter before this stage;
                                      mapping present as a safety net)
  <anything else>-> 'local'   (unknown regions treated as local, not national)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional, Union

import numpy as np
from sentence_transformers import util

from config.loader import get_settings
from parsing.extractor import ParsedArticle
from utils.model_registry import get_model

logger = logging.getLogger(__name__)

# Only add a top_k cap above this article count to bound memory.
# Below 2000 articles the full similarity matrix is ~30MB of float32 — fine.
_TOPK_CAP_THRESHOLD = 2000
_TOPK_CAP = 200


def _to_utc(dt: datetime) -> datetime:
    """Return dt as a tz-aware UTC datetime. Naive datetimes are assumed UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class UnionFind:
    """Path-compressed Union-Find (disjoint-set) with union-by-rank.

    Used to merge article indices into clusters transitively.
    All operations are effectively O(α(n)) ≈ O(1).
    """

    def __init__(self, n: int) -> None:
        self._parent = list(range(n))
        self._rank = [0] * n

    def find(self, x: int) -> int:
        """Return the root of x's component with path compression."""
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]  # path halving
            x = self._parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        """Merge the components of x and y."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1


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

    # importance_score is intentionally NOT settable via the constructor.
    # It is computed post-construction by scoring.scorer.score_clusters()
    # and written back in place. Using field(init=False) prevents accidental
    # override if StoryCluster is constructed with a keyword argument.
    importance_score: float = field(init=False, default=0.0)

    representative_headline: str = field(init=False)

    def __post_init__(self) -> None:
        self.source_count = len(self.articles)
        self.bias_spread = list({
            a.raw.bias_lean for a in self.articles if a.raw.bias_lean != "unknown"
        })
        # Normalize to UTC before comparison — RSS feeds produce a mix of
        # tz-aware and tz-naive datetimes; min() raises TypeError on mixed lists.
        published_dates = [
            _to_utc(a.raw.published_at)
            for a in self.articles
            if a.raw.published_at is not None
        ]
        self.earliest_published = min(published_dates) if published_dates else None
        credibility_order = {"high": 0, "medium": 1, "low": 2}
        best = min(
            self.articles,
            key=lambda a: credibility_order.get(a.raw.credibility, 2)
        )
        self.representative_headline = best.raw.headline

    @property
    def is_singleton(self) -> bool:
        """True if this cluster contains only one article."""
        return len(self.articles) == 1

    @property
    def is_single_source(self) -> bool:
        """Compatibility alias used by older tests/renderers."""
        return len({a.raw.source_name for a in self.articles}) <= 1

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

    Uses complete-linkage via Union-Find rather than single-linkage greedy
    assignment. All pairs above similarity_threshold are merged transitively,
    so same-story articles separated by one hop are no longer left as
    singletons.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        cfg = self.settings.get("clustering", {})
        self._threshold: float = float(cfg["similarity_threshold"])
        self._age_delta_cfg: Union[dict, float] = cfg.get("max_age_delta_hours", 48)
        self._drop_singleton_threshold: float = float(
            cfg.get("drop_singletons_below_importance", 0.4)
        )

    def _age_delta_for_tier(self, tier: str) -> timedelta:
        """Return the max age delta for a given tier string.

        Accepts both the new dict form and legacy scalar form from settings.yaml.
        """
        cfg = self._age_delta_cfg
        if isinstance(cfg, dict):
            hours = float(cfg.get(tier, cfg.get("national", 72)))
        else:
            hours = float(cfg)
        return timedelta(hours=hours)

    def cluster(self, articles: list[ParsedArticle]) -> list[StoryCluster]:
        if not articles:
            return []
        if len(articles) == 1:
            return [self._make_cluster(0, articles)]

        model = get_model(self.settings["clustering"]["model"])
        texts = [a.cluster_text for a in articles]
        logger.info("Encoding %d articles for clustering (cluster_text)...", len(texts))
        embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=False)

        # Full pairwise ANN — no top_k cap below _TOPK_CAP_THRESHOLD articles
        n = len(articles)
        top_k = _TOPK_CAP if n > _TOPK_CAP_THRESHOLD else n
        hits = util.semantic_search(embeddings, embeddings, top_k=top_k)

        # Build Union-Find over all above-threshold pairs
        uf = UnionFind(n)
        for i, neighbors in enumerate(hits):
            tier_i = self._tier(articles[i].raw.region)
            for hit in neighbors:
                j = hit["corpus_id"]
                if j == i:
                    continue
                if hit["score"] < self._threshold:
                    continue
                tier_j = self._tier(articles[j].raw.region)
                delta = min(
                    self._age_delta_for_tier(tier_i),
                    self._age_delta_for_tier(tier_j),
                )
                if self._within_age_window(articles[i], articles[j], delta):
                    uf.union(i, j)

        # Group articles by their Union-Find root
        cluster_map: dict[int, list[ParsedArticle]] = {}
        for idx, article in enumerate(articles):
            root = uf.find(idx)
            cluster_map.setdefault(root, []).append(article)

        clusters = [
            self._make_cluster(cid, members)
            for cid, members in cluster_map.items()
        ]

        # Score-dependent singleton filtering now belongs to the digest
        # orchestrator after score_clusters() has populated importance_score.
        dropped = 0

        self._log_quality(articles, clusters, dropped)
        return clusters

    def _log_quality(self,
                     articles: list[ParsedArticle],
                     clusters: list[StoryCluster],
                     dropped_singletons: int) -> None:
        n_clusters = len(clusters)
        n_singletons = sum(1 for c in clusters if c.is_singleton)
        multi = n_clusters - n_singletons
        sizes = [len(c.articles) for c in clusters]
        avg_size = sum(sizes) / len(sizes) if sizes else 0.0
        cross_lean = sum(1 for c in clusters if c.has_cross_lean_coverage)
        logger.info(
            "Clustered %d articles → %d clusters "
            "(singletons kept: %d, dropped: %d | multi-source: %d | "
            "avg size: %.1f | cross-lean: %d)",
            len(articles), n_clusters,
            n_singletons, dropped_singletons,
            multi, avg_size, cross_lean,
        )

    def _within_age_window(
        self,
        a: ParsedArticle,
        b: ParsedArticle,
        delta: timedelta,
    ) -> bool:
        """Return True if both articles fall within the given age delta."""
        ts_a = a.raw.published_at
        ts_b = b.raw.published_at
        if ts_a is None or ts_b is None:
            return True
        return abs(_to_utc(ts_a) - _to_utc(ts_b)) <= delta

    def _make_cluster(self, cid: int, members: list[ParsedArticle]) -> StoryCluster:
        topic = self._dominant_topic(members)
        tiers = list({self._tier(a.raw.geo_tier or a.raw.region) for a in members})
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

        Explicit mapping prevents unknown regions from accidentally receiving
        the local_tier scorer boost (1.5x). Unknown regions fall to 'local'
        which is conservative — they won't be treated as high-value national
        content. International is mapped explicitly so future international
        sources don't accidentally pollute the national tier.
        """
        if region == "national":
            return "national"
        if region == "north_carolina":
            return "state"
        if region == "international":
            return "international"
        # All other regions (lee_county_nc, unknown, etc.) → local
        return "local"
