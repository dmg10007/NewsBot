"""Deduplication of raw articles before NLP processing.

Two-pass strategy:
1. Exact URL hash dedup (fast, catches reposts of the same URL)
2. Headline semantic similarity dedup using ANN (util.semantic_search)

Semantic dedup passes:
  Pass A — within-source: removes rewrites of the same story published
            multiple times by the same outlet.
  Pass B — cross-source wire detection: removes articles from different
            outlets that are near-identical (sim >= wire_syndication_threshold,
            default 0.99). Without this, syndicated AP/Reuters wire copy
            appears as independent corroboration in clustering, inflating
            the source_count and importance_score of those clusters.

Per-tier thresholds
-------------------
Within-source dedup uses per-tier thresholds from settings.yaml:
  deduplication.headline_similarity_threshold_by_tier:
    local:    0.85  (lower — local outlets reuse similar phrasing for
                    follow-up stories; collapse more aggressively)
    state:    0.92
    national: 0.95
Falls back to deduplication.headline_similarity_threshold (global) if a
tier is not in the map or the map key is absent.

Performance
-----------
Uses util.semantic_search() — one vectorized matrix operation — instead of
the original O(n²) double-loop calling cos_sim() individually per pair.

Lifecycle
---------
Deduplicator exposes a close() method as a forward-compatible lifecycle hook.
Currently a no-op; callers should still call it in a finally block.
"""

from __future__ import annotations

import logging

from sentence_transformers import util

from config.loader import get_settings
from ingestion.fetcher import RawArticle
from utils.model_registry import get_model

logger = logging.getLogger(__name__)

_WIRE_SYNDICATION_THRESHOLD = 0.95


class Deduplicator:
    def __init__(self) -> None:
        self.settings = get_settings()
        dedup_cfg = self.settings["deduplication"]
        self._global_threshold: float = float(
            dedup_cfg["headline_similarity_threshold"]
        )
        self._tier_thresholds: dict[str, float] = {
            tier: float(val)
            for tier, val in dedup_cfg.get(
                "headline_similarity_threshold_by_tier", {}
            ).items()
        }
        self._wire_threshold: float = float(
            dedup_cfg.get("wire_syndication_threshold", _WIRE_SYNDICATION_THRESHOLD)
        )

    def deduplicate(self, articles: list[RawArticle]) -> list[RawArticle]:
        """Remove duplicate articles. Returns deduplicated list."""
        after_url = self._dedup_by_url(articles)
        logger.info(
            "URL dedup: %d -> %d articles", len(articles), len(after_url)
        )
        after_semantic = self._dedup_by_headline(after_url)
        logger.info(
            "Semantic dedup: %d -> %d articles", len(after_url), len(after_semantic)
        )
        return after_semantic

    def _threshold_for(self, region: str) -> float:
        """Return the within-source dedup threshold for a given region string.

        Maps region -> tier -> threshold. Falls back to global threshold if
        the region or tier is not in headline_similarity_threshold_by_tier.

        Region-to-tier mapping mirrors clustering/clusterer.py _tier():
          national        -> national
          north_carolina  -> state
          international   -> national (safe fallback; shouldn't reach dedup)
          anything else   -> local
        """
        if region == "national":
            tier = "national"
        elif region == "north_carolina":
            tier = "state"
        else:
            tier = "local"
        return self._tier_thresholds.get(tier, self._global_threshold)

    def _dedup_by_url(self, articles: list[RawArticle]) -> list[RawArticle]:
        seen: set[str] = set()
        unique: list[RawArticle] = []
        for article in articles:
            if article.url_hash not in seen:
                seen.add(article.url_hash)
                unique.append(article)
        return unique

    def _dedup_by_headline(self, articles: list[RawArticle]) -> list[RawArticle]:
        if len(articles) < 2:
            return articles

        model = get_model(self.settings["clustering"]["model"])
        headlines = [a.headline for a in articles]
        embeddings = model.encode(
            headlines, convert_to_tensor=True, show_progress_bar=False
        )

        top_k = min(len(articles), 50)
        hits = util.semantic_search(embeddings, embeddings, top_k=top_k)

        keep: list[bool] = [True] * len(articles)

        for i, article_hits in enumerate(hits):
            if not keep[i]:
                continue
            for hit in article_hits:
                j = hit["corpus_id"]
                if j <= i or not keep[j]:
                    continue
                sim = hit["score"]
                same_source = articles[i].source_name == articles[j].source_name

                if same_source:
                    # Use per-tier threshold for within-source dedup.
                    threshold = self._threshold_for(articles[i].region)
                    if sim >= threshold:
                        keep[j] = False
                        logger.debug(
                            "Same-source duplicate removed: '%s' (sim=%.3f, "
                            "tier=%s, threshold=%.2f)",
                            articles[j].headline, sim,
                            articles[i].region, threshold,
                        )
                    continue

                # Cross-source wire syndication: very high threshold only
                if sim >= self._wire_threshold:
                    keep[j] = False
                    logger.debug(
                        "Wire syndication duplicate removed: '%s' [%s] (sim=%.3f)",
                        articles[j].headline, articles[j].source_name, sim,
                    )

        return [a for a, k in zip(articles, keep) if k]

    def close(self) -> None:
        """Lifecycle hook for cleanup. Currently a no-op."""
