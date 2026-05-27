"""Deduplication of raw articles before NLP processing.

Two-pass strategy:
1. Exact URL hash dedup (fast, catches reposts of the same URL)
2. Headline semantic similarity dedup using ANN (util.semantic_search)

Semantic dedup passes:
  Pass A — within-source: removes rewrites of the same story published
            multiple times by the same outlet.
  Pass B — cross-source wire detection: removes articles from different
            outlets that are near-identical (sim >= wire_syndication_threshold,
            default 0.95). Without this, syndicated AP/Reuters wire copy
            appears as independent corroboration in clustering, inflating
            the source_count and importance_score of those clusters.

Performance
-----------
The original implementation used an O(n²) double-loop calling cos_sim()
individually for every article pair. This version uses util.semantic_search()
— one vectorized matrix operation identical to the clustering approach —
which is dramatically faster for n > ~100 articles.

Lifecycle
---------
Deduplicator exposes a close() method as a forward-compatible lifecycle hook.
It is currently a no-op because the model reference is owned by model_registry
(not by Deduplicator), but callers (ingestion/pipeline.py) should still call
it in a finally block so that any future persistent state (e.g., a seen-URL
database) can be cleaned up without changing call sites.
"""

from __future__ import annotations

import logging

from sentence_transformers import util

from config.loader import get_settings
from ingestion.fetcher import RawArticle
from utils.model_registry import get_model

logger = logging.getLogger(__name__)

# High-confidence threshold for cross-source wire syndication detection.
# At 0.95+ the articles are effectively identical content; safe to collapse.
_WIRE_SYNDICATION_THRESHOLD = 0.95


class Deduplicator:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._threshold: float = self.settings["deduplication"]["headline_similarity_threshold"]
        self._wire_threshold: float = self.settings["deduplication"].get(
            "wire_syndication_threshold", _WIRE_SYNDICATION_THRESHOLD
        )

    def deduplicate(self, articles: list[RawArticle]) -> list[RawArticle]:
        """Remove duplicate articles. Returns deduplicated list."""
        after_url = self._dedup_by_url(articles)
        logger.info(
            "URL dedup: %d → %d articles", len(articles), len(after_url)
        )
        after_semantic = self._dedup_by_headline(after_url)
        logger.info(
            "Semantic dedup: %d → %d articles", len(after_url), len(after_semantic)
        )
        return after_semantic

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

        # ANN search: find the top-k most similar headlines for every article.
        # top_k capped at len(articles) so all pairs above threshold are visible.
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

                # Within-source dedup: threshold from settings (default ~0.95)
                if same_source and sim >= self._threshold:
                    keep[j] = False
                    logger.debug(
                        "Same-source duplicate removed: '%s' (sim=%.3f)",
                        articles[j].headline, sim,
                    )
                    continue

                # Cross-source wire syndication dedup: very high threshold (0.95+)
                # to only collapse truly identical content, not just similar stories.
                if not same_source and sim >= self._wire_threshold:
                    keep[j] = False
                    logger.debug(
                        "Wire syndication duplicate removed: '%s' [%s] (sim=%.3f)",
                        articles[j].headline, articles[j].source_name, sim,
                    )

        return [a for a, k in zip(articles, keep) if k]

    def close(self) -> None:
        """Lifecycle hook for cleanup. Currently a no-op.

        The model reference is owned by utils.model_registry, not by this
        class, so there is nothing to release here. This method exists as a
        forward-compatible hook: if Deduplicator ever acquires persistent state
        (e.g., a seen-URL SQLite store), callers do not need to change.
        """
