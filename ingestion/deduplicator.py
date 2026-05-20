"""Deduplication of raw articles before NLP processing.

Two-pass strategy:
1. Exact URL hash dedup (fast, catches reposts of the same URL)
2. Headline semantic similarity dedup (catches rewrites of the same story
   within the same source tier)
"""

from __future__ import annotations

import logging
from collections import defaultdict

from sentence_transformers import SentenceTransformer, util

from config.loader import get_settings
from ingestion.fetcher import RawArticle

logger = logging.getLogger(__name__)


class Deduplicator:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._threshold: float = self.settings["deduplication"]["headline_similarity_threshold"]
        self._model: SentenceTransformer | None = None  # Lazy-loaded

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

        model = self._get_model()
        headlines = [a.headline for a in articles]
        embeddings = model.encode(headlines, convert_to_tensor=True, show_progress_bar=False)
        
        keep: list[bool] = [True] * len(articles)
        for i in range(len(articles)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(articles)):
                if not keep[j]:
                    continue
                # Only dedup within same source — cross-source duplicates
                # are actually useful for clustering and corroboration scoring
                if articles[i].source_name == articles[j].source_name:
                    sim = float(util.cos_sim(embeddings[i], embeddings[j]))
                    if sim >= self._threshold:
                        keep[j] = False
                        logger.debug(
                            "Duplicate headline removed: '%s' (sim=%.3f)",
                            articles[j].headline, sim
                        )

        return [a for a, k in zip(articles, keep) if k]

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            model_name = self.settings["clustering"]["model"]
            logger.info("Loading sentence-transformer model: %s", model_name)
            self._model = SentenceTransformer(model_name)
        return self._model
