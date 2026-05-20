"""Normalization of parsed articles before clustering.

Handles:
- Canonical entity name resolution (e.g., 'Joe Biden' / 'Biden' -> 'Joe Biden')
- Date normalization to UTC
- Headline cleanup (strip HTML entities, excessive whitespace)
- Deduplication of entity lists per article
"""

from __future__ import annotations

import html
import logging
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Optional

from parsing.extractor import ParsedArticle

logger = logging.getLogger(__name__)


class ArticleNormalizer:
    """Normalizes a list of ParsedArticles for downstream consistency."""

    def normalize_all(self, articles: list[ParsedArticle]) -> list[ParsedArticle]:
        normalized = [self._normalize(a) for a in articles]
        logger.info("Normalized %d articles", len(normalized))
        return normalized

    def _normalize(self, article: ParsedArticle) -> ParsedArticle:
        raw = article.raw

        # Clean headline and summary
        raw.headline = _clean_text(raw.headline)
        raw.summary = _clean_text(raw.summary)

        # Normalize published_at to UTC if naive
        if raw.published_at and raw.published_at.tzinfo is None:
            raw.published_at = raw.published_at.replace(tzinfo=timezone.utc)

        # Deduplicate entities, keep most-frequent canonical form
        article.entities = _canonical_entities(article.entities)

        return article


def _clean_text(text: str) -> str:
    """Strip HTML entities, tags, and normalize whitespace."""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", "", text)       # Strip HTML tags
    text = re.sub(r"\s+", " ", text).strip()  # Normalize whitespace
    return text


def _canonical_entities(
    entities: list[tuple[str, str]]
) -> list[tuple[str, str]]:
    """Deduplicate entities, prefer the longest form of the same name.

    e.g. [("Biden", "PERSON"), ("Joe Biden", "PERSON")] -> [("Joe Biden", "PERSON")]
    """
    if not entities:
        return []

    # Group by label
    by_label: dict[str, list[str]] = {}
    for text, label in entities:
        by_label.setdefault(label, []).append(text)

    canonical: list[tuple[str, str]] = []
    for label, names in by_label.items():
        # For each cluster of names that share a token, keep the longest
        kept: list[str] = []
        for name in sorted(set(names), key=len, reverse=True):
            # Only keep if not already a substring of a kept name
            if not any(name.lower() in k.lower() for k in kept):
                kept.append(name)
        for name in kept:
            canonical.append((name, label))

    return canonical
