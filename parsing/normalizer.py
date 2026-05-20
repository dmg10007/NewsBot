"""Entity and date normalization across sources.

Ensures that the same real-world entity referred to differently across
outlets ("GOP", "Republican Party", "Republicans") maps to a single
canonical form for accurate cross-source comparison.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone, timedelta
from typing import Optional

from parsing.extractor import ParsedArticle

# Canonical entity map — maps surface form variants to a single canonical string.
# Expand this over time as new aliases are observed in the wild.
_ENTITY_ALIASES: dict[str, str] = {
    # Parties
    "gop": "Republican Party",
    "republicans": "Republican Party",
    "the republican party": "Republican Party",
    "democrats": "Democratic Party",
    "the democratic party": "Democratic Party",
    "dems": "Democratic Party",
    # Executive
    "the white house": "White House",
    "the president": "President of the United States",
    "potus": "President of the United States",
    # Legislature
    "the senate": "U.S. Senate",
    "the house": "U.S. House of Representatives",
    "congress": "U.S. Congress",
    "capitol hill": "U.S. Congress",
    # Federal agencies
    "the fed": "Federal Reserve",
    "federal reserve": "Federal Reserve",
    "doj": "Department of Justice",
    "fbi": "Federal Bureau of Investigation",
    "dhs": "Department of Homeland Security",
    "irs": "Internal Revenue Service",
    # NC-specific
    "nc": "North Carolina",
    "n.c.": "North Carolina",
    "the general assembly": "NC General Assembly",
    "raleigh": "Raleigh, NC",
    "sanford": "Sanford, NC",
    "lee county": "Lee County, NC",
}


class Normalizer:
    """Normalizes entities and dates in ParsedArticles."""

    def normalize_all(self, articles: list[ParsedArticle]) -> list[ParsedArticle]:
        for article in articles:
            article.entities = self._normalize_entities(article.entities)
            if article.raw.published_at:
                article.raw.published_at = self._normalize_date(article.raw.published_at)
        return articles

    def _normalize_entities(self, entities: list[tuple[str, str]]) -> list[tuple[str, str]]:
        normalized = []
        seen: set[str] = set()
        for text, label in entities:
            canonical = _ENTITY_ALIASES.get(text.lower(), text)
            key = (canonical.lower(), label)
            if key not in seen:
                seen.add(key)
                normalized.append((canonical, label))
        return normalized

    def _normalize_date(self, dt: datetime) -> datetime:
        """Ensure timezone-aware UTC datetime."""
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
