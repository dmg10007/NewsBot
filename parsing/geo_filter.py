"""Geographic filter: drops articles with no detectable US content signal.

Runs after ArticleExtractor, before StoryClusterer. Articles tagged
geo_tier='international' are removed from the pipeline entirely.

Approach
--------
Two-signal check:
  1. spaCy GPE/NORP entities — any US place name or demonym triggers pass
  2. Dateline pattern — "CITY, State —" or "WASHINGTON" at article start

An article passes if EITHER signal fires. This is intentionally permissive:
we'd rather let one genuine international story slip through than drop a
domestic story about US foreign policy (e.g. "US imposes tariffs on China").

Articles that fail both checks are logged at DEBUG level and removed.
The filter logs a summary count at INFO level for monitoring.

Geo_tier field
--------------
RawArticle.geo_tier is written to 'domestic' on pass or 'international'
on fail. Downstream stages (scorer, renderer) can use this field for
additional weighting but the filter itself is the enforcement gate —
anything tagged 'international' never reaches the clusterer.

Limitations
-----------
- Relies on spaCy entity recognition: en_core_web_sm has ~85% NER F1.
  Stories about foreign events affecting US interests may be incorrectly
  dropped if the US entity appears only late in the body (not in headline
  or summary lead). Tune _MIN_US_ENTITY_HITS if over-filtering occurs.
- The US_SIGNALS lexicon is US-English centric. Non-English content from
  US sources will likely be dropped — acceptable given the bot's scope.
- Does not replace feed-level geo scoping in sources.yaml. Both defences
  should be active: sources.yaml reduces international volume at ingest;
  GeoFilter catches what leaks through.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from parsing.extractor import ParsedArticle

logger = logging.getLogger(__name__)

# --- US geographic signals ------------------------------------------------
# State names, major cities, US-specific institutions and demonyms.
# Lowercase — matched against lowercased text.
_US_GPE_SIGNALS: frozenset[str] = frozenset({
    # States (full names)
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
    "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana",
    "maine", "maryland", "massachusetts", "michigan", "minnesota",
    "mississippi", "missouri", "montana", "nebraska", "nevada",
    "new hampshire", "new jersey", "new mexico", "new york", "north carolina",
    "north dakota", "ohio", "oklahoma", "oregon", "pennsylvania",
    "rhode island", "south carolina", "south dakota", "tennessee", "texas",
    "utah", "vermont", "virginia", "washington", "west virginia",
    "wisconsin", "wyoming",
    # State abbreviations as standalone tokens (spaCy tags these as GPE)
    "n.c.", "s.c.", "n.y.", "l.a.", "d.c.",
    # US territories
    "puerto rico", "guam", "washington d.c.", "washington, d.c.",
    # Major cities
    "new york city", "los angeles", "chicago", "houston", "phoenix",
    "philadelphia", "san antonio", "san diego", "dallas", "san jose",
    "austin", "jacksonville", "san francisco", "columbus", "charlotte",
    "indianapolis", "seattle", "denver", "nashville", "boston",
    "raleigh", "durham", "greensboro", "winston-salem", "fayetteville",
    "cary", "wilmington", "high point", "concord", "gastonia", "sanford",
    # US institutions / demonyms
    "united states", "u.s.", "us ", "american", "americans",
    "congress", "senate", "white house", "pentagon", "capitol hill",
    "federal reserve", "supreme court", "u.s. house", "u.s. senate",
    "department of", "trump", "biden",
})

# Regex for common US dateline patterns at the start of a summary
_DATELINE_RE = re.compile(
    r"^(washington|new york|los angeles|chicago|houston|atlanta|boston|"
    r"dallas|denver|miami|seattle|philadelphia|phoenix|san francisco|"
    r"raleigh|charlotte|durham|fayetteville|sanford)"
    r"\s*[,—\-]",
    re.IGNORECASE,
)

# Minimum number of distinct US entity hits required from spaCy GPE/NORP
# entities to pass without a dateline match. Set to 1 to be permissive.
_MIN_US_ENTITY_HITS = 1


class GeoFilter:
    """Filters ParsedArticles to US-relevant content only.

    Call filter() after ArticleExtractor.extract_all(). Returns only articles
    that pass the US signal check. Writes geo_tier to each RawArticle.
    """

    def filter(self, articles: list["ParsedArticle"]) -> list["ParsedArticle"]:
        """Return articles that pass the US geographic signal check.

        Args:
            articles: Extracted ParsedArticle objects from ArticleExtractor.

        Returns:
            Filtered list — international articles removed.
        """
        passed: list[ParsedArticle] = []
        dropped_count = 0

        for article in articles:
            if self._is_domestic(article):
                article.raw.geo_tier = "domestic"
                passed.append(article)
            else:
                article.raw.geo_tier = "international"
                dropped_count += 1
                logger.debug(
                    "[GeoFilter] DROPPED (international): %s",
                    article.raw.headline,
                )

        logger.info(
            "[GeoFilter] %d/%d articles passed (dropped %d international)",
            len(passed),
            len(articles),
            dropped_count,
        )
        return passed

    def _is_domestic(self, article: "ParsedArticle") -> bool:
        """Return True if the article has at least one US geographic signal."""
        # Signal 1: spaCy GPE/NORP entities overlapping US signals lexicon
        us_entity_hits = sum(
            1
            for text, label in article.entities
            if label in ("GPE", "NORP", "ORG")
            and self._matches_us_signal(text)
        )
        if us_entity_hits >= _MIN_US_ENTITY_HITS:
            return True

        # Signal 2: dateline pattern in summary lead
        summary_lead = (article.raw.summary or "")[:200]
        if _DATELINE_RE.match(summary_lead.strip()):
            return True

        # Signal 3: explicit US keywords in headline (belt-and-suspenders)
        headline_lower = (article.raw.headline or "").lower()
        if any(sig in headline_lower for sig in (
            "u.s.", "united states", "american", "congress", "white house",
            "trump", "biden", "senate", "federal",
        )):
            return True

        return False

    @staticmethod
    def _matches_us_signal(entity_text: str) -> bool:
        """Check if an entity text overlaps with the US signals lexicon."""
        text_lower = entity_text.lower().strip()
        # Direct match
        if text_lower in _US_GPE_SIGNALS:
            return True
        # Substring match for multi-word signals (e.g. "North Carolina" in entity)
        return any(sig in text_lower or text_lower in sig for sig in _US_GPE_SIGNALS)
