"""Story-level geographic classification and filtering.

Replaces source-level region assignment as the primary tier signal. The
classifier runs a fast keyword scan over each article's headline and any
available topic tags, then assigns one of four tiers:

  local       — story is specifically about Lee County / Sanford / Harnett
  state       — story is specifically about North Carolina
  national    — story is US-level with no strong local/state signal
  international — story is primarily about a foreign country or region

Geo signals take precedence in the order above (local > state > national >
international). If no signal is found in the headline, the classifier falls
back to the article’s source-level region field.

Filtering
---------
When geo_filter.exclude_international is true in settings.yaml (default:
true), stories classified as international are dropped from the pipeline
before clustering. This is the primary mechanism for suppressing the
international news that was leaking through via national wire sources.

Keywords are configured in settings.yaml under geo_filter.keywords so
operators can tune them without touching Python.

Design note
-----------
This is intentionally a fast, zero-cost keyword pass — no embeddings, no
LLM calls. Precision is more important than recall here: a false
classification that drops a legitimate local story is worse than one that
lets a borderline national story through. When in doubt, the classifier
defaults to the source-level region.
"""

from __future__ import annotations

import logging
import re
from typing import Literal

from config.loader import get_settings
from parsing.extractor import ParsedArticle

logger = logging.getLogger(__name__)

GeoTier = Literal["local", "state", "national", "international"]

# Default keyword lists — overridden by settings.yaml geo_filter.keywords
_DEFAULT_LOCAL_KEYWORDS: list[str] = [
    "lee county", "sanford", "harnett", "broadway nc", "angier", "fuquay",
    "chatham county",
]
_DEFAULT_STATE_KEYWORDS: list[str] = [
    "north carolina", " nc ", "\bnc\b", "raleigh", "charlotte", "durham",
    "chapel hill", "wilmington nc", "asheville", "greensboro", "winston-salem",
    "fayetteville nc", "cary nc",
]
# International signals: if headline contains any of these AND no US signal
# is present, classify as international.
_DEFAULT_INTERNATIONAL_KEYWORDS: list[str] = [
    "ukraine", "russia", "china", "beijing", "moscow", "london", "paris",
    "berlin", "tokyo", "seoul", "israel", "gaza", "iran", "iraq", "syria",
    "india", "pakistan", "brazil", "mexico", "canada", "australia",
    "european union", "nato", "united nations", "un security council",
    "kremlin", "white house press",  # "white house" alone is too broad
    "parliament", "prime minister", "chancellor",
]


class GeoFilter:
    """Classifies and optionally filters articles by geographic relevance."""

    def __init__(self) -> None:
        settings = get_settings()
        geo_cfg = settings.get("geo_filter", {})
        kw = geo_cfg.get("keywords", {})

        self._local_kw: list[str] = kw.get("local", _DEFAULT_LOCAL_KEYWORDS)
        self._state_kw: list[str] = kw.get("state", _DEFAULT_STATE_KEYWORDS)
        self._intl_kw: list[str] = kw.get("international", _DEFAULT_INTERNATIONAL_KEYWORDS)
        self._exclude_international: bool = geo_cfg.get("exclude_international", True)

    def classify(self, article: ParsedArticle) -> GeoTier:
        """Return the geo tier for a single article."""
        text = (article.raw.headline or "").lower()
        if article.raw.tags:
            text += " " + " ".join(article.raw.tags).lower()

        if self._matches(text, self._local_kw):
            return "local"
        if self._matches(text, self._state_kw):
            return "state"
        if self._matches(text, self._intl_kw):
            return "international"

        # No geographic signal in headline — fall back to source-level region
        region = getattr(article.raw, "region", "national") or "national"
        if region in ("local", "state", "national"):
            return region  # type: ignore[return-value]
        return "national"

    def filter(self, articles: list[ParsedArticle]) -> list[ParsedArticle]:
        """Classify all articles, attach geo_tier, and optionally drop international.

        Sets article.raw.geo_tier on every article as a side-effect so
        downstream modules (clusterer, renderer) can read the tier without
        re-running classification.
        """
        kept: list[ParsedArticle] = []
        dropped = 0
        for article in articles:
            tier = self.classify(article)
            # Attach tier to the raw article object for downstream use
            article.raw.geo_tier = tier  # type: ignore[attr-defined]
            if tier == "international" and self._exclude_international:
                dropped += 1
                logger.debug(
                    "Dropped international story: %s", article.raw.headline
                )
                continue
            kept.append(article)

        if dropped:
            logger.info(
                "GeoFilter: dropped %d international stories, kept %d",
                dropped, len(kept),
            )
        return kept

    @staticmethod
    def _matches(text: str, keywords: list[str]) -> bool:
        """Return True if any keyword is found as a word-boundary match in text."""
        for kw in keywords:
            # Use word-boundary regex only for short tokens to avoid false positives
            # e.g. " nc " in "fence" without boundaries
            if len(kw) <= 3:
                if re.search(rf"\b{re.escape(kw)}\b", text):
                    return True
            else:
                if kw in text:
                    return True
        return False
