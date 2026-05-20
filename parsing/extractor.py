"""NLP extraction: named entities, claims, keywords, and topic classification.

Uses spaCy for entity recognition and keyword extraction.
Topic classification uses a lightweight keyword-matching approach
before any LLM call is made — keeping this stage fast and free.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from ingestion.fetcher import RawArticle

logger = logging.getLogger(__name__)

# Load once at module level — spaCy model load is expensive
try:
    _NLP = spacy.load("en_core_web_sm")
except OSError:
    logger.warning(
        "spaCy model 'en_core_web_sm' not found. "
        "Run: python -m spacy download en_core_web_sm"
    )
    _NLP = None

_VADER = SentimentIntensityAnalyzer()

# Lightweight topic keyword map — used before LLM escalation
_TOPIC_KEYWORDS: dict[str, list[str]] = {
    "politics": [
        "congress", "senate", "president", "white house", "democrat", "republican",
        "election", "vote", "legislation", "governor", "bill", "law", "policy",
        "supreme court", "general assembly", "legislature", "campaign", "trump",
        "biden", "political", "government", "federal", "state government",
    ],
    "economy": [
        "inflation", "gdp", "unemployment", "jobs", "market", "stock", "fed",
        "interest rate", "tariff", "trade", "economy", "economic", "recession",
        "budget", "deficit", "debt", "tax", "treasury", "wages", "cost of living",
        "housing", "mortgage", "bank", "finance", "fiscal",
    ],
    "current_events": [
        "weather", "storm", "hurricane", "flood", "crime", "arrest", "shooting",
        "accident", "fire", "emergency", "health", "hospital", "school", "community",
        "local", "county", "city", "town", "public safety", "obituary", "sports",
        "development", "construction", "environment",
    ],
}


@dataclass
class ParsedArticle:
    """An article after NLP processing, ready for clustering and bias analysis."""
    raw: RawArticle
    entities: list[tuple[str, str]]   # (text, label) e.g. ("NC", "GPE")
    keywords: list[str]
    detected_topics: list[str]        # Intersection of source topics + keyword match
    sentiment_compound: float         # VADER compound score: -1.0 to 1.0
    sentiment_label: str              # positive | negative | neutral
    word_count: int
    full_text: str                    # headline + summary concatenated for embedding


class ArticleExtractor:
    """Runs NLP extraction on a list of RawArticles."""

    def extract_all(self, articles: list[RawArticle]) -> list[ParsedArticle]:
        results = []
        for article in articles:
            try:
                results.append(self._extract(article))
            except Exception as exc:
                logger.error("Extraction failed for '%s': %s", article.headline, exc)
        logger.info("Extracted %d articles", len(results))
        return results

    def _extract(self, article: RawArticle) -> ParsedArticle:
        full_text = f"{article.headline}. {article.summary}".strip()
        entities: list[tuple[str, str]] = []
        keywords: list[str] = []

        if _NLP is not None:
            doc = _NLP(full_text[:1000])  # Cap to avoid slow processing on long summaries
            entities = [
                (ent.text.strip(), ent.label_)
                for ent in doc.ents
                if ent.label_ in {
                    "PERSON", "ORG", "GPE", "LOC", "EVENT",
                    "LAW", "NORP", "FAC", "PRODUCT"
                }
            ]
            keywords = [
                token.lemma_.lower()
                for token in doc
                if not token.is_stop
                and not token.is_punct
                and token.is_alpha
                and len(token.text) > 3
            ]

        sentiment = _VADER.polarity_scores(full_text)
        compound = sentiment["compound"]
        if compound >= 0.05:
            sentiment_label = "positive"
        elif compound <= -0.05:
            sentiment_label = "negative"
        else:
            sentiment_label = "neutral"

        detected_topics = self._classify_topics(full_text, article.topics)

        return ParsedArticle(
            raw=article,
            entities=entities,
            keywords=keywords,
            detected_topics=detected_topics,
            sentiment_compound=compound,
            sentiment_label=sentiment_label,
            word_count=len(full_text.split()),
            full_text=full_text,
        )

    def _classify_topics(self, text: str, source_topics: list[str]) -> list[str]:
        """Intersect source-declared topics with keyword evidence from the text."""
        text_lower = text.lower()
        matched: list[str] = []
        for topic, keywords in _TOPIC_KEYWORDS.items():
            if topic not in source_topics:
                continue
            if any(kw in text_lower for kw in keywords):
                matched.append(topic)
        # Fall back to source-declared topics if no keywords match (short summaries)
        return matched if matched else source_topics
