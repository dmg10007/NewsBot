"""NLP extraction: named entities, claims, topics, and sentiment from raw articles.

Uses spaCy for entity extraction and VADER for sentiment scoring.
Outputs a ParsedArticle that augments RawArticle with NLP fields.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from ingestion.fetcher import RawArticle

logger = logging.getLogger(__name__)

# Loaded word / loaded phrase lexicon — emotionally charged terms
# used for heuristic bias flagging in the bias layer.
# Extend this list as you discover patterns in digest output.
LOADED_WORD_LEXICON: set[str] = {
    # Fear / threat framing
    "invasion", "infest", "plague", "crisis", "chaos", "catastrophe",
    "disaster", "threat", "danger", "alarming", "shocking", "outrage",
    # Positive spin
    "triumph", "historic", "landmark", "breakthrough", "unprecedented",
    "hero", "champion", "savior",
    # Dismissive framing
    "so-called", "alleged", "claims", "radical", "extreme", "fringe",
    "conspiracy", "hoax", "fake",
    # Loaded identity terms (context-dependent — flagged for LLM review)
    "regime", "mob", "thugs", "illegals", "elites", "globalists",
    # Economic framing
    "explode", "skyrocket", "plummet", "collapse", "surge", "soar",
    "tank", "wreck", "burden", "drain",
}


@dataclass
class ParsedArticle:
    """RawArticle enriched with NLP extraction results."""
    raw: RawArticle

    # spaCy entities: list of (text, label) tuples
    # e.g. [("Joe Biden", "PERSON"), ("North Carolina", "GPE")]
    entities: list[tuple[str, str]] = field(default_factory=list)

    # VADER compound score: -1.0 (most negative) to +1.0 (most positive)
    sentiment_score: float = 0.0

    # Loaded words found in headline + summary
    loaded_words_found: list[str] = field(default_factory=list)

    # Detected topic (overrides source-level topic if NLP is more specific)
    detected_topic: Optional[str] = None

    # Embedding placeholder — populated by the clustering layer
    embedding: Optional[list[float]] = None


class ArticleExtractor:
    """Runs NLP extraction on a list of RawArticles."""

    def __init__(self) -> None:
        self._nlp: Optional[spacy.language.Language] = None
        self._vader = SentimentIntensityAnalyzer()

    def extract_all(self, articles: list[RawArticle]) -> list[ParsedArticle]:
        nlp = self._get_nlp()
        parsed: list[ParsedArticle] = []
        for article in articles:
            try:
                parsed.append(self._extract(article, nlp))
            except Exception as exc:
                logger.warning("Extraction failed for '%s': %s", article.headline, exc)
        logger.info("Extracted NLP data for %d articles", len(parsed))
        return parsed

    def _extract(self, article: RawArticle, nlp: spacy.language.Language) -> ParsedArticle:
        text = f"{article.headline}. {article.summary}"

        # Named entity recognition
        doc = nlp(text[:1000])  # Cap at 1000 chars for speed
        entities = [(ent.text, ent.label_) for ent in doc.ents]

        # Sentiment
        scores = self._vader.polarity_scores(text)
        sentiment_score = scores["compound"]

        # Loaded word scan
        text_lower = text.lower()
        loaded = [w for w in LOADED_WORD_LEXICON if w in text_lower]

        # Topic detection — keyword-based override
        detected_topic = self._detect_topic(text_lower, article.topics)

        return ParsedArticle(
            raw=article,
            entities=entities,
            sentiment_score=sentiment_score,
            loaded_words_found=loaded,
            detected_topic=detected_topic,
        )

    def _detect_topic(
        self, text_lower: str, source_topics: list[str]
    ) -> Optional[str]:
        """Simple keyword-based topic classifier. Returns most specific match."""
        TOPIC_KEYWORDS: dict[str, list[str]] = {
            "politics": [
                "congress", "senate", "house", "president", "governor",
                "election", "vote", "legislation", "bill", "policy",
                "democrat", "republican", "trump", "white house", "legislature",
                "general assembly", "mayor", "county commissioner",
            ],
            "economy": [
                "inflation", "gdp", "unemployment", "fed", "interest rate",
                "stock", "market", "tariff", "trade", "budget", "deficit",
                "jobs", "wage", "recession", "economy", "economic",
                "housing", "mortgage", "price", "cost of living",
            ],
            "current_events": [
                "weather", "hurricane", "tornado", "flood", "storm",
                "crime", "arrest", "shooting", "fire", "accident",
                "school", "hospital", "community", "local", "county",
                "road", "infrastructure", "development",
            ],
        }
        for topic, keywords in TOPIC_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                return topic
        # Fall back to first source-declared topic
        return source_topics[0] if source_topics else None

    def _get_nlp(self) -> spacy.language.Language:
        if self._nlp is None:
            try:
                self._nlp = spacy.load("en_core_web_sm")
                logger.info("Loaded spaCy model: en_core_web_sm")
            except OSError:
                logger.error(
                    "spaCy model not found. Run: python -m spacy download en_core_web_sm"
                )
                raise
        return self._nlp
