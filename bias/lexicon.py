"""Stage 1 bias detection: loaded word lexicon and sentiment heuristics.

This runs on every cluster — fast, free, no API calls.
Output feeds into the escalation decision for LLM analysis.

Detects:
  - Loaded/charged language (words with strong connotative weight)
  - Sentiment asymmetry across sources covering the same story
  - Framing signal words (hedge words, attribution language)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from clustering.clusterer import StoryCluster
from parsing.extractor import ParsedArticle

logger = logging.getLogger(__name__)

# Loaded word lists — words that carry strong ideological or emotional charge
# These are not inherently wrong to use, but their presence across a cluster
# warrants deeper analysis. Sourced from journalism bias research.
_LOADED_WORDS: set[str] = {
    # Politically charged
    "radical", "extremist", "socialist", "communist", "fascist", "globalist",
    "elites", "regime", "propaganda", "indoctrination", "woke", "agenda",
    "invasion", "crisis", "catastrophe", "disaster", "collapse", "chaos",
    "corrupt", "rigged", "stolen", "illegitimate", "fraud",
    # Immigration framing
    "illegal alien", "illegal immigrant", "undocumented", "migrant", "refugee",
    "border crisis", "open border",
    # Economic framing
    "job-killing", "tax hike", "handout", "entitlement", "bailout", "bloated",
    "burden", "wasteful", "irresponsible",
    # Emotional loading
    "horrific", "outrageous", "shameful", "disgusting", "dangerous", "alarming",
    "shocking", "stunning", "explosive", "bombshell",
}

# Hedge / epistemic markers — signal uncertain or attributed claims
_HEDGE_WORDS: set[str] = {
    "allegedly", "reportedly", "claims", "said to", "sources say",
    "according to", "unverified", "disputed", "questioned", "appears to",
    "may have", "could be", "might be", "is believed to",
}


@dataclass
class LexiconResult:
    cluster_id: int
    loaded_words_found: dict[str, list[str]]   # source_name -> [loaded words found]
    sentiment_variance: float                   # Variance of compound scores across sources
    sentiment_by_source: dict[str, float]       # source_name -> compound score
    hedge_words_found: dict[str, list[str]]     # source_name -> [hedge words found]
    escalate: bool                              # True if LLM analysis is warranted
    escalation_reasons: list[str]


class LexiconAnalyzer:
    """Runs fast lexicon and sentiment analysis on a StoryCluster."""

    def __init__(self, escalation_threshold: float = 0.35) -> None:
        self.escalation_threshold = escalation_threshold

    def analyze(self, cluster: StoryCluster) -> LexiconResult:
        loaded_found: dict[str, list[str]] = {}
        hedge_found: dict[str, list[str]] = {}
        sentiment_by_source: dict[str, float] = {}

        for article in cluster.articles:
            source = article.raw.source_name
            text_lower = article.full_text.lower()

            loaded_found[source] = [
                w for w in _LOADED_WORDS if w in text_lower
            ]
            hedge_found[source] = [
                w for w in _HEDGE_WORDS if w in text_lower
            ]
            sentiment_by_source[source] = article.sentiment_compound

        scores = list(sentiment_by_source.values())
        variance = float(_variance(scores)) if len(scores) > 1 else 0.0

        escalation_reasons: list[str] = []
        if variance > self.escalation_threshold:
            escalation_reasons.append(
                f"Sentiment variance {variance:.3f} exceeds threshold {self.escalation_threshold}"
            )
        any_loaded = any(v for v in loaded_found.values())
        if any_loaded:
            escalation_reasons.append("Loaded language detected in one or more sources")
        if cluster.has_cross_lean_coverage:
            escalation_reasons.append("Cross-lean coverage detected — framing comparison warranted")

        escalate = bool(escalation_reasons)

        return LexiconResult(
            cluster_id=cluster.cluster_id,
            loaded_words_found=loaded_found,
            sentiment_variance=variance,
            sentiment_by_source=sentiment_by_source,
            hedge_words_found=hedge_found,
            escalate=escalate,
            escalation_reasons=escalation_reasons,
        )


def _variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)
