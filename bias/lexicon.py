"""Lexicon and sentiment heuristics — first pass of bias detection.

Runs on every cluster, cheaply and offline.
Flags clusters that warrant LLM escalation based on:
  - Loaded word density
  - Sentiment variance across sources
  - Presence of known framing patterns

Outputs a BiasSignal per cluster that the LLM analyzer uses to decide
whether to escalate.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field

from clustering.clusterer import StoryCluster
from config.loader import get_settings
from parsing.extractor import LOADED_WORD_LEXICON

logger = logging.getLogger(__name__)


@dataclass
class BiasSignal:
    """Heuristic bias indicators for a story cluster."""
    cluster_id: int
    loaded_word_hits: dict[str, list[str]] = field(default_factory=dict)
    # {source_name: [loaded_words_found]}
    sentiment_scores: dict[str, float] = field(default_factory=dict)
    # {source_name: compound_score}
    sentiment_variance: float = 0.0
    escalate_to_llm: bool = False
    escalation_reasons: list[str] = field(default_factory=list)


class LexiconAnalyzer:
    """Runs heuristic bias detection on all clusters."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._threshold: float = self.settings["bias_detection"][
            "llm_escalation_threshold"
        ]

    def analyze_all(self, clusters: list[StoryCluster]) -> list[BiasSignal]:
        signals = [self._analyze(cluster) for cluster in clusters]
        escalated = sum(1 for s in signals if s.escalate_to_llm)
        logger.info(
            "Lexicon analysis: %d clusters, %d flagged for LLM escalation",
            len(signals), escalated,
        )
        return signals

    def _analyze(self, cluster: StoryCluster) -> BiasSignal:
        signal = BiasSignal(cluster_id=cluster.cluster_id)

        for article in cluster.articles:
            name = article.raw.source_name
            signal.sentiment_scores[name] = article.sentiment_score
            signal.loaded_word_hits[name] = article.loaded_words_found

        # Sentiment variance across sources in this cluster
        scores = list(signal.sentiment_scores.values())
        if len(scores) >= 2:
            signal.sentiment_variance = statistics.variance(scores)
        elif len(scores) == 1:
            signal.sentiment_variance = 0.0

        # Escalation decision
        reasons: list[str] = []

        if signal.sentiment_variance > self._threshold:
            reasons.append(
                f"High sentiment variance across sources: {signal.sentiment_variance:.3f}"
            )

        total_loaded = sum(len(v) for v in signal.loaded_word_hits.values())
        if total_loaded >= 3:
            all_hits = [w for hits in signal.loaded_word_hits.values() for w in hits]
            reasons.append(f"Loaded language detected: {', '.join(set(all_hits))}")

        # Multi-source clusters with opposing sentiment polarity warrant review
        if len(scores) >= 2:
            has_positive = any(s > 0.2 for s in scores)
            has_negative = any(s < -0.2 for s in scores)
            if has_positive and has_negative:
                reasons.append("Opposing sentiment polarity across sources")

        signal.escalation_reasons = reasons
        signal.escalate_to_llm = len(reasons) > 0
        return signal
