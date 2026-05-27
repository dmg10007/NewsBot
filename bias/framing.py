"""Stage 2 bias detection: cross-source entity and framing comparison.

Compares how different outlets frame the same story:
  - Entity label differences (same person/place described differently)
  - Omission detection (entities present in some sources but absent in others)
  - Attribution asymmetry (one source hedges a claim, another states it as fact)

This runs only on clusters flagged for escalation by lexicon.py.
Output is passed to llm_analyzer.py for final LLM-assisted analysis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from collections import Counter

from bias.lexicon import _HEDGE_WORDS
from clustering.clusterer import StoryCluster
from parsing.extractor import ParsedArticle

logger = logging.getLogger(__name__)


@dataclass
class FramingResult:
    cluster_id: int
    entity_omissions: list[str]           # Entities mentioned in >50% of sources but absent in some
    framing_differences: list[str]        # Human-readable descriptions of detected framing gaps
    attribution_asymmetry: list[str]      # Sources that hedge where others state as fact
    cross_source_summary: str             # Short text description for LLM prompt context


class FramingAnalyzer:
    """Detects framing differences and omissions across sources in a cluster."""

    def analyze(self, cluster: StoryCluster) -> FramingResult:
        articles = cluster.articles
        n = len(articles)

        # Entity omission: entities appearing in majority of sources but not all
        entity_counter: Counter[str] = Counter()
        entity_by_source: dict[str, set[str]] = {}
        for article in articles:
            source = article.raw.source_name
            entity_by_source[source] = {e[0] for e in article.entities}
            for entity_text, _ in article.entities:
                entity_counter[entity_text] += 1

        majority_threshold = max(2, int(n * 0.5))
        majority_entities = {
            e for e, count in entity_counter.items() if count >= majority_threshold
        }
        omissions: list[str] = []
        for entity in majority_entities:
            missing_in = [
                source for source, entities in entity_by_source.items()
                if entity not in entities
            ]
            if missing_in:
                omissions.append(
                    f"'{entity}' mentioned in {entity_counter[entity]}/{n} sources "
                    f"but absent in: {', '.join(missing_in)}"
                )

        # Attribution asymmetry: hedge words present in some sources but not others
        hedge_by_source: dict[str, bool] = {}
        for article in articles:
            text_lower = article.full_text.lower()
            hedge_by_source[article.raw.source_name] = any(
                w in text_lower for w in _HEDGE_WORDS
            )
        hedgers = [s for s, h in hedge_by_source.items() if h]
        asserters = [s for s, h in hedge_by_source.items() if not h]
        attribution_asymmetry: list[str] = []
        if hedgers and asserters and n > 1:
            attribution_asymmetry.append(
                f"Sources using hedge language: {', '.join(hedgers)}. "
                f"Sources stating as fact: {', '.join(asserters)}."
            )

        # Framing differences: bias-lean divergence on same story
        framing_differences: list[str] = []
        if cluster.has_cross_lean_coverage:
            left_sources = [
                a.raw.source_name for a in articles
                if a.raw.bias_lean in ("left", "center-left")
            ]
            right_sources = [
                a.raw.source_name for a in articles
                if a.raw.bias_lean in ("right", "center-right")
            ]
            if left_sources and right_sources:
                framing_differences.append(
                    f"Left-leaning coverage: {', '.join(left_sources)}. "
                    f"Right-leaning coverage: {', '.join(right_sources)}."
                )

        cross_source_summary = self._build_summary(cluster, omissions, attribution_asymmetry)

        return FramingResult(
            cluster_id=cluster.cluster_id,
            entity_omissions=omissions,
            framing_differences=framing_differences,
            attribution_asymmetry=attribution_asymmetry,
            cross_source_summary=cross_source_summary,
        )

    def _build_summary(self, cluster: StoryCluster, omissions: list[str], asymmetry: list[str]) -> str:
        lines = [
            f"Story: {cluster.representative_headline}",
            f"Sources ({cluster.source_count}): " +
            ", ".join(a.raw.source_name for a in cluster.articles),
        ]
        if omissions:
            lines.append("Entity omissions: " + "; ".join(omissions[:3]))
        if asymmetry:
            lines.append("Attribution: " + "; ".join(asymmetry))
        return "\n".join(lines)
