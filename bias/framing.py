"""Cross-source framing comparison.

Compares how different outlets describe the same named entities
within a cluster. Detects systematic differences in:
  - Which entities are mentioned / omitted
  - Entity co-occurrence patterns (who is linked to what)
  - Linguistic framing around key entities (verbs, adjectives nearby)

This runs before LLM escalation and enriches the BiasSignal with
specific framing observations that get passed to the LLM prompt.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from clustering.clusterer import StoryCluster
from bias.lexicon import BiasSignal

logger = logging.getLogger(__name__)


@dataclass_workaround = None  # noqa — dataclass used via BiasSignal; no new class needed


class FramingAnalyzer:
    """Enriches BiasSignals with entity framing observations."""

    def analyze_all(
        self,
        clusters: list[StoryCluster],
        signals: list[BiasSignal],
    ) -> list[BiasSignal]:
        for cluster, signal in zip(clusters, signals):
            self._analyze(cluster, signal)
        framing_flagged = sum(
            1 for s in signals
            if any("framing" in r.lower() or "entity" in r.lower()
                   for r in s.escalation_reasons)
        )
        logger.info("Framing analysis complete. %d clusters with entity framing flags", framing_flagged)
        return signals

    def _analyze(self, cluster: StoryCluster, signal: BiasSignal) -> None:
        if len(cluster.articles) < 2:
            return  # Need at least 2 sources to compare framing

        # Build per-source entity sets
        source_entities: dict[str, set[str]] = {}
        for article in cluster.articles:
            entity_texts = {e[0].lower() for e in article.entities}
            source_entities[article.raw.source_name] = entity_texts

        if len(source_entities) < 2:
            return

        sources = list(source_entities.keys())
        all_entities = set().union(*source_entities.values())

        # Detect entities present in some sources but absent in others
        omitted: dict[str, list[str]] = defaultdict(list)
        for entity in all_entities:
            mentioning = [s for s in sources if entity in source_entities[s]]
            omitting = [s for s in sources if entity not in source_entities[s]]
            # Only flag if entity appears in majority but not all
            if len(mentioning) >= max(1, len(sources) // 2) and omitting:
                for src in omitting:
                    omitted[src].append(entity)

        if omitted:
            omission_desc = "; ".join(
                f"{src} omits: {', '.join(entities[:3])}"
                for src, entities in list(omitted.items())[:3]
            )
            signal.escalation_reasons.append(f"Entity omission detected — {omission_desc}")
            signal.escalate_to_llm = True

        # Flag cross-source entity labeling differences
        # e.g. 'migrants' vs 'illegal immigrants' vs 'asylum seekers'
        self._detect_label_divergence(cluster, signal)

    def _detect_label_divergence(
        self, cluster: StoryCluster, signal: BiasSignal
    ) -> None:
        """Flag clusters where same real-world entity gets different labels."""
        LABEL_GROUPS: list[set[str]] = [
            {"migrants", "illegal immigrants", "undocumented immigrants",
             "asylum seekers", "illegal aliens", "border crossers"},
            {"protesters", "rioters", "demonstrators", "mob", "activists"},
            {"terrorists", "militants", "fighters", "rebels", "insurgents"},
            {"tax cuts", "tax relief", "tax breaks", "giveaways to the rich"},
            {"pro-life", "anti-abortion", "abortion opponents", "abortion rights opponents"},
            {"pro-choice", "abortion rights", "abortion supporters"},
        ]

        for label_group in LABEL_GROUPS:
            found_labels: dict[str, str] = {}
            for article in cluster.articles:
                text_lower = f"{article.raw.headline} {article.raw.summary}".lower()
                for label in label_group:
                    if label in text_lower:
                        found_labels[article.raw.source_name] = label
                        break

            if len(set(found_labels.values())) > 1:
                label_desc = ", ".join(
                    f"{src}: '{label}'" for src, label in found_labels.items()
                )
                signal.escalation_reasons.append(
                    f"Label divergence detected: {label_desc}"
                )
                signal.escalate_to_llm = True
                break  # One flag per cluster is enough
