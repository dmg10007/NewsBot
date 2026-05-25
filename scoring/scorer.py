"""Story importance scoring.

Assigns a numeric importance_score to each StoryCluster before summarization.
The score is used by:
  - StoryClusterer: to decide whether to keep singleton clusters
  - Summarizer: to route high-importance clusters to Perplexity
  - Summarizer: to decide whether to enrich a cluster with Brave Search

Scoring factors (all weights configurable in config/settings.yaml):

  base = len(cluster.articles) * source_count_weight
  tier_multiplier:
    local    -> local_tier weight  (default 1.5 — local stories are rare)
    state    -> state_tier weight  (default 1.2)
    national -> national_tier weight (default 1.0)
  recency_decay:
    score *= max(0.1, 1.0 - age_hours * recency_decay_weight)
    Older stories lose relevance; floor of 0.1 prevents complete zeroing.

Final score is clamped to [0.0, 1.0] and written back to
cluster.importance_score in place.

Usage (see scheduler/scheduler.py)::

    from scoring.scorer import score_clusters
    clusters = score_clusters(clusters, settings)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from clustering.clusterer import StoryCluster

logger = logging.getLogger(__name__)

# Normalization ceiling — raw scores above this value are treated as 1.0.
# Adjust upward if average cluster sizes grow significantly.
_SCORE_CEILING = 5.0


def score_clusters(
    clusters: list["StoryCluster"],
    settings: dict,
) -> list["StoryCluster"]:
    """Score each cluster and write the result to cluster.importance_score.

    Args:
        clusters: List of StoryCluster objects to score (mutated in place).
        settings: Full settings dict from config.loader.get_settings().

    Returns:
        The same list, with importance_score set on every cluster.
    """
    weights = settings.get("scoring", {}).get("weights", {})
    source_count_w: float = float(weights.get("source_count", 0.3))
    local_tier_w: float = float(weights.get("local_tier", 1.5))
    state_tier_w: float = float(weights.get("state_tier", 1.2))
    national_tier_w: float = float(weights.get("national_tier", 1.0))
    recency_decay_w: float = float(weights.get("recency_decay", 0.05))

    now = datetime.now(timezone.utc)

    for cluster in clusters:
        # Base score: more corroborating sources = higher base
        score: float = len(cluster.articles) * source_count_w

        # Tier multiplier: local stories are rare and high-value
        if "local" in cluster.tiers:
            score *= local_tier_w
        elif "state" in cluster.tiers:
            score *= state_tier_w
        else:
            score *= national_tier_w

        # Recency decay: older stories lose importance
        if cluster.earliest_published is not None:
            pub = cluster.earliest_published
            if pub.tzinfo is None:
                pub = pub.replace(tzinfo=timezone.utc)
            age_hours = (now - pub).total_seconds() / 3600.0
            decay = max(0.1, 1.0 - age_hours * recency_decay_w)
            score *= decay

        # Normalize to [0.0, 1.0]
        cluster.importance_score = min(score / _SCORE_CEILING, 1.0)

    scored_above_threshold = sum(
        1 for c in clusters if c.importance_score >= 0.4
    )
    logger.info(
        "Scored %d clusters — %d above singleton threshold (0.4)",
        len(clusters),
        scored_above_threshold,
    )
    return clusters
