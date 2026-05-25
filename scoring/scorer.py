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

Normalization ceiling
---------------------
normalization_ceiling (settings.yaml scoring.weights.normalization_ceiling,
default 5.0) is the raw score value that maps to 1.0. If source_count_weight
or tier multipliers are tuned upward, raise this ceiling proportionally to
keep scores in the 0–1 range. Without this, clusters with many sources will
all score 1.0 and lose relative distinction.

Score floor
-----------
A minimum score of 0.1 is applied after normalization. This prevents clusters
where all articles lack a published_at timestamp from scoring 0.0 and being
silently dropped by the singleton filter. A cluster with no pubdate is
unknown-age, not worthless.

Mutates in place, returns None
-------------------------------
score_clusters() mutates cluster.importance_score directly and returns None.
Callers should NOT reassign the return value (there is none). This avoids the
mutate-and-return anti-pattern where a function both mutates its argument and
returns it, creating a false impression that a transformed copy is produced.

Usage (see scheduler/scheduler.py)::

    from scoring.scorer import score_clusters
    score_clusters(clusters, settings)
    # clusters now have importance_score set — no reassignment needed
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from clustering.clusterer import StoryCluster

logger = logging.getLogger(__name__)

_DEFAULT_SCORE_CEILING = 5.0
_SCORE_FLOOR = 0.1


def score_clusters(
    clusters: list["StoryCluster"],
    settings: dict,
) -> None:
    """Score each cluster and write the result to cluster.importance_score.

    Mutates clusters in place. Returns None — do not reassign.

    Args:
        clusters: List of StoryCluster objects to score (mutated in place).
        settings: Full settings dict from config.loader.get_settings().
    """
    weights = settings.get("scoring", {}).get("weights", {})
    source_count_w: float = float(weights.get("source_count", 0.3))
    local_tier_w: float = float(weights.get("local_tier", 1.5))
    state_tier_w: float = float(weights.get("state_tier", 1.2))
    national_tier_w: float = float(weights.get("national_tier", 1.0))
    recency_decay_w: float = float(weights.get("recency_decay", 0.05))
    # Config-driven normalization ceiling. Raise if weight tuning pushes raw
    # scores above 5.0 and clusters lose relative distinction at the top.
    score_ceiling: float = float(weights.get("normalization_ceiling", _DEFAULT_SCORE_CEILING))

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

        # Recency decay: older stories lose importance.
        # If published_at is missing entirely, skip decay — the article is
        # unknown-age, not worthless. The score floor below prevents it from
        # being treated as zero-value.
        if cluster.earliest_published is not None:
            pub = cluster.earliest_published
            if pub.tzinfo is None:
                pub = pub.replace(tzinfo=timezone.utc)
            age_hours = (now - pub).total_seconds() / 3600.0
            decay = max(0.1, 1.0 - age_hours * recency_decay_w)
            score *= decay

        # Normalize to [0.0, 1.0] then apply floor.
        # Floor of _SCORE_FLOOR (0.1) ensures clusters with missing pubdate
        # are not silently eliminated by the singleton importance filter.
        cluster.importance_score = max(
            min(score / score_ceiling, 1.0),
            _SCORE_FLOOR,
        )

    scored_above_threshold = sum(
        1 for c in clusters if c.importance_score >= 0.4
    )
    logger.info(
        "Scored %d clusters — %d above singleton threshold (0.4)",
        len(clusters),
        scored_above_threshold,
    )
