"""Source bias resolver: domain -> merged SourceRating.

Usage
-----
    from bias.resolver import BiasResolver

    resolver = BiasResolver()          # loads + scrapes at init (cached)
    rating = resolver.resolve("foxnews.com")
    print(rating.bias_lean)            # 'right'
    print(rating.factuality)           # 'mixed'
    print(rating.confidence)           # 0.67

The resolver merges AllSides + MBFC ratings:
- If both agree on bias direction: confidence = 1.0
- If they differ by one step on the 5-point scale: confidence = 0.67, use AllSides
- If they differ by 2+ steps: confidence = 0.33, flag in notes
- Factuality comes from MBFC (more granular); falls back to credibility field
  from sources.yaml if MBFC has no data.

The lookup table is refreshed via refresh() which should be called weekly.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

import httpx

from bias.source_ratings import (
    BIAS_SCALE,
    FACTUALITY_SCALE,
    SourceRating,
    _OUTLET_DOMAIN_MAP,
    scrape_allsides,
    scrape_mbfc_bulk,
)

logger = logging.getLogger(__name__)

# Fallback ratings for sources we know but that may not appear in scraped data.
# Keyed by domain. These are used when both scrapers return nothing.
_FALLBACK_RATINGS: dict[str, dict] = {
    "apnews.com":            {"bias_lean": "center",       "factuality": "very-high"},
    "reuters.com":           {"bias_lean": "center",       "factuality": "very-high"},
    "npr.org":               {"bias_lean": "center-left",  "factuality": "high"},
    "pbs.org":               {"bias_lean": "center-left",  "factuality": "high"},
    "cnn.com":               {"bias_lean": "center-left",  "factuality": "mostly-factual"},
    "foxnews.com":           {"bias_lean": "right",         "factuality": "mixed"},
    "foxbusiness.com":       {"bias_lean": "right",         "factuality": "mostly-factual"},
    "thehill.com":           {"bias_lean": "center",        "factuality": "mostly-factual"},
    "axios.com":             {"bias_lean": "center",        "factuality": "high"},
    "wsj.com":               {"bias_lean": "center-right",  "factuality": "high"},
    "wral.com":              {"bias_lean": "center",        "factuality": "high"},
    "charlotteobserver.com": {"bias_lean": "center-left",  "factuality": "high"},
    "ncpolicywatch.com":     {"bias_lean": "center-left",  "factuality": "mostly-factual"},
    "carolinapublicpress.org": {"bias_lean": "center",     "factuality": "high"},
    "wcnc.com":              {"bias_lean": "center",        "factuality": "mostly-factual"},
    "sanfordherald.com":     {"bias_lean": "center",        "factuality": "mostly-factual"},
    "rantnc.com":            {"bias_lean": "center",        "factuality": "mixed"},
}

# credibility field from sources.yaml -> factuality scale
_CREDIBILITY_TO_FACTUALITY: dict[str, str] = {
    "high": "high",
    "medium": "mostly-factual",
    "low": "mixed",
}


class BiasResolver:
    """Thread-safe bias + factuality resolver with lazy initialization."""

    def __init__(self, auto_scrape: bool = True) -> None:
        self._lock = threading.Lock()
        self._allsides: dict[str, str] = {}
        self._mbfc: dict[str, dict] = {}
        self._cache: dict[str, SourceRating] = {}
        self._initialized = False

        if auto_scrape:
            self.refresh()

    def refresh(self) -> None:
        """Re-scrape AllSides and MBFC; rebuild the cache."""
        logger.info("BiasResolver: refreshing source ratings...")
        client = httpx.Client(
            timeout=20,
            headers={"User-Agent": "NewsBot/1.0 (bias-ratings-lookup; educational use)"},
            follow_redirects=True,
        )
        try:
            allsides = scrape_allsides(client)

            # Only MBFC-scrape domains we know about to avoid hammering the site
            known_domains = list(
                set(list(allsides.keys()) + list(_FALLBACK_RATINGS.keys()))
            )
            mbfc = scrape_mbfc_bulk(client, known_domains, delay=1.5)
        finally:
            client.close()

        with self._lock:
            self._allsides = allsides
            self._mbfc = mbfc
            self._cache.clear()
            self._initialized = True

        logger.info(
            "BiasResolver: ready. AllSides=%d, MBFC=%d domains loaded.",
            len(allsides), len(mbfc),
        )

    def resolve(self, domain: str, credibility: str = "medium") -> SourceRating:
        """Return a SourceRating for the given domain.

        Falls back gracefully through: live scrape -> fallback table -> neutral defaults.
        """
        domain = _normalize_domain(domain)

        with self._lock:
            if domain in self._cache:
                return self._cache[domain]

        rating = self._build_rating(domain, credibility)

        with self._lock:
            self._cache[domain] = rating

        return rating

    def _build_rating(self, domain: str, credibility: str) -> SourceRating:
        allsides_bias = self._allsides.get(domain)
        mbfc_data = self._mbfc.get(domain, {})
        mbfc_bias = mbfc_data.get("bias")
        mbfc_factuality = mbfc_data.get("factuality")

        # Pull from fallback table if live scrape came up empty
        fallback = _FALLBACK_RATINGS.get(domain, {})
        if not allsides_bias and not mbfc_bias:
            bias_lean = fallback.get("bias_lean", "center")
            factuality = fallback.get(
                "factuality",
                _CREDIBILITY_TO_FACTUALITY.get(credibility, "mostly-factual"),
            )
            return SourceRating(
                domain=domain,
                bias_lean=bias_lean,
                factuality=factuality,
                confidence=0.5,
                notes=["fallback: no live scrape data available"],
            )

        # Merge AllSides + MBFC bias
        bias_lean, confidence, notes = _merge_bias(allsides_bias, mbfc_bias)

        # Factuality: MBFC is authoritative, fall back to credibility field
        factuality = (
            mbfc_factuality
            or fallback.get("factuality")
            or _CREDIBILITY_TO_FACTUALITY.get(credibility, "mostly-factual")
        )

        return SourceRating(
            domain=domain,
            bias_lean=bias_lean,
            factuality=factuality,
            confidence=confidence,
            allsides_bias=allsides_bias,
            mbfc_bias=mbfc_bias,
            mbfc_factuality=mbfc_factuality,
            notes=notes,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_domain(domain: str) -> str:
    """Strip scheme, www, and path to get bare domain."""
    import re
    domain = re.sub(r"^https?://", "", domain)
    domain = re.sub(r"^www\.", "", domain)
    domain = domain.split("/")[0].lower().strip()
    return domain


def _merge_bias(
    allsides: Optional[str],
    mbfc: Optional[str],
) -> tuple[str, float, list[str]]:
    """Merge two bias labels into one with a confidence score."""
    if allsides and not mbfc:
        return allsides, 0.75, ["allsides only"]
    if mbfc and not allsides:
        return mbfc, 0.75, ["mbfc only"]
    if allsides == mbfc:
        return allsides, 1.0, []

    # Both present but differ — resolve by scale distance
    try:
        as_idx = BIAS_SCALE.index(allsides)
        mb_idx = BIAS_SCALE.index(mbfc)
    except ValueError:
        # One label not in scale — trust AllSides
        return allsides, 0.5, [f"scale mismatch: allsides={allsides} mbfc={mbfc}"]

    distance = abs(as_idx - mb_idx)
    if distance == 1:
        # Off by one step — use AllSides, moderate confidence
        return allsides, 0.67, [f"minor disagreement: allsides={allsides} mbfc={mbfc}"]
    else:
        # Significant disagreement — use AllSides but flag it
        return allsides, 0.33, [
            f"significant disagreement: allsides={allsides} mbfc={mbfc} (distance={distance})"
        ]
