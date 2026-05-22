"""Source bias resolver: domain -> merged SourceRating.

Usage
-----
    from bias.resolver import BiasResolver

    resolver = BiasResolver()          # scrapes at init (async internally)
    rating = resolver.resolve("foxnews.com")
    print(rating.bias_lean)            # 'right'
    print(rating.factuality)           # 'mixed'
    print(rating.confidence)           # 0.67

The resolver merges AllSides + MBFC ratings:
  - Both agree:             confidence = 1.0
  - Differ by 1 scale step: confidence = 0.67, use AllSides
  - Differ by 2+ steps:     confidence = 0.33, flag in notes

Async refresh
-------------
refresh_async() runs AllSides (in a thread, single request) and
MBFC (async, concurrent) simultaneously via asyncio.gather, cutting
startup time from 90s+ down to ~10s for a 30-domain list.

The synchronous refresh() wrapper is kept so existing call sites
(tests, CLI) continue to work without modification.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
from typing import Optional

import httpx

from bias.source_ratings import (
    BIAS_SCALE,
    FACTUALITY_SCALE,
    SourceRating,
    _OUTLET_DOMAIN_MAP,
    scrape_allsides,
    scrape_mbfc_bulk_async,
)

logger = logging.getLogger(__name__)

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

_CREDIBILITY_TO_FACTUALITY: dict[str, str] = {
    "high": "high",
    "medium": "mostly-factual",
    "low": "mixed",
}


class BiasResolver:
    """Thread-safe bias + factuality resolver with async-backed lazy initialization."""

    def __init__(self, auto_scrape: bool = True) -> None:
        self._lock = threading.Lock()
        self._allsides: dict[str, str] = {}
        self._mbfc: dict[str, dict] = {}
        self._cache: dict[str, SourceRating] = {}
        self._initialized = False

        if auto_scrape:
            self.refresh()  # sync wrapper — fine at startup

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """Synchronous refresh wrapper.

        Blocks until AllSides + MBFC scraping completes.
        Calls refresh_async() via asyncio.run().
        Existing call sites (tests, CLI) work unchanged.
        """
        asyncio.run(self.refresh_async())

    async def refresh_async(self) -> None:
        """Async refresh: AllSides (thread) + MBFC (async) run concurrently.

        AllSides is a single sync HTTP request — run in a thread so it
        doesn't block the event loop. MBFC runs natively async with up to
        5 concurrent domain lookups.
        """
        logger.info("BiasResolver: refreshing source ratings (async)...")

        sync_client = httpx.Client(
            timeout=20,
            headers={"User-Agent": "NewsBot/1.0 (bias-ratings-lookup; educational use)"},
            follow_redirects=True,
        )

        known_domains = list(
            set(list(self._allsides.keys()) + list(_FALLBACK_RATINGS.keys()))
        ) or list(_FALLBACK_RATINGS.keys())

        try:
            # Run AllSides (sync, in thread) and MBFC (async) in parallel
            allsides_result, mbfc_result = await asyncio.gather(
                asyncio.to_thread(scrape_allsides, sync_client),
                scrape_mbfc_bulk_async(known_domains, delay=1.0, max_concurrent=5),
            )
        finally:
            sync_client.close()

        with self._lock:
            self._allsides = allsides_result
            self._mbfc = mbfc_result
            self._cache.clear()
            self._initialized = True

        logger.info(
            "BiasResolver: ready. AllSides=%d, MBFC=%d domains loaded.",
            len(allsides_result), len(mbfc_result),
        )

    def resolve(self, domain: str, credibility: str = "medium") -> SourceRating:
        """Return a SourceRating for the given domain.

        Falls back gracefully: live scrape -> fallback table -> neutral defaults.
        """
        domain = _normalize_domain(domain)

        with self._lock:
            if domain in self._cache:
                return self._cache[domain]

        rating = self._build_rating(domain, credibility)

        with self._lock:
            self._cache[domain] = rating

        return rating

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_rating(self, domain: str, credibility: str) -> SourceRating:
        allsides_bias = self._allsides.get(domain)
        mbfc_data = self._mbfc.get(domain, {})
        mbfc_bias = mbfc_data.get("bias")
        mbfc_factuality = mbfc_data.get("factuality")

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

        bias_lean, confidence, notes = _merge_bias(allsides_bias, mbfc_bias)
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

    try:
        as_idx = BIAS_SCALE.index(allsides)
        mb_idx = BIAS_SCALE.index(mbfc)
    except ValueError:
        return allsides, 0.5, [f"scale mismatch: allsides={allsides} mbfc={mbfc}"]

    distance = abs(as_idx - mb_idx)
    if distance == 1:
        return allsides, 0.67, [f"minor disagreement: allsides={allsides} mbfc={mbfc}"]
    return allsides, 0.33, [
        f"significant disagreement: allsides={allsides} mbfc={mbfc} (distance={distance})"
    ]
