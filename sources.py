"""Source configuration loading helpers."""

from __future__ import annotations

from collections.abc import Mapping

from domain.models import Source

_TIERS = ("national", "state", "local")


def load_sources_from_config(config: Mapping) -> list[Source]:
    """Flatten sources.yaml into typed Source objects."""
    sources: list[Source] = []
    for tier in _TIERS:
        tier_data = config.get(tier, {}) or {}
        if isinstance(tier_data, list):
            rss_items = tier_data
            scraper_items = []
        else:
            rss_items = tier_data.get("rss", [])
            scraper_items = tier_data.get("scrapers", [])

        for item in rss_items:
            sources.append(_source_from_dict(item, tier=tier, source_type="rss"))
        for item in scraper_items:
            sources.append(_source_from_dict(item, tier=tier, source_type="scraper"))
    return sources


def _source_from_dict(item: Mapping, *, tier: str, source_type: str) -> Source:
    region = item.get("region") or ("national" if tier == "national" else tier)
    return Source(
        name=item["name"],
        url=item["url"],
        source_type=source_type,  # type: ignore[arg-type]
        tier=tier,  # type: ignore[arg-type]
        bias_lean=item.get("bias_lean", "unknown"),
        credibility=item.get("credibility", "medium"),
        topics=list(item.get("topics", [])),
        region=region,
        scraper_class=item.get("scraper_class"),
        rss_url=item.get("rss_url"),
        selectors=dict(item.get("selectors", {})),
    )
