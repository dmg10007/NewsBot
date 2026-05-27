"""Config-driven geographic classification."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping

from domain.models import Article, GeoTier


@dataclass(frozen=True)
class GeographyProfile:
    name: str
    labels: dict[str, str]
    local_keywords: list[str]
    state_keywords: list[str]
    international_keywords: list[str]
    exclude_international: bool = True


def profile_from_settings(settings: Mapping) -> GeographyProfile:
    geo = settings.get("geography", {}) or {}
    geo_filter = settings.get("geo_filter", {}) or {}
    keywords = geo_filter.get("keywords", {}) or {}
    labels = geo.get("labels") or {
        "national": "National",
        "state": "North Carolina",
        "local": "Local - Lee County",
    }
    return GeographyProfile(
        name=geo.get("profile", "default"),
        labels=dict(labels),
        local_keywords=list(keywords.get("local", [])),
        state_keywords=list(keywords.get("state", [])),
        international_keywords=list(keywords.get("international", [])),
        exclude_international=bool(geo_filter.get("exclude_international", True)),
    )


class GeographyClassifier:
    def __init__(self, profile: GeographyProfile) -> None:
        self.profile = profile

    def classify_article(self, article: Article) -> GeoTier:
        text = " ".join([
            article.headline or "",
            article.summary or "",
            " ".join(article.tags),
        ]).lower()

        if _matches(text, self.profile.local_keywords):
            return "local"
        if _matches(text, self.profile.state_keywords):
            return "state"
        if _matches(text, self.profile.international_keywords):
            return "international"
        if article.region in ("national", "state", "local"):
            return article.region  # type: ignore[return-value]
        if article.region == "north_carolina":
            return "state"
        return "national"

    def classify_all(self, articles: list[Article]) -> list[Article]:
        kept: list[Article] = []
        for article in articles:
            article.geo_profile = self.profile.name
            article.geo_tier = self.classify_article(article)
            if article.geo_tier == "international" and self.profile.exclude_international:
                continue
            kept.append(article)
        return kept


def _matches(text: str, keywords: list[str]) -> bool:
    for keyword in keywords:
        kw = keyword.strip().lower()
        if not kw:
            continue
        if len(kw) <= 3:
            if re.search(rf"\b{re.escape(kw)}\b", text):
                return True
        elif kw in text:
            return True
    return False
