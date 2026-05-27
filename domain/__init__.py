"""Stable domain contracts for the NewsBot digest pipeline."""

from domain.models import (
    Article,
    ArticleDraft,
    DigestRun,
    DigestStory,
    ReportingComparison,
    Source,
    SourceLink,
    StoryCluster,
    canonical_url_hash,
    normalize_article_url,
)

__all__ = [
    "Article",
    "ArticleDraft",
    "DigestRun",
    "DigestStory",
    "ReportingComparison",
    "Source",
    "SourceLink",
    "StoryCluster",
    "canonical_url_hash",
    "normalize_article_url",
]
