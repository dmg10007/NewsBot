"""Single orchestrator for the durable digest pipeline."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from bias.resolver import BiasResolver
from config.loader import BASE_DIR, get_settings, get_sources
from delivery.email_renderer import EmailRenderer
from delivery.email_sender import EmailSender
from delivery.telegram_bot import TelegramSender
from domain.models import Article, ArticleDraft, DigestRun, DigestStory, Source, SourceLink
from geography import GeographyClassifier, profile_from_settings
from ingestion.fetcher import FeedFetcher
from ingestion.scraper import SCRAPER_REGISTRY
from llm_clients import ComparisonLLMClient, SummaryLLMClient
from sources import load_sources_from_config
from storage import SQLiteStore
from story_clustering import (
    StoryClusterer,
    filter_reportable_clusters,
    score_clusters,
    select_digest_clusters,
)

logger = logging.getLogger(__name__)


class DigestPipeline:
    """Owns the end-to-end digest flow."""

    def __init__(
        self,
        *,
        settings: Optional[dict] = None,
        sources_config: Optional[dict] = None,
        store: Optional[SQLiteStore] = None,
        comparison_client: Optional[ComparisonLLMClient] = None,
        summary_client: Optional[SummaryLLMClient] = None,
    ) -> None:
        self.settings = dict(settings or get_settings())
        self.sources_config = sources_config or get_sources()
        self.sources = load_sources_from_config(self.sources_config)
        self.store = store or SQLiteStore(_sqlite_path(self.settings))
        self.store.migrate()
        self.comparison_client = comparison_client or ComparisonLLMClient(self.settings)
        self.summary_client = summary_client or SummaryLLMClient(self.settings)
        self.geo = GeographyClassifier(profile_from_settings(self.settings))
        self.bias_resolver = BiasResolver(auto_scrape=False)

    def close(self) -> None:
        self.store.close()
        self.comparison_client.close()
        self.summary_client.close()

    def migrate(self) -> None:
        self.store.migrate()

    def check_sources(self) -> list[str]:
        problems = []
        seen = set()
        for source in self.sources:
            if source.name in seen:
                problems.append(f"Duplicate source name: {source.name}")
            seen.add(source.name)
            if not source.url:
                problems.append(f"{source.name}: missing url")
            if source.source_type == "scraper" and source.scraper_class not in SCRAPER_REGISTRY:
                problems.append(f"{source.name}: unknown scraper_class {source.scraper_class}")
        return problems

    def ingest(self, *, dry_run: bool = False) -> list[Article]:
        self.store.upsert_sources(self.sources)
        drafts = self._collect_article_drafts()
        articles = [Article.from_draft(draft, geo_profile=self.geo.profile.name) for draft in drafts]
        self._enrich_bias(articles)
        articles = self.geo.classify_all(articles)
        if not dry_run:
            articles = self.store.upsert_articles(articles)
        logger.info("Ingested %d articles after geographic filtering", len(articles))
        return articles

    def run_digest(self, period: str, *, deliver: bool = True) -> list[DigestStory]:
        start = datetime.now(timezone.utc)
        run = self.store.create_digest_run(DigestRun(run_id=None, period=period, started_at=start))
        stories: list[DigestStory] = []
        try:
            articles = self.ingest(dry_run=False)
            articles = self._suppress_recent_articles(articles, period)
            clusters = StoryClusterer(
                similarity_threshold=float(self.settings.get("clustering", {}).get("similarity_threshold", 0.58))
            ).cluster(articles)
            score_clusters(clusters, self.settings)
            clusters = filter_reportable_clusters(clusters, self.settings)
            clusters = select_digest_clusters(clusters, self.settings)
            clusters.sort(key=lambda c: c.importance_score, reverse=True)

            for cluster in clusters:
                comparison = self.comparison_client.compare(cluster)
                summary = self.summary_client.summarize(cluster, comparison)
                stories.append(DigestStory(
                    story_id=None,
                    headline=cluster.representative_headline,
                    summary=summary.text,
                    geo_tier=cluster.geo_tier,
                    topic=cluster.topic,
                    importance_score=cluster.importance_score,
                    source_links=[
                        SourceLink(
                            source_name=a.source_name,
                            url=a.url,
                            bias_lean=a.bias_lean or "unknown",
                            credibility=a.credibility,
                        )
                        for a in cluster.articles
                    ],
                    comparison=comparison,
                    source_count=cluster.source_count,
                    is_single_source=cluster.is_single_source,
                    summary_provider=summary.provider_used,
                    fallback_used=summary.fallback_used or comparison.fallback_used,
                    article_ids=[
                        a.article_id for a in cluster.articles if a.article_id is not None
                    ],
                ))

            self.store.save_digest_stories(run, stories)
            if deliver:
                self._deliver(stories, run)
        except Exception as exc:
            run.failures.append(f"{type(exc).__name__}: {exc}")
            logger.exception("Digest pipeline failed")
            raise
        finally:
            self.store.finish_digest_run(run, story_count=len(stories))
        return stories

    def _collect_article_drafts(self) -> list[ArticleDraft]:
        drafts: list[ArticleDraft] = []
        fetcher = FeedFetcher()
        try:
            for source in self.sources:
                try:
                    if source.source_type == "rss":
                        fetched = fetcher.fetch_source(source)
                    else:
                        fetched = self._scrape_source(source)
                    drafts.extend(fetched)
                    self.store.record_source_success(source.name, len(fetched))
                except Exception as exc:
                    logger.warning("Source failed: %s: %s", source.name, exc)
                    self.store.record_source_failure(source.name, str(exc))
        finally:
            fetcher.close()
        return drafts

    def _scrape_source(self, source: Source) -> list[ArticleDraft]:
        scraper_cls = SCRAPER_REGISTRY.get(source.scraper_class or "")
        if not scraper_cls:
            raise ValueError(f"Unknown scraper_class: {source.scraper_class}")
        scraper = scraper_cls({
            "name": source.name,
            "url": source.url,
            "rss_url": source.rss_url,
            "bias_lean": source.bias_lean,
            "credibility": source.credibility,
            "region": source.region,
            "scraper_class": source.scraper_class,
            "selectors": source.selectors,
        })
        try:
            raw_articles = scraper.scrape()
            return [
                ArticleDraft(
                    source=source,
                    headline=a.headline,
                    url=a.url,
                    summary=a.summary,
                    published_at=a.published_at,
                    tags=a.tags,
                )
                for a in raw_articles
            ]
        finally:
            scraper.close()

    def _enrich_bias(self, articles: list[Article]) -> None:
        for article in articles:
            domain = urlparse(article.canonical_url).netloc
            if not domain:
                continue
            rating = self.bias_resolver.resolve(domain, credibility=article.credibility)
            article.bias_lean = rating.bias_lean or article.bias_lean or "unknown"
            article.bias_metadata = rating

    def _suppress_recent_articles(self, articles: list[Article], period: str) -> list[Article]:
        schedule_cfg = self.settings.get("schedule", {}).get(f"{period}_digest", {})
        lookback_hours = int(schedule_cfg.get("lookback_hours", 12))
        seen = self.store.recently_delivered_hashes(lookback_hours)
        return [article for article in articles if article.url_hash not in seen]

    def _deliver(self, stories: list[DigestStory], run: DigestRun) -> None:
        delivery_cfg = self.settings.get("delivery", {}).get(run.period, {})
        if delivery_cfg.get("email", {}).get("enabled"):
            html = EmailRenderer(self.settings).render(stories, run)
            EmailSender().send(html, period=run.period, run_date=run.started_at)
        if delivery_cfg.get("telegram", {}).get("enabled"):
            TelegramSender().send_digest(stories, period=run.period)


def _sqlite_path(settings: dict) -> Path:
    configured = settings.get("storage", {}).get("sqlite_path", "data/newsbot.sqlite")
    path = Path(configured)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path
