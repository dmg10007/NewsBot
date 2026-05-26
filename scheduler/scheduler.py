"""Digest scheduler: runs the full NewsBot pipeline on a cron schedule.

Import strategy
---------------
All pipeline imports are at module level (not deferred inside run_digest).
This means a broken or missing module raises ImportError at process startup,
not silently at 6 AM when the first digest fires. _validate_imports() is
called by start_scheduler() to make this fail-fast guarantee explicit.

Failure alerting
----------------
If run_digest() raises an unhandled exception, _send_failure_alert() is
called before re-raising. It attempts to deliver an alert via TelegramSender.
EmailSender is NOT used for alerts because it requires fully-rendered HTML
and a valid digest period — it is not designed for plain-text error strings.

Email delivery
--------------
Email digests are a two-step process:
  1. EmailRenderer.render(summaries, period) -> HTML string
  2. EmailSender.send(html, period, run_date) -> bool
Do not call EmailSender directly with summaries — it does not accept them.

Summarizer lifecycle
--------------------
Summarizer is used as a context manager (`with Summarizer() as s`) so its
three httpx.Client instances are always released, even when an exception
occurs mid-run.

Scoring
-------
score_clusters() mutates clusters in place and returns None. Call it between
clustering (Stage 3) and bias analysis (Stage 4); do NOT reassign its return
value. Downstream consumers (Summarizer, LLMAnalyzer) read importance_score
to decide whether to use Perplexity, Brave enrichment, or local fallback.

GeoFilter
---------
GeoFilter runs between ArticleExtractor (Stage 2) and StoryClusterer (Stage 3).
It drops articles with no detectable US geographic signal, preventing Reuters
and AP international wire content from consuming national story slots. The
filter writes geo_tier='domestic' or 'international' to each RawArticle for
downstream observability. See parsing/geo_filter.py for tuning details.
"""

from __future__ import annotations

import logging

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from bias.llm_analyzer import LLMAnalyzer
from bias.framing import FramingAnalyzer
from clustering.clusterer import StoryClusterer
from config.loader import get_settings, get_sources
from delivery.email_renderer import EmailRenderer
from delivery.email_sender import EmailSender
from delivery.telegram_bot import TelegramSender
from ingestion.pipeline import ingest_all_sources
from monitoring.health import record_run
from parsing.extractor import ArticleExtractor
from parsing.geo_filter import GeoFilter
from scoring.scorer import score_clusters
from summarizer.summarizer import Summarizer

logger = logging.getLogger(__name__)


def run_digest(period: str = "morning") -> None:
    pipeline = DigestPipeline()
    try:
        _run_digest_inner(period)
    except Exception as exc:
        logger.critical(
            "Digest run failed for period '%s': %s", period, exc, exc_info=True
        )
        _send_failure_alert(period, exc)
        raise


def _run_digest_inner(period: str) -> None:
    """Inner implementation of the digest pipeline (no exception wrapping)."""
    settings = get_settings()
    sources = get_sources()
    start_time = datetime.now(timezone.utc)
    logger.info("Starting %s digest at %s", period, start_time.isoformat())

    # Stage 1: Ingest + deduplicate
    articles = ingest_all_sources(sources)
    if not articles:
        logger.warning("No articles ingested — aborting digest run")
        return

    # Stage 2: Parse / NLP extraction
    extractor = ArticleExtractor()
    parsed = extractor.extract_all(articles)

    # Stage 2b: Geographic filter — drop international articles before clustering.
    # Runs after extraction so spaCy entity data is available for signal detection.
    # Writes geo_tier='domestic'|'international' to each RawArticle for observability.
    geo_filter = GeoFilter()
    parsed = geo_filter.filter(parsed)
    if not parsed:
        logger.warning("GeoFilter removed all articles — check filter thresholds")
        return

    # Stage 3: Cluster into stories
    clusterer = StoryClusterer()
    clusters = clusterer.cluster(parsed)
    logger.info("Produced %d story clusters", len(clusters))

    # Stage 3b: Score clusters (mutates in place, returns None)
    # Must run before any downstream consumer checks importance_score.
    # Fixes three silent cascade failures:
    #   - Singleton filter uses 0.0 score without this → drops all singletons
    #   - Summarizer Perplexity gate: 0.0 >= pplx_min_importance_score is False
    #   - Brave Search enrichment gate: 0.0 >= brave_enrich_threshold is False
    score_clusters(clusters, settings)

    # Stage 4: Framing analysis (lexicon-based, no LLM)
    framing_analyzer = FramingAnalyzer()
    framing_results = {c.cluster_id: framing_analyzer.analyze(c) for c in clusters}

    # Stage 5: LLM bias analysis (capped, degrades gracefully)
    llm_analyzer = LLMAnalyzer(
        max_calls=settings["bias"].get("max_llm_calls_per_run", 50)
    )
    llm_results = {}
    try:
        for cluster in clusters:
            llm_results[cluster.cluster_id] = llm_analyzer.analyze(
                cluster, framing_results[cluster.cluster_id]
            )
    finally:
        pipeline.close()


def start_scheduler() -> None:
    settings = get_settings()
    scheduler = BlockingScheduler(timezone=settings.get("app", {}).get("timezone", "America/New_York"))
    for period, cron_expr in settings["scheduler"]["schedule"].items():
        scheduler.add_job(
            run_digest,
            trigger=CronTrigger.from_crontab(cron_expr),
            kwargs={"period": period},
            id=f"digest_{period}",
            name=f"NewsBot {period} digest",
            misfire_grace_time=300,
            coalesce=True,
        )
        logger.info("Scheduled %s digest: %s", period, cron_expr)
    scheduler.start()
