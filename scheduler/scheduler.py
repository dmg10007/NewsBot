"""APScheduler-based job scheduler.

Runs the full pipeline twice daily at 6:00 AM and 6:00 PM ET.
The scheduler is blocking and designed to run as a long-lived process
(e.g., a systemd service or a simple 'python -m scheduler.scheduler' call).

Timezone handling: APScheduler CronTrigger accepts a pytz timezone string.
Using 'America/New_York' handles EDT/EST transitions automatically.
"""

from __future__ import annotations

import logging
import signal
import sys
from datetime import datetime

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def run_digest(period: str) -> None:
    """Full pipeline: ingest → parse → cluster → bias → summarize → deliver."""
    from datetime import timezone
    from config.loader import get_settings
    from ingestion.fetcher import FeedFetcher
    from ingestion.scraper import SCRAPER_REGISTRY
    from ingestion.deduplicator import Deduplicator
    from parsing.extractor import ArticleExtractor
    from parsing.normalizer import Normalizer
    from clustering.clusterer import StoryClusterer
    from bias.lexicon import LexiconAnalyzer
    from bias.framing import FramingAnalyzer
    from bias.llm_analyzer import LLMAnalyzer, LLMAnalysisResult
    from summarizer.summarizer import Summarizer
    from delivery.email_renderer import render_digest
    from delivery.email_sender import EmailSender
    from delivery.telegram_bot import TelegramSender

    settings = get_settings()
    run_date = datetime.now(tz=timezone.utc)
    logger.info("=== NewsBot %s digest run started ===", period.upper())

    # --- 1. Ingest ---
    from config.loader import get_sources
    sources = get_sources()
    fetcher = FeedFetcher()
    raw_articles = []

    for tier_name in ("national", "state"):
        tier = sources.get(tier_name, {})
        rss_sources = tier.get("rss", [])
        if rss_sources:
            raw_articles.extend(fetcher.fetch_all(rss_sources))
        for scraper_source in tier.get("scrapers", []):
            cls = SCRAPER_REGISTRY.get(scraper_source.get("scraper_class", ""))
            if cls:
                scraper = cls(scraper_source)
                try:
                    raw_articles.extend(scraper.scrape())
                finally:
                    scraper.close()

    local_tier = sources.get("local", {})
    for scraper_source in local_tier.get("scrapers", []):
        cls = SCRAPER_REGISTRY.get(scraper_source.get("scraper_class", ""))
        if cls:
            scraper = cls(scraper_source)
            try:
                raw_articles.extend(scraper.scrape())
            finally:
                scraper.close()

    fetcher.close()
    logger.info("Ingested %d raw articles", len(raw_articles))

    # --- 2. Deduplicate ---
    raw_articles = Deduplicator().deduplicate(raw_articles)
    logger.info("%d articles after deduplication", len(raw_articles))

    # Filter by lookback window
    from datetime import timedelta
    lookback_hours = settings["schedule"][f"{period}_digest"]["lookback_hours"]
    cutoff = run_date - timedelta(hours=lookback_hours)
    raw_articles = [
        a for a in raw_articles
        if a.published_at is None or a.published_at >= cutoff
    ]
    logger.info("%d articles within %dh lookback window", len(raw_articles), lookback_hours)

    if not raw_articles:
        logger.warning("No articles found in lookback window. Skipping digest.")
        return

    # --- 3. Parse ---
    extractor = ArticleExtractor()
    parsed = extractor.extract_all(raw_articles)
    parsed = Normalizer().normalize_all(parsed)

    # --- 4. Cluster ---
    clusters = StoryClusterer().cluster(parsed)

    # Score clusters: credibility + source count + tier + recency
    import math
    credibility_map = {"high": 1.0, "medium": 0.7, "low": 0.4}
    weights = settings["scoring"]["weights"]
    tier_weights = {"national": weights["national_tier"], "state": weights["state_tier"], "local": weights["local_tier"]}
    decay = weights["recency_decay"]
    for cluster in clusters:
        score = 0.0
        for article in cluster.articles:
            score += credibility_map.get(article.raw.credibility, 0.7)
            score += weights["source_count"]
        for tier in cluster.tiers:
            score *= tier_weights.get(tier, 1.0)
        if cluster.earliest_published:
            age_hours = (run_date - cluster.earliest_published).total_seconds() / 3600
            score *= max(0.1, 1 - decay * age_hours)
        cluster.importance_score = score

    clusters.sort(key=lambda c: c.importance_score, reverse=True)

    # --- 5. Bias analysis ---
    lexicon_analyzer = LexiconAnalyzer(
        escalation_threshold=settings["bias_detection"]["llm_escalation_threshold"]
    )
    framing_analyzer = FramingAnalyzer()
    llm_analyzer = LLMAnalyzer(
        max_calls=settings["bias_detection"]["max_llm_calls_per_run"]
    )

    analysis_map: dict[int, LLMAnalysisResult] = {}
    for cluster in clusters:
        lexicon_result = lexicon_analyzer.analyze(cluster)
        if lexicon_result.escalate and cluster.source_count > 1:
            framing_result = framing_analyzer.analyze(cluster)
            analysis = llm_analyzer.analyze(cluster, framing_result)
        else:
            from bias.framing import FramingResult
            empty_framing = FramingResult(
                cluster_id=cluster.cluster_id,
                entity_omissions=[],
                framing_differences=[],
                attribution_asymmetry=[],
                cross_source_summary="",
            )
            analysis = llm_analyzer.analyze(cluster, empty_framing)
        analysis_map[cluster.cluster_id] = analysis

    llm_analyzer.close()

    # --- 6. Summarize ---
    summarizer = Summarizer()
    summaries = [
        summarizer.summarize(cluster, analysis_map[cluster.cluster_id])
        for cluster in clusters
    ]
    summarizer.close()

    # --- 7. Deliver ---
    html = render_digest(summaries, period, run_date)

    email_cfg = settings["delivery"]["email"]
    if email_cfg.get("enabled", True):
        try:
            EmailSender().send(html, period, run_date)
        except EnvironmentError as exc:
            logger.error("Email delivery skipped: %s", exc)

    telegram_cfg = settings["delivery"]["telegram"]
    if telegram_cfg.get("enabled", False):
        TelegramSender().send(summaries, period, run_date)

    logger.info("=== NewsBot %s digest complete ===", period.upper())


def main() -> None:
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    scheduler = BlockingScheduler(timezone="America/New_York")

    scheduler.add_job(
        run_digest,
        trigger=CronTrigger(hour=6, minute=0, timezone="America/New_York"),
        args=["morning"],
        id="morning_digest",
        name="Morning Digest (6:00 AM ET)",
        misfire_grace_time=300,  # 5 min grace window if system was briefly down
    )
    scheduler.add_job(
        run_digest,
        trigger=CronTrigger(hour=18, minute=0, timezone="America/New_York"),
        args=["evening"],
        id="evening_digest",
        name="Evening Digest (6:00 PM ET)",
        misfire_grace_time=300,
    )

    def _shutdown(sig, frame):
        logger.info("Shutdown signal received. Stopping scheduler.")
        scheduler.shutdown(wait=False)
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    logger.info("NewsBot scheduler started. Morning: 6:00 AM ET | Evening: 6:00 PM ET")
    scheduler.start()


if __name__ == "__main__":
    main()
