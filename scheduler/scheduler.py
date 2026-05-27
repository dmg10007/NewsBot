"""Thin scheduler entrypoints for the durable DigestPipeline."""

from __future__ import annotations

import logging

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from config.loader import get_settings
from pipeline import DigestPipeline

logger = logging.getLogger(__name__)


def run_digest(period: str = "morning") -> None:
    pipeline = DigestPipeline()
    try:
        stories = pipeline.run_digest(period=period, deliver=True)
        logger.info("Digest complete: period=%s stories=%d", period, len(stories))
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
