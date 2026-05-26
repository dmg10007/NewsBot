"""NewsBot entry point.

This is the sole place where load_dotenv() is called. It must run before
any other module is imported, so environment variables from .env are visible
to config.loader and all pipeline modules.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env FIRST, before any project imports that read os.getenv()
load_dotenv(Path(__file__).resolve().parent / ".env", override=False)

from config.loader import get_settings, get_sources  # noqa: E402
from ingestion.pipeline import ingest_all_sources    # noqa: E402
from pipeline import DigestPipeline                  # noqa: E402
from scheduler.scheduler import run_digest, start_scheduler  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _run_test_ingest() -> None:
    """Quick sanity check: ingest all sources and report article counts.

    Delegates to ingestion.pipeline.ingest_all_sources() — the single
    authoritative implementation shared with the scheduler.
    """
    sources = get_sources()
    articles = ingest_all_sources(sources)
    logger.info("Test ingest complete: %d articles after deduplication", len(articles))
    for article in articles[:5]:
        logger.info("  [%s] %s", article.source_name, article.headline)


def main() -> None:
    parser = argparse.ArgumentParser(description="NewsBot digest runner")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run a single digest immediately")
    run_parser.add_argument(
        "--period",
        choices=["morning", "evening"],
        default="morning",
        help="Digest period to run (default: morning)",
    )

    subparsers.add_parser("schedule", help="Start the scheduler (blocking)")
    subparsers.add_parser("test-ingest", help="Ingest sources and print article counts")
    ingest_parser = subparsers.add_parser("ingest", help="Run ingestion without delivery")
    ingest_parser.add_argument("--dry-run", action="store_true", help="Do not persist articles")

    sources_parser = subparsers.add_parser("sources", help="Source utilities")
    sources_sub = sources_parser.add_subparsers(dest="sources_command")
    sources_sub.add_parser("check", help="Validate configured sources")

    db_parser = subparsers.add_parser("db", help="Database utilities")
    db_sub = db_parser.add_subparsers(dest="db_command")
    db_sub.add_parser("migrate", help="Create or update the SQLite schema")

    args = parser.parse_args()

    if args.command == "run":
        run_digest(period=args.period)
    elif args.command == "schedule":
        start_scheduler()
    elif args.command == "test-ingest":
        _run_test_ingest()
    elif args.command == "ingest":
        pipeline = DigestPipeline()
        try:
            articles = pipeline.ingest(dry_run=args.dry_run)
            logger.info("Ingest complete: %d articles", len(articles))
        finally:
            pipeline.close()
    elif args.command == "sources" and args.sources_command == "check":
        pipeline = DigestPipeline()
        try:
            problems = pipeline.check_sources()
            if problems:
                for problem in problems:
                    logger.error(problem)
                sys.exit(2)
            logger.info("Source check passed: %d sources", len(pipeline.sources))
        finally:
            pipeline.close()
    elif args.command == "db" and args.db_command == "migrate":
        pipeline = DigestPipeline()
        try:
            pipeline.migrate()
            logger.info("Database migration complete")
        finally:
            pipeline.close()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
