"""SQLite persistence for articles, clusters, comparisons, and digest runs."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from domain.models import (
    Article,
    DigestRun,
    DigestStory,
    ReportingComparison,
    Source,
    SourceLink,
    StoryCluster,
)


class SQLiteStore:
    """Small repository wrapper around sqlite3.

    The store is intentionally boring: one connection per instance, explicit
    migrations, JSON only for small list fields, and idempotent upserts for
    articles and sources.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")

    def close(self) -> None:
        self.conn.close()

    def migrate(self) -> None:
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def upsert_sources(self, sources: Iterable[Source]) -> None:
        self.conn.executemany(
            """
            INSERT INTO sources (
                name, url, source_type, tier, bias_lean, credibility,
                topics, region, scraper_class, rss_url
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                url=excluded.url,
                source_type=excluded.source_type,
                tier=excluded.tier,
                bias_lean=excluded.bias_lean,
                credibility=excluded.credibility,
                topics=excluded.topics,
                region=excluded.region,
                scraper_class=excluded.scraper_class,
                rss_url=excluded.rss_url
            """,
            [
                (
                    s.name,
                    s.url,
                    s.source_type,
                    s.tier,
                    s.bias_lean,
                    s.credibility,
                    json.dumps(s.topics),
                    s.region,
                    s.scraper_class,
                    s.rss_url,
                )
                for s in sources
            ],
        )
        self.conn.commit()

    def upsert_articles(self, articles: Iterable[Article]) -> list[Article]:
        saved: list[Article] = []
        for article in articles:
            self.conn.execute(
                """
                INSERT INTO articles (
                    url_hash, canonical_url, url, source_name, source_url,
                    headline, summary, body_text, published_at, region,
                    geo_tier, geo_profile, bias_lean, credibility, topics,
                    tags, fetch_status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(url_hash) DO UPDATE SET
                    headline=excluded.headline,
                    summary=excluded.summary,
                    body_text=excluded.body_text,
                    published_at=COALESCE(excluded.published_at, articles.published_at),
                    region=excluded.region,
                    geo_tier=excluded.geo_tier,
                    geo_profile=excluded.geo_profile,
                    bias_lean=excluded.bias_lean,
                    credibility=excluded.credibility,
                    topics=excluded.topics,
                    tags=excluded.tags,
                    fetch_status=excluded.fetch_status
                """,
                _article_params(article),
            )
            row = self.conn.execute(
                "SELECT id FROM articles WHERE url_hash = ?", (article.url_hash,)
            ).fetchone()
            article.article_id = int(row["id"])
            saved.append(article)
        self.conn.commit()
        return saved

    def create_digest_run(self, run: DigestRun) -> DigestRun:
        cur = self.conn.execute(
            "INSERT INTO digest_runs (period, started_at, status) VALUES (?, ?, ?)",
            (run.period, _dt(run.started_at), "running"),
        )
        run.run_id = int(cur.lastrowid)
        self.conn.commit()
        return run

    def finish_digest_run(self, run: DigestRun, *, story_count: int) -> None:
        run.completed_at = datetime.now(timezone.utc)
        self.conn.execute(
            """
            UPDATE digest_runs
            SET completed_at = ?, story_count = ?, failures = ?, status = ?
            WHERE id = ?
            """,
            (
                _dt(run.completed_at),
                story_count,
                json.dumps(run.failures),
                "failed" if run.failures else "complete",
                run.run_id,
            ),
        )
        self.conn.commit()

    def save_digest_stories(self, run: DigestRun, stories: list[DigestStory]) -> None:
        if run.run_id is None:
            raise ValueError("DigestRun must be persisted before stories can be saved")
        for story in stories:
            cluster_id = self.save_story_cluster(_cluster_from_digest(story))
            comparison_id = self.save_reporting_comparison(story.comparison, cluster_id)
            cur = self.conn.execute(
                """
                INSERT INTO digest_run_stories (
                    run_id, cluster_id, comparison_id, headline, summary,
                    geo_tier, topic, importance_score, source_links,
                    source_count, is_single_source, summary_provider, fallback_used
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.run_id,
                    cluster_id,
                    comparison_id,
                    story.headline,
                    story.summary,
                    story.geo_tier,
                    story.topic,
                    story.importance_score,
                    json.dumps([link.__dict__ for link in story.source_links]),
                    story.source_count,
                    int(story.is_single_source),
                    story.summary_provider,
                    int(story.fallback_used),
                ),
            )
            story.story_id = int(cur.lastrowid)
            run.delivered_story_ids.append(story.story_id)
        self.conn.commit()

    def save_story_cluster(self, cluster: StoryCluster) -> int:
        article_ids = [
            article.article_id for article in cluster.articles if article.article_id is not None
        ] or list(getattr(cluster, "_article_ids", []))
        cur = self.conn.execute(
            """
            INSERT INTO story_clusters (
                representative_headline, topic, geo_tier, importance_score,
                article_count, source_count, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cluster.representative_headline,
                cluster.topic,
                cluster.geo_tier,
                cluster.importance_score,
                len(article_ids) or len(cluster.articles),
                cluster.source_count,
                _dt(datetime.now(timezone.utc)),
            ),
        )
        cluster_id = int(cur.lastrowid)
        for article_id in article_ids:
            self.conn.execute(
                "INSERT OR IGNORE INTO cluster_articles (cluster_id, article_id) VALUES (?, ?)",
                (cluster_id, article_id),
            )
        return cluster_id

    def save_reporting_comparison(
        self, comparison: ReportingComparison, cluster_id: int
    ) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO reporting_comparisons (
                cluster_id, shared_facts, source_specific_claims, omissions,
                framing_differences, bias_notes, provider_used, confidence,
                fallback_used, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cluster_id,
                json.dumps(comparison.shared_facts),
                json.dumps(comparison.source_specific_claims),
                json.dumps(comparison.omissions),
                json.dumps(comparison.framing_differences),
                comparison.bias_notes,
                comparison.provider_used,
                comparison.confidence,
                int(comparison.fallback_used),
                _dt(datetime.now(timezone.utc)),
            ),
        )
        return int(cur.lastrowid)

    def record_source_success(self, source_name: str, article_count: int) -> None:
        now = _dt(datetime.now(timezone.utc))
        self.conn.execute(
            """
            INSERT INTO source_health (
                source_name, last_success, consecutive_failures, total_fetches,
                total_failures, last_article_count
            )
            VALUES (?, ?, 0, 1, 0, ?)
            ON CONFLICT(source_name) DO UPDATE SET
                last_success=excluded.last_success,
                consecutive_failures=0,
                total_fetches=source_health.total_fetches + 1,
                last_article_count=excluded.last_article_count
            """,
            (source_name, now, article_count),
        )
        self.conn.commit()

    def record_source_failure(self, source_name: str, error: str) -> None:
        now = _dt(datetime.now(timezone.utc))
        self.conn.execute(
            """
            INSERT INTO source_health (
                source_name, last_failure, consecutive_failures, total_fetches,
                total_failures, last_error
            )
            VALUES (?, ?, 1, 1, 1, ?)
            ON CONFLICT(source_name) DO UPDATE SET
                last_failure=excluded.last_failure,
                consecutive_failures=source_health.consecutive_failures + 1,
                total_fetches=source_health.total_fetches + 1,
                total_failures=source_health.total_failures + 1,
                last_error=excluded.last_error
            """,
            (source_name, now, error),
        )
        self.conn.commit()

    def recently_delivered_hashes(self, lookback_hours: int) -> set[str]:
        rows = self.conn.execute(
            """
            SELECT DISTINCT a.url_hash
            FROM articles a
            JOIN cluster_articles ca ON ca.article_id = a.id
            JOIN digest_run_stories drs ON drs.cluster_id = ca.cluster_id
            JOIN digest_runs dr ON dr.id = drs.run_id
            WHERE dr.started_at >= datetime('now', ?)
            """,
            (f"-{lookback_hours} hours",),
        ).fetchall()
        return {row["url_hash"] for row in rows}


def _article_params(article: Article) -> tuple:
    return (
        article.url_hash,
        article.canonical_url,
        article.url,
        article.source_name,
        article.source_url,
        article.headline,
        article.summary,
        article.body_text,
        _dt(article.published_at),
        article.region,
        article.geo_tier,
        article.geo_profile,
        article.bias_lean,
        article.credibility,
        json.dumps(article.topics),
        json.dumps(article.tags),
        article.fetch_status,
    )


def _dt(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    return value.isoformat()


def _cluster_from_digest(story: DigestStory) -> StoryCluster:
    cluster = StoryCluster(
        cluster_id=story.story_id,
        articles=[],
        topic=story.topic,
        geo_tier=story.geo_tier,
        representative_headline=story.headline,
        importance_score=story.importance_score,
    )
    cluster._article_ids = story.article_ids  # type: ignore[attr-defined]
    return cluster


_SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    url TEXT NOT NULL,
    source_type TEXT NOT NULL,
    tier TEXT NOT NULL,
    bias_lean TEXT NOT NULL,
    credibility TEXT NOT NULL,
    topics TEXT NOT NULL,
    region TEXT NOT NULL,
    scraper_class TEXT,
    rss_url TEXT
);

CREATE TABLE IF NOT EXISTS articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url_hash TEXT NOT NULL UNIQUE,
    canonical_url TEXT NOT NULL,
    url TEXT NOT NULL,
    source_name TEXT NOT NULL,
    source_url TEXT NOT NULL,
    headline TEXT NOT NULL,
    summary TEXT NOT NULL DEFAULT '',
    body_text TEXT NOT NULL DEFAULT '',
    published_at TEXT,
    region TEXT NOT NULL,
    geo_tier TEXT NOT NULL,
    geo_profile TEXT NOT NULL,
    bias_lean TEXT NOT NULL,
    credibility TEXT NOT NULL,
    topics TEXT NOT NULL,
    tags TEXT NOT NULL,
    fetch_status TEXT NOT NULL DEFAULT 'ok',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS story_clusters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    representative_headline TEXT NOT NULL,
    topic TEXT NOT NULL,
    geo_tier TEXT NOT NULL,
    importance_score REAL NOT NULL DEFAULT 0,
    article_count INTEGER NOT NULL,
    source_count INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cluster_articles (
    cluster_id INTEGER NOT NULL REFERENCES story_clusters(id) ON DELETE CASCADE,
    article_id INTEGER NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
    PRIMARY KEY (cluster_id, article_id)
);

CREATE TABLE IF NOT EXISTS reporting_comparisons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cluster_id INTEGER NOT NULL REFERENCES story_clusters(id) ON DELETE CASCADE,
    shared_facts TEXT NOT NULL,
    source_specific_claims TEXT NOT NULL,
    omissions TEXT NOT NULL,
    framing_differences TEXT NOT NULL,
    bias_notes TEXT NOT NULL,
    provider_used TEXT NOT NULL,
    confidence REAL NOT NULL,
    fallback_used INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS digest_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    period TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    story_count INTEGER NOT NULL DEFAULT 0,
    failures TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS digest_run_stories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES digest_runs(id) ON DELETE CASCADE,
    cluster_id INTEGER NOT NULL REFERENCES story_clusters(id) ON DELETE CASCADE,
    comparison_id INTEGER REFERENCES reporting_comparisons(id) ON DELETE SET NULL,
    headline TEXT NOT NULL,
    summary TEXT NOT NULL,
    geo_tier TEXT NOT NULL,
    topic TEXT NOT NULL,
    importance_score REAL NOT NULL,
    source_links TEXT NOT NULL,
    source_count INTEGER NOT NULL,
    is_single_source INTEGER NOT NULL,
    summary_provider TEXT NOT NULL,
    fallback_used INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS source_health (
    source_name TEXT PRIMARY KEY,
    last_success TEXT,
    last_failure TEXT,
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    total_fetches INTEGER NOT NULL DEFAULT 0,
    total_failures INTEGER NOT NULL DEFAULT 0,
    last_article_count INTEGER NOT NULL DEFAULT 0,
    last_error TEXT
);
"""
