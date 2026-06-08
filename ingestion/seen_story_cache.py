"""Cross-run seen-story cache.

Persists url_hash values across digest runs so a story that appeared in a
previous briefing cannot reappear while it is still live in source RSS feeds.

Design
------
Storage: JSON file (default: data/seen_stories.json). No database dependency.
Key:     RawArticle.url_hash (SHA-256 hex of the normalized URL).
Value:   ISO 8601 UTC timestamp of when the story was first seen.
Expiry:  TTL-based. Entries older than ttl_days are pruned on every write.

Thread safety: single-process use only. The scheduler runs one digest at a
time (coalesce=True in APScheduler config), so no locking is needed.

Atomic writes
-------------
The JSON file is written to a .tmp sibling first, then renamed into place.
This prevents a corrupt cache file if the process crashes mid-write.

Empty / missing file
--------------------
If the file does not exist or is empty/corrupt, the cache starts fresh and
logs at WARNING level. Worst outcome: one run where already-seen stories
can reappear.

Usage (see ingestion/pipeline.py)::

    cache = SeenStoryCache(settings)
    articles = cache.filter_seen(articles)
    # ... dedup, extract, cluster, deliver ...
    cache.mark_seen(delivered_articles)
    cache.save()
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ingestion.fetcher import RawArticle

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_PATH = "data/seen_stories.json"
_DEFAULT_TTL_DAYS = 7


class SeenStoryCache:
    """Persists seen url_hashes across digest runs with TTL expiry.

    Args:
        settings: Full settings dict from config.loader.get_settings().
    """

    def __init__(self, settings: dict) -> None:
        cfg = settings.get("seen_story_cache", {})
        self._enabled: bool = bool(cfg.get("enabled", True))
        self._path = Path(cfg.get("path", _DEFAULT_CACHE_PATH))
        self._ttl = timedelta(days=int(cfg.get("ttl_days", _DEFAULT_TTL_DAYS)))
        self._seen: dict[str, str] = {}  # {url_hash: iso_timestamp_str}
        if self._enabled:
            self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def filter_seen(self, articles: list["RawArticle"]) -> list["RawArticle"]:
        """Return only articles whose url_hash has not been seen before.

        Args:
            articles: Ingested RawArticle objects from ingest_all_sources().

        Returns:
            Filtered list with previously-seen stories removed.
        """
        if not self._enabled:
            return articles

        fresh, skipped = [], 0
        for article in articles:
            if article.url_hash in self._seen:
                skipped += 1
                logger.debug(
                    "[SeenCache] Skipping already-seen story: %s",
                    article.headline,
                )
            else:
                fresh.append(article)

        logger.info(
            "[SeenCache] %d/%d articles fresh (skipped %d already-seen)",
            len(fresh), len(articles), skipped,
        )
        return fresh

    def mark_seen(self, articles: list["RawArticle"]) -> None:
        """Record url_hashes as seen with the current UTC timestamp.

        Call after successful delivery so only stories that were actually
        included in a briefing are suppressed in future runs.

        Args:
            articles: Articles delivered in this run.
        """
        if not self._enabled:
            return
        now_iso = datetime.now(timezone.utc).isoformat()
        for article in articles:
            self._seen[article.url_hash] = now_iso

    def save(self) -> None:
        """Prune expired entries and atomically write the cache to disk."""
        if not self._enabled:
            return
        self._prune()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd, tmp_path = tempfile.mkstemp(
                dir=self._path.parent, suffix=".tmp"
            )
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self._seen, f, indent=2)
            os.replace(tmp_path, self._path)
            logger.debug(
                "[SeenCache] Saved %d entries to %s", len(self._seen), self._path
            )
        except OSError as exc:
            logger.warning("[SeenCache] Failed to save cache: %s", exc)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load the cache file from disk. Starts fresh on any error."""
        if not self._path.exists():
            logger.debug("[SeenCache] No cache file at %s — starting fresh", self._path)
            return
        try:
            with self._path.open(encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("Expected a JSON object")
            self._seen = data
            self._prune()
            logger.info(
                "[SeenCache] Loaded %d entries from %s", len(self._seen), self._path
            )
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            logger.warning(
                "[SeenCache] Cache file corrupt or unreadable (%s) — starting fresh", exc
            )
            self._seen = {}

    def _prune(self) -> None:
        """Remove entries older than ttl_days."""
        cutoff = datetime.now(timezone.utc) - self._ttl
        before = len(self._seen)
        self._seen = {
            h: ts
            for h, ts in self._seen.items()
            if self._parse_ts(ts) >= cutoff
        }
        pruned = before - len(self._seen)
        if pruned:
            logger.debug("[SeenCache] Pruned %d expired entries", pruned)

    @staticmethod
    def _parse_ts(ts_str: str) -> datetime:
        """Parse ISO 8601 timestamp to tz-aware UTC datetime.

        Falls back to epoch on parse failure so malformed entries are
        treated as maximally old and pruned immediately.
        """
        try:
            dt = datetime.fromisoformat(ts_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, TypeError):
            return datetime.fromtimestamp(0, tz=timezone.utc)
