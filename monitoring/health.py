"""Source health monitoring.

Tracks fetch success/failure rates per source over time.
Writes a JSON health log that can be inspected manually or by a future
dashboard. Alerts via logging.WARNING when a source is consistently failing.

Design: no external dependencies — plain Python stdlib only.
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

HEALTH_LOG_PATH = Path(os.getenv("NEWSBOT_HEALTH_LOG", "logs/source_health.json"))
FAILURE_ALERT_THRESHOLD = 3  # Consecutive failures before WARNING is logged


@dataclass
class SourceHealth:
    source_name: str
    last_success: Optional[str] = None    # ISO 8601 string
    last_failure: Optional[str] = None
    consecutive_failures: int = 0
    total_fetches: int = 0
    total_failures: int = 0
    last_article_count: int = 0

    @property
    def success_rate(self) -> float:
        if self.total_fetches == 0:
            return 0.0
        return (self.total_fetches - self.total_failures) / self.total_fetches

    @property
    def is_degraded(self) -> bool:
        return self.consecutive_failures >= FAILURE_ALERT_THRESHOLD


class SourceHealthMonitor:
    """Records fetch outcomes and persists health state to a JSON log file."""

    def __init__(self) -> None:
        self._state: dict[str, SourceHealth] = {}
        self._load()

    def record_success(self, source_name: str, article_count: int) -> None:
        h = self._get_or_create(source_name)
        h.last_success = _now_iso()
        h.consecutive_failures = 0
        h.total_fetches += 1
        h.last_article_count = article_count
        if article_count == 0:
            logger.warning(
                "[health] %s: fetch succeeded but returned 0 articles — "
                "selectors may need updating.", source_name
            )
        self._save()

    def record_failure(self, source_name: str, error: str) -> None:
        h = self._get_or_create(source_name)
        h.last_failure = _now_iso()
        h.consecutive_failures += 1
        h.total_fetches += 1
        h.total_failures += 1
        if h.is_degraded:
            logger.warning(
                "[health] %s: %d consecutive failures. Last error: %s",
                source_name, h.consecutive_failures, error,
            )
        self._save()

    def report(self) -> list[dict]:
        """Return current health state as a list of dicts. Useful for logging."""
        return [asdict(h) for h in self._state.values()]

    def degraded_sources(self) -> list[str]:
        return [name for name, h in self._state.items() if h.is_degraded]

    def _get_or_create(self, source_name: str) -> SourceHealth:
        if source_name not in self._state:
            self._state[source_name] = SourceHealth(source_name=source_name)
        return self._state[source_name]

    def _save(self) -> None:
        HEALTH_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(HEALTH_LOG_PATH, "w", encoding="utf-8") as f:
                json.dump(
                    {name: asdict(h) for name, h in self._state.items()},
                    f, indent=2
                )
        except OSError as exc:
            logger.error("Failed to write health log: %s", exc)

    def _load(self) -> None:
        if not HEALTH_LOG_PATH.exists():
            return
        try:
            with open(HEALTH_LOG_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for name, data in raw.items():
                self._state[name] = SourceHealth(**data)
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            logger.warning("Could not load health log: %s. Starting fresh.", exc)


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()
