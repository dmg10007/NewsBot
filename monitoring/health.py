"""Source health monitoring.

Tracks fetch success/failure rates per source over time.
Writes a JSON health log that can be inspected manually or by a future
dashboard. Alerts via logging.WARNING when a source is consistently failing.

Design: no external dependencies — plain Python stdlib only.

Module-level helpers
--------------------
record_run() is a convenience wrapper so callers (e.g. scheduler.py) can
log a completed digest run without instantiating SourceHealthMonitor directly.

Atomic writes
-------------
Both _save() and record_run() write via a temp file + os.replace() so that
a process crash mid-write never leaves the JSON file truncated or empty.
On POSIX, _flock() adds LOCK_EX / LOCK_SH guards against two simultaneous
processes clobbering each other's entries. On Windows, _flock() is a no-op;
the temp-file + os.replace() pattern still provides crash-safety.

Run history rotation
--------------------
record_run() caps the run history at monitoring.max_run_history_entries
(default 500) entries.

Failure alert threshold
-----------------------
Read from monitoring.failure_alert_threshold in settings.yaml (default 3).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config.loader import get_settings

logger = logging.getLogger(__name__)

HEALTH_LOG_PATH = Path(os.getenv("NEWSBOT_HEALTH_LOG", "logs/source_health.json"))
RUN_LOG_PATH = Path(os.getenv("NEWSBOT_RUN_LOG", "logs/run_history.json"))

_DEFAULT_FAILURE_ALERT_THRESHOLD = 3
_DEFAULT_MAX_RUN_HISTORY = 500

# ---------------------------------------------------------------------------
# Cross-platform file locking shim
# ---------------------------------------------------------------------------
# fcntl is POSIX-only.  On Windows we fall back to a no-op because the
# atomic temp-file + os.replace() write already guarantees crash-safety;
# the lock only guards against two *concurrent* processes — an edge case
# that is uncommon on Windows developer machines.

if sys.platform != "win32":
    import fcntl as _fcntl

    def _flock(file_obj, exclusive: bool) -> None:
        op = _fcntl.LOCK_EX if exclusive else _fcntl.LOCK_SH
        _fcntl.flock(file_obj, op)
else:
    def _flock(file_obj, exclusive: bool) -> None:  # type: ignore[misc]
        pass  # no-op on Windows


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SourceHealth:
    source_name: str
    last_success: Optional[str] = None
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
        return self.consecutive_failures >= _failure_alert_threshold()


def _failure_alert_threshold() -> int:
    return int(
        get_settings()
        .get("monitoring", {})
        .get("failure_alert_threshold", _DEFAULT_FAILURE_ALERT_THRESHOLD)
    )


def _max_run_history() -> int:
    return int(
        get_settings()
        .get("monitoring", {})
        .get("max_run_history_entries", _DEFAULT_MAX_RUN_HISTORY)
    )


# ---------------------------------------------------------------------------
# Atomic JSON write
# ---------------------------------------------------------------------------

def _atomic_json_write(path: Path, data: object) -> None:
    """Write *data* to *path* atomically via a sibling temp file + os.replace()."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as tmp_f:
            _flock(tmp_f, exclusive=True)
            json.dump(data, tmp_f, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# SourceHealthMonitor
# ---------------------------------------------------------------------------

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
        return [asdict(h) for h in self._state.values()]

    def degraded_sources(self) -> list[str]:
        return [name for name, h in self._state.items() if h.is_degraded]

    def _get_or_create(self, source_name: str) -> SourceHealth:
        if source_name not in self._state:
            self._state[source_name] = SourceHealth(source_name=source_name)
        return self._state[source_name]

    def _save(self) -> None:
        try:
            _atomic_json_write(
                HEALTH_LOG_PATH,
                {name: asdict(h) for name, h in self._state.items()},
            )
        except OSError as exc:
            logger.error("Failed to write health log: %s", exc)

    def _load(self) -> None:
        if not HEALTH_LOG_PATH.exists():
            return
        try:
            with open(HEALTH_LOG_PATH, "r", encoding="utf-8") as f:
                _flock(f, exclusive=False)
                raw = json.load(f)
            for name, data in raw.items():
                self._state[name] = SourceHealth(**data)
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            logger.warning("Could not load health log: %s. Starting fresh.", exc)


# ---------------------------------------------------------------------------
# record_run convenience wrapper
# ---------------------------------------------------------------------------

def record_run(
    period: str,
    story_count: int,
    elapsed_seconds: float,
) -> None:
    """Append a digest run entry to run_history.json."""
    entry = {
        "timestamp": _now_iso(),
        "period": period,
        "story_count": story_count,
        "elapsed_seconds": round(elapsed_seconds, 2),
    }

    history: list[dict] = []
    if RUN_LOG_PATH.exists():
        try:
            with open(RUN_LOG_PATH, "r", encoding="utf-8") as f:
                _flock(f, exclusive=False)
                history = json.load(f)
        except (OSError, json.JSONDecodeError):
            logger.warning("Could not read run history log — starting fresh.")

    history.append(entry)

    cap = _max_run_history()
    if len(history) > cap:
        history = history[-cap:]

    try:
        _atomic_json_write(RUN_LOG_PATH, history)
        logger.debug(
            "run_history: period=%s stories=%d elapsed=%.1fs",
            period, story_count, elapsed_seconds,
        )
    except OSError as exc:
        logger.error("Failed to write run history log: %s", exc)


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()
