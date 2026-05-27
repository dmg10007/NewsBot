"""Digest run logging.

Appends a structured JSON entry for each digest run, recording:
  - run timestamp and period
  - article counts at each pipeline stage
  - LLM calls made and provider used
  - delivery success/failure
  - any degraded sources at time of run

This is your paper trail for debugging quality issues and tracking
whether the bot is actually running cleanly.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DIGEST_LOG_PATH = Path(os.getenv("NEWSBOT_DIGEST_LOG", "logs/digest_runs.jsonl"))

# Uses a plain dict instead of a dataclass for stdlib compatibility.


class DigestRunLogger:
    """Appends one JSONL entry per digest run to the digest log file."""

    def __init__(self) -> None:
        self._entry: dict[str, Any] = {
            "run_at": datetime.now(tz=timezone.utc).isoformat(),
            "period": None,
            "raw_articles": 0,
            "after_dedup": 0,
            "after_lookback_filter": 0,
            "clusters": 0,
            "llm_calls_made": 0,
            "llm_provider": None,
            "summaries_generated": 0,
            "email_sent": False,
            "telegram_sent": False,
            "degraded_sources": [],
            "errors": [],
        }

    def set(self, **kwargs: Any) -> None:
        """Update one or more log fields."""
        self._entry.update(kwargs)

    def add_error(self, error: str) -> None:
        self._entry["errors"].append(error)

    def commit(self) -> None:
        """Write the completed entry to the JSONL log file."""
        DIGEST_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(DIGEST_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(self._entry) + "\n")
            logger.debug("Digest run logged to %s", DIGEST_LOG_PATH)
        except OSError as exc:
            logger.error("Failed to write digest log: %s", exc)
