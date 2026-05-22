"""Configuration loader for NewsBot.

Loads YAML config files from config/ and caches them in memory.
Keeps settings separate from secrets, which must come from environment variables.

Important: both get_settings() and get_sources() use @lru_cache, which means
they return the same dict object for the entire lifetime of the process. This
is intentional — config is read once at startup and held in memory.

Operational note: if you edit settings.yaml or sources.yaml while the
scheduler is running, the changes will NOT take effect until the process
is restarted. There is no hot-reload. This is by design to avoid
mid-run config drift, but it means `systemctl restart newsbot` is
required after any config change.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"


@lru_cache(maxsize=1)
def get_settings() -> dict[str, Any]:
    """Load and cache settings.yaml. Requires process restart to pick up changes."""
    with open(CONFIG_DIR / "settings.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def get_sources() -> dict[str, Any]:
    """Load and cache sources.yaml. Requires process restart to pick up changes."""
    with open(CONFIG_DIR / "sources.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
