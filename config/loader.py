"""Configuration loader for NewsBot.

Loads YAML config files from config/ and caches them in memory.
Keeps settings separate from secrets, which must come from environment variables.
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
    with open(CONFIG_DIR / "settings.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def get_sources() -> dict[str, Any]:
    with open(CONFIG_DIR / "sources.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
