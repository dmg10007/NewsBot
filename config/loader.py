"""Configuration loader for NewsBot.

Loads YAML config files from config/ and caches them in memory.
Keeps settings separate from secrets, which must come from environment variables.

Dotenv loading
--------------
This module does NOT call load_dotenv(). Environment loading is the sole
responsibility of the process entry point (main.py). Doing it here would:
  1. Double-load when invoked through main.py.
  2. Silently pollute test environments with values from .env.
  3. Make import-order affect whether env vars are visible.
Callers must ensure load_dotenv() (or equivalent) has run before importing
this module in any context where .env values are needed.

Important: both get_settings() and get_sources() use @lru_cache, which means
they return the same object for the entire lifetime of the process. The
returned MappingProxyType is read-only — callers cannot accidentally mutate
the shared cached config.

Operational note: if you edit settings.yaml or sources.yaml while the
scheduler is running, the changes will NOT take effect until the process
is restarted. There is no hot-reload. This is by design to avoid
mid-run config drift, but it means `systemctl restart newsbot` is
required after any config change.
"""

from __future__ import annotations

import types
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"


@lru_cache(maxsize=1)
def get_settings() -> types.MappingProxyType:
    """Load and cache settings.yaml as a read-only mapping.

    Returns a MappingProxyType so callers cannot accidentally mutate the
    shared cached config object. Requires process restart to pick up changes.
    """
    with open(CONFIG_DIR / "settings.yaml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return types.MappingProxyType(data)


@lru_cache(maxsize=1)
def get_sources() -> types.MappingProxyType:
    """Load and cache sources.yaml as a read-only mapping.

    Returns a MappingProxyType so callers cannot accidentally mutate the
    shared cached config object. Requires process restart to pick up changes.
    """
    with open(CONFIG_DIR / "sources.yaml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return types.MappingProxyType(data)
