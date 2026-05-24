"""Shared SentenceTransformer model registry.

Problem solved
--------------
Both Deduplicator and StoryClusterer previously lazy-loaded their own
SentenceTransformer instance. The same model (~400 MB for all-MiniLM-L6-v2)
was loaded into RAM twice on every run — wasting ~800 MB and adding
significant startup time.

This module provides a process-lifetime singleton registry. Any component
that needs a sentence-transformer model calls get_model(name) and gets back
the same cached instance. The registry is populated lazily: the first call
for a given model name triggers the load; all subsequent calls return the
cached object immediately.

Thread safety
-------------
The registry uses a simple dict. If two threads request the same model
simultaneously before it is loaded, both will trigger a load and the second
will overwrite the first. This is acceptable — the worst case is two loads
on the very first call, which converges to one instance afterward. A Lock
would prevent this but adds complexity not warranted for the current
single-threaded scheduler architecture.
"""

from __future__ import annotations

import logging

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_registry: dict[str, SentenceTransformer] = {}


def get_model(name: str) -> SentenceTransformer:
    """Return a cached SentenceTransformer for *name*, loading it on first use.

    Args:
        name: Model name or path accepted by SentenceTransformer(), e.g.
              'all-MiniLM-L6-v2'.

    Returns:
        The shared SentenceTransformer instance for that model.
    """
    if name not in _registry:
        logger.info("Loading sentence-transformer model: %s", name)
        _registry[name] = SentenceTransformer(name)
    return _registry[name]
