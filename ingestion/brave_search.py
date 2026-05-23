"""Brave Web Search API wrapper.

Used as a fallback enrichment source when:
  1. The RSS description is empty or a headline echo, AND
  2. Direct article body fetching fails or returns too little text.

The Brave Web Search API returns result snippets (150-200 chars each)
for a query. We concatenate the top snippets into a pseudo-body that
the summarizer can use instead of falling back to the entity last-resort.

Docs: https://api.search.brave.com/app/documentation/web-search/get-started
Free tier: 2,000 queries/month (as of 2026)

Design notes:
  - Only instantiated if BRAVE_SEARCH_API_KEY is set; callers check .available
  - Returns empty string on any failure
  - Queries are the story headline — short, specific, safe
  - max_results=5 is enough for a summary; keeps response payload small
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.search.brave.com/res/v1/web/search"
_TIMEOUT = 8.0
_MAX_RESULTS = 5


class BraveSearch:
    """Thin wrapper around the Brave Web Search API."""

    def __init__(self) -> None:
        self._api_key: Optional[str] = os.getenv("BRAVE_SEARCH_API_KEY", "") or None
        self._client = httpx.Client(
            timeout=_TIMEOUT,
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": self._api_key or "",
            },
        )

    @property
    def available(self) -> bool:
        """True if a Brave API key is configured."""
        return bool(self._api_key)

    def search_snippets(self, query: str) -> str:
        """Search for query and return concatenated result snippets.

        Returns empty string if the API is unavailable, the key is missing,
        or the request fails for any reason.
        """
        if not self.available:
            return ""
        try:
            response = self._client.get(
                _BASE_URL,
                params={
                    "q": query,
                    "count": _MAX_RESULTS,
                    "text_decorations": False,
                    "search_lang": "en",
                    "country": "us",
                    "safesearch": "moderate",
                },
            )
            response.raise_for_status()
            data = response.json()
            results = data.get("web", {}).get("results", [])
            snippets = [
                r.get("description", "").strip()
                for r in results
                if r.get("description", "").strip()
            ]
            combined = " ".join(snippets)
            logger.debug(
                "Brave search for '%s' returned %d snippets (%d chars)",
                query[:60], len(snippets), len(combined),
            )
            return combined
        except Exception as exc:
            logger.warning("Brave search failed for '%s': %s", query[:60], exc)
            return ""

    def close(self) -> None:
        self._client.close()
