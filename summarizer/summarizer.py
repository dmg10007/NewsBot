"""Article summarizer: fetches full article text and produces bias-free summaries.

Pipeline per cluster:
  1. Attempt to fetch full article text via ArticleFetcher (httpx + trafilatura)
  2. Fall back to RSS summary if fetch fails or yields < MIN_CONTENT_CHARS
  3. Optionally enrich with Brave Search context for high-importance clusters
  4. Call Perplexity Sonar (or extractive fallback) to produce a neutral summary
  5. Return a SummarizedCluster ready for digest rendering

Connection management
---------------------
Summarizer owns three httpx.Client instances (via ArticleFetcher and
BraveSearchClient). Always call summarizer.close() when done, or use it
as a context manager:

    with Summarizer() as s:
        results = s.summarize_all(clusters)

Failure modes
-------------
The summarizer degrades gracefully:
  - Full text fetch fails  -> uses RSS summary
  - Brave enrichment fails -> skips enrichment, continues with available text
  - LLM call fails         -> returns extractive fallback (first 3 sentences)
A SummarizedCluster is always returned; nothing is silently dropped.

Token budget
------------
Input text is truncated to MAX_INPUT_CHARS (4000) before the LLM call.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

import httpx

from clustering.clusterer import StoryCluster
from config.loader import get_settings

logger = logging.getLogger(__name__)

MIN_CONTENT_CHARS = 200
MAX_INPUT_CHARS = 4000
_PPLX_API_URL = "https://api.perplexity.ai/chat/completions"
_PPLX_MODEL = "sonar"


@dataclass
class SummarizedCluster:
    """A story cluster with a bias-free summary ready for digest rendering."""
    cluster_id: int
    headline: str
    summary: str
    sources: list[str]
    bias_notes: str
    importance_score: float
    tiers: list[str]
    source_count: int
    has_cross_lean_coverage: bool
    fallback_used: bool = False


class ArticleFetcher:
    """Fetches and extracts full article text from URLs using trafilatura."""

    def __init__(self, timeout: float = 15.0, user_agent: str = "NewsBot/1.0") -> None:
        self._client = httpx.Client(
            timeout=timeout,
            headers={"User-Agent": user_agent},
            follow_redirects=True,
        )

    def fetch(self, url: str) -> Optional[str]:
        """Return extracted article text, or None on failure."""
        try:
            import trafilatura
            response = self._client.get(url)
            response.raise_for_status()
            text = trafilatura.extract(response.text)
            if text and len(text) >= MIN_CONTENT_CHARS:
                return text
        except Exception as exc:
            logger.debug("ArticleFetcher failed for %s: %s", url, exc)
        return None

    def close(self) -> None:
        self._client.close()


class BraveSearchClient:
    """Fetches additional context for high-importance clusters via Brave Search API."""

    def __init__(self) -> None:
        self._api_key = os.getenv("BRAVE_SEARCH_API_KEY", "")
        self._client = httpx.Client(timeout=10.0)

    def search(self, query: str, count: int = 3) -> list[str]:
        """Return a list of snippet strings for the query, or [] on failure."""
        if not self._api_key:
            return []
        try:
            response = self._client.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": count},
                headers={
                    "Accept": "application/json",
                    "X-Subscription-Token": self._api_key,
                },
            )
            response.raise_for_status()
            results = response.json().get("web", {}).get("results", [])
            return [r.get("description", "") for r in results if r.get("description")]
        except Exception as exc:
            logger.debug("Brave search failed for query '%s': %s", query, exc)
            return []

    def close(self) -> None:
        self._client.close()


class Summarizer:
    """Produces neutral summaries for each StoryCluster.

    Lifecycle: call close() (or use as a context manager) when done to release
    the underlying httpx connection pools.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._pplx_key = os.getenv("PPLX_API_KEY", "")
        timeout = self.settings["summarizer"]["request_timeout_seconds"]
        user_agent = self.settings["ingestion"]["user_agent"]
        self._client = httpx.Client(timeout=timeout)
        self._article_fetcher = ArticleFetcher(
            timeout=self.settings["summarizer"]["article_fetch_timeout_seconds"],
            user_agent=user_agent,
        )
        self._brave = BraveSearchClient()

    def __enter__(self) -> "Summarizer":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        """Release all underlying httpx connection pools.

        Must be called when the Summarizer is no longer needed, unless the
        instance is used as a context manager (which calls this automatically).
        The scheduler should use:

            with Summarizer() as s:
                results = s.summarize_all(clusters)

        Or with explicit try/finally:

            summarizer = Summarizer()
            try:
                results = summarizer.summarize_all(clusters)
            finally:
                summarizer.close()
        """
        self._client.close()
        self._article_fetcher.close()
        self._brave.close()

    def summarize_all(self, clusters: list[StoryCluster]) -> list[SummarizedCluster]:
        """Summarize every cluster. Returns results in input order."""
        results = []
        for cluster in clusters:
            try:
                results.append(self._summarize_cluster(cluster))
            except Exception as exc:
                logger.error("Summarization failed for cluster %d: %s", cluster.cluster_id, exc)
                results.append(self._fallback_summary(cluster))
        return results

    def _summarize_cluster(self, cluster: StoryCluster) -> SummarizedCluster:
        input_text = self._build_input_text(cluster)

        if cluster.importance_score >= self.settings["summarizer"].get("brave_enrich_threshold", 0.7):
            snippets = self._brave.search(cluster.representative_headline, count=3)
            if snippets:
                input_text += "\n\nADDITIONAL CONTEXT:\n" + "\n".join(snippets)

        input_text = input_text[:MAX_INPUT_CHARS]

        if self._pplx_key:
            try:
                summary = self._call_perplexity(input_text)
                fallback = False
            except Exception as exc:
                logger.warning("LLM summarization failed for cluster %d: %s", cluster.cluster_id, exc)
                summary = self._extractive_fallback(input_text)
                fallback = True
        else:
            summary = self._extractive_fallback(input_text)
            fallback = True

        return SummarizedCluster(
            cluster_id=cluster.cluster_id,
            headline=cluster.representative_headline,
            summary=summary,
            sources=list({a.raw.source_name for a in cluster.articles}),
            bias_notes="",
            importance_score=cluster.importance_score,
            tiers=cluster.tiers,
            source_count=cluster.source_count,
            has_cross_lean_coverage=cluster.has_cross_lean_coverage,
            fallback_used=fallback,
        )

    def _build_input_text(self, cluster: StoryCluster) -> str:
        parts = []
        for article in cluster.articles:
            full_text = self._article_fetcher.fetch(article.raw.url)
            text = full_text if full_text else article.raw.summary
            parts.append(f"SOURCE: {article.raw.source_name}\n{text}")
        return "\n\n---\n\n".join(parts)

    def _call_perplexity(self, input_text: str) -> str:
        payload = {
            "model": _PPLX_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a neutral news summarizer. Produce a concise 3-5 sentence "
                        "summary containing only verifiable facts. Remove editorial framing, "
                        "emotional language, and opinion. Use plain, direct language."
                    ),
                },
                {"role": "user", "content": input_text},
            ],
            "max_tokens": 300,
            "temperature": 0.1,
        }
        response = self._client.post(
            _PPLX_API_URL,
            json=payload,
            headers={
                "Authorization": f"Bearer {self._pplx_key}",
                "Content-Type": "application/json",
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    def _extractive_fallback(self, text: str) -> str:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return " ".join(sentences[:3])

    def _fallback_summary(self, cluster: StoryCluster) -> SummarizedCluster:
        headline = cluster.representative_headline
        return SummarizedCluster(
            cluster_id=cluster.cluster_id,
            headline=headline,
            summary=headline,
            sources=list({a.raw.source_name for a in cluster.articles}),
            bias_notes="",
            importance_score=cluster.importance_score,
            tiers=cluster.tiers,
            source_count=cluster.source_count,
            has_cross_lean_coverage=cluster.has_cross_lean_coverage,
            fallback_used=True,
        )
