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
  - LLM call fails or times out -> returns extractive fallback (first 3 sentences)
A SummarizedCluster is always returned; nothing is silently dropped.

Token budget
------------
Input text is truncated to MAX_INPUT_CHARS (4000) before the LLM call.

Concurrency
-----------
Clusters are summarized in parallel via ThreadPoolExecutor. The worker
count is controlled by settings.summarizer.max_concurrent_summaries
(default 5). Each future has an individual timeout as a final backstop
against a single hanging cluster blocking the pool.

Timeout configuration (settings.yaml)
--------------------------------------
summarizer:
  request_timeout_seconds: 45       # httpx read/write timeout per LLM call
  connect_timeout_seconds: 10       # httpx connect timeout
  article_fetch_timeout_seconds: 15 # per-article fetch timeout
  max_concurrent_summaries: 5       # ThreadPoolExecutor worker cap
  cluster_timeout_seconds: 120      # per-cluster wall-clock timeout guard
"""

from __future__ import annotations

import logging
import os
import re
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from typing import Optional

import httpx

from clustering.clusterer import StoryCluster
from config.loader import get_settings

logger = logging.getLogger(__name__)

MIN_CONTENT_CHARS = 200
MAX_INPUT_CHARS = 4000
_PPLX_API_URL = "https://api.perplexity.ai/chat/completions"
_PPLX_MODEL = "sonar"

_DEFAULT_READ_TIMEOUT = 45.0
_DEFAULT_CONNECT_TIMEOUT = 10.0
_DEFAULT_MAX_WORKERS = 5
_DEFAULT_CLUSTER_TIMEOUT = 120.0


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
    # (source_name, article_url) pairs — one per article in the cluster.
    # Used by the email renderer to build linked source chips.
    source_links: list[tuple[str, str]] = field(default_factory=list)
    # {source_name: lean_label} map for bias color rendering in the email.
    source_bias: dict[str, str] = field(default_factory=dict)
    fallback_used: bool = False


class ArticleFetcher:
    """Fetches and extracts full article text from URLs using trafilatura."""

    def __init__(self, timeout: float = 15.0, user_agent: str = "NewsBot/1.0") -> None:
        self._client = httpx.Client(
            timeout=httpx.Timeout(connect=10.0, read=timeout, write=10.0, pool=5.0),
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
        self._client = httpx.Client(
            timeout=httpx.Timeout(connect=5.0, read=10.0, write=5.0, pool=5.0)
        )

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

        s = self.settings["summarizer"]
        read_timeout: float = float(s.get("request_timeout_seconds", _DEFAULT_READ_TIMEOUT))
        connect_timeout: float = float(s.get("connect_timeout_seconds", _DEFAULT_CONNECT_TIMEOUT))
        article_timeout: float = float(s.get("article_fetch_timeout_seconds", 15.0))
        self._max_workers: int = int(s.get("max_concurrent_summaries", _DEFAULT_MAX_WORKERS))
        self._cluster_timeout: float = float(s.get("cluster_timeout_seconds", _DEFAULT_CLUSTER_TIMEOUT))

        user_agent = self.settings["ingestion"]["user_agent"]

        self._client = httpx.Client(
            timeout=httpx.Timeout(
                connect=connect_timeout,
                read=read_timeout,
                write=connect_timeout,
                pool=5.0,
            )
        )
        self._article_fetcher = ArticleFetcher(
            timeout=article_timeout,
            user_agent=user_agent,
        )
        self._brave = BraveSearchClient()

    def __enter__(self) -> "Summarizer":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()
        self._article_fetcher.close()
        self._brave.close()

    def summarize_all(self, clusters: list[StoryCluster]) -> list[SummarizedCluster]:
        """Summarize every cluster in parallel. Returns results in input order."""
        results: list[Optional[SummarizedCluster]] = [None] * len(clusters)

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            future_to_index: dict[Future[SummarizedCluster], int] = {
                pool.submit(self._summarize_cluster_safe, cluster): idx
                for idx, cluster in enumerate(clusters)
            }
            for future, idx in future_to_index.items():
                cluster = clusters[idx]
                try:
                    results[idx] = future.result(timeout=self._cluster_timeout)
                except FutureTimeoutError:
                    logger.error(
                        "Cluster %d timed out after %.0fs — using extractive fallback",
                        cluster.cluster_id,
                        self._cluster_timeout,
                    )
                    results[idx] = self._fallback_summary(cluster)
                except Exception as exc:
                    logger.error("Cluster %d raised unexpected error: %s", cluster.cluster_id, exc)
                    results[idx] = self._fallback_summary(cluster)

        return [r for r in results if r is not None]

    def _summarize_cluster_safe(self, cluster: StoryCluster) -> SummarizedCluster:
        try:
            return self._summarize_cluster(cluster)
        except Exception as exc:
            logger.error("Summarization failed for cluster %d: %s", cluster.cluster_id, exc)
            return self._fallback_summary(cluster)

    def _summarize_cluster(self, cluster: StoryCluster) -> SummarizedCluster:
        input_text = self._build_input_text(cluster)

        enrich_threshold = float(
            self.settings["summarizer"].get("brave_enrich_threshold", 0.7)
        )
        if cluster.importance_score >= enrich_threshold:
            snippets = self._brave.search(cluster.representative_headline, count=3)
            if snippets:
                input_text += "\n\nADDITIONAL CONTEXT:\n" + "\n".join(snippets)

        input_text = input_text[:MAX_INPUT_CHARS]

        if self._pplx_key:
            try:
                summary = self._call_perplexity(input_text)
                fallback = False
            except Exception as exc:
                logger.warning(
                    "LLM summarization failed for cluster %d: %s",
                    cluster.cluster_id, exc,
                )
                summary = self._extractive_fallback(input_text)
                fallback = True
        else:
            summary = self._extractive_fallback(input_text)
            fallback = True

        # Build source_links and source_bias for the renderer
        source_links: list[tuple[str, str]] = [
            (a.raw.source_name, a.raw.url)
            for a in cluster.articles
            if a.raw.url
        ]
        source_bias: dict[str, str] = {
            a.raw.source_name: (a.raw.bias_lean or "unknown")
            for a in cluster.articles
        }

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
            source_links=source_links,
            source_bias=source_bias,
            fallback_used=fallback,
        )

    def _build_input_text(self, cluster: StoryCluster) -> str:
        parts = []
        for article in cluster.articles:
            full_text = self._article_fetcher.fetch(article.raw.url)
            text = full_text if full_text else article.raw.summary
            # Source names excluded from prompt — see code review item #9
            parts.append(text or "")
        return "\n\n---\n\n".join(parts)

    def _call_perplexity(self, input_text: str) -> str:
        payload = {
            "model": _PPLX_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a neutral news summarizer. Produce a concise 2-3 sentence "
                        "summary containing only verifiable facts. Remove editorial framing, "
                        "emotional language, and opinion. Use plain, direct language."
                    ),
                },
                {"role": "user", "content": input_text},
            ],
            "max_tokens": 150,
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
        return SummarizedCluster(
            cluster_id=cluster.cluster_id,
            headline=cluster.representative_headline,
            summary=cluster.representative_headline,
            sources=list({a.raw.source_name for a in cluster.articles}),
            bias_notes="",
            importance_score=cluster.importance_score,
            tiers=cluster.tiers,
            source_count=cluster.source_count,
            has_cross_lean_coverage=cluster.has_cross_lean_coverage,
            source_links=[(a.raw.source_name, a.raw.url) for a in cluster.articles if a.raw.url],
            source_bias={a.raw.source_name: (a.raw.bias_lean or "unknown") for a in cluster.articles},
            fallback_used=True,
        )
