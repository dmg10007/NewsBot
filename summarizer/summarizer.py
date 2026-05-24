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
  - Full text fetch fails       -> uses RSS summary
  - Brave enrichment fails      -> skips enrichment, continues
  - LLM call fails after retry  -> extractive fallback (first 3 sentences)
  - Per-run cap exceeded        -> extractive fallback
  - Importance score too low    -> extractive fallback (no API call made)
A SummarizedCluster is always returned; nothing is silently dropped.

Token budget
------------
Input text is truncated to MAX_INPUT_CHARS (4000) before the LLM call.

Concurrency
-----------
Clusters are summarized in parallel via ThreadPoolExecutor. The worker
count is controlled by settings.summarizer.max_concurrent_summaries
(default 3). Each future has an individual timeout as a final backstop
against a single hanging cluster blocking the pool.

Perplexity rate-limit strategy
-------------------------------
Three overlapping guards prevent 429 errors:

  1. _pplx_throttle_lock  — a threading.Lock serialising the inter-call
     delay so parallel workers queue behind each other rather than firing
     simultaneously.
  2. _pplx_calls / pplx_max_calls_per_run  — hard cap; clusters over the
     cap receive extractive fallback immediately.
  3. _call_perplexity_with_retry()  — catches 429 (and 5xx) responses and
     retries with exponential backoff before giving up. Respects the
     Retry-After header when present.
  4. Importance gate  — clusters below pplx_min_importance_score never
     reach the API; extractive fallback is adequate for low-signal stories.

Timeout configuration (settings.yaml)
--------------------------------------
summarizer:
  request_timeout_seconds: 45
  connect_timeout_seconds: 10
  article_fetch_timeout_seconds: 15
  max_concurrent_summaries: 3
  cluster_timeout_seconds: 120
  pplx_call_delay_seconds: 1.2
  pplx_max_calls_per_run: 15
  pplx_min_importance_score: 0.3
  pplx_retry_attempts: 3
  pplx_retry_base_delay: 2.0
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
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
_DEFAULT_MAX_WORKERS = 3
_DEFAULT_CLUSTER_TIMEOUT = 120.0
_DEFAULT_PPLX_DELAY = 1.2
_DEFAULT_PPLX_MAX_CALLS = 15
_DEFAULT_PPLX_MIN_SCORE = 0.3
_DEFAULT_PPLX_RETRY_ATTEMPTS = 3
_DEFAULT_PPLX_RETRY_BASE_DELAY = 2.0


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

        # Rate-limit controls
        self._pplx_call_delay: float = float(s.get("pplx_call_delay_seconds", _DEFAULT_PPLX_DELAY))
        self._pplx_max_calls: int = int(s.get("pplx_max_calls_per_run", _DEFAULT_PPLX_MAX_CALLS))
        self._pplx_min_score: float = float(s.get("pplx_min_importance_score", _DEFAULT_PPLX_MIN_SCORE))
        self._pplx_retry_attempts: int = int(s.get("pplx_retry_attempts", _DEFAULT_PPLX_RETRY_ATTEMPTS))
        self._pplx_retry_base_delay: float = float(s.get("pplx_retry_base_delay", _DEFAULT_PPLX_RETRY_BASE_DELAY))

        # Thread-safe call counter and throttle lock
        self._pplx_calls: int = 0
        self._pplx_call_lock: threading.Lock = threading.Lock()
        # Shared lock that serialises the inter-call delay across workers
        self._pplx_throttle_lock: threading.Lock = threading.Lock()

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

        summary: str
        fallback: bool

        if not self._pplx_key:
            summary = self._extractive_fallback(input_text)
            fallback = True
        elif cluster.importance_score < self._pplx_min_score:
            # Importance gate: low-signal clusters don't warrant an API call
            logger.debug(
                "Cluster %d below importance threshold (%.2f < %.2f) — using extractive fallback",
                cluster.cluster_id, cluster.importance_score, self._pplx_min_score,
            )
            summary = self._extractive_fallback(input_text)
            fallback = True
        else:
            # Check and increment the per-run cap atomically
            with self._pplx_call_lock:
                if self._pplx_calls >= self._pplx_max_calls:
                    logger.info(
                        "Cluster %d: Perplexity cap (%d) reached — using extractive fallback",
                        cluster.cluster_id, self._pplx_max_calls,
                    )
                    summary = self._extractive_fallback(input_text)
                    fallback = True
                else:
                    self._pplx_calls += 1
                    do_call = True

            if not fallback:  # type: ignore[possibly-undefined]
                summary, fallback = self._call_with_throttle_and_retry(
                    cluster.cluster_id, input_text
                )

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

    def _call_with_throttle_and_retry(
        self, cluster_id: int, input_text: str
    ) -> tuple[str, bool]:
        """Acquire throttle lock, enforce inter-call delay, then call with retry.

        The throttle lock serialises all workers so the delay is wall-clock
        time between consecutive calls, not just per-worker.
        """
        with self._pplx_throttle_lock:
            time.sleep(self._pplx_call_delay)
            try:
                summary = self._call_perplexity_with_retry(input_text)
                return summary, False
            except Exception as exc:
                logger.warning(
                    "Perplexity failed for cluster %d after retries: %s — using extractive fallback",
                    cluster_id, exc,
                )
                # Decrement counter so a retry run can use the slot
                with self._pplx_call_lock:
                    self._pplx_calls = max(0, self._pplx_calls - 1)
                return self._extractive_fallback(input_text), True

    def _call_perplexity_with_retry(self, input_text: str) -> str:
        """Call Perplexity with exponential backoff on 429 / 5xx responses.

        Retries up to self._pplx_retry_attempts times. On a 429, respects
        the Retry-After header if present; otherwise uses exponential backoff
        starting at pplx_retry_base_delay seconds.

        Raises the last exception if all attempts are exhausted.
        """
        last_exc: Exception = RuntimeError("No attempts made")
        for attempt in range(self._pplx_retry_attempts + 1):
            try:
                return self._call_perplexity(input_text)
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                status = exc.response.status_code
                if status == 429 or status >= 500:
                    if attempt < self._pplx_retry_attempts:
                        retry_after = exc.response.headers.get("Retry-After")
                        if retry_after:
                            wait = float(retry_after)
                        else:
                            wait = self._pplx_retry_base_delay * (2 ** attempt)
                        logger.warning(
                            "Perplexity returned %d — retrying in %.1fs (attempt %d/%d)",
                            status, wait, attempt + 1, self._pplx_retry_attempts,
                        )
                        time.sleep(wait)
                        continue
                # Non-retryable status or final attempt
                raise
            except Exception as exc:
                last_exc = exc
                raise
        raise last_exc

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

    def _build_input_text(self, cluster: StoryCluster) -> str:
        parts = []
        for article in cluster.articles:
            full_text = self._article_fetcher.fetch(article.raw.url)
            text = full_text if full_text else article.raw.summary
            # Source names excluded from prompt — see code review item #9
            parts.append(text or "")
        return "\n\n---\n\n".join(parts)

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
