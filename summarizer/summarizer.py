"""Article summarizer: fetches full article text and produces bias-free summaries.

Pipeline per cluster:
  1. Attempt to fetch full article text via ArticleFetcher (httpx + trafilatura)
  2. Fall back to RSS summary if fetch fails or yields < MIN_CONTENT_CHARS
  3. Optionally enrich with Brave Search context for high-importance clusters
  4. Route to the appropriate LLM based on importance score and API caps:
       - High importance + cap not reached  -> Perplexity Sonar
       - Low importance OR cap reached      -> local llama.cpp
       - Local model unavailable            -> extractive fallback (3 sentences)
  5. Return a SummarizedCluster ready for digest rendering

Connection management
---------------------
Summarizer owns four httpx.Client instances (via ArticleFetcher,
BraveSearchClient, and LocalLLMClient). Always call summarizer.close()
when done, or use it as a context manager:

    with Summarizer() as s:
        results = s.summarize_all(clusters)

Failure modes
-------------
The summarizer degrades gracefully:
  - Full text fetch fails              -> uses RSS summary
  - Brave enrichment fails             -> skips enrichment, continues
  - Perplexity fails after retry       -> routes to local LLM
  - Local LLM unavailable / fails      -> extractive fallback (3 sentences)
  - Per-run Perplexity cap exceeded    -> routes to local LLM
  - Importance score below threshold   -> routes to local LLM
A SummarizedCluster is always returned; nothing is silently dropped.

Token budget
------------
Input text is truncated to MAX_INPUT_CHARS (4000) before any LLM call.

Concurrency
-----------
Clusters are summarized in parallel via ThreadPoolExecutor. The worker
count is controlled by settings.summarizer.max_concurrent_summaries
(default 3). Each future has an individual timeout as a final backstop
against a single hanging cluster blocking the pool.

Perplexity rate-limit strategy
-------------------------------
Four overlapping guards prevent 429 errors:

  1. Importance gate — clusters below pplx_min_importance_score are
     routed to the local LLM without touching the Perplexity API.
  2. Per-run cap — _pplx_calls counter (thread-safe); clusters over
     pplx_max_calls_per_run are routed to the local LLM.
  3. _pplx_throttle_lock — serialises the inter-call delay across all
     workers so they queue rather than burst.
  4. _call_perplexity_with_retry() — exponential backoff on 429/5xx;
     on final failure routes to local LLM (not bare extractive).

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
  model: llama3           # local model name passed to llama.cpp

Environment variables
---------------------
  PPLX_API_KEY         — enables Perplexity path (required for cloud summarization)
  LLAMA_CPP_BASE_URL   — base URL of local llama.cpp server
                         (default: http://localhost:8080)
  LLAMA_CPP_MODEL      — overrides settings.summarizer.model for local path
"""

from __future__ import annotations

import html as html_module
import logging
import os
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Optional

import httpx

from clustering.clusterer import StoryCluster
from config.loader import get_settings

logger = logging.getLogger(__name__)

MIN_CONTENT_CHARS = 200
MAX_INPUT_CHARS = 4000
_PPLX_API_URL = "https://api.perplexity.ai/chat/completions"
_PPLX_MODEL = "sonar"
_LOCAL_LLM_DEFAULT_BASE = "http://localhost:8080"

_DEFAULT_READ_TIMEOUT = 45.0
_DEFAULT_CONNECT_TIMEOUT = 10.0
_DEFAULT_MAX_WORKERS = 3
_DEFAULT_CLUSTER_TIMEOUT = 120.0
_DEFAULT_PPLX_DELAY = 1.2
_DEFAULT_PPLX_MAX_CALLS = 15
_DEFAULT_PPLX_MIN_SCORE = 0.3
_DEFAULT_PPLX_RETRY_ATTEMPTS = 3
_DEFAULT_PPLX_RETRY_BASE_DELAY = 2.0

_NEUTRAL_SYSTEM_PROMPT = (
    "You are a neutral news summarizer. Produce a concise 2-3 sentence "
    "summary containing only verifiable facts. Remove editorial framing, "
    "emotional language, and opinion. Use plain, direct language."
)


# ---------------------------------------------------------------------------
# HTML stripping
# ---------------------------------------------------------------------------

class _TextExtractor(HTMLParser):
    """stdlib HTMLParser subclass that discards all tags and collects text."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:  # noqa: D401
        stripped = data.strip()
        if stripped:
            self._parts.append(stripped)

    def get_text(self) -> str:
        return " ".join(self._parts)


def _strip_html(text: str) -> str:
    """Return plain text with all HTML tags removed and entities decoded.

    Uses stdlib html.parser — no extra dependencies. Safe to call on
    strings that contain no HTML; returns them unchanged.
    """
    if not text or "<" not in text:
        return text
    parser = _TextExtractor()
    try:
        parser.feed(text)
        return parser.get_text()
    except Exception:
        # Malformed HTML edge case — fall back to regex strip
        return re.sub(r"<[^>]+>", " ", html_module.unescape(text)).strip()


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
    # True only for pure extractive (3-sentence) fallback — not for local LLM.
    fallback_used: bool = False
    # Which backend produced this summary: "perplexity" | "local" | "extractive"
    summary_backend: str = "extractive"


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


class LocalLLMClient:
    """Calls a local llama.cpp server via its OpenAI-compatible chat endpoint.

    The server must be running and expose POST /v1/chat/completions.
    Base URL is read from LLAMA_CPP_BASE_URL (default: http://localhost:8080).
    Model name is read from LLAMA_CPP_MODEL env var, falling back to
    settings.summarizer.model, then to 'llama3'.
    """

    def __init__(self, model: str, timeout: float = 45.0) -> None:
        base = os.getenv("LLAMA_CPP_BASE_URL", _LOCAL_LLM_DEFAULT_BASE).rstrip("/")
        self._url = f"{base}/v1/chat/completions"
        self._model = os.getenv("LLAMA_CPP_MODEL", model)
        self._client = httpx.Client(
            timeout=httpx.Timeout(connect=5.0, read=timeout, write=10.0, pool=5.0)
        )

    def available(self) -> bool:
        """Return True if LLAMA_CPP_BASE_URL is configured (non-default or explicitly set)."""
        return bool(os.getenv("LLAMA_CPP_BASE_URL", ""))

    def summarize(self, input_text: str, max_tokens: int = 150) -> str:
        """Return a neutral summary from the local model, or raise on failure."""
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _NEUTRAL_SYSTEM_PROMPT},
                {"role": "user", "content": input_text},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.1,
        }
        response = self._client.post(self._url, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    def close(self) -> None:
        self._client.close()


class Summarizer:
    """Produces neutral summaries for each StoryCluster."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._pplx_key = os.getenv("PPLX_API_KEY", "")

        s = self.settings["summarizer"]
        read_timeout: float = float(s.get("request_timeout_seconds", _DEFAULT_READ_TIMEOUT))
        connect_timeout: float = float(s.get("connect_timeout_seconds", _DEFAULT_CONNECT_TIMEOUT))
        article_timeout: float = float(s.get("article_fetch_timeout_seconds", 15.0))
        self._max_workers: int = int(s.get("max_concurrent_summaries", _DEFAULT_MAX_WORKERS))
        self._cluster_timeout: float = float(s.get("cluster_timeout_seconds", _DEFAULT_CLUSTER_TIMEOUT))
        self._max_summary_tokens: int = int(s.get("max_summary_tokens", 150))

        self._pplx_call_delay: float = float(s.get("pplx_call_delay_seconds", _DEFAULT_PPLX_DELAY))
        self._pplx_max_calls: int = int(s.get("pplx_max_calls_per_run", _DEFAULT_PPLX_MAX_CALLS))
        self._pplx_min_score: float = float(s.get("pplx_min_importance_score", _DEFAULT_PPLX_MIN_SCORE))
        self._pplx_retry_attempts: int = int(s.get("pplx_retry_attempts", _DEFAULT_PPLX_RETRY_ATTEMPTS))
        self._pplx_retry_base_delay: float = float(s.get("pplx_retry_base_delay", _DEFAULT_PPLX_RETRY_BASE_DELAY))

        self._pplx_calls: int = 0
        self._pplx_call_lock: threading.Lock = threading.Lock()
        self._pplx_throttle_lock: threading.Lock = threading.Lock()

        user_agent = self.settings["ingestion"]["user_agent"]
        local_model: str = s.get("model", "llama3")

        self._client = httpx.Client(
            timeout=httpx.Timeout(
                connect=connect_timeout,
                read=read_timeout,
                write=connect_timeout,
                pool=5.0,
            )
        )
        self._article_fetcher = ArticleFetcher(timeout=article_timeout, user_agent=user_agent)
        self._brave = BraveSearchClient()
        self._local_llm = LocalLLMClient(model=local_model, timeout=read_timeout)

    def __enter__(self) -> "Summarizer":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()
        self._article_fetcher.close()
        self._brave.close()
        self._local_llm.close()

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

        enrich_threshold = float(self.settings["summarizer"].get("brave_enrich_threshold", 0.7))
        if cluster.importance_score >= enrich_threshold:
            snippets = self._brave.search(cluster.representative_headline, count=3)
            if snippets:
                input_text += "\n\nADDITIONAL CONTEXT:\n" + "\n".join(snippets)

        input_text = input_text[:MAX_INPUT_CHARS]

        summary, backend = self._route_summary(cluster, input_text)

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
            fallback_used=(backend == "extractive"),
            summary_backend=backend,
        )

    def _route_summary(self, cluster: StoryCluster, input_text: str) -> tuple[str, str]:
        use_perplexity = False

        if self._pplx_key and cluster.importance_score >= self._pplx_min_score:
            with self._pplx_call_lock:
                if self._pplx_calls < self._pplx_max_calls:
                    self._pplx_calls += 1
                    use_perplexity = True
                else:
                    logger.info(
                        "Cluster %d: Perplexity cap (%d) reached — routing to local LLM",
                        cluster.cluster_id, self._pplx_max_calls,
                    )
        elif self._pplx_key and cluster.importance_score < self._pplx_min_score:
            logger.debug(
                "Cluster %d: importance %.2f below threshold %.2f — routing to local LLM",
                cluster.cluster_id, cluster.importance_score, self._pplx_min_score,
            )

        if use_perplexity:
            summary, succeeded = self._call_with_throttle_and_retry(cluster.cluster_id, input_text)
            if succeeded:
                return summary, "perplexity"
            logger.warning(
                "Cluster %d: Perplexity failed — falling through to local LLM",
                cluster.cluster_id,
            )

        if self._local_llm.available():
            try:
                summary = self._local_llm.summarize(input_text, max_tokens=self._max_summary_tokens)
                return summary, "local"
            except Exception as exc:
                logger.warning(
                    "Cluster %d: local LLM failed (%s) — using extractive fallback",
                    cluster.cluster_id, exc,
                )

        return self._extractive_fallback(input_text), "extractive"

    def _call_with_throttle_and_retry(
        self, cluster_id: int, input_text: str
    ) -> tuple[str, bool]:
        with self._pplx_throttle_lock:
            time.sleep(self._pplx_call_delay)
            try:
                summary = self._call_perplexity_with_retry(input_text)
                return summary, True
            except Exception as exc:
                logger.warning(
                    "Perplexity failed for cluster %d after retries: %s",
                    cluster_id, exc,
                )
                with self._pplx_call_lock:
                    self._pplx_calls = max(0, self._pplx_calls - 1)
                return self._extractive_fallback(input_text), False

    def _call_perplexity_with_retry(self, input_text: str) -> str:
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
                        wait = float(retry_after) if retry_after else (
                            self._pplx_retry_base_delay * (2 ** attempt)
                        )
                        logger.warning(
                            "Perplexity returned %d — retrying in %.1fs (attempt %d/%d)",
                            status, wait, attempt + 1, self._pplx_retry_attempts,
                        )
                        time.sleep(wait)
                        continue
                raise
            except Exception as exc:
                last_exc = exc
                raise
        raise last_exc

    def _call_perplexity(self, input_text: str) -> str:
        payload = {
            "model": _PPLX_MODEL,
            "messages": [
                {"role": "system", "content": _NEUTRAL_SYSTEM_PROMPT},
                {"role": "user", "content": input_text},
            ],
            "max_tokens": self._max_summary_tokens,
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
        """Build plain-text input for the LLM from article full text or RSS summary.

        RSS summaries from Reuters, AP, CNN etc. are HTML documents. Strip all
        tags before feeding to the LLM so the model sees clean prose, not markup.
        """
        parts = []
        for article in cluster.articles:
            full_text = self._article_fetcher.fetch(article.raw.url)
            if full_text:
                # trafilatura already returns clean plain text
                parts.append(full_text)
            else:
                # RSS fallback — may contain HTML; strip it
                parts.append(_strip_html(article.raw.summary or ""))
        return "\n\n---\n\n".join(p for p in parts if p)

    def _extractive_fallback(self, text: str) -> str:
        """Return the first 3 sentences of text as a plain-text summary.

        Strip HTML first — defence-in-depth in case anything HTML-containing
        reaches this path from outside _build_input_text.
        """
        clean = _strip_html(text)
        sentences = re.split(r'(?<=[.!?])\s+', clean.strip())
        return " ".join(sentences[:3])

    def _fallback_summary(self, cluster: StoryCluster) -> SummarizedCluster:
        """Emergency fallback used only when _summarize_cluster itself raises."""
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
            summary_backend="extractive",
        )
