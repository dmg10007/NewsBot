"""Neutral digest summarization using local llama.cpp.

Takes a StoryCluster + LLMAnalysisResult and produces a short,
bias-stripped paragraph suitable for the email digest.

Summary quality hierarchy (best to worst):
  1. LLM summary from cross-source extracted facts  (multi-source clusters)
  2. RSS description text (cleaned, echo-checked)   (any cluster)
  3. Full article body scraped from the source URL  (fallback enrichment)
  4. Brave Search snippets for the headline         (fallback enrichment)
  5. Entity-based sentence                          (last resort)
  6. Bare "full details via link" message           (absolute last resort)

Levels 3-4 only fire when BRAVE_SEARCH_API_KEY is set and levels 1-2 fail.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional

import httpx

from bias.llm_analyzer import LLMAnalysisResult
from clustering.clusterer import StoryCluster
from config.loader import get_settings
from ingestion.article_fetcher import ArticleFetcher
from ingestion.brave_search import BraveSearch

logger = logging.getLogger(__name__)

_LLAMA_DEFAULT_URL = "http://localhost:8080"

# Minimum word count for an RSS summary to be considered useful.
_MIN_SUMMARY_WORDS = 8

# If this fraction of the summary's words are also in the headline, treat
# it as a headline echo and discard it.
_HEADLINE_OVERLAP_THRESHOLD = 0.6

SYSTEM_PROMPT = """You are a neutral news summarizer. Your output will appear in a bias-free daily briefing.
Rules:
- Write 2-4 sentences maximum
- Use only the facts provided — do not add context or inference
- Use neutral, plain language
- Do not name a political winner or loser
- Do not use loaded, emotional, or charged words
- Write in third-person past tense
- Do not repeat the source names in the summary"""


@dataclass
class SourceLink:
    """A single article link with its source label and bias lean."""
    source_name: str
    url: str
    bias_lean: Optional[str] = None


@dataclass
class SummaryResult:
    cluster_id: int
    summary: str
    source_count: int
    tiers_covered: list[str]
    is_single_source: bool
    topic: str
    representative_headline: str
    bias_notes: str
    provider_used: str
    sources: list[SourceLink] = field(default_factory=list)


class Summarizer:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._llama_url = os.getenv("LLAMA_CPP_BASE_URL", _LLAMA_DEFAULT_URL)
        self._llama_model = os.getenv("LLAMA_CPP_MODEL", "llama3")
        self._max_tokens = self.settings["summarization"]["max_summary_tokens"]
        self._client = httpx.Client(timeout=60.0)
        self._article_fetcher = ArticleFetcher()
        self._brave = BraveSearch()
        if self._brave.available:
            logger.info("Brave Search enrichment enabled.")
        else:
            logger.info(
                "Brave Search enrichment disabled (BRAVE_SEARCH_API_KEY not set)."
            )

    def summarize(self, cluster: StoryCluster, analysis: LLMAnalysisResult) -> SummaryResult:
        facts = analysis.extracted_facts
        bias_notes = analysis.bias_notes

        # Build per-source links, one per unique source.
        seen_sources: set[str] = set()
        source_links: list[SourceLink] = []
        for article in cluster.articles:
            name = article.raw.source_name
            url = article.raw.url
            if name not in seen_sources and url:
                seen_sources.add(name)
                source_links.append(SourceLink(
                    source_name=name,
                    url=url,
                    bias_lean=getattr(article.raw, "bias_lean", None),
                ))
        source_links.sort(key=lambda s: s.source_name.lower())

        if cluster.source_count > 1 and facts:
            summary = self._summarize_from_facts(facts, cluster)
            provider = "local"
        else:
            summary = self._fallback_summary(cluster)
            provider = "fallback"

        return SummaryResult(
            cluster_id=cluster.cluster_id,
            summary=summary,
            source_count=cluster.source_count,
            tiers_covered=cluster.tiers,
            is_single_source=cluster.source_count == 1,
            topic=cluster.topic,
            representative_headline=cluster.representative_headline,
            bias_notes=bias_notes,
            provider_used=provider,
            sources=source_links,
        )

    # ------------------------------------------------------------------
    # LLM path
    # ------------------------------------------------------------------

    def _summarize_from_facts(self, facts: list[str], cluster: StoryCluster) -> str:
        facts_text = "\n".join(f"- {f}" for f in facts[:10])
        user_prompt = (
            f"Summarize the following verified facts from a news story into 2-4 neutral sentences:\n\n"
            f"{facts_text}\n\n"
            f"Story headline for context (do not copy verbatim): {cluster.representative_headline}"
        )
        try:
            return self._call_local(user_prompt)
        except Exception as exc:
            logger.warning("Local LLM summarization failed: %s. Using fallback.", exc)
            return self._fallback_summary(cluster)

    def _call_local(self, user_prompt: str) -> str:
        payload = {
            "model": self._llama_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": self._max_tokens,
            "temperature": 0.1,
        }
        response = self._client.post(
            f"{self._llama_url.rstrip('/')}/v1/chat/completions",
            json=payload,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    # ------------------------------------------------------------------
    # Fallback path: RSS → article body → Brave snippets → entities
    # ------------------------------------------------------------------

    def _fallback_summary(self, cluster: StoryCluster) -> str:
        """Multi-tier fallback when the LLM path is unavailable or skipped.

        Tier 1: RSS description (cleaned + echo-checked)
        Tier 2: Full article body scraped from the source URL
        Tier 3: Brave Search snippets for the headline
        Tier 4: Entity-based sentence
        Tier 5: Bare fallback message
        """
        headline = cluster.representative_headline

        # --- Tier 1: RSS description ---
        credibility_order = {"high": 0, "medium": 1, "low": 2}
        sorted_articles = sorted(
            cluster.articles,
            key=lambda a: credibility_order.get(a.raw.credibility, 2),
        )
        for article in sorted_articles:
            raw = article.raw.summary.strip()
            if not raw:
                continue
            cleaned = self._clean_rss_summary(raw)
            if cleaned and not self._is_headline_echo(cleaned, headline):
                return cleaned

        # --- Tier 2: Full article body scrape ---
        for article in sorted_articles:
            url = article.raw.url
            if not url:
                continue
            body = self._article_fetcher.fetch_body(url)
            if not body:
                continue
            summary = self._extract_summary_from_body(body, headline)
            if summary:
                logger.debug("Used article body for summary: %s", headline[:60])
                return summary

        # --- Tier 3: Brave Search snippets ---
        if self._brave.available:
            snippets = self._brave.search_snippets(headline)
            if snippets:
                summary = self._extract_summary_from_body(snippets, headline)
                if summary:
                    logger.debug("Used Brave snippets for summary: %s", headline[:60])
                    return summary

        # --- Tier 4: Entity-based sentence ---
        entities = list({
            e[0] for a in cluster.articles for e in a.entities
            if e[1] in ("PERSON", "ORG", "GPE", "LOC")
        })[:5]
        if entities:
            return (
                f"Key subjects: {', '.join(entities)}. "
                f"Full details available via the source link(s) below."
            )

        # --- Tier 5: Bare fallback ---
        return "Full details available via the source link(s) below."

    def _extract_summary_from_body(self, text: str, headline: str) -> str:
        """Extract a clean 1-2 sentence summary from a body of text.

        Splits into sentences, discards ones that are headline echoes or too
        short, and returns the first 1-2 that pass. This avoids feeding the
        entire body to the LLM when we just need a brief summary snippet.
        """
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return ""

        sentences = re.split(r"(?<=[.!?])\s+", text)
        good: list[str] = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent.split()) < _MIN_SUMMARY_WORDS:
                continue
            if self._is_headline_echo(sent, headline):
                continue
            good.append(sent)
            if len(good) == 2:
                break

        return " ".join(good).strip()

    # ------------------------------------------------------------------
    # Shared text-cleaning utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_rss_summary(text: str) -> str:
        """Strip HTML, remove source-name suffixes, collapse whitespace, cap at 2 sentences."""
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s{3,}[A-Z][^\n]{1,40}$", "", text).strip()
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return ""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        result = " ".join(sentences[:2]).strip()
        if len(result.split()) < _MIN_SUMMARY_WORDS:
            return ""
        return result

    @staticmethod
    def _is_headline_echo(summary: str, headline: str) -> bool:
        """Return True if summary is a near-duplicate of the headline."""
        def meaningful_words(s: str) -> set[str]:
            return {w.lower() for w in re.findall(r"[a-zA-Z]{4,}", s)}

        summary_words = meaningful_words(summary)
        if not summary_words:
            return True
        headline_words = meaningful_words(headline)
        overlap = summary_words & headline_words
        return len(overlap) / len(summary_words) >= _HEADLINE_OVERLAP_THRESHOLD

    def close(self) -> None:
        self._client.close()
        self._article_fetcher.close()
        self._brave.close()
