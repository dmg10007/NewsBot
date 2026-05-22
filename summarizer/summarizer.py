"""Neutral digest summarization using local llama.cpp.

Takes a StoryCluster + LLMAnalysisResult and produces a short,
bias-stripped paragraph suitable for the email digest.

Design rules:
  - Summarize only facts extracted by the bias layer
  - Never editorialize or infer intent
  - Always note how many sources covered the story
  - Flag if coverage is single-source (lower confidence)
  - If LLM is unavailable, or the story is single-source, fall back to
    the RSS description text from the best-credibility article.
    Single-source stories have nothing to cross-compare, so the fallback
    is equally informative at zero cost.
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
    bias_lean: Optional[str] = None  # e.g. "left", "center", "right", "center-left", "center-right"


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

    def summarize(self, cluster: StoryCluster, analysis: LLMAnalysisResult) -> SummaryResult:
        facts = analysis.extracted_facts
        bias_notes = analysis.bias_notes

        # Build per-source links, one per unique source (pick first article URL per source).
        seen_sources: set[str] = set()
        source_links: list[SourceLink] = []
        for article in cluster.articles:
            name = article.raw.source_name
            url = article.raw.url
            if name not in seen_sources and url:
                seen_sources.add(name)
                bias_lean = getattr(article.raw, "bias_lean", None)
                source_links.append(SourceLink(
                    source_name=name,
                    url=url,
                    bias_lean=bias_lean,
                ))
        source_links.sort(key=lambda s: s.source_name.lower())

        # Skip local LLM for single-source stories or clusters with no
        # extracted facts — there is nothing to cross-compare, so the
        # RSS description fallback is equally informative at zero cost.
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

    def _fallback_summary(self, cluster: StoryCluster) -> str:
        """Use the RSS description from the best-credibility article as the summary.

        Priority:
          1. RSS summary text from articles sorted by credibility, accepting the
             first one that passes the headline-echo and minimum-length checks.
          2. Last resort: a sentence built from extracted named entities.
        """
        credibility_order = {"high": 0, "medium": 1, "low": 2}
        sorted_articles = sorted(
            cluster.articles,
            key=lambda a: credibility_order.get(a.raw.credibility, 2),
        )

        headline = cluster.representative_headline
        for article in sorted_articles:
            raw_summary = article.raw.summary.strip()
            if not raw_summary:
                continue
            cleaned = self._clean_rss_summary(raw_summary)
            if not cleaned:
                continue
            if self._is_headline_echo(cleaned, headline):
                logger.debug(
                    "Discarding RSS summary (headline echo) for: %s", headline[:60]
                )
                continue
            return cleaned

        # Last resort: entity-based sentence
        entities = list({
            e[0] for a in cluster.articles for e in a.entities
            if e[1] in ("PERSON", "ORG", "GPE", "LOC")
        })[:5]
        if entities:
            return f"Key subjects: {', '.join(entities)}. Full details available via the source link(s) below."
        return "Full details available via the source link(s) below."

    @staticmethod
    def _clean_rss_summary(text: str) -> str:
        """Strip HTML, remove source-name suffixes, collapse whitespace, cap at 2 sentences.

        Some RSS feeds (Reuters, Axios) set the description field to just
        "{headline}    {source name}" with no real summary text. We strip
        trailing chunks that look like bare source attributions (all caps or
        title-case words after 3+ spaces) before the headline-echo check.
        """
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", text)
        # Strip trailing "   Source Name" patterns (3+ spaces then a short label)
        text = re.sub(r"\s{3,}[A-Z][^\n]{1,40}$", "", text).strip()
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return ""
        # Cap at 2 sentences
        sentences = re.split(r"(?<=[.!?])\s+", text)
        result = " ".join(sentences[:2]).strip()
        # Reject if too short to be informative
        if len(result.split()) < _MIN_SUMMARY_WORDS:
            return ""
        return result

    @staticmethod
    def _is_headline_echo(summary: str, headline: str) -> bool:
        """Return True if the summary is just a restatement of the headline.

        Computes the fraction of meaningful summary words (4+ chars, alpha)
        that also appear in the headline. If >= _HEADLINE_OVERLAP_THRESHOLD,
        the summary adds no new information and should be discarded.
        """
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
