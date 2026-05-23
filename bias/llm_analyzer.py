"""Stage 3 bias detection: LLM-assisted cross-source claim analysis.

Only called for clusters that passed lexicon.py escalation threshold.
Uses Perplexity Sonar as primary, with llama.cpp as fallback.

Outputs:
  - A list of factual claims extracted from the cluster
  - Identified framing differences between sources
  - A bias_notes field for the digest: plain-language explanation of what
    was stripped and why, without introducing a new directional lean

Design principle: The LLM is a LABELER, not a judge. It identifies and
names differences; it does not declare which source is correct.

Prompt design: source names and bias-lean labels are intentionally excluded
from the prompt. Providing that metadata primes the LLM to reason about a
source's political identity rather than the actual content, which works
against the goal of neutral framing analysis. The LLM sees only anonymized
ARTICLE_N labels, headlines, and summaries.

Perplexity model history:
  sonar-reasoning      -> DEPRECATED (400 Bad Request)
  sonar                -> Current standard model (used here)
  sonar-reasoning-pro  -> Higher quality, higher cost alternative

LLM call cap:
  Default is 50 per run. Override via LLMAnalyzer(max_calls=N) or expose
  in settings.yaml under bias.max_llm_calls_per_run.

Prompt token budget:
  Article summaries are truncated to _SUMMARY_PREVIEW_CHARS (300) in the
  prompt. This keeps input tokens reasonable: a 10-article cluster costs
  ~1,800 input tokens instead of ~5,000+ with full summaries, while still
  giving the model sufficient context for framing analysis.

Security note:
  The httpx client uses a request hook to redact the Authorization header
  value before it can appear in any log output, preventing accidental key
  exposure in debug-level httpx logs.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

import httpx

from bias.framing import FramingResult
from clustering.clusterer import StoryCluster

logger = logging.getLogger(__name__)

_PPLX_API_URL = "https://api.perplexity.ai/chat/completions"
_PPLX_MODEL = "sonar"  # sonar-reasoning was deprecated; sonar is the current standard model
_LLAMA_DEFAULT_BASE_URL = "http://localhost:8080"
_DEFAULT_MAX_CALLS = 50
_SUMMARY_PREVIEW_CHARS = 300  # Max chars per article summary in prompt — controls input token cost


def _mask_auth_header(request: httpx.Request) -> None:
    """httpx request hook: redact Authorization header value in any log output.

    httpx can log full request details at DEBUG level, which would expose
    the Bearer token. This hook replaces the value with a fixed placeholder
    so the key never appears in logs regardless of the log level in use.
    """
    if "authorization" in request.headers:
        # headers are case-insensitive but the assignment must use the exact
        # key format that httpx stores internally.
        request.headers["authorization"] = "Bearer [REDACTED]"


@dataclass
class LLMAnalysisResult:
    cluster_id: int
    extracted_facts: list[str]        # Verifiable factual claims found across all sources
    framing_notes: list[str]          # Plain-language descriptions of framing differences
    bias_notes: str                   # Short paragraph for digest — what was stripped and why
    provider_used: str                # "perplexity" | "local" | "none"
    skipped: bool = False             # True if max_llm_calls cap was hit


class LLMAnalyzer:
    """Calls Perplexity Sonar or local llama.cpp for bias/framing analysis."""

    def __init__(self, max_calls: int = _DEFAULT_MAX_CALLS) -> None:
        self.max_calls = max_calls
        self._calls_made = 0
        self._pplx_key = os.getenv("PPLX_API_KEY", "")
        self._llama_model = os.getenv("LLAMA_CPP_MODEL", "llama3")

        # Normalize the base URL: strip trailing slash and any /v1 suffix
        # so we can always safely append /v1/chat/completions ourselves.
        raw_base = os.getenv("LLAMA_CPP_BASE_URL", _LLAMA_DEFAULT_BASE_URL)
        self._llama_base_url = re.sub(r"(/v1)?/?$", "", raw_base.rstrip("/"))

        # event_hooks ensure the Authorization header is redacted before httpx
        # can include it in any debug-level log output.
        self._client = httpx.Client(
            timeout=60.0,
            event_hooks={"request": [_mask_auth_header]},
        )

    def analyze(
        self,
        cluster: StoryCluster,
        framing: FramingResult,
    ) -> LLMAnalysisResult:
        if self._calls_made >= self.max_calls:
            logger.warning(
                "LLM call cap (%d) reached — skipping cluster %d",
                self.max_calls, cluster.cluster_id,
            )
            return LLMAnalysisResult(
                cluster_id=cluster.cluster_id,
                extracted_facts=[],
                framing_notes=[],
                bias_notes="Analysis skipped: daily LLM call limit reached.",
                provider_used="none",
                skipped=True,
            )

        prompt = self._build_prompt(cluster, framing)

        if self._pplx_key:
            try:
                result = self._call_perplexity(cluster.cluster_id, prompt)
                self._calls_made += 1
                return result
            except Exception as exc:
                logger.warning(
                    "Perplexity call failed for cluster %d: %s. Falling back to local.",
                    cluster.cluster_id, exc,
                )

        try:
            result = self._call_local(cluster.cluster_id, prompt)
            self._calls_made += 1
            return result
        except Exception as exc:
            logger.error("Local LLM call also failed for cluster %d: %s", cluster.cluster_id, exc)
            return LLMAnalysisResult(
                cluster_id=cluster.cluster_id,
                extracted_facts=[],
                framing_notes=[framing.cross_source_summary],
                bias_notes="Automated bias analysis unavailable for this story.",
                provider_used="none",
            )

    def _build_prompt(self, cluster: StoryCluster, framing: FramingResult) -> str:
        """Build the analysis prompt.

        Source names and bias-lean labels are intentionally excluded.
        Providing them primes the LLM to reason about a source's political
        identity rather than the actual content. Articles are labelled
        ARTICLE_1, ARTICLE_2, etc. only.

        Article summaries are truncated to _SUMMARY_PREVIEW_CHARS to keep
        input token usage predictable. Headlines are never truncated.
        """
        articles_text = "\n\n".join(
            f"ARTICLE_{idx + 1}\n"
            f"HEADLINE: {a.raw.headline}\n"
            f"SUMMARY: {(a.raw.summary or '')[:_SUMMARY_PREVIEW_CHARS]}"
            for idx, a in enumerate(cluster.articles)
        )
        return f"""You are a neutral fact-extraction assistant. Your job is to analyze how multiple news sources cover the same story and identify:
1. The core verifiable factual claims present across sources
2. Framing differences: word choices, emphasis, or omissions that differ between sources
3. A short, neutral "bias notes" paragraph (2-3 sentences) describing what editorial choices were detected, WITHOUT declaring which source is correct

Rules:
- Do not introduce your own political lean
- Distinguish between verified facts and attributed claims
- Use plain language
- Do not use source outlet names or political labels in your analysis
- Return your response in this exact format:

FACTS:
- [fact 1]
- [fact 2]

FRAMING:
- [framing observation 1]
- [framing observation 2]

BIAS NOTES:
[2-3 sentence paragraph]

---
ARTICLES TO ANALYZE:
{articles_text}

PREVIOUS ANALYSIS CONTEXT:
{framing.cross_source_summary}
"""

    def _call_perplexity(self, cluster_id: int, prompt: str) -> LLMAnalysisResult:
        payload = {
            "model": _PPLX_MODEL,
            "messages": [
                {"role": "system", "content": "You are a neutral journalism analysis assistant."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 600,
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
        content = response.json()["choices"][0]["message"]["content"]
        logger.info("Perplexity analysis complete for cluster %d (model: %s)", cluster_id, _PPLX_MODEL)
        return self._parse_llm_response(cluster_id, content, "perplexity")

    def _call_local(self, cluster_id: int, prompt: str) -> LLMAnalysisResult:
        payload = {
            "model": self._llama_model,
            "messages": [
                {"role": "system", "content": "You are a neutral journalism analysis assistant."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 600,
            "temperature": 0.1,
        }
        url = f"{self._llama_base_url}/v1/chat/completions"
        response = self._client.post(url, json=payload)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        logger.info("Local LLM analysis complete for cluster %d", cluster_id)
        return self._parse_llm_response(cluster_id, content, "local")

    def _parse_llm_response(
        self, cluster_id: int, content: str, provider: str
    ) -> LLMAnalysisResult:
        """Parse structured LLM output into an LLMAnalysisResult."""
        facts: list[str] = []
        framing_notes: list[str] = []
        bias_notes = ""

        facts_match = re.search(r"FACTS:\s*(.+?)(?=FRAMING:|BIAS NOTES:|$)", content, re.DOTALL)
        framing_match = re.search(r"FRAMING:\s*(.+?)(?=BIAS NOTES:|$)", content, re.DOTALL)
        bias_match = re.search(r"BIAS NOTES:\s*(.+?)$", content, re.DOTALL)

        if facts_match:
            facts = [
                line.lstrip("- ").strip()
                for line in facts_match.group(1).strip().splitlines()
                if line.strip() and line.strip() != "-"
            ]
        if framing_match:
            framing_notes = [
                line.lstrip("- ").strip()
                for line in framing_match.group(1).strip().splitlines()
                if line.strip() and line.strip() != "-"
            ]
        if bias_match:
            bias_notes = bias_match.group(1).strip()

        if not bias_notes:
            bias_notes = "Bias analysis completed but no specific notes were generated."

        return LLMAnalysisResult(
            cluster_id=cluster_id,
            extracted_facts=facts,
            framing_notes=framing_notes,
            bias_notes=bias_notes,
            provider_used=provider,
        )

    def close(self) -> None:
        """Release the underlying httpx connection pool."""
        self._client.close()
