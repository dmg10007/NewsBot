"""Separate LLM clients with consistent result contracts."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

import httpx

from domain.models import ReportingComparison, StoryCluster

logger = logging.getLogger(__name__)

_PPLX_API_URL = "https://api.perplexity.ai/chat/completions"


@dataclass
class SummaryResult:
    text: str
    provider_used: str
    fallback_used: bool = False


class SummaryLLMClient:
    """Produces neutral summaries from clustered article content."""

    def __init__(self, settings: dict) -> None:
        cfg = settings.get("summarizer", {})
        self.model = cfg.get("model", "llama3")
        self.max_tokens = int(cfg.get("max_summary_tokens", 150))
        self._pplx_key = os.getenv("PPLX_API_KEY", "")
        self._local_base = os.getenv("LLAMA_CPP_BASE_URL", "").rstrip("/")
        timeout = float(cfg.get("request_timeout_seconds", 45))
        self._client = httpx.Client(timeout=httpx.Timeout(connect=10, read=timeout, write=10, pool=5))

    def close(self) -> None:
        self._client.close()

    def summarize(self, cluster: StoryCluster, comparison: ReportingComparison) -> SummaryResult:
        prompt = _summary_prompt(cluster, comparison)
        if self._pplx_key:
            try:
                return SummaryResult(self._chat(_PPLX_API_URL, "sonar", prompt, self._pplx_key), "perplexity")
            except Exception as exc:
                logger.warning("Perplexity summary failed for cluster %s: %s", cluster.cluster_id, exc)
        if self._local_base:
            try:
                return SummaryResult(self._chat(f"{self._local_base}/v1/chat/completions", self.model, prompt), "local")
            except Exception as exc:
                logger.warning("Local summary failed for cluster %s: %s", cluster.cluster_id, exc)
        return SummaryResult(_extractive_summary(cluster), "extractive", fallback_used=True)

    def _chat(self, url: str, model: str, prompt: str, api_key: str = "") -> str:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        response = self._client.post(
            url,
            json={
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You summarize news neutrally. Bias-free means minimizing loaded "
                            "language and labeling attribution; it does not mean guessing certainty."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": self.max_tokens,
                "temperature": 0.1,
            },
            headers=headers,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()


class ComparisonLLMClient:
    """Compares how sources report the same story."""

    def __init__(self, settings: dict) -> None:
        cfg = settings.get("bias", {})
        self.max_tokens = int(cfg.get("max_tokens", 600))
        self._pplx_key = os.getenv("PPLX_API_KEY", "")
        self._local_base = os.getenv("LLAMA_CPP_BASE_URL", "").rstrip("/")
        self._local_model = os.getenv("LLAMA_CPP_MODEL", "llama3")
        self._client = httpx.Client(timeout=httpx.Timeout(connect=10, read=45, write=10, pool=5))

    def close(self) -> None:
        self._client.close()

    def compare(self, cluster: StoryCluster) -> ReportingComparison:
        if cluster.is_single_source:
            article = cluster.articles[0]
            return ReportingComparison(
                cluster_id=cluster.cluster_id,
                shared_facts=[article.headline],
                source_specific_claims=[f"{article.source_name} is the only source in this cluster."],
                bias_notes="Single-source story; claims are not corroborated by other configured outlets.",
                provider_used="heuristic",
                confidence=0.5,
            )

        prompt = _comparison_prompt(cluster)
        if self._pplx_key:
            try:
                return self._parse(cluster.cluster_id, self._chat(_PPLX_API_URL, "sonar", prompt, self._pplx_key), "perplexity")
            except Exception as exc:
                logger.warning("Perplexity comparison failed for cluster %s: %s", cluster.cluster_id, exc)
        if self._local_base:
            try:
                return self._parse(cluster.cluster_id, self._chat(f"{self._local_base}/v1/chat/completions", self._local_model, prompt), "local")
            except Exception as exc:
                logger.warning("Local comparison failed for cluster %s: %s", cluster.cluster_id, exc)
        return _heuristic_comparison(cluster)

    def _chat(self, url: str, model: str, prompt: str, api_key: str = "") -> str:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        response = self._client.post(
            url,
            json={
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You compare news reporting. Distinguish verified shared facts "
                            "from attributed claims and do not decide which source is correct."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": self.max_tokens,
                "temperature": 0.1,
            },
            headers=headers,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    def _parse(self, cluster_id: int | None, content: str, provider: str) -> ReportingComparison:
        shared = _section(content, "SHARED FACTS")
        specific = _section(content, "SOURCE-SPECIFIC CLAIMS")
        omissions = _section(content, "OMISSIONS")
        # SOURCE PERSPECTIVES replaces the old FRAMING + BIAS NOTES split.
        # Each bullet is "Outlet (Bias): one-sentence angle." Joined with
        # newlines so the renderer can display them line-by-line.
        perspectives = _section(content, "SOURCE PERSPECTIVES")
        if perspectives:
            bias_notes = "\n".join(perspectives)
        else:
            # Fallback: try old BIAS NOTES section for backward compat.
            bias_notes = _paragraph(content, "BIAS NOTES") or "No framing differences identified."
        return ReportingComparison(
            cluster_id=cluster_id,
            shared_facts=shared,
            source_specific_claims=specific,
            omissions=omissions,
            framing_differences=[],   # retired; kept for contract compatibility
            bias_notes=bias_notes,
            provider_used=provider,
            confidence=0.8,
        )


def _summary_prompt(cluster: StoryCluster, comparison: ReportingComparison) -> str:
    shared = "\n".join(f"- {f}" for f in comparison.shared_facts) or "None extracted."
    specific = "\n".join(f"- {c}" for c in comparison.source_specific_claims) or "None extracted."

    rep = cluster.representative_article
    other_articles = [a for a in cluster.articles if a is not rep]
    ordered = ([rep] if rep else []) + other_articles

    excerpts = "\n\n".join(
        f"[{a.source_name} / {a.bias_lean}]\n{a.summary or a.headline}"
        for a in ordered
    )
    return f"""Write a 2-3 sentence neutral digest summary of this story.

Headline: "{cluster.representative_headline}"

Your summary MUST:
- Directly expand on the headline above. Do not pivot to a different angle.
- Synthesize information ACROSS all sources listed below, not just one.
- Preserve attribution for single-source or disputed claims.
- Remove loaded adjectives, speculation, opinion framing, and unsupported causal language.

Base your summary primarily on these cross-source facts:

SHARED FACTS (reported by multiple sources):
{shared}

SOURCE-SPECIFIC CLAIMS (single-source details worth including):
{specific}

SUPPORTING EXCERPTS (for context only — do not summarize just one):
{excerpts[:3000]}
"""


def _comparison_prompt(cluster: StoryCluster) -> str:
    articles = "\n\n".join(
        f"SOURCE: {a.source_name}\nBIAS LABEL: {a.bias_lean}\nHEADLINE: {a.headline}\nEXCERPT: {a.summary or a.body_text}"
        for a in cluster.articles
    )
    return f"""Compare how each source below covers this story.

Return exactly these sections:

SHARED FACTS:
- bullet per fact reported by most/all sources

SOURCE-SPECIFIC CLAIMS:
- "Source: claim" for details only that source includes

OMISSIONS:
- meaningful facts one or more sources leave out

SOURCE PERSPECTIVES:
- One bullet per source in this format:
  "Source Name (Bias Label): one sentence describing their specific angle, framing, or emphasis."
- Be concrete — name the specific language, omission, or emphasis that distinguishes each outlet.
- Do not repeat the shared facts here.

ARTICLES:
{articles[:6000]}
"""


def _heuristic_comparison(cluster: StoryCluster) -> ReportingComparison:
    headlines = [a.headline for a in cluster.articles]
    leans = sorted({a.bias_lean for a in cluster.articles if a.bias_lean})
    perspectives = [
        f"{a.source_name} ({a.bias_lean}): {a.summary or a.headline}"
        for a in cluster.articles[:6]
    ]
    return ReportingComparison(
        cluster_id=cluster.cluster_id,
        shared_facts=headlines[:3],
        source_specific_claims=[
            f"{a.source_name}: {a.summary or a.headline}" for a in cluster.articles[:5]
        ],
        framing_differences=[],
        bias_notes="\n".join(perspectives) or "Automated comparison unavailable.",
        provider_used="heuristic",
        confidence=0.4,
        fallback_used=True,
    )


def _extractive_summary(cluster: StoryCluster) -> str:
    text = " ".join(a.summary or a.headline for a in cluster.articles).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(sentences[:3]) or cluster.representative_headline


def _section(content: str, title: str) -> list[str]:
    match = re.search(rf"{re.escape(title)}:\s*(.+?)(?=\n[A-Z][A-Z\-\s]+:|$)", content, re.S)
    if not match:
        return []
    return [
        line.lstrip("- ").strip()
        for line in match.group(1).splitlines()
        if line.strip() and line.strip() != "-"
    ]


def _paragraph(content: str, title: str) -> str:
    match = re.search(
        rf"{re.escape(title)}:\s*\n?(.*?)(?=\n[A-Z][A-Z\-\s]+:|$)",
        content, re.S
    )
    return match.group(1).strip() if match else ""
