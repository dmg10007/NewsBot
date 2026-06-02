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
        framing = _section(content, "FRAMING")
        raw_bias = _paragraph(content, "BIAS NOTES")
        # Promote framing differences into bias_notes when the LLM returns an
        # empty or missing BIAS NOTES section, so we never silently fall back
        # to the default string when real framing data was parsed.
        if not raw_bias and framing:
            raw_bias = " ".join(framing[:2])
        bias_notes = raw_bias or "No specific framing differences were identified."
        return ReportingComparison(
            cluster_id=cluster_id,
            shared_facts=_section(content, "SHARED FACTS"),
            source_specific_claims=_section(content, "SOURCE-SPECIFIC CLAIMS"),
            omissions=_section(content, "OMISSIONS"),
            framing_differences=framing,
            bias_notes=bias_notes,
            provider_used=provider,
            confidence=0.8,
        )


def _summary_prompt(cluster: StoryCluster, comparison: ReportingComparison) -> str:
    # Place the representative article first so it anchors the truncation
    # window and the LLM treats it as the primary frame of reference.
    rep = cluster.representative_article
    other_articles = [a for a in cluster.articles if a is not rep]
    ordered = ([rep] if rep else []) + other_articles

    articles = "\n\n".join(
        f"SOURCE: {a.source_name} ({a.bias_lean})\nHEADLINE: {a.headline}\nSUMMARY: {a.summary or a.body_text}"
        for a in ordered
    )
    return f"""Write a concise 2-3 sentence neutral digest summary of this story.

The headline for this story is:
"{cluster.representative_headline}"

Rules:
- Your summary MUST directly support and expand on the headline above.
- Do not summarize a different angle or sub-topic from the articles.
- Prefer facts shared across sources.
- Preserve attribution for disputed or single-source claims.
- Remove loaded adjectives, speculation, opinion framing, and unsupported causal language.

Shared facts:
{'; '.join(comparison.shared_facts)}

Source-specific claims:
{'; '.join(comparison.source_specific_claims)}

Articles:
{articles[:5000]}
"""


def _comparison_prompt(cluster: StoryCluster) -> str:
    articles = "\n\n".join(
        f"SOURCE: {a.source_name}\nBIAS: {a.bias_lean}\nHEADLINE: {a.headline}\nSUMMARY: {a.summary or a.body_text}"
        for a in cluster.articles
    )
    return f"""Compare these articles about the same story.

Return exactly:
SHARED FACTS:
- facts most/all sources report
SOURCE-SPECIFIC CLAIMS:
- claims or details only some sources include, with attribution
OMISSIONS:
- meaningful omissions or missing details
FRAMING:
- differences in numbers, dates, attribution, emphasis, or language
BIAS NOTES:
2-3 neutral sentences. Mention source bias labels only as labels, not proof of accuracy.

ARTICLES:
{articles[:6000]}
"""


def _heuristic_comparison(cluster: StoryCluster) -> ReportingComparison:
    headlines = [a.headline for a in cluster.articles]
    leans = sorted({a.bias_lean for a in cluster.articles if a.bias_lean})
    return ReportingComparison(
        cluster_id=cluster.cluster_id,
        shared_facts=headlines[:3],
        source_specific_claims=[
            f"{a.source_name}: {a.summary or a.headline}" for a in cluster.articles[:5]
        ],
        framing_differences=[
            f"Configured source bias labels represented: {', '.join(leans) or 'unknown'}."
        ],
        bias_notes="Automated LLM comparison was unavailable; this note is based on source metadata and article summaries.",
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
    # Capture content starting on the same line OR the next line after the
    # header. The old pattern required content immediately after the colon,
    # missing cases where the model emitted a newline before the text.
    match = re.search(
        rf"{re.escape(title)}:\s*\n?(.*?)(?=\n[A-Z][A-Z\-\s]+:|$)",
        content, re.S
    )
    return match.group(1).strip() if match else ""
