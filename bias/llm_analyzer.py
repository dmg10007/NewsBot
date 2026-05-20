"""LLM-assisted bias analysis — Perplexity Sonar with llama.cpp fallback.

Only called for clusters flagged by lexicon.py or framing.py.
Outputs a BiasReport per cluster: what was stripped, what remains as fact.

Perplexity Sonar (sonar-reasoning) is used as the primary provider because
its web-grounded reasoning is well-suited to cross-source claim comparison.
llama.cpp is the offline fallback when Sonar is unavailable or the daily
call cap is reached.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import httpx

from bias.lexicon import BiasSignal
from clustering.clusterer import StoryCluster
from config.loader import get_settings

logger = logging.getLogger(__name__)

PPLX_API_URL = "https://api.perplexity.ai/chat/completions"
LLAMA_DEFAULT_URL = "http://localhost:8080/v1/chat/completions"


@dataclass
class BiasReport:
    """Full bias analysis result for a single story cluster."""
    cluster_id: int
    provider_used: str  # 'perplexity' | 'local' | 'heuristic_only'
    factual_summary: str = ""         # Stripped, neutral facts only
    stripped_elements: list[str] = field(default_factory=list)   # What was removed and why
    confidence: str = "low"           # low | medium | high
    escalated: bool = False
    error: Optional[str] = None


class LLMBiasAnalyzer:
    """Runs LLM-assisted bias analysis on flagged clusters."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._calls_made = 0
        self._call_cap: int = self.settings["bias_detection"]["max_llm_calls_per_run"]
        self._pplx_key: Optional[str] = os.getenv("PPLX_API_KEY")
        self._llama_url: str = os.getenv("LLAMA_CPP_BASE_URL", LLAMA_DEFAULT_URL)
        self._llama_model: str = os.getenv(
            "LLAMA_CPP_MODEL",
            self.settings["bias_detection"]["llm_fallback_model"]
        )
        self._client = httpx.Client(timeout=60)

    def analyze_flagged(
        self,
        clusters: list[StoryCluster],
        signals: list[BiasSignal],
    ) -> list[BiasReport]:
        reports: list[BiasReport] = []
        for cluster, signal in zip(clusters, signals):
            if signal.escalate_to_llm and self._calls_made < self._call_cap:
                report = self._run_llm_analysis(cluster, signal)
            else:
                # Heuristic-only report — no LLM call
                report = BiasReport(
                    cluster_id=cluster.cluster_id,
                    provider_used="heuristic_only",
                    escalated=signal.escalate_to_llm,
                    confidence="low" if signal.escalate_to_llm else "medium",
                )
            reports.append(report)

        llm_used = sum(1 for r in reports if r.provider_used != "heuristic_only")
        logger.info(
            "Bias analysis complete: %d LLM calls made (%d cap)",
            llm_used, self._call_cap,
        )
        return reports

    def _run_llm_analysis(
        self, cluster: StoryCluster, signal: BiasSignal
    ) -> BiasReport:
        prompt = self._build_prompt(cluster, signal)
        provider = self.settings["bias_detection"]["llm_provider"]

        # Try primary provider
        if provider == "perplexity" and self._pplx_key:
            try:
                result = self._call_perplexity(prompt)
                self._calls_made += 1
                return self._parse_response(
                    cluster.cluster_id, result, provider="perplexity"
                )
            except Exception as exc:
                logger.warning("Perplexity call failed, falling back to local: %s", exc)

        # Fallback to llama.cpp
        try:
            result = self._call_llama(prompt)
            self._calls_made += 1
            return self._parse_response(
                cluster.cluster_id, result, provider="local"
            )
        except Exception as exc:
            logger.error("Local LLM also failed for cluster %d: %s", cluster.cluster_id, exc)
            return BiasReport(
                cluster_id=cluster.cluster_id,
                provider_used="none",
                error=str(exc),
                escalated=True,
                confidence="low",
            )

    def _build_prompt(self, cluster: StoryCluster, signal: BiasSignal) -> str:
        articles_text = "\n".join(
            f"- [{a.raw.source_name} / {a.raw.bias_lean}] \"{a.raw.headline}\""
            f"{' | ' + a.raw.summary[:200] if a.raw.summary else ''}"
            for a in cluster.articles[:6]  # Cap at 6 articles per prompt
        )
        flags_text = "\n".join(f"  * {r}" for r in signal.escalation_reasons)

        return f"""You are a neutral fact-extraction assistant. Your job is to strip bias, spin, and editorial framing from news coverage and return only verifiable facts.

The following articles cover the same story from different sources. Bias flags were raised:
{flags_text}

Articles:
{articles_text}

Your task:
1. FACTUAL SUMMARY: Write 2-3 sentences containing only verifiable facts agreed upon across sources. No loaded language. No editorial framing. Active voice. No opinion.
2. STRIPPED ELEMENTS: List each specific bias element removed and why (e.g. 'loaded word: \"invasion\" — replaced with neutral \"border crossing increase\"').
3. CONFIDENCE: Rate your confidence in the factual summary as low / medium / high based on how much sources agreed.

Respond in exactly this format:
FACTUAL_SUMMARY: <your summary>
STRIPPED: <item 1> | <item 2> | ...
CONFIDENCE: <low|medium|high>"""

    def _call_perplexity(self, prompt: str) -> str:
        model = self.settings["bias_detection"]["llm_model"]
        response = self._client.post(
            PPLX_API_URL,
            headers={
                "Authorization": f"Bearer {self._pplx_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 400,
                "temperature": 0.1,  # Low temp for factual tasks
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def _call_llama(self, prompt: str) -> str:
        response = self._client.post(
            f"{self._llama_url.rstrip('/')}/v1/chat/completions",
            json={
                "model": self._llama_model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 400,
                "temperature": 0.1,
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def _parse_response(
        self, cluster_id: int, response_text: str, provider: str
    ) -> BiasReport:
        factual_summary = ""
        stripped: list[str] = []
        confidence = "low"

        for line in response_text.strip().splitlines():
            if line.startswith("FACTUAL_SUMMARY:"):
                factual_summary = line.replace("FACTUAL_SUMMARY:", "").strip()
            elif line.startswith("STRIPPED:"):
                raw_stripped = line.replace("STRIPPED:", "").strip()
                stripped = [s.strip() for s in raw_stripped.split("|") if s.strip()]
            elif line.startswith("CONFIDENCE:"):
                confidence = line.replace("CONFIDENCE:", "").strip().lower()
                if confidence not in ("low", "medium", "high"):
                    confidence = "low"

        return BiasReport(
            cluster_id=cluster_id,
            provider_used=provider,
            factual_summary=factual_summary,
            stripped_elements=stripped,
            confidence=confidence,
            escalated=True,
        )

    def close(self) -> None:
        self._client.close()
