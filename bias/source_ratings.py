"""Scrapers for AllSides and Media Bias Fact Check source ratings.

Builds a normalized lookup table: domain -> {bias_lean, factuality, confidence}.
Designed to be called once at startup (or weekly via cron) and cached in memory.

No paid API required:
  AllSides  — scrapes public media bias ratings page (single request, sync)
  MBFC      — scrapes public search results page (async, rate-limited)

MBFC async design
-----------------
Each domain requires 2 HTTP requests (search + detail page). The original
synchronous implementation serialised these with a 1.5s sleep between each
domain, taking 90+ seconds for 30 domains.

scrape_mbfc_bulk_async() runs requests concurrently behind an asyncio.Semaphore
(default max_concurrent=5) with a per-domain minimum delay of 1.0s. For 30
domains this reduces wall-clock time to ~8s while remaining a polite crawler.

The synchronous scrape_mbfc_bulk() wrapper is kept for backwards compatibility
(tests, one-off CLI use) and delegates to the async version via asyncio.run().
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

BIAS_SCALE = ("left", "center-left", "center", "center-right", "right")
FACTUALITY_SCALE = ("very-low", "low", "mixed", "mostly-factual", "high", "very-high")


@dataclass
class SourceRating:
    """Merged bias + factuality rating for a single news domain."""
    domain: str
    bias_lean: str
    factuality: str
    confidence: float
    allsides_bias: Optional[str] = None
    mbfc_bias: Optional[str] = None
    mbfc_factuality: Optional[str] = None
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# AllSides scraper (synchronous — single request)
# ---------------------------------------------------------------------------

_ALLSIDES_BIAS_MAP: dict[str, str] = {
    "left": "left",
    "lean left": "center-left",
    "center": "center",
    "lean right": "center-right",
    "right": "right",
    "allsides lean left": "center-left",
    "allsides lean right": "center-right",
    "allsides left": "left",
    "allsides right": "right",
    "allsides center": "center",
}


def _normalize_allsides_bias(raw: str) -> Optional[str]:
    return _ALLSIDES_BIAS_MAP.get(raw.strip().lower())


def scrape_allsides(client: httpx.Client) -> dict[str, str]:
    """Scrape AllSides media bias ratings page.

    Returns: {domain: normalized_bias_lean}
    This is a single HTTP request so it stays synchronous.
    Call via asyncio.to_thread() if you need it inside an async context.
    """
    url = "https://www.allsides.com/media-bias/ratings"
    ratings: dict[str, str] = {}

    try:
        resp = client.get(url, timeout=20)
        resp.raise_for_status()
    except Exception as exc:
        logger.error("AllSides scrape failed: %s", exc)
        return ratings

    soup = BeautifulSoup(resp.text, "lxml")
    table = soup.find("table", {"class": re.compile(r"views-table")})
    if not table:
        logger.warning("AllSides: could not find ratings table — page structure may have changed")
        return ratings

    for row in table.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 3:
            continue
        domain = _extract_domain_allsides(cells[0], cells[2])
        if not domain:
            continue
        bias_raw = ""
        bias_img = cells[2].find("img")
        if bias_img and bias_img.get("alt"):
            bias_raw = bias_img["alt"]
        else:
            bias_div = cells[2].find(attrs={"class": re.compile(r"allsides-")})
            if bias_div:
                bias_raw = " ".join(bias_div.get("class", []))
        normalized = _normalize_allsides_bias(bias_raw)
        if normalized:
            ratings[domain] = normalized

    logger.info("AllSides: scraped %d source ratings", len(ratings))
    return ratings


def _extract_domain_allsides(name_cell, bias_cell) -> Optional[str]:
    for a in name_cell.find_all("a", href=True):
        href = a["href"]
        if href.startswith("http"):
            return _domain_from_url(href)
    text = name_cell.get_text(separator=" ", strip=True).lower()
    return _OUTLET_DOMAIN_MAP.get(text)


def _domain_from_url(url: str) -> Optional[str]:
    match = re.search(r"https?://(?:www\.)?([^/]+)", url)
    return match.group(1).lower() if match else None


_OUTLET_DOMAIN_MAP: dict[str, str] = {
    "ap": "apnews.com",
    "associated press": "apnews.com",
    "reuters": "reuters.com",
    "npr": "npr.org",
    "pbs newshour": "pbs.org",
    "cnn": "cnn.com",
    "fox news": "foxnews.com",
    "fox business": "foxbusiness.com",
    "the hill": "thehill.com",
    "axios": "axios.com",
    "wall street journal": "wsj.com",
    "washington post": "washingtonpost.com",
    "new york times": "nytimes.com",
    "abc news": "abcnews.go.com",
    "nbc news": "nbcnews.com",
    "cbs news": "cbsnews.com",
    "msnbc": "msnbc.com",
    "breitbart": "breitbart.com",
    "daily wire": "dailywire.com",
    "mother jones": "motherjones.com",
    "the nation": "thenation.com",
    "reason": "reason.com",
    "politico": "politico.com",
    "the atlantic": "theatlantic.com",
    "vox": "vox.com",
    "buzzfeed news": "buzzfeednews.com",
    "wral": "wral.com",
    "charlotte observer": "charlotteobserver.com",
    "nc policy watch": "ncpolicywatch.com",
    "carolina public press": "carolinapublicpress.org",
}


# ---------------------------------------------------------------------------
# MBFC scrapers
# ---------------------------------------------------------------------------

_MBFC_BIAS_MAP: dict[str, str] = {
    "left": "left",
    "left-center": "center-left",
    "least biased": "center",
    "right-center": "center-right",
    "right": "right",
    "extreme left": "left",
    "extreme right": "right",
    "conspiracy-pseudoscience": "right",
    "questionable": "right",
    "pro-science": "center",
    "satire": "center",
}

_MBFC_FACTUALITY_MAP: dict[str, str] = {
    "very high": "very-high",
    "high": "high",
    "mostly factual": "mostly-factual",
    "mixed": "mixed",
    "low": "low",
    "very low": "very-low",
}

_MBFC_HEADERS = {
    "User-Agent": "NewsBot/1.0 (bias-ratings-lookup; educational use)"
}


# --- Async implementation ---

async def scrape_mbfc_source_async(
    client: httpx.AsyncClient,
    domain: str,
) -> Optional[dict]:
    """Async: scrape MBFC rating for a single domain via their search.

    Returns: {bias: str, factuality: str} or None
    Two HTTP requests are made: search page, then the first result detail page.
    """
    search_url = f"https://mediabiasfactcheck.com/?s={domain.split('.')[0]}"
    try:
        resp = await client.get(search_url, timeout=15)
        resp.raise_for_status()
    except Exception as exc:
        logger.debug("MBFC search failed for %s: %s", domain, exc)
        return None

    soup = BeautifulSoup(resp.text, "lxml")
    result = soup.select_one("h2.entry-title a, .post-title a")
    if not result:
        return None

    detail_url = result["href"]
    try:
        detail_resp = await client.get(detail_url, timeout=15)
        detail_resp.raise_for_status()
    except Exception as exc:
        logger.debug("MBFC detail fetch failed for %s: %s", detail_url, exc)
        return None

    detail_soup = BeautifulSoup(detail_resp.text, "lxml")
    return _parse_mbfc_detail(detail_soup)


async def scrape_mbfc_bulk_async(
    domains: list[str],
    delay: float = 1.0,
    max_concurrent: int = 5,
) -> dict[str, dict]:
    """Async: scrape MBFC for a list of domains with concurrency control.

    Runs up to max_concurrent domains in parallel. Each domain still waits
    at least `delay` seconds (asyncio.sleep — non-blocking) before its
    requests fire, providing a polite crawl rate.

    Wall-clock time: ~(len(domains) / max_concurrent) * delay + fetch overhead
    For 30 domains at max_concurrent=5, delay=1.0s: ~8s vs 90s synchronous.

    Returns: {domain: {bias, factuality}}
    """
    sem = asyncio.Semaphore(max_concurrent)
    results: dict[str, dict] = {}

    async def _fetch_one(client: httpx.AsyncClient, domain: str, index: int) -> None:
        async with sem:
            # Stagger initial requests to avoid burst on first tick
            await asyncio.sleep(index * (delay / max_concurrent))
            result = await scrape_mbfc_source_async(client, domain)
            if result:
                results[domain] = result
                logger.debug("MBFC: %s -> %s", domain, result)
            else:
                logger.debug("MBFC: no result for %s", domain)

    async with httpx.AsyncClient(
        headers=_MBFC_HEADERS,
        follow_redirects=True,
        timeout=20,
    ) as client:
        await asyncio.gather(*[
            _fetch_one(client, domain, i)
            for i, domain in enumerate(domains)
        ])

    logger.info("MBFC: scraped %d/%d domains", len(results), len(domains))
    return results


# --- Synchronous wrappers (backwards compatibility) ---

def scrape_mbfc_source(
    client: httpx.Client,
    domain: str,
) -> Optional[dict]:
    """Synchronous wrapper around scrape_mbfc_source_async.

    Kept for backwards compatibility and one-off CLI use.
    For bulk lookups, prefer scrape_mbfc_bulk_async() directly.
    """
    return asyncio.run(scrape_mbfc_bulk_async([domain])).get(domain)


def scrape_mbfc_bulk(
    client: httpx.Client,
    domains: list[str],
    delay: float = 1.5,
) -> dict[str, dict]:
    """Synchronous wrapper around scrape_mbfc_bulk_async.

    Kept for backwards compatibility and one-off CLI use.
    The `client` and `delay` arguments are accepted but the async version
    manages its own client internally; `client` is ignored here.
    """
    return asyncio.run(scrape_mbfc_bulk_async(domains, delay=delay))


# ---------------------------------------------------------------------------
# MBFC detail page parser (shared by sync and async paths)
# ---------------------------------------------------------------------------

def _parse_mbfc_detail(soup: BeautifulSoup) -> Optional[dict]:
    """Extract bias and factuality from an MBFC detail page."""
    text = soup.get_text(separator=" ", strip=True).lower()

    bias = None
    factuality = None

    bias_match = re.search(
        r"bias(?:\s+rating)?[:\s]+([a-z\s\-]+?)(?:[\|\n\r]|factual)",
        text,
    )
    if bias_match:
        raw = bias_match.group(1).strip().rstrip("-")
        bias = _MBFC_BIAS_MAP.get(raw)

    fact_match = re.search(
        r"factual\s+reporting[:\s]+([a-z\s]+?)(?:[\|\n\r]|country|world|press)",
        text,
    )
    if fact_match:
        raw = fact_match.group(1).strip()
        factuality = _MBFC_FACTUALITY_MAP.get(raw)

    if not bias and not factuality:
        return None
    return {"bias": bias, "factuality": factuality}
