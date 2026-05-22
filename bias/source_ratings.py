"""Scrapers for AllSides and Media Bias Fact Check source ratings.

Builds a normalized lookup table: domain -> {bias_lean, factuality, confidence}.
Designed to be called once at startup (or weekly via cron) and cached in memory.

No paid API required:
  AllSides  — scrapes public media bias ratings page (single request, sync)
  MBFC      — scrapes public search + detail pages (sequential, rate-limited)

MBFC crawl posture
------------------
MBFC is a small WordPress site. We scrape it sequentially (max_concurrent=1)
with a 2.0s delay between domains for the weekly cron path. The semaphore
architecture is kept so max_concurrent can be raised if needed, but the
default is deliberately conservative to avoid 429s.

Each domain requires 2 HTTP requests: search page + detail page.
For 17 domains at 2.0s delay: ~35s total — acceptable for a weekly job.

AllSides anti-bot mitigation
----------------------------
AllSides uses Cloudflare bot detection. We send a full browser-like header
set (UA + Accept + Accept-Language + Sec-Fetch-*) on all outbound requests.
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

# Full browser header set — required to pass Cloudflare / AllSides bot checks.
# UA alone is insufficient; Sec-Fetch-* and Accept-Language are also checked.
_BROWSER_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,image/apng,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
    "Cache-Control": "max-age=0",
}

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

    Uses a full browser header set (not just UA) to pass Cloudflare checks.
    """
    url = "https://www.allsides.com/media-bias/ratings"
    ratings: dict[str, str] = {}

    try:
        resp = client.get(url, timeout=20, headers=_BROWSER_HEADERS)
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

# Map domain -> preferred MBFC search term when the domain slug gives bad results.
# MBFC's search is fuzzy; some outlets only surface correctly under their full name.
_MBFC_SEARCH_MAP: dict[str, str] = {
    "apnews.com": "associated press",
    "wsj.com": "wall street journal",
    "pbs.org": "pbs newshour",
    "foxnews.com": "fox news",
    "foxbusiness.com": "fox business",
    "cnn.com": "cnn",
    "npr.org": "npr",
    "thehill.com": "the hill",
    "axios.com": "axios",
    "wcnc.com": "wcnc charlotte",
    "charlotteobserver.com": "charlotte observer",
    "ncpolicywatch.com": "nc policy watch",
    "carolinapublicpress.org": "carolina public press",
    "rantnc.com": "rant nc",
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

# A profile URL has exactly one path segment (the slug), e.g.:
#   mediabiasfactcheck.com/reuters/         -> GOOD
#   mediabiasfactcheck.com/left/cnn-bias/   -> BAD (sub-directory)
#   mediabiasfactcheck.com/2017/02/26/...   -> BAD (date-based article)
_MBFC_PROFILE_RE = re.compile(
    r"https?://mediabiasfactcheck\.com/([^/]+)/$"
)


def _is_mbfc_profile_url(url: str) -> bool:
    """Return True only for source-profile URLs (single slug, no sub-paths)."""
    return bool(_MBFC_PROFILE_RE.match(url))


# --- Async implementation ---

async def scrape_mbfc_source_async(
    client: httpx.AsyncClient,
    domain: str,
) -> Optional[dict]:
    """Async: scrape MBFC rating for a single domain via their search.

    Returns: {bias: str, factuality: str} or None.
    Makes 2 HTTP requests: search page, then first *profile* result page.
    """
    search_term = _MBFC_SEARCH_MAP.get(domain, domain.split(".")[0])
    search_url = f"https://mediabiasfactcheck.com/?s={search_term.replace(' ', '+')}"

    try:
        resp = await client.get(search_url, timeout=15)
        resp.raise_for_status()
    except Exception as exc:
        logger.debug("MBFC search failed for %s: %s", domain, exc)
        return None

    soup = BeautifulSoup(resp.text, "lxml")

    # Walk all article links and pick the first that looks like a source profile.
    detail_url: Optional[str] = None
    for a in soup.select("h2.entry-title a, h3.entry-title a, .post-title a, article a"):
        href = a.get("href", "")
        if _is_mbfc_profile_url(href):
            detail_url = href
            break

    if not detail_url:
        logger.debug(
            "MBFC: no profile link found for %s (search_term=%r). HTML snippet: %s",
            domain,
            search_term,
            resp.text[:600].replace("\n", " "),
        )
        logger.warning("MBFC: no result for %s", domain)
        return None

    try:
        detail_resp = await client.get(detail_url, timeout=15)
        detail_resp.raise_for_status()
    except Exception as exc:
        logger.debug("MBFC detail fetch failed for %s: %s", detail_url, exc)
        return None

    detail_soup = BeautifulSoup(detail_resp.text, "lxml")
    return _parse_mbfc_detail(detail_soup, domain)


async def scrape_mbfc_bulk_async(
    domains: list[str],
    delay: float = 2.0,
    max_concurrent: int = 1,
) -> dict[str, dict]:
    """Async: scrape MBFC for a list of domains with concurrency control.

    Default is max_concurrent=1 (fully sequential) with a 2.0s delay —
    the safe posture for a weekly cron hitting a small WordPress site.
    Raise max_concurrent only if you know the target can handle it.

    Returns: {domain: {bias, factuality}}
    """
    sem = asyncio.Semaphore(max_concurrent)
    results: dict[str, dict] = {}

    async def _fetch_one(client: httpx.AsyncClient, domain: str, index: int) -> None:
        async with sem:
            if index > 0:
                await asyncio.sleep(delay)
            result = await scrape_mbfc_source_async(client, domain)
            if result:
                results[domain] = result
                logger.info("MBFC: %s -> %s", domain, result)
            else:
                logger.warning("MBFC: no result for %s", domain)

    async with httpx.AsyncClient(
        headers=_BROWSER_HEADERS,
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
    """Synchronous wrapper — delegates to scrape_mbfc_bulk_async."""
    return asyncio.run(scrape_mbfc_bulk_async([domain])).get(domain)


def scrape_mbfc_bulk(
    client: httpx.Client,
    domains: list[str],
    delay: float = 2.0,
) -> dict[str, dict]:
    """Synchronous wrapper — delegates to scrape_mbfc_bulk_async.

    `client` is accepted for API compatibility but ignored;
    the async version manages its own client internally.
    """
    return asyncio.run(scrape_mbfc_bulk_async(domains, delay=delay))


# ---------------------------------------------------------------------------
# MBFC detail page parser (shared by sync and async paths)
# ---------------------------------------------------------------------------

def _parse_mbfc_detail(soup: BeautifulSoup, domain: str = "") -> Optional[dict]:
    """Extract bias and factuality from an MBFC detail page.

    MBFC formats the Detailed Report block as a <p> or <li> containing lines like:
        Bias Rating: LEAST BIASED (-0.5)
        Factual Reporting: VERY HIGH (0.0)

    We locate the element containing 'Bias Rating:' and parse from its text
    rather than running a regex over the entire page (which previously matched
    comment section text and failed on bold markers).
    """
    bias: Optional[str] = None
    factuality: Optional[str] = None

    # Strategy 1: find the structured "Detailed Report" block.
    # MBFC wraps it in a <p> or <li> that contains both labels.
    for tag in soup.find_all(["p", "li", "div"]):
        text = tag.get_text(separator=" ", strip=True)
        if "Bias Rating:" not in text and "bias rating:" not in text.lower():
            continue

        # Extract bias from this block
        b_match = re.search(
            r"[Bb]ias\s+[Rr]ating\s*:\s*\**([A-Z][A-Z\s\-]+?)\**\s*(?:\(|[A-Z]{2,}|\n|$)",
            text,
        )
        if b_match:
            raw_bias = b_match.group(1).strip().rstrip("-").lower()
            bias = _MBFC_BIAS_MAP.get(raw_bias)

        # Extract factuality from the same block or the next sibling
        f_match = re.search(
            r"[Ff]actual\s+[Rr]eporting\s*:\s*\**([A-Z][A-Z\s]+?)\**\s*(?:\(|[A-Z]{2,}|\n|$)",
            text,
        )
        if f_match:
            raw_fact = f_match.group(1).strip().lower()
            factuality = _MBFC_FACTUALITY_MAP.get(raw_fact)

        if bias or factuality:
            break

    # Strategy 2: fallback — scan the full page text with relaxed patterns.
    # Handles pages where the report block uses unusual markup.
    if not bias and not factuality:
        full_text = soup.get_text(separator="\n", strip=True)

        b_match = re.search(
            r"[Bb]ias\s+[Rr]ating\s*[:\-]\s*([A-Za-z][A-Za-z\s\-]+?)(?:\s*\(|\s*\n|\s{3,})",
            full_text,
        )
        if b_match:
            bias = _MBFC_BIAS_MAP.get(b_match.group(1).strip().lower())

        f_match = re.search(
            r"[Ff]actual\s+[Rr]eporting\s*[:\-]\s*([A-Za-z][A-Za-z\s]+?)(?:\s*\(|\s*\n|\s{3,})",
            full_text,
        )
        if f_match:
            factuality = _MBFC_FACTUALITY_MAP.get(f_match.group(1).strip().lower())

    if not bias and not factuality:
        logger.debug("MBFC: detail parser found nothing for %s", domain)
        return None

    return {"bias": bias, "factuality": factuality}
