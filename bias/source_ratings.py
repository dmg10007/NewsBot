"""Scrapers for AllSides and Media Bias Fact Check source ratings.

Builds a normalized lookup table: domain -> {bias_lean, factuality, confidence, sources}.
Designed to be called once at startup (or weekly via cron) and cached in memory.

No paid API required:
- AllSides: scrapes public media bias ratings page
- MBFC: scrapes public search results page
"""

from __future__ import annotations

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

# Normalized 5-point bias scale used internally
BIAS_SCALE = ("left", "center-left", "center", "center-right", "right")

# Normalized factuality scale
FACTUALITY_SCALE = ("very-low", "low", "mixed", "mostly-factual", "high", "very-high")


@dataclass
class SourceRating:
    """Merged bias + factuality rating for a single news domain."""
    domain: str
    bias_lean: str                        # normalized BIAS_SCALE value
    factuality: str                       # normalized FACTUALITY_SCALE value
    confidence: float                     # 0.0-1.0; agreement between sources
    allsides_bias: Optional[str] = None  # raw AllSides label
    mbfc_bias: Optional[str] = None      # raw MBFC label
    mbfc_factuality: Optional[str] = None
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# AllSides scraper
# ---------------------------------------------------------------------------

# Map AllSides display labels -> internal BIAS_SCALE
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
    key = raw.strip().lower()
    return _ALLSIDES_BIAS_MAP.get(key)


def scrape_allsides(client: httpx.Client) -> dict[str, str]:
    """Scrape AllSides media bias ratings page.

    Returns: {domain: normalized_bias_lean}
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

    # AllSides renders ratings in a table with class 'views-table'
    table = soup.find("table", {"class": re.compile(r"views-table")})
    if not table:
        logger.warning("AllSides: could not find ratings table — page structure may have changed")
        return ratings

    for row in table.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 3:
            continue

        # Column 0: outlet name+link, Column 2: bias rating image/cell
        name_cell = cells[0]
        bias_cell = cells[2]

        # Extract domain from the outlet link
        link = name_cell.find("a", href=True)
        outlet_href = link["href"] if link else ""
        # AllSides links are like /news-source/ap or have an external URL attr
        # Try to extract domain from a data attribute or the outlet name
        domain = _extract_domain_allsides(name_cell, bias_cell)
        if not domain:
            continue

        # Bias is encoded as an image alt text or a div class like 'allsides-left'
        bias_raw = ""
        bias_img = bias_cell.find("img")
        if bias_img and bias_img.get("alt"):
            bias_raw = bias_img["alt"]
        else:
            bias_div = bias_cell.find(attrs={"class": re.compile(r"allsides-")})
            if bias_div:
                cls = " ".join(bias_div.get("class", []))
                bias_raw = cls

        normalized = _normalize_allsides_bias(bias_raw)
        if normalized:
            ratings[domain] = normalized
            logger.debug("AllSides: %s -> %s", domain, normalized)

    logger.info("AllSides: scraped %d source ratings", len(ratings))
    return ratings


def _extract_domain_allsides(name_cell: BeautifulSoup, bias_cell: BeautifulSoup) -> Optional[str]:
    """Best-effort domain extraction from an AllSides table row."""
    # Try external link in name cell
    for a in name_cell.find_all("a", href=True):
        href = a["href"]
        if href.startswith("http"):
            return _domain_from_url(href)

    # Fall back to outlet name -> hardcoded map for top sources
    text = name_cell.get_text(separator=" ", strip=True).lower()
    return _OUTLET_DOMAIN_MAP.get(text)


def _domain_from_url(url: str) -> Optional[str]:
    match = re.search(r"https?://(?:www\.)?([^/]+)", url)
    return match.group(1).lower() if match else None


# Hardcoded domain map for outlets AllSides links internally (no external href)
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
# Media Bias Fact Check (MBFC) scraper
# ---------------------------------------------------------------------------

_MBFC_BIAS_MAP: dict[str, str] = {
    "left": "left",
    "left-center": "center-left",
    "least biased": "center",
    "right-center": "center-right",
    "right": "right",
    "extreme left": "left",
    "extreme right": "right",
    "conspiracy-pseudoscience": "right",  # treat as right-fringe for our purposes
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


def scrape_mbfc_source(client: httpx.Client, domain: str) -> Optional[dict]:
    """Scrape MBFC rating for a single domain via their search.

    Returns: {bias: str, factuality: str} or None
    """
    search_url = f"https://mediabiasfactcheck.com/?s={domain.split('.')[0]}"
    try:
        resp = client.get(search_url, timeout=15)
        resp.raise_for_status()
    except Exception as exc:
        logger.debug("MBFC search failed for %s: %s", domain, exc)
        return None

    soup = BeautifulSoup(resp.text, "lxml")

    # Find the first result link and follow it
    result = soup.select_one("h2.entry-title a, .post-title a")
    if not result:
        return None

    detail_url = result["href"]
    try:
        detail_resp = client.get(detail_url, timeout=15)
        detail_resp.raise_for_status()
    except Exception as exc:
        logger.debug("MBFC detail fetch failed for %s: %s", detail_url, exc)
        return None

    detail_soup = BeautifulSoup(detail_resp.text, "lxml")
    return _parse_mbfc_detail(detail_soup)


def _parse_mbfc_detail(soup: BeautifulSoup) -> Optional[dict]:
    """Extract bias and factuality from an MBFC detail page."""
    text = soup.get_text(separator=" ", strip=True).lower()

    bias = None
    factuality = None

    # MBFC pages contain lines like "Bias Rating: Left-Center"
    bias_match = re.search(
        r"bias(?:\s+rating)?[:\s]+([a-z\s\-]+?)(?:[\|\n\r]|factual)",
        text,
    )
    if bias_match:
        raw = bias_match.group(1).strip().rstrip("-")
        bias = _MBFC_BIAS_MAP.get(raw)

    # "Factual Reporting: High"
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


def scrape_mbfc_bulk(
    client: httpx.Client,
    domains: list[str],
    delay: float = 1.5,
) -> dict[str, dict]:
    """Scrape MBFC for a list of domains with polite rate limiting.

    Returns: {domain: {bias, factuality}}
    """
    results: dict[str, dict] = {}
    for i, domain in enumerate(domains):
        result = scrape_mbfc_source(client, domain)
        if result:
            results[domain] = result
            logger.debug("MBFC: %s -> %s", domain, result)
        else:
            logger.debug("MBFC: no result for %s", domain)
        if i < len(domains) - 1:
            time.sleep(delay)  # polite crawl delay
    logger.info("MBFC: scraped %d/%d domains", len(results), len(domains))
    return results
