"""Scrapers for AllSides and Media Bias Fact Check source ratings.

Builds a normalized lookup table: domain -> {bias_lean, factuality, confidence}.
Designed to be called once at startup (or weekly via cron) and cached in memory.

Both AllSides and MBFC now use Playwright headless Chromium. Both sites run
Cloudflare bot detection that blocks plain httpx requests. A real browser
instance passes TLS fingerprinting and JS challenge checks that Cloudflare
uses to distinguish browsers from scripts.

A single Playwright browser instance is shared across all scraping in one
refresh run. Use scrape_all() to get both sources in one browser session.
The individual scrape_allsides() and scrape_mbfc_bulk() entry points are
kept for backwards compatibility and each launch their own browser.

Install:
    pip install playwright
    playwright install chromium

Fallback behaviour
------------------
If Playwright is not installed or Chromium fails to launch:
  - AllSides  → _ALLSIDES_STATIC  (hardcoded May 2026 ratings)
  - MBFC      → empty dict  (resolver falls back to _FALLBACK_RATINGS)

MBFC crawl posture
------------------
MBFC is a small WordPress site. Pages are loaded sequentially with a
configurable delay (default 1.5s). Each domain requires 2 page loads:
search page + detail page. For 17 domains: ~51s total.
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

# Retained for backwards compatibility — used by refresh.py for AllSides client
# (now unused at runtime, but kept so existing imports don’t break).
_BROWSER_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

BIAS_SCALE = ("left", "center-left", "center", "center-right", "right")
FACTUALITY_SCALE = ("very-low", "low", "mixed", "mostly-factual", "high", "very-high")

MBFC_MATCH_THRESHOLD = 0.85  # minimum string similarity for MBFC search result acceptance


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
# AllSides static fallback
# ---------------------------------------------------------------------------
# Used when Playwright is unavailable. Update when adding new sources.
# Ratings sourced from allsides.com/media-bias/ratings (May 2026).

_ALLSIDES_STATIC: dict[str, str] = {
    "apnews.com":              "center",
    "reuters.com":             "center",
    "npr.org":                 "center-left",
    "pbs.org":                 "center",
    "cnn.com":                 "center-left",
    "foxnews.com":             "right",
    "foxbusiness.com":         "right",
    "thehill.com":             "center",
    "axios.com":               "center",
    "wsj.com":                 "center-right",
    "wral.com":                "center",
    "charlotteobserver.com":   "center-left",
    "ncpolicywatch.com":       "center-left",
    "carolinapublicpress.org": "center",
    "wcnc.com":                "center",
    "sanfordherald.com":       "center",
    "rantnc.com":              "center",
}


# ---------------------------------------------------------------------------
# Shared Playwright browser context factory
# ---------------------------------------------------------------------------

def _make_playwright_context(pw):
    """Return a (browser, context) pair with a realistic desktop profile."""
    browser = pw.chromium.launch(headless=True)
    context = browser.new_context(
        viewport={"width": 1280, "height": 800},
        locale="en-US",
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
    )
    return browser, context


# ---------------------------------------------------------------------------
# AllSides scraper — Playwright
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


def _parse_allsides_html(html: str) -> dict[str, str]:
    """Parse the AllSides ratings table from raw HTML."""
    ratings: dict[str, str] = {}
    soup = BeautifulSoup(html, "lxml")
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


def _scrape_allsides_with_context(context) -> dict[str, str]:
    """Scrape AllSides using an existing Playwright browser context."""
    from playwright.sync_api import TimeoutError as PWTimeout
    url = "https://www.allsides.com/media-bias/ratings"
    try:
        page = context.new_page()
        page.goto(url, wait_until="networkidle", timeout=30_000)
        html = page.content()
        page.close()
    except PWTimeout:
        logger.warning("AllSides: page load timed out — using static fallback")
        return dict(_ALLSIDES_STATIC)
    except Exception as exc:
        logger.warning("AllSides: page load failed (%s) — using static fallback", exc)
        return dict(_ALLSIDES_STATIC)

    ratings = _parse_allsides_html(html)
    if not ratings:
        logger.warning("AllSides: parsed 0 ratings — using static fallback")
        return dict(_ALLSIDES_STATIC)
    return ratings


def scrape_allsides(client: httpx.Client) -> dict[str, str]:
    """Scrape AllSides ratings using Playwright headless Chromium.

    `client` is accepted for API compatibility but not used.
    Falls back to _ALLSIDES_STATIC if Playwright is unavailable.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        logger.warning(
            "AllSides: playwright not installed — using static fallback. "
            "Run: pip install playwright && playwright install chromium"
        )
        return dict(_ALLSIDES_STATIC)

    try:
        with sync_playwright() as pw:
            browser, context = _make_playwright_context(pw)
            try:
                return _scrape_allsides_with_context(context)
            finally:
                browser.close()
    except Exception as exc:
        logger.warning("AllSides: Playwright launch failed (%s) — using static fallback", exc)
        return dict(_ALLSIDES_STATIC)


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
# MBFC scraper — Playwright
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

_MBFC_PROFILE_RE = re.compile(
    r"https?://mediabiasfactcheck\.com/([^/]+)/$"
)


def _is_mbfc_profile_url(url: str) -> bool:
    return bool(_MBFC_PROFILE_RE.match(url))


def _scrape_mbfc_domain_with_context(context, domain: str) -> Optional[dict]:
    """Scrape MBFC for a single domain using an existing Playwright context.

    Two page loads: MBFC search → first profile result.
    Returns {bias, factuality} or None.
    """
    from playwright.sync_api import TimeoutError as PWTimeout

    search_term = _MBFC_SEARCH_MAP.get(domain, domain.split(".")[0])
    search_url = f"https://mediabiasfactcheck.com/?s={search_term.replace(' ', '+')}"

    # --- Search page ---
    try:
        page = context.new_page()
        page.goto(search_url, wait_until="domcontentloaded", timeout=20_000)
        search_html = page.content()
        page.close()
    except PWTimeout:
        logger.debug("MBFC: search page timed out for %s", domain)
        return None
    except Exception as exc:
        logger.debug("MBFC: search page failed for %s: %s", domain, exc)
        return None

    soup = BeautifulSoup(search_html, "lxml")
    detail_url: Optional[str] = None
    for a in soup.select("h2.entry-title a, h3.entry-title a, .post-title a, article a"):
        href = a.get("href", "")
        if _is_mbfc_profile_url(href):
            detail_url = href
            break

    if not detail_url:
        logger.warning("MBFC: no profile result for %s (searched %r)", domain, search_term)
        return None

    # --- Detail page ---
    try:
        page = context.new_page()
        page.goto(detail_url, wait_until="domcontentloaded", timeout=20_000)
        detail_html = page.content()
        page.close()
    except PWTimeout:
        logger.debug("MBFC: detail page timed out for %s", domain)
        return None
    except Exception as exc:
        logger.debug("MBFC: detail page failed for %s: %s", domain, exc)
        return None

    return _parse_mbfc_detail(BeautifulSoup(detail_html, "lxml"), domain)


def scrape_mbfc_bulk(client: httpx.Client, domains: list[str], delay: float = 1.5) -> dict[str, dict]:
    """Scrape MBFC for a list of domains using Playwright.

    Sequential with `delay` seconds between domains to avoid hammering the
    server. `client` is accepted for API compatibility but not used.

    Returns {domain: {bias, factuality}}.
    Falls back to an empty dict if Playwright is unavailable.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        logger.warning(
            "MBFC: playwright not installed — skipping MBFC scrape. "
            "Run: pip install playwright && playwright install chromium"
        )
        return {}

    results: dict[str, dict] = {}

    try:
        with sync_playwright() as pw:
            browser, context = _make_playwright_context(pw)
            try:
                for i, domain in enumerate(domains):
                    if i > 0:
                        time.sleep(delay)
                    result = _scrape_mbfc_domain_with_context(context, domain)
                    if result:
                        results[domain] = result
                        logger.info("MBFC: %s -> %s", domain, result)
                    else:
                        logger.warning("MBFC: no result for %s", domain)
            finally:
                browser.close()
    except Exception as exc:
        logger.error("MBFC: Playwright session failed: %s", exc)

    logger.info("MBFC: scraped %d/%d domains", len(results), len(domains))
    return results


# Synchronous wrapper kept for any callers that used the old async path.
def scrape_mbfc_source(client: httpx.Client, domain: str) -> Optional[dict]:
    """Scrape MBFC for a single domain. Convenience wrapper around scrape_mbfc_bulk."""
    return scrape_mbfc_bulk(client, [domain]).get(domain)


# ---------------------------------------------------------------------------
# Combined scraper — single browser session for both sources
# ---------------------------------------------------------------------------

def scrape_all(client: httpx.Client, domains: list[str], delay: float = 1.5) -> tuple[dict[str, str], dict[str, dict]]:
    """Scrape AllSides and MBFC in a single shared Playwright browser session.

    More efficient than calling scrape_allsides() + scrape_mbfc_bulk() separately
    because only one Chromium instance is launched.

    Returns: (allsides_ratings, mbfc_ratings)
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        logger.warning(
            "playwright not installed — using AllSides static fallback, skipping MBFC. "
            "Run: pip install playwright && playwright install chromium"
        )
        return dict(_ALLSIDES_STATIC), {}

    allsides_result: dict[str, str] = {}
    mbfc_result: dict[str, dict] = {}

    try:
        with sync_playwright() as pw:
            browser, context = _make_playwright_context(pw)
            try:
                # AllSides first
                logger.info("Scraping AllSides via Playwright...")
                allsides_result = _scrape_allsides_with_context(context)

                # MBFC sequentially
                logger.info("Scraping MBFC for %d domains (%.1fs delay)...", len(domains), delay)
                for i, domain in enumerate(domains):
                    if i > 0:
                        time.sleep(delay)
                    result = _scrape_mbfc_domain_with_context(context, domain)
                    if result:
                        mbfc_result[domain] = result
                        logger.info("MBFC: %s -> %s", domain, result)
                    else:
                        logger.warning("MBFC: no result for %s", domain)
            finally:
                browser.close()
    except Exception as exc:
        logger.error("scrape_all: Playwright session failed: %s", exc)
        if not allsides_result:
            allsides_result = dict(_ALLSIDES_STATIC)

    logger.info("MBFC: scraped %d/%d domains", len(mbfc_result), len(domains))
    return allsides_result, mbfc_result


# ---------------------------------------------------------------------------
# MBFC detail page parser
# ---------------------------------------------------------------------------

def _parse_mbfc_detail(soup: BeautifulSoup, domain: str = "") -> Optional[dict]:
    """Extract bias and factuality from an MBFC detail page."""
    bias: Optional[str] = None
    factuality: Optional[str] = None

    for tag in soup.find_all(["p", "li", "div"]):
        text = tag.get_text(separator=" ", strip=True)
        if "Bias Rating:" not in text and "bias rating:" not in text.lower():
            continue

        b_match = re.search(
            r"[Bb]ias\s+[Rr]ating\s*:\s*\**([A-Z][A-Z\s\-]+?)\**\s*(?:\(|[A-Z]{2,}|\n|$)",
            text,
        )
        if b_match:
            raw_bias = b_match.group(1).strip().rstrip("-").lower()
            bias = _MBFC_BIAS_MAP.get(raw_bias)

        f_match = re.search(
            r"[Ff]actual\s+[Rr]eporting\s*:\s*\**([A-Z][A-Z\s]+?)\**\s*(?:\(|[A-Z]{2,}|\n|$)",
            text,
        )
        if f_match:
            raw_fact = f_match.group(1).strip().lower()
            factuality = _MBFC_FACTUALITY_MAP.get(raw_fact)

        if bias or factuality:
            break

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
