"""Scrapers for AllSides and Media Bias Fact Check source ratings.

Builds a normalized lookup table: domain -> {bias_lean, factuality, confidence}.
Designed to be called once at startup (or weekly via cron) and cached to disk.

Both AllSides and MBFC use Playwright headless Chromium.

Install:
    pip install playwright
    playwright install chromium

Fallback behaviour
------------------
  AllSides  -> _ALLSIDES_STATIC  (hardcoded May 2026 ratings)
  MBFC      -> empty dict  (resolver falls back to _FALLBACK_RATINGS)
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

MBFC_MATCH_THRESHOLD = 0.85


@dataclass
class SourceRating:
    domain: str
    bias_lean: str
    factuality: str
    confidence: float
    allsides_bias: Optional[str] = None
    mbfc_bias: Optional[str] = None
    mbfc_factuality: Optional[str] = None
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# AllSides static fallback (May 2026)
# ---------------------------------------------------------------------------

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
# AllSides scraper
# ---------------------------------------------------------------------------
# The Tailwind redesign (2025) no longer uses CSS classes to encode bias.
# The bias label is rendered as plain visible text in the third <td>, e.g.:
#   "Left", "Lean Left", "Center", "Lean Right", "Right"
# We extract that text directly.

_ALLSIDES_TEXT_MAP: dict[str, str] = {
    "left":        "left",
    "lean left":   "center-left",
    "center":      "center",
    "lean right":  "center-right",
    "right":       "right",
    # legacy img-alt labels kept as fallback
    "allsides lean left":  "center-left",
    "allsides lean right": "center-right",
    "allsides left":       "left",
    "allsides right":      "right",
    "allsides center":     "center",
}

# Words that appear in the bias cell but are NOT the bias label.
_ALLSIDES_NOISE = frozenset({
    "agree", "disagree", "allsides", "bias", "rating",
    "news", "media", "source", "type",
})


def _extract_allsides_bias(bias_cell) -> Optional[str]:
    """Extract normalized bias from the third <td> of an AllSides row.

    Strategy 1: color-* CSS class on any descendant (future-proof).
    Strategy 2: visible plain text in the cell.
    Strategy 3: img alt text.
    Strategy 4: aria-label.
    """
    _CLASS_MAP = {
        "color-left":         "left",
        "color-center-left":  "center-left",
        "color-center":       "center",
        "color-center-right": "center-right",
        "color-right":        "right",
    }

    # Strategy 1: color-* class
    for tag in bias_cell.find_all(True):
        for cls in tag.get("class", []):
            if cls in _CLASS_MAP:
                return _CLASS_MAP[cls]

    # Strategy 2: plain text — strip noise words, match what remains
    raw = bias_cell.get_text(separator=" ", strip=True).lower()
    # Remove parenthesised notes like "(agree/disagree)"
    raw = re.sub(r"\([^)]*\)", "", raw).strip()
    for noise in _ALLSIDES_NOISE:
        raw = re.sub(rf"\b{re.escape(noise)}\b", "", raw)
    raw = " ".join(raw.split())  # collapse whitespace
    if raw in _ALLSIDES_TEXT_MAP:
        return _ALLSIDES_TEXT_MAP[raw]

    # Strategy 3: img alt
    img = bias_cell.find("img")
    if img and img.get("alt"):
        return _ALLSIDES_TEXT_MAP.get(img["alt"].strip().lower())

    # Strategy 4: aria-label
    for tag in bias_cell.find_all(True):
        label = (tag.get("aria-label") or "").strip().lower()
        if label in _ALLSIDES_TEXT_MAP:
            return _ALLSIDES_TEXT_MAP[label]

    return None


def _parse_allsides_html(html: str) -> dict[str, str]:
    ratings: dict[str, str] = {}
    soup = BeautifulSoup(html, "lxml")

    table = soup.find("table", {"class": lambda c: c and "w-full" in c})
    if not table:
        table = soup.find("table", {"class": re.compile(r"views-table")})
    if not table:
        logger.warning("AllSides: could not find ratings table")
        return ratings

    for row in table.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 3:
            continue
        domain = _extract_domain_allsides(cells[0])
        if not domain:
            continue
        bias = _extract_allsides_bias(cells[2])
        if bias:
            ratings[domain] = bias

    logger.info("AllSides: scraped %d source ratings", len(ratings))
    return ratings


def _extract_domain_allsides(name_cell) -> Optional[str]:
    for a in name_cell.find_all("a", href=True):
        href = a["href"]
        if href.startswith("http"):
            return _domain_from_url(href)
    text = name_cell.get_text(separator=" ", strip=True).lower()
    return _OUTLET_DOMAIN_MAP.get(text)


def _domain_from_url(url: str) -> Optional[str]:
    match = re.search(r"https?://(?:www\.)?([^/]+)", url)
    return match.group(1).lower() if match else None


def _scrape_allsides_with_context(context) -> dict[str, str]:
    from playwright.sync_api import TimeoutError as PWTimeout
    url = "https://www.allsides.com/media-bias/ratings"
    try:
        page = context.new_page()
        page.goto(url, wait_until="networkidle", timeout=30_000)
        try:
            page.wait_for_selector("table", timeout=10_000)
        except PWTimeout:
            logger.warning("AllSides: table not found after networkidle")
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
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        logger.warning("AllSides: playwright not installed — using static fallback")
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
# MBFC scraper
# ---------------------------------------------------------------------------
# Profile URLs changed structure: previously /slug/, now /bias-category/slug/
# e.g.  https://mediabiasfactcheck.com/left/cnn-bias/
# We detect profiles by excluding known nav/utility path prefixes.

# Slugs that are navigation or utility pages, not source profiles.
_MBFC_NAV_SLUGS = frozenset({
    "membership-account", "gift-memberships", "support-media-bias-fact-check",
    "filtered-search", "mbfcs-data-api", "login", "about", "funding",
    "methodology", "corrections-policy", "changes-corrections", "news",
    "category", "search", "fact-check-search", "country-profiles",
    "world-leaders-facts-and-bias", "united-states-governors-bias-ratings",
    "united-states-senators-facts-and-bias-ratings", "journalist-bias",
    "educational-class-materials-by-media-bias-fact-check",
    "educational-media-sources", "interactive-maps-and-charts-by-mbfc",
    "interactive-political-orientation-map-of-the-world",
    "interactive-country-freedom-map-of-the-world",
    "trump-administration-performance-tracker-chart",
    "electoral-college-map-2024-biden-vs-trump", "re-evaluated-sources",
    "appsextensions", "sources-pending", "submit-source",
    "pseudoscience-dictionary", "membership-questions",
    "mbfc-ratings-by-the-numbers", "election-center-2024",
    "media-categories", "wp-login.php", "page",
    # bias category index pages (not individual profiles)
    "center", "leftcenter", "left", "right-center", "right",
    "fake-news", "conspiracy", "pro-science", "satire",
})


def _is_mbfc_profile_url(url: str) -> bool:
    """Return True if url looks like an MBFC source profile page.

    Valid forms (as of May 2026):
        https://mediabiasfactcheck.com/slug/
        https://mediabiasfactcheck.com/bias-category/slug/

    Rejected: nav pages, category indexes, date-based posts, fact-checks,
    journalist profiles, login pages, pagination.
    """
    if "mediabiasfactcheck.com" not in url:
        return False
    # Strip scheme + domain
    path = re.sub(r"https?://mediabiasfactcheck\.com", "", url).strip("/")
    if not path:
        return False
    parts = [p for p in path.split("/") if p]
    # Must be 1 or 2 path segments
    if len(parts) not in (1, 2):
        return False
    # Reject known nav slugs
    if parts[0] in _MBFC_NAV_SLUGS:
        return False
    # Reject date-based URLs (/2026/05/...)
    if re.match(r"^\d{4}$", parts[0]):
        return False
    # Reject fact-check posts
    if parts[0].startswith("fact-check-"):
        return False
    # Reject journalist profiles
    if parts[-1].endswith("-bias-rating"):
        return False
    # Reject query strings
    if "?" in url:
        return False
    return True


_MBFC_BIAS_MAP: dict[str, str] = {
    "left":                     "left",
    "left-center":              "center-left",
    "least biased":             "center",
    "right-center":             "center-right",
    "right":                    "right",
    "extreme left":             "left",
    "extreme right":            "right",
    "conspiracy-pseudoscience": "right",
    "questionable":             "right",
    "pro-science":              "center",
    "satire":                   "center",
}

_MBFC_FACTUALITY_MAP: dict[str, str] = {
    "very high":      "very-high",
    "high":           "high",
    "mostly factual": "mostly-factual",
    "mixed":          "mixed",
    "low":            "low",
    "very low":       "very-low",
}


def _wait_past_cloudflare(page, timeout_ms: int = 15_000) -> None:
    from playwright.sync_api import TimeoutError as PWTimeout
    try:
        page.wait_for_selector("article, .search-results, .entry-title", timeout=timeout_ms)
    except PWTimeout:
        logger.debug("_wait_past_cloudflare: content selector not found within %dms", timeout_ms)


def _scrape_mbfc_domain_with_context(context, domain: str) -> Optional[dict]:
    from playwright.sync_api import TimeoutError as PWTimeout

    search_term = _MBFC_SEARCH_MAP.get(domain, domain.split(".")[0])
    search_url = f"https://mediabiasfactcheck.com/?s={search_term.replace(' ', '+')}"

    # --- Search page ---
    try:
        page = context.new_page()
        page.goto(search_url, wait_until="networkidle", timeout=25_000)
        _wait_past_cloudflare(page)
        search_html = page.content()
        page.close()
    except PWTimeout:
        logger.debug("MBFC: search page timed out for %s", domain)
        return None
    except Exception as exc:
        logger.debug("MBFC: search page failed for %s: %s", domain, exc)
        return None

    if "Checking your browser" in search_html and "<article" not in search_html:
        logger.warning("MBFC: Cloudflare challenge not resolved for %s", domain)
        return None

    soup = BeautifulSoup(search_html, "lxml")
    detail_url: Optional[str] = None
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if _is_mbfc_profile_url(href):
            detail_url = href
            break

    if not detail_url:
        logger.warning("MBFC: no profile result for %s (searched %r)", domain, search_term)
        return None

    # --- Detail page ---
    try:
        page = context.new_page()
        page.goto(detail_url, wait_until="domcontentloaded", timeout=25_000)
        # Detail pages load fast; domcontentloaded is enough
        detail_html = page.content()
        page.close()
    except PWTimeout:
        logger.debug("MBFC: detail page timed out for %s", domain)
        return None
    except Exception as exc:
        logger.debug("MBFC: detail page failed for %s: %s", domain, exc)
        return None

    return _parse_mbfc_detail(BeautifulSoup(detail_html, "lxml"), domain)


def scrape_mbfc_bulk(
    client: httpx.Client,
    domains: list[str],
    delay: float = 1.5,
) -> dict[str, dict]:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        logger.warning("MBFC: playwright not installed — skipping MBFC scrape")
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


def scrape_mbfc_source(client: httpx.Client, domain: str) -> Optional[dict]:
    return scrape_mbfc_bulk(client, [domain]).get(domain)


# ---------------------------------------------------------------------------
# Combined scraper
# ---------------------------------------------------------------------------

def scrape_all(
    client: httpx.Client,
    domains: list[str],
    delay: float = 1.5,
) -> tuple[dict[str, str], dict[str, dict]]:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        logger.warning("playwright not installed — using AllSides static fallback, skipping MBFC")
        return dict(_ALLSIDES_STATIC), {}

    allsides_result: dict[str, str] = {}
    mbfc_result: dict[str, dict] = {}

    try:
        with sync_playwright() as pw:
            browser, context = _make_playwright_context(pw)
            try:
                logger.info("Scraping AllSides via Playwright...")
                allsides_result = _scrape_allsides_with_context(context)

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
# MBFC detail parser
# ---------------------------------------------------------------------------

def _parse_mbfc_detail(soup: BeautifulSoup, domain: str = "") -> Optional[dict]:
    bias: Optional[str] = None
    factuality: Optional[str] = None

    full_text = soup.get_text(separator="\n", strip=True)

    for line in full_text.splitlines():
        ll = line.strip().lower()
        if not bias and re.search(r"bias\s+rating", ll):
            m = re.search(r"bias\s+rating\s*[:\-]\s*(.+)", ll)
            if m:
                bias = _MBFC_BIAS_MAP.get(m.group(1).strip().strip("*"))
        if not factuality and re.search(r"factual\s+reporting", ll):
            m = re.search(r"factual\s+reporting\s*[:\-]\s*(.+)", ll)
            if m:
                factuality = _MBFC_FACTUALITY_MAP.get(m.group(1).strip().strip("*"))
        if bias and factuality:
            break

    if not bias or not factuality:
        for tag in soup.find_all(["p", "td", "li", "div"]):
            text = tag.get_text(separator=" ", strip=True).lower()
            if not bias:
                m = re.search(r"bias\s+rating\s*[:\-]\s*([a-z][a-z\s\-]+)", text)
                if m:
                    bias = _MBFC_BIAS_MAP.get(m.group(1).strip().strip("*"))
            if not factuality:
                m = re.search(r"factual\s+reporting\s*[:\-]\s*([a-z][a-z\s]+)", text)
                if m:
                    factuality = _MBFC_FACTUALITY_MAP.get(m.group(1).strip().strip("*"))
            if bias and factuality:
                break

    if not bias and not factuality:
        logger.debug("MBFC: detail parser found nothing for %s", domain)
        return None

    return {"bias": bias, "factuality": factuality}
