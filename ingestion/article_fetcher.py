"""Full article body fetcher.

Attempts to extract clean main-content text from an article URL using
httpx + BeautifulSoup. Used as the first enrichment step when the RSS
description is junk or a headline echo.

Design notes:
  - Only fires when RSS content fails quality checks (opt-in, not default)
  - Hard timeout (10s) so a slow site never blocks the pipeline
  - Strips boilerplate: nav, header, footer, aside, ads, scripts, styles
  - Returns at most _MAX_BODY_CHARS characters to keep downstream LLM costs low
  - Returns empty string on any failure — callers must handle gracefully
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_TIMEOUT = 10.0          # seconds — never block the pipeline for a slow site
_MAX_BODY_CHARS = 4000   # enough for 2-4 paragraph summary; keeps LLM tokens low

# Tags that are reliably boilerplate — stripped before text extraction
_STRIP_TAGS = {
    "script", "style", "noscript", "nav", "header", "footer",
    "aside", "form", "button", "iframe", "figure", "figcaption",
    "advertisement", "ads",
}

# CSS class/id fragments that signal ad or nav content
_NOISE_PATTERNS = re.compile(
    r"(ad|ads|advert|advertisement|banner|breadcrumb|byline|"
    r"comment|cookie|footer|header|menu|modal|nav|newsletter|"
    r"popup|promo|related|sidebar|social|subscribe|widget)",
    re.IGNORECASE,
)


class ArticleFetcher:
    """Fetches and extracts main body text from article URLs."""

    def __init__(self) -> None:
        self._client = httpx.Client(
            timeout=_TIMEOUT,
            follow_redirects=True,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (compatible; NewsBot/1.0; "
                    "+https://github.com/dmg10007/NewsBot)"
                )
            },
        )

    def fetch_body(self, url: str) -> str:
        """Return cleaned article body text, or empty string on failure."""
        try:
            response = self._client.get(url)
            response.raise_for_status()
            return self._extract(response.text)
        except Exception as exc:
            logger.debug("Article fetch failed for %s: %s", url, exc)
            return ""

    def _extract(self, html: str) -> str:
        soup = BeautifulSoup(html, "lxml")

        # Remove boilerplate tags entirely
        for tag in soup.find_all(_STRIP_TAGS):
            tag.decompose()

        # Remove elements whose class or id looks like noise
        for tag in soup.find_all(True):
            classes = " ".join(tag.get("class", []))
            tag_id = tag.get("id", "")
            if _NOISE_PATTERNS.search(classes) or _NOISE_PATTERNS.search(tag_id):
                tag.decompose()

        # Try semantic content containers first
        body_text = ""
        for selector in ("article", "main", '[role="main"]', ".article-body",
                         ".story-body", ".post-content", ".entry-content"):
            container = soup.select_one(selector)
            if container:
                body_text = container.get_text(separator=" ", strip=True)
                break

        # Fallback: grab all <p> tags from body
        if not body_text:
            paragraphs = soup.find_all("p")
            body_text = " ".join(p.get_text(strip=True) for p in paragraphs)

        # Collapse whitespace
        body_text = re.sub(r"\s+", " ", body_text).strip()
        return body_text[:_MAX_BODY_CHARS]

    def close(self) -> None:
        self._client.close()
