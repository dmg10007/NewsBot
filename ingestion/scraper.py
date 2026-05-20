"""HTML scrapers for outlets without RSS feeds.

Each scraper is a self-contained class that fetches and parses
a specific outlet's public-facing pages. Add new outlets by
subclassing BaseScraper and registering in SCRAPER_REGISTRY.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Optional

import httpx
from bs4 import BeautifulSoup

from config.loader import get_settings
from ingestion.fetcher import RawArticle

logger = logging.getLogger(__name__)


class BaseScraper(ABC):
    """Base class for all outlet scrapers."""

    def __init__(self, source_config: dict) -> None:
        self.source = source_config
        self.settings = get_settings()
        self._client = httpx.Client(
            timeout=self.settings["ingestion"]["request_timeout_seconds"],
            headers={"User-Agent": self.settings["ingestion"]["user_agent"]},
            follow_redirects=True,
        )

    @abstractmethod
    def scrape(self) -> list[RawArticle]:
        """Fetch and parse articles from the outlet. Returns list of RawArticles."""
        ...

    def _get_html(self, url: str) -> Optional[BeautifulSoup]:
        max_retries = self.settings["ingestion"]["max_retries"]
        backoff = self.settings["ingestion"]["retry_backoff_seconds"]
        for attempt in range(1, max_retries + 1):
            try:
                response = self._client.get(url)
                response.raise_for_status()
                return BeautifulSoup(response.text, "lxml")
            except Exception as exc:
                logger.warning(
                    "Scrape attempt %d/%d failed for %s: %s",
                    attempt, max_retries, url, exc
                )
                if attempt < max_retries:
                    time.sleep(backoff * attempt)
        return None

    def _make_article(
        self,
        url: str,
        headline: str,
        summary: str = "",
        published_at: Optional[datetime] = None,
    ) -> Optional[RawArticle]:
        if not url or not headline:
            return None
        return RawArticle(
            url=url,
            headline=headline.strip(),
            summary=summary.strip(),
            source_name=self.source["name"],
            bias_lean=self.source.get("bias_lean", "unknown"),
            credibility=self.source.get("credibility", "medium"),
            topics=self.source.get("topics", []),
            region=self.source.get("region", "national"),
            published_at=published_at,
        )

    def close(self) -> None:
        self._client.close()


class GenericRSSBackedScraper(BaseScraper):
    """For outlets that have an RSS feed URL listed in sources.yaml as 'rss_url'.
    Falls back to HTML scraping if RSS is unavailable.
    """

    def scrape(self) -> list[RawArticle]:
        import feedparser
        rss_url = self.source.get("rss_url")
        if not rss_url:
            logger.error("%s: no rss_url configured", self.source["name"])
            return []
        feed = feedparser.parse(rss_url)
        articles = []
        max_per_source = self.settings["ingestion"]["max_articles_per_source"]
        for entry in feed.entries[:max_per_source]:
            url = getattr(entry, "link", "").strip()
            headline = getattr(entry, "title", "").strip()
            summary = getattr(entry, "summary", "").strip()
            published_at = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                published_at = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
            article = self._make_article(url, headline, summary, published_at)
            if article:
                articles.append(article)
        logger.info("Scraped %d articles from %s (RSS)", len(articles), self.source["name"])
        return articles


class CharlotteObserverScraper(GenericRSSBackedScraper):
    """Charlotte Observer — uses their RSS feed."""
    pass  # GenericRSSBackedScraper handles it via rss_url in sources.yaml


class SanfordHeraldScraper(BaseScraper):
    """Sanford Herald (Lee County, NC) — HTML scraper from public pages."""

    def scrape(self) -> list[RawArticle]:
        base_url = self.source["url"]
        soup = self._get_html(base_url)
        if not soup:
            logger.error("SanfordHeraldScraper: could not fetch %s", base_url)
            return []

        selectors = self.source.get("selectors", {})
        articles = []
        max_per_source = self.settings["ingestion"]["max_articles_per_source"]

        for item in soup.select(selectors.get("article_list", "article"))[:max_per_source]:
            headline_el = item.select_one(selectors.get("headline", "h2 a"))
            if not headline_el:
                continue
            headline = headline_el.get_text(strip=True)
            href = headline_el.get("href", "")
            url = href if href.startswith("http") else base_url.rstrip("/") + href

            summary_el = item.select_one(selectors.get("summary", "p"))
            summary = summary_el.get_text(strip=True) if summary_el else ""

            date_el = item.select_one(selectors.get("date", "time"))
            published_at = None
            if date_el:
                dt_str = date_el.get("datetime") or date_el.get_text(strip=True)
                published_at = _parse_date_safe(dt_str)

            article = self._make_article(url, headline, summary, published_at)
            if article:
                articles.append(article)

        logger.info("Scraped %d articles from Sanford Herald", len(articles))
        return articles


class RantNCScraper(BaseScraper):
    """The Rant NC (rantnc.com) — WordPress-based HTML scraper."""

    def scrape(self) -> list[RawArticle]:
        base_url = self.source["url"]
        # The Rant NC is WordPress — try the native WP JSON API first
        wp_api_url = f"{base_url}/wp-json/wp/v2/posts?per_page=20&_fields=title,link,excerpt,date"
        try:
            response = self._client.get(wp_api_url)
            if response.status_code == 200:
                return self._parse_wp_json(response.json())
        except Exception as exc:
            logger.warning("RantNCScraper: WP JSON API failed, falling back to HTML: %s", exc)

        # HTML fallback
        return self._parse_html(base_url)

    def _parse_wp_json(self, posts: list[dict]) -> list[RawArticle]:
        articles = []
        for post in posts:
            headline = BeautifulSoup(post.get("title", {}).get("rendered", ""), "lxml").get_text()
            url = post.get("link", "")
            excerpt_html = post.get("excerpt", {}).get("rendered", "")
            summary = BeautifulSoup(excerpt_html, "lxml").get_text(strip=True)
            published_at = _parse_date_safe(post.get("date", ""))
            article = self._make_article(url, headline, summary, published_at)
            if article:
                articles.append(article)
        logger.info("Scraped %d articles from The Rant NC (WP JSON)", len(articles))
        return articles

    def _parse_html(self, base_url: str) -> list[RawArticle]:
        soup = self._get_html(base_url)
        if not soup:
            return []
        selectors = self.source.get("selectors", {})
        articles = []
        max_per_source = self.settings["ingestion"]["max_articles_per_source"]
        for item in soup.select(selectors.get("article_list", "article"))[:max_per_source]:
            headline_el = item.select_one(selectors.get("headline", "h2 a"))
            if not headline_el:
                continue
            headline = headline_el.get_text(strip=True)
            href = headline_el.get("href", "")
            url = href if href.startswith("http") else base_url.rstrip("/") + href
            summary_el = item.select_one(selectors.get("summary", ".entry-summary p"))
            summary = summary_el.get_text(strip=True) if summary_el else ""
            date_el = item.select_one(selectors.get("date", "time"))
            published_at = _parse_date_safe(date_el.get("datetime", "") if date_el else "")
            article = self._make_article(url, headline, summary, published_at)
            if article:
                articles.append(article)
        logger.info("Scraped %d articles from The Rant NC (HTML)", len(articles))
        return articles


# Registry — maps scraper_class string in sources.yaml to the class
SCRAPER_REGISTRY: dict[str, type[BaseScraper]] = {
    "CharlotteObserverScraper": CharlotteObserverScraper,
    "SanfordHeraldScraper": SanfordHeraldScraper,
    "RantNCScraper": RantNCScraper,
}


def _parse_date_safe(date_str: str) -> Optional[datetime]:
    """Best-effort ISO 8601 / common date string parser."""
    if not date_str:
        return None
    from dateutil import parser as dateutil_parser
    try:
        return dateutil_parser.parse(date_str).replace(tzinfo=timezone.utc)
    except Exception:
        return None
