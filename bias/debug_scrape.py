"""Debug script: dumps raw HTML structure and all hrefs from MBFC search results.

Run:  python -m bias.debug_scrape

Outputs:
    debug_allsides.html
    debug_mbfc_search.html   (MBFC search for 'cnn')
    debug_mbfc_detail.html   (first linked page, whatever it is)

Key console output:
    - AllSides table class names + bias CSS classes found
    - Every <a href> on the MBFC search page (no filtering)
    - Text content of every line containing 'bias' or 'factual' on MBFC detail page
"""

from __future__ import annotations
from pathlib import Path


def _wait_for_content(page, timeout_ms: int = 15_000) -> None:
    from playwright.sync_api import TimeoutError as PWTimeout
    try:
        page.wait_for_selector("article, .entry-title, .search-results, h2", timeout=timeout_ms)
    except PWTimeout:
        pass


def main() -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright not installed. Run: pip install playwright && playwright install chromium")
        return

    from bs4 import BeautifulSoup

    with sync_playwright() as pw:
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

        # ----------------------------------------------------------------
        # AllSides
        # ----------------------------------------------------------------
        print("=" * 60)
        print("ALLSIDES")
        print("=" * 60)
        page = context.new_page()
        page.goto("https://www.allsides.com/media-bias/ratings", wait_until="networkidle", timeout=30_000)
        page.wait_for_selector("table", timeout=10_000)
        html = page.content()
        page.close()
        Path("debug_allsides.html").write_text(html, encoding="utf-8")
        print(f"Saved debug_allsides.html ({len(html):,} bytes)")

        soup = BeautifulSoup(html, "lxml")
        tables = soup.find_all("table")
        print(f"Tables found: {len(tables)}")
        for t in tables:
            print(f"  class={t.get('class')}  id={t.get('id')}")

        # Sample first 3 data rows
        for t in tables[:1]:
            rows = t.find_all("tr")
            print(f"  Row count: {len(rows)}")
            for row in rows[1:4]:
                cells = row.find_all("td")
                if cells:
                    all_classes = []
                    for tag in row.find_all(True):
                        all_classes += tag.get("class", [])
                    bias_classes = [c for c in all_classes if any(
                        k in c for k in ("color-", "bias", "left", "right", "center")
                    )]
                    print(f"  Row text: {row.get_text(' ', strip=True)[:80]}")
                    print(f"    bias-related classes: {list(set(bias_classes))}")

        # ----------------------------------------------------------------
        # MBFC search
        # ----------------------------------------------------------------
        print()
        print("=" * 60)
        print("MBFC SEARCH — 'cnn'")
        print("=" * 60)
        page = context.new_page()
        page.goto("https://mediabiasfactcheck.com/?s=cnn", wait_until="networkidle", timeout=25_000)
        _wait_for_content(page)
        search_html = page.content()
        page.close()
        Path("debug_mbfc_search.html").write_text(search_html, encoding="utf-8")
        print(f"Saved debug_mbfc_search.html ({len(search_html):,} bytes)")

        soup2 = BeautifulSoup(search_html, "lxml")

        # Cloudflare check
        h1s = [h.get_text(strip=True) for h in soup2.find_all("h1")]
        print(f"H1 tags: {h1s}")

        # Print EVERY href on the page — no filtering
        all_hrefs = [a["href"] for a in soup2.find_all("a", href=True)]
        print(f"Total <a href> count: {len(all_hrefs)}")
        print("All hrefs:")
        for href in all_hrefs:
            print(f"  {href}")

        # ----------------------------------------------------------------
        # MBFC detail — follow first mediabiasfactcheck.com link
        # ----------------------------------------------------------------
        import re
        detail_url = None
        for href in all_hrefs:
            if "mediabiasfactcheck.com" in href and href != "https://mediabiasfactcheck.com/":
                detail_url = href
                break

        if detail_url:
            print()
            print("=" * 60)
            print(f"MBFC DETAIL — {detail_url}")
            print("=" * 60)
            page = context.new_page()
            page.goto(detail_url, wait_until="networkidle", timeout=25_000)
            _wait_for_content(page)
            detail_html = page.content()
            page.close()
            Path("debug_mbfc_detail.html").write_text(detail_html, encoding="utf-8")
            print(f"Saved debug_mbfc_detail.html ({len(detail_html):,} bytes)")

            soup3 = BeautifulSoup(detail_html, "lxml")
            full_text = soup3.get_text(separator="\n", strip=True)
            keywords = ("bias", "factual", "rating", "reporting")
            print("Lines containing bias/factual keywords:")
            for line in full_text.splitlines():
                if any(k in line.lower() for k in keywords) and len(line.strip()) > 3:
                    print(f"  {line.strip()[:160]}")
        else:
            print("No mediabiasfactcheck.com links found at all.")

        browser.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
