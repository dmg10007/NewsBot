"""One-shot debug script to capture raw HTML from AllSides and MBFC.

Run:  python -m bias.debug_scrape

Outputs:
    debug_allsides.html     — full AllSides ratings page HTML
    debug_mbfc_search.html  — MBFC search results page for 'cnn'
    debug_mbfc_detail.html  — first MBFC profile page found in search results

Use these to inspect the real page structure and update the CSS selectors
in source_ratings.py accordingly.
"""

from __future__ import annotations

from pathlib import Path


def main() -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright not installed. Run: pip install playwright && playwright install chromium")
        return

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

        # --- AllSides ---
        print("Loading AllSides ratings page...")
        page = context.new_page()
        page.goto("https://www.allsides.com/media-bias/ratings", wait_until="networkidle", timeout=30_000)
        html = page.content()
        page.close()
        Path("debug_allsides.html").write_text(html, encoding="utf-8")
        print(f"  Saved debug_allsides.html ({len(html):,} bytes)")

        # Quick structural hints
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "lxml")
        tables = soup.find_all("table")
        print(f"  Tables found: {len(tables)}")
        for t in tables[:5]:
            print(f"    <table class='{t.get('class', [])}'>  id='{t.get('id', '')}'")
        rows_sample = []
        for t in tables[:3]:
            rows_sample += t.find_all("tr")[:3]
        if rows_sample:
            print(f"  First row sample: {rows_sample[0]}"[:300])

        # Bias-related class names in the page
        bias_classes = set()
        for tag in soup.find_all(class_=True):
            for cls in tag.get("class", []):
                if any(k in cls.lower() for k in ("bias", "rating", "lean", "left", "right", "center")):
                    bias_classes.add(cls)
        print(f"  Bias-related CSS classes: {sorted(bias_classes)[:30]}")

        # --- MBFC search ---
        print("\nLoading MBFC search for 'cnn'...")
        page = context.new_page()
        page.goto("https://mediabiasfactcheck.com/?s=cnn", wait_until="domcontentloaded", timeout=20_000)
        search_html = page.content()
        page.close()
        Path("debug_mbfc_search.html").write_text(search_html, encoding="utf-8")
        print(f"  Saved debug_mbfc_search.html ({len(search_html):,} bytes)")

        # Show all <a href> links that look like profile pages
        soup2 = BeautifulSoup(search_html, "lxml")
        profile_links = [
            a["href"] for a in soup2.find_all("a", href=True)
            if "mediabiasfactcheck.com/" in a.get("href", "")
            and a["href"].count("/") >= 4
        ]
        print(f"  MBFC profile links found: {profile_links[:10]}")

        # Show heading/title tags used in search results
        for sel in ["h1", "h2", "h3", "h4", ".entry-title", ".post-title", ".result-title"]:
            found = soup2.select(sel)
            if found:
                print(f"  Selector '{sel}': {len(found)} hits — first: {found[0].get_text(strip=True)[:80]}")

        # --- MBFC detail ---
        detail_url = next(
            (a["href"] for a in soup2.find_all("a", href=True)
             if "mediabiasfactcheck.com/" in a.get("href", "")
             and a["href"].count("/") >= 4),
            None,
        )
        if detail_url:
            print(f"\nLoading MBFC detail page: {detail_url}")
            page = context.new_page()
            page.goto(detail_url, wait_until="domcontentloaded", timeout=20_000)
            detail_html = page.content()
            page.close()
            Path("debug_mbfc_detail.html").write_text(detail_html, encoding="utf-8")
            print(f"  Saved debug_mbfc_detail.html ({len(detail_html):,} bytes)")

            soup3 = BeautifulSoup(detail_html, "lxml")
            full_text = soup3.get_text(separator="\n", strip=True)
            # Find lines containing bias/factuality keywords
            keywords = ("bias", "factual", "rating", "left", "right", "center", "least")
            for line in full_text.splitlines():
                if any(k in line.lower() for k in keywords) and len(line.strip()) > 5:
                    print(f"  TEXT MATCH: {line.strip()[:120]}")
        else:
            print("\nNo MBFC detail link found in search results.")

        browser.close()

    print("\nDone. Open debug_allsides.html / debug_mbfc_search.html / debug_mbfc_detail.html in a browser.")


if __name__ == "__main__":
    main()
