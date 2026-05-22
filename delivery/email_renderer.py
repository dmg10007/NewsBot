"""Builds the HTML email digest from a list of SummaryResults.

Output is a clean, dense briefing — not a feed.
Design principles:
  - Signal over noise: no ads, no images, no clutter
  - Scannable: topic sections with clear headers
  - Transparent: source count, tier badges, bias notes visible but unobtrusive
  - Single-source stories are flagged with a low-confidence indicator
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from summarizer.summarizer import SummaryResult

DigestPeriod = Literal["morning", "evening"]

TOPIC_ORDER = ["politics", "economy", "current_events"]
TOPIC_LABELS = {
    "politics": "Politics",
    "economy": "Economy",
    "current_events": "Current Events",
}
TIER_BADGE_COLOR = {
    "national": "#2c7be5",
    "state": "#00897b",
    "local": "#e65100",
}


def _format_date(dt: datetime) -> str:
    """Return a date string like 'Friday, May 22, 2026'.

    Uses str(dt.day) instead of %-d so it works on Windows (no GNU strftime).
    """
    return dt.strftime("%A, %B ") + str(dt.day) + dt.strftime(", %Y")


def render_digest(summaries: list[SummaryResult], period: DigestPeriod, run_date: datetime) -> str:
    """Render a full HTML email digest string."""
    date_str = _format_date(run_date)
    period_label = "Morning" if period == "morning" else "Evening"

    # Group by topic, respecting max_stories limits set in settings.yaml
    from config.loader import get_settings
    settings = get_settings()
    topic_limits = {
        "politics": settings["topics"]["politics"]["max_stories"],
        "economy": settings["topics"]["economy"]["max_stories"],
        "current_events": settings["topics"]["current_events"]["max_stories"],
    }

    grouped: dict[str, list[SummaryResult]] = {t: [] for t in TOPIC_ORDER}
    for s in summaries:
        topic = s.topic if s.topic in grouped else "current_events"
        limit = topic_limits.get(topic, 10)
        if len(grouped[topic]) < limit:
            grouped[topic].append(s)

    sections_html = ""
    for topic in TOPIC_ORDER:
        stories = grouped[topic]
        if not stories:
            continue
        sections_html += _render_section(topic, stories)

    total = sum(len(v) for v in grouped.values())

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NewsBot {period_label} Briefing</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         font-size: 15px; line-height: 1.6; color: #1a1a1a; background: #f5f5f0;
         margin: 0; padding: 0; }}
  .wrapper {{ max-width: 680px; margin: 0 auto; padding: 24px 16px; }}
  .header {{ border-bottom: 2px solid #1a1a1a; padding-bottom: 12px; margin-bottom: 28px; }}
  .header h1 {{ font-size: 20px; font-weight: 700; margin: 0 0 4px 0; letter-spacing: -0.3px; }}
  .header .meta {{ font-size: 13px; color: #666; }}
  .section-header {{ font-size: 11px; font-weight: 700; letter-spacing: 1.2px;
                     text-transform: uppercase; color: #888; margin: 32px 0 14px 0;
                     padding-bottom: 6px; border-bottom: 1px solid #ddd; }}
  .story {{ background: #fff; border-radius: 6px; padding: 16px 18px;
            margin-bottom: 10px; border: 1px solid #e8e8e5; }}
  .story-headline {{ font-size: 15px; font-weight: 600; margin: 0 0 8px 0;
                     line-height: 1.3; color: #111; }}
  .story-summary {{ font-size: 14px; color: #333; margin: 0 0 10px 0; line-height: 1.55; }}
  .story-meta {{ font-size: 12px; color: #888; display: flex; flex-wrap: wrap; gap: 8px;
                 align-items: center; }}
  .badge {{ display: inline-block; font-size: 10px; font-weight: 600;
            letter-spacing: 0.5px; text-transform: uppercase; padding: 2px 7px;
            border-radius: 3px; color: #fff; }}
  .badge-unverified {{ background: #999; }}
  .bias-notes {{ font-size: 12px; color: #777; background: #f9f9f7;
                 border-left: 3px solid #ddd; padding: 8px 10px;
                 margin-top: 10px; border-radius: 0 4px 4px 0; font-style: italic; }}
  .footer {{ margin-top: 40px; padding-top: 16px; border-top: 1px solid #ddd;
             font-size: 12px; color: #aaa; text-align: center; }}
</style>
</head>
<body>
<div class="wrapper">
  <div class="header">
    <h1>NewsBot {period_label} Briefing</h1>
    <div class="meta">{date_str} &nbsp;&middot;&nbsp; {total} stories across {len([t for t in grouped if grouped[t]])} topics</div>
  </div>
  {sections_html}
  <div class="footer">
    NewsBot &mdash; bias-aware news digest for Lee County, NC &middot; National &middot; State<br>
    Sources: AP, Reuters, NPR, PBS, Fox News, CNN, WSJ, The Hill, WRAL, Carolina Public Press,
    NC Policy Watch, Charlotte Observer, Sanford Herald, The Rant NC
  </div>
</div>
</body>
</html>
""".strip()


def _render_section(topic: str, stories: list[SummaryResult]) -> str:
    label = TOPIC_LABELS.get(topic, topic.replace("_", " ").title())
    items = "".join(_render_story(s) for s in stories)
    return f'<div class="section-header">{label}</div>{items}'


def _render_story(s: SummaryResult) -> str:
    tier_badges = "".join(
        f'<span class="badge" style="background:{TIER_BADGE_COLOR.get(t, "#888")}">{t}</span>'
        for t in sorted(s.tiers_covered)
    )
    unverified = (
        '<span class="badge badge-unverified" title="Single source — lower confidence">'
        '1 source</span>'
        if s.is_single_source else
        f'<span style="font-size:12px;color:#888">{s.source_count} sources</span>'
    )
    bias_block = ""
    if s.bias_notes and s.bias_notes not in (
        "No significant framing differences detected.",
        "Automated bias analysis unavailable for this story.",
        "Analysis skipped: daily LLM call limit reached.",
    ):
        bias_block = f'<div class="bias-notes">&#9432; {s.bias_notes}</div>'

    return f"""
<div class="story">
  <div class="story-headline">{s.representative_headline}</div>
  <div class="story-summary">{s.summary}</div>
  <div class="story-meta">{tier_badges}{unverified}</div>
  {bias_block}
</div>"""
