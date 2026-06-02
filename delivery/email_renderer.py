"""HTML email rendering for the stable DigestStory contract."""

from __future__ import annotations

import html
from datetime import datetime, timezone

from config.loader import get_settings
from domain.models import DigestRun, DigestStory, SourceLink

_DEFAULT_BIAS_COLORS: dict[str, str] = {
    "left": "#1565c0",
    "center-left": "#1976d2",
    "center": "#388e3c",
    "center-right": "#e64a19",
    "right": "#b71c1c",
    "unknown": "#757575",
}
_DEFAULT_BIAS_LABELS: dict[str, str] = {
    "left": "Left",
    "center-left": "Lean Left",
    "center": "Center",
    "center-right": "Lean Right",
    "right": "Right",
    "unknown": "Unknown",
}
_SECTION_ORDER = ["national", "state", "local"]
_FONT_STACK = "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif"


class EmailRenderer:
    def __init__(self, settings: dict | None = None) -> None:
        self.settings = settings or get_settings()
        email_cfg = self.settings.get("delivery", {}).get("email", {})
        self._max_per_section = int(email_cfg.get("max_stories_per_category", 7))
        self._bias_colors = email_cfg.get("bias_tag_colors") or _DEFAULT_BIAS_COLORS
        self._section_labels = (
            self.settings.get("geography", {}).get("labels")
            or {
                "national": "National",
                "state": "North Carolina",
                "local": "Local - Lee County",
            }
        )

    def render(self, stories: list[DigestStory], run: DigestRun | str) -> str:
        if isinstance(run, str):
            run = DigestRun(run_id=None, period=run, started_at=datetime.now(timezone.utc))
        date_str = run.started_at.strftime("%B %d, %Y")
        title = f"NewsBot {run.period.title()} Briefing"
        sections = {tier: [] for tier in _SECTION_ORDER}
        for story in stories:
            bucket = story.geo_tier if story.geo_tier in sections else "national"
            sections[bucket].append(story)

        sections_html = ""
        for tier in _SECTION_ORDER:
            tier_stories = sorted(
                sections[tier],
                key=lambda s: s.importance_score,
                reverse=True,
            )[: self._max_per_section]
            if not tier_stories:
                continue
            cards = "\n".join(self._render_story(story) for story in tier_stories)
            sections_html += f"""
<tr><td style="padding:20px 0 6px;">
  <h2 style="margin:0;font-size:12px;font-weight:700;text-transform:uppercase;
             letter-spacing:0.08em;color:#666;border-bottom:1px solid #e8e8e8;
             padding-bottom:8px;font-family:{_FONT_STACK};">
    {html.escape(self._section_labels.get(tier, tier.title()))}
  </h2>
</td></tr>
{cards}
"""

        if not sections_html:
            sections_html = f"""
<tr><td style="padding:24px 0;font-family:{_FONT_STACK};color:#555;">
  No reportable stories found for this run.
</td></tr>
"""

        story_count = len(stories)
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1.0">
  <title>{html.escape(title)}</title>
</head>
<body style="margin:0;padding:0;background:#f4f4f0;">
  <div style="background:#f4f4f0;padding:16px;">
    <div style="background:#fff;border:1px solid #ddd;border-radius:6px;max-width:680px;width:100%;margin:0 auto;">
      <div style="background:#1a1a1a;padding:18px 20px;border-radius:6px 6px 0 0;">
        <p style="margin:0;font-size:10px;color:#aaa;letter-spacing:0.1em;text-transform:uppercase;font-family:{_FONT_STACK};">
          NewsBot Daily Briefing
        </p>
        <p style="margin:4px 0 0;font-size:20px;font-weight:700;color:#fff;font-family:{_FONT_STACK};">
          {html.escape(run.period.title())} Briefing - {html.escape(date_str)}
        </p>
        <p style="margin:6px 0 0;font-size:12px;color:#bbb;font-family:{_FONT_STACK};">
          {story_count} {'story' if story_count == 1 else 'stories'}
        </p>
      </div>
      <div style="padding:0 20px 20px;">
        <table width="100%" cellpadding="0" cellspacing="0">{sections_html}</table>
      </div>
      <div style="padding:12px 20px;border-top:1px solid #eee;">
        <p style="margin:0;font-size:11px;color:#888;font-family:{_FONT_STACK};">
          Summaries minimize loaded language and label source attribution; perfect objectivity is not guaranteed.
        </p>
      </div>
    </div>
  </div>
</body>
</html>"""

    def _render_story(self, story: DigestStory) -> str:
        top_url = story.source_links[0].article_url or story.source_links[0].url if story.source_links else ""
        headline = html.escape(story.headline)
        headline_html = (
            f'<a href="{html.escape(top_url)}" style="color:#1a1a1a;text-decoration:none;'
            f'font-weight:700;font-size:17px;line-height:1.35;font-family:{_FONT_STACK};">{headline}</a>'
            if top_url else
            f'<span style="font-weight:700;font-size:17px;line-height:1.35;font-family:{_FONT_STACK};">{headline}</span>'
        )
        source_note = (
            "Single source"
            if story.is_single_source
            else f"{story.source_count} sources"
        )
        # Render a single unified Source Perspectives callout block.
        # bias_notes now contains one line per outlet from the SOURCE PERSPECTIVES
        # section of the comparison prompt, replacing the old split
        # 'Reporting differences' / 'Bias note' display.
        perspectives_html = ""
        if story.comparison.bias_notes and not story.is_single_source:
            # Each perspective is on its own line; render as individual rows
            # so the callout stays scannable with many sources.
            lines = [
                l.strip() for l in story.comparison.bias_notes.splitlines() if l.strip()
            ]
            rows = "<br>".join(html.escape(l) for l in lines)
            perspectives_html = (
                f'<p style="margin:10px 0 0;font-size:12px;color:#444;'
                f'border-left:3px solid #e0e0e0;padding-left:10px;'
                f'font-family:{_FONT_STACK};line-height:1.7;">'
                f'<strong>Source perspectives:</strong><br>{rows}</p>'
            )
        return f"""
<tr><td style="padding:16px 0;border-bottom:1px solid #f0f0f0;">
  <p style="margin:0 0 5px;">{headline_html}</p>
  <p style="margin:0 0 8px;font-size:12px;color:#777;font-family:{_FONT_STACK};">{html.escape(source_note)}</p>
  <p style="margin:0 0 10px;font-size:14px;line-height:1.7;color:#333;font-family:{_FONT_STACK};">{html.escape(story.summary)}</p>
  <div style="line-height:2.0;">{self._render_sources(story.source_links)}</div>
  {perspectives_html}
</td></tr>"""

    def _render_sources(self, links: list[SourceLink]) -> str:
        chips = []
        for link in links:
            href = html.escape(link.article_url or link.url)
            chips.append(
                f'<a href="{href}" style="display:inline-block;margin:2px 4px 2px 0;'
                f'font-size:12px;color:#333;text-decoration:none;background:#f0f0f0;'
                f'padding:3px 8px;border-radius:4px;font-family:{_FONT_STACK};">'
                f'{html.escape(link.source_name)}</a>{_bias_badge(link.bias_lean, self._bias_colors)}'
            )
        return "".join(chips)


def _bias_badge(lean: str, colors: dict[str, str]) -> str:
    color = colors.get(lean, _DEFAULT_BIAS_COLORS["unknown"])
    label = _DEFAULT_BIAS_LABELS.get(lean, lean.title())
    return (
        f'<span style="display:inline-block;padding:2px 6px;border-radius:3px;'
        f'background:{color};color:#fff;font-size:11px;font-weight:700;'
        f'font-family:{_FONT_STACK};vertical-align:middle;margin-left:3px;line-height:1.4;">'
        f'{html.escape(label)}</span>'
    )


def render_digest(stories: list[DigestStory], period: str, run_date: datetime) -> str:
    """Compatibility wrapper around the new renderer."""
    run = DigestRun(run_id=None, period=period, started_at=run_date)
    return EmailRenderer().render(stories, run)
