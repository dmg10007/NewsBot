"""HTML email renderer for digest output.

Layout
------
Stories are grouped into three geographic sections rendered in this order:
  1. National
  2. North Carolina
  3. Local (Lee County / Sanford)

Each section shows at most max_stories_per_category stories (settings.yaml
delivery.email.max_stories_per_category, default 7), sorted by
importance_score descending.

Each story card renders:
  - Headline linked to the top source URL
  - 2-3 sentence summary
  - Per-source chips: linked outlet name + colored bias badge side-by-side
  - Framing note (if bias analysis produced one)

Bias tag colors
---------------
Colors for each bias lean label are read from settings.yaml under
delivery.email.bias_tag_colors. Hardcoded defaults are used as fallbacks.
_bias_colors is resolved once at EmailRenderer.__init__() — not on every
badge render — so get_settings() dict traversal is not repeated for each of
the ~80 badges in a typical digest.

Example settings.yaml entry::

    delivery:
      email:
        max_stories_per_category: 7
        bias_tag_colors:
          left:         "#1565c0"
          center-left:  "#1976d2"
          center:       "#388e3c"
          center-right: "#e64a19"
          right:        "#b71c1c"
          unknown:      "#757575"
"""

from __future__ import annotations

import html

from config.loader import get_settings
from summarizer.summarizer import SummarizedCluster

_DEFAULT_BIAS_COLORS: dict[str, str] = {
    "left":          "#1565c0",
    "center-left":   "#1976d2",
    "center":        "#388e3c",
    "center-right":  "#e64a19",
    "right":         "#b71c1c",
    "unknown":       "#757575",
}
_DEFAULT_BIAS_LABELS: dict[str, str] = {
    "left":          "Left",
    "center-left":   "Lean Left",
    "center":        "Center",
    "center-right":  "Lean Right",
    "right":         "Right",
    "unknown":       "Unknown",
}

# Tier label → section heading displayed in the email
_SECTION_LABELS: dict[str, str] = {
    "national": "National",
    "state":    "North Carolina",
    "local":    "Local — Lee County",
}
_SECTION_ORDER: list[str] = ["national", "state", "local"]


def _bias_badge(lean: str, colors: dict[str, str]) -> str:
    """Render a small colored inline badge for a bias lean label.

    Args:
        lean:   Bias lean string (e.g. 'left', 'center-right', 'unknown').
        colors: Pre-resolved color map from EmailRenderer._bias_colors.
                Passed as a parameter so callers do not re-traverse settings
                on every badge render.
    """
    color = colors.get(lean, _DEFAULT_BIAS_COLORS["unknown"])
    label = _DEFAULT_BIAS_LABELS.get(lean, lean.title())
    return (
        f'<span style="display:inline-block;padding:1px 5px;border-radius:2px;'
        f'background:{color};color:#fff;font-size:10px;font-weight:700;'
        f'vertical-align:middle;margin-left:3px;">'
        f'{html.escape(label)}</span>'
    )


class EmailRenderer:
    """Renders a list of SummarizedClusters into an HTML email digest."""

    def __init__(self) -> None:
        settings = get_settings()
        email_cfg = settings.get("delivery", {}).get("email", {})
        self._max_per_section: int = int(
            email_cfg.get("max_stories_per_category", 7)
        )
        # Resolve bias colors once at construction time.
        # A typical digest renders ~80 badges; resolving the color map here
        # avoids 80 repeated settings dict traversals per digest run.
        self._bias_colors: dict[str, str] = (
            email_cfg.get("bias_tag_colors") or _DEFAULT_BIAS_COLORS
        )

    def render(self, summaries: list[SummarizedCluster], period: str) -> str:
        """Return a complete HTML email string for the given digest period."""
        date_str = __import__("datetime").date.today().strftime("%B %d, %Y")
        title = f"NewsBot — {period.title()} Briefing"

        # Group stories by primary tier (first tier in the list)
        sections: dict[str, list[SummarizedCluster]] = {
            "national": [], "state": [], "local": []
        }
        for story in summaries:
            primary_tier = story.tiers[0] if story.tiers else "national"
            bucket = primary_tier if primary_tier in sections else "national"
            sections[bucket].append(story)

        # Sort each section by importance and cap at max_per_section
        sections_html = ""
        for tier in _SECTION_ORDER:
            stories = sorted(
                sections[tier],
                key=lambda s: s.importance_score,
                reverse=True,
            )[:self._max_per_section]
            if not stories:
                continue
            heading = _SECTION_LABELS[tier]
            cards = "\n".join(self._render_story(s) for s in stories)
            sections_html += f"""
<tr><td style="padding:24px 0 8px;">
  <h2 style="margin:0;font-size:15px;font-weight:700;text-transform:uppercase;
             letter-spacing:0.08em;color:#555;border-bottom:1px solid #e0e0e0;
             padding-bottom:6px;">{html.escape(heading)}</h2>
</td></tr>
{cards}
"""

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1.0">
  <title>{html.escape(title)}</title>
</head>
<body style="margin:0;padding:0;background:#f4f4f0;font-family:Georgia,serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#f4f4f0;">
    <tr><td align="center" style="padding:24px 16px;">
      <table width="600" cellpadding="0" cellspacing="0"
             style="max-width:600px;width:100%;background:#fff;
                    border:1px solid #ddd;border-radius:4px;">

        <!-- Header -->
        <tr><td style="background:#1a1a1a;padding:20px 24px;border-radius:4px 4px 0 0;">
          <p style="margin:0;font-size:11px;color:#888;letter-spacing:0.1em;
                    text-transform:uppercase;">NewsBot Daily Briefing</p>
          <p style="margin:4px 0 0;font-size:20px;font-weight:700;color:#fff;">
            {html.escape(period.title())} Edition &mdash; {html.escape(date_str)}</p>
        </td></tr>

        <!-- Body -->
        <tr><td style="padding:0 24px 24px;">
          <table width="100%" cellpadding="0" cellspacing="0">
            {sections_html}
          </table>
        </td></tr>

        <!-- Footer -->
        <tr><td style="padding:12px 24px;border-top:1px solid #eee;">
          <p style="margin:0;font-size:10px;color:#aaa;">Generated by NewsBot &mdash; facts only, framing stripped.</p>
        </td></tr>

      </table>
    </td></tr>
  </table>
</body>
</html>"""

    def _render_story(self, story: SummarizedCluster) -> str:
        # Headline linked to the first available source URL
        top_url = story.source_links[0][1] if story.source_links else ""
        if top_url:
            headline_html = (
                f'<a href="{html.escape(top_url)}" '
                f'style="color:#1a1a1a;text-decoration:none;font-weight:700;">'
                f'{html.escape(story.headline)}</a>'
            )
        else:
            headline_html = (
                f'<span style="font-weight:700;">{html.escape(story.headline)}</span>'
            )

        # Per-source chips: linked name + bias badge.
        # Skip entries where source_name or url is empty — these are malformed
        # RawArticle entries where source_url was not set during scraping.
        source_chips = ""
        seen: set[str] = set()
        for source_name, url in story.source_links:
            if not source_name or not url:
                continue
            if source_name in seen:
                continue
            seen.add(source_name)
            lean = story.source_bias.get(source_name, "unknown")
            badge = _bias_badge(lean, self._bias_colors)
            chip = (
                f'<a href="{html.escape(url)}" '
                f'style="display:inline-block;margin:2px 4px 2px 0;'
                f'font-size:11px;color:#333;text-decoration:none;'
                f'background:#f0f0f0;padding:2px 6px;border-radius:3px;">'
                f'{html.escape(source_name)}</a>'
                f'{badge}'
            )
            source_chips += chip

        # Framing note
        bias_note_html = ""
        if story.bias_notes and not story.bias_notes.startswith("Automated bias"):
            bias_note_html = (
                f'<p style="margin:6px 0 0;font-size:11px;color:#666;'
                f'font-style:italic;border-left:2px solid #ddd;padding-left:8px;">'
                f'<strong>Framing:</strong> {html.escape(story.bias_notes)}</p>'
            )

        return f"""
<tr><td style="padding:14px 0;border-bottom:1px solid #f0f0f0;">
  <p style="margin:0 0 6px;font-size:15px;line-height:1.4;">{headline_html}</p>
  <p style="margin:0 0 8px;font-size:13px;line-height:1.6;color:#444;">{html.escape(story.summary)}</p>
  <div style="line-height:1.8;">{source_chips}</div>
  {bias_note_html}
</td></tr>"""
