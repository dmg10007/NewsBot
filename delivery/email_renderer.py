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
  - 2–3 sentence summary
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

_SECTION_LABELS: dict[str, str] = {
    "national": "National",
    "state":    "North Carolina",
    "local":    "Local — Lee County",
}
_SECTION_ORDER: list[str] = ["national", "state", "local"]

# System font stack: renders crisply on iOS without loading external fonts.
_FONT_STACK = (
    "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,"
    "'Helvetica Neue',Arial,sans-serif"
)


def _bias_badge(lean: str, colors: dict[str, str]) -> str:
    color = colors.get(lean, _DEFAULT_BIAS_COLORS["unknown"])
    label = _DEFAULT_BIAS_LABELS.get(lean, lean.title())
    return (
        f'<span style="display:inline-block;padding:2px 6px;border-radius:3px;'
        f'background:{color};color:#fff;font-size:11px;font-weight:700;'
        f'vertical-align:middle;margin-left:3px;line-height:1.4;">'
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
        self._bias_colors: dict[str, str] = (
            email_cfg.get("bias_tag_colors") or _DEFAULT_BIAS_COLORS
        )

    def render(self, summaries: list[SummarizedCluster], period: str) -> str:
        """Return a complete HTML email string for the given digest period."""
        date_str = __import__("datetime").date.today().strftime("%B %d, %Y")
        title = f"NewsBot — {period.title()} Briefing"

        sections: dict[str, list[SummarizedCluster]] = {
            "national": [], "state": [], "local": []
        }
        for story in summaries:
            primary_tier = story.tiers[0] if story.tiers else "national"
            bucket = primary_tier if primary_tier in sections else "national"
            sections[bucket].append(story)

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
<tr><td style="padding:20px 0 6px;">
  <h2 style="margin:0;font-size:11px;font-weight:700;text-transform:uppercase;
             letter-spacing:0.1em;color:#888;border-bottom:1px solid #e8e8e8;
             padding-bottom:8px;font-family:{_FONT_STACK};">
    {html.escape(heading)}
  </h2>
</td></tr>
{cards}
"""

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1.0">
  <title>{html.escape(title)}</title>
  <style>
    body {{ margin:0; padding:0; background:#f4f4f0; }}
    .wrapper {{ background:#f4f4f0; padding:16px; }}
    .card {{ background:#fff; border:1px solid #ddd; border-radius:6px;
             max-width:600px; width:100%; margin:0 auto; }}
    .header {{ background:#1a1a1a; padding:18px 20px; border-radius:6px 6px 0 0; }}
    .header-label {{ margin:0; font-size:10px; color:#888; letter-spacing:0.1em;
                     text-transform:uppercase; font-family:{_FONT_STACK}; }}
    .header-title {{ margin:4px 0 0; font-size:20px; font-weight:700; color:#fff;
                     font-family:{_FONT_STACK}; }}
    .body-pad {{ padding:0 20px 20px; }}
    .footer {{ padding:12px 20px; border-top:1px solid #eee; }}
    .footer p {{ margin:0; font-size:11px; color:#aaa; font-family:{_FONT_STACK}; }}
    @media (max-width:600px) {{
      .wrapper {{ padding:0 !important; }}
      .card {{ border-radius:0 !important; border-left:none !important;
               border-right:none !important; }}
      .header {{ padding:16px !important; }}
      .header-title {{ font-size:18px !important; }}
      .body-pad {{ padding:0 16px 16px !important; }}
      .story-headline {{ font-size:16px !important; }}
      .story-body {{ font-size:14px !important; line-height:1.7 !important; }}
      .chip {{ font-size:12px !important; padding:3px 8px !important; }}
    }}
  </style>
</head>
<body>
  <div class="wrapper">
    <div class="card">
      <div class="header">
        <p class="header-label">NewsBot Daily Briefing</p>
        <p class="header-title">{html.escape(period.title())} Edition &mdash; {html.escape(date_str)}</p>
      </div>
      <div class="body-pad">
        <table width="100%" cellpadding="0" cellspacing="0">
          {sections_html}
        </table>
      </div>
      <div class="footer">
        <p>Generated by NewsBot &mdash; facts only, framing stripped.</p>
      </div>
    </div>
  </div>
</body>
</html>"""

    def _render_story(self, story: SummarizedCluster) -> str:
        top_url = story.source_links[0][1] if story.source_links else ""
        if top_url:
            headline_html = (
                f'<a href="{html.escape(top_url)}" class="story-headline" '
                f'style="color:#1a1a1a;text-decoration:none;font-weight:700;'
                f'font-size:17px;line-height:1.35;font-family:{_FONT_STACK};">'
                f'{html.escape(story.headline)}</a>'
            )
        else:
            headline_html = (
                f'<span class="story-headline" style="font-weight:700;font-size:17px;'
                f'line-height:1.35;font-family:{_FONT_STACK};">'
                f'{html.escape(story.headline)}</span>'
            )

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
                f'<a href="{html.escape(url)}" class="chip" '
                f'style="display:inline-block;margin:2px 4px 2px 0;'
                f'font-size:12px;color:#333;text-decoration:none;'
                f'background:#f0f0f0;padding:3px 8px;border-radius:4px;'
                f'font-family:{_FONT_STACK};vertical-align:middle;">'
                f'{html.escape(source_name)}</a>'
                f'{badge}'
            )
            source_chips += chip

        bias_note_html = ""
        if story.bias_notes and not story.bias_notes.startswith("Automated bias"):
            bias_note_html = (
                f'<p style="margin:8px 0 0;font-size:12px;color:#666;'
                f'font-style:italic;border-left:3px solid #e0e0e0;'
                f'padding-left:10px;font-family:{_FONT_STACK};line-height:1.5;">'
                f'<strong>Framing:</strong> {html.escape(story.bias_notes)}</p>'
            )

        return f"""
<tr><td style="padding:16px 0;border-bottom:1px solid #f0f0f0;">
  <p style="margin:0 0 8px;">{headline_html}</p>
  <p class="story-body" style="margin:0 0 10px;font-size:14px;line-height:1.7;
     color:#333;font-family:{_FONT_STACK};">{html.escape(story.summary)}</p>
  <div style="line-height:2.0;">{source_chips}</div>
  {bias_note_html}
</td></tr>"""
