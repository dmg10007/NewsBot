"""Tests for delivery.email_renderer."""

from __future__ import annotations

from datetime import datetime, timezone

from summarizer.summarizer import SummaryResult
from delivery.email_renderer import render_digest


def _make_summary(
    headline: str,
    topic: str = "politics",
    source_count: int = 3,
    tiers: list[str] | None = None,
    bias_notes: str = "No significant framing differences detected.",
) -> SummaryResult:
    return SummaryResult(
        cluster_id=1,
        summary=f"This is a neutral summary of: {headline}",
        source_count=source_count,
        tiers_covered=tiers or ["national"],
        is_single_source=source_count == 1,
        topic=topic,
        representative_headline=headline,
        bias_notes=bias_notes,
        provider_used="local",
    )


def test_render_produces_valid_html():
    summaries = [
        _make_summary("Congress debates spending bill", topic="politics"),
        _make_summary("Inflation ticks up in April", topic="economy"),
        _make_summary("Storm warning for Lee County", topic="current_events", tiers=["local"]),
    ]
    run_date = datetime(2026, 5, 21, 10, 0, 0, tzinfo=timezone.utc)
    html = render_digest(summaries, "morning", run_date)

    assert "<!DOCTYPE html>" in html
    assert "Morning Briefing" in html
    assert "Congress debates spending bill" in html
    assert "Inflation ticks up in April" in html
    assert "Storm warning for Lee County" in html


def test_render_shows_single_source_warning():
    summaries = [_make_summary("Local road closure", topic="current_events",
                                source_count=1, tiers=["local"])]
    run_date = datetime(2026, 5, 21, 18, 0, 0, tzinfo=timezone.utc)
    html = render_digest(summaries, "evening", run_date)
    assert "1 source" in html


def test_render_shows_bias_notes_when_present():
    summaries = [_make_summary(
        "Immigration policy debate",
        bias_notes="Left-leaning sources emphasized humanitarian impact; right-leaning sources emphasized enforcement.",
    )]
    run_date = datetime(2026, 5, 21, 10, 0, 0, tzinfo=timezone.utc)
    html = render_digest(summaries, "morning", run_date)
    assert "humanitarian impact" in html


def test_render_omits_bias_notes_when_empty():
    summaries = [_make_summary("Routine senate vote",
                                bias_notes="No significant framing differences detected.")]
    run_date = datetime(2026, 5, 21, 10, 0, 0, tzinfo=timezone.utc)
    html = render_digest(summaries, "morning", run_date)
    assert "No significant framing" not in html


def test_render_evening_label():
    summaries = [_make_summary("Evening story")]
    run_date = datetime(2026, 5, 21, 18, 0, 0, tzinfo=timezone.utc)
    html = render_digest(summaries, "evening", run_date)
    assert "Evening Briefing" in html


def test_render_empty_summaries():
    html = render_digest([], "morning", datetime(2026, 5, 21, tzinfo=timezone.utc))
    assert "<!DOCTYPE html>" in html
    assert "0 stories" in html
