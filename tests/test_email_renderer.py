"""Tests for the DigestStory email renderer contract."""

from __future__ import annotations

from datetime import datetime, timezone

from delivery.email_renderer import EmailRenderer
from domain.models import DigestRun, DigestStory, ReportingComparison, SourceLink


def _story() -> DigestStory:
    return DigestStory(
        story_id=None,
        headline="Congress debates spending bill",
        summary="Lawmakers debated a spending bill and scheduled another vote.",
        geo_tier="national",
        topic="politics",
        importance_score=0.8,
        source_links=[
            SourceLink("AP News", "https://apnews.com/story", "center", "high"),
            SourceLink("Fox News", "https://foxnews.com/story", "right", "medium"),
        ],
        comparison=ReportingComparison(
            cluster_id=1,
            framing_differences=["One source emphasized cost while another emphasized timing."],
            bias_notes="Coverage used different emphasis but reported the same core vote.",
        ),
        source_count=2,
        is_single_source=False,
    )


def test_render_includes_source_links_and_bias_badges():
    run = DigestRun(run_id=1, period="morning", started_at=datetime(2026, 5, 21, tzinfo=timezone.utc))
    html = EmailRenderer().render([_story()], run)

    assert "<!DOCTYPE html>" in html
    assert "Morning Briefing" in html
    assert "Congress debates spending bill" in html
    assert "https://apnews.com/story" in html
    assert "AP News" in html
    assert "Center" in html
    assert "Fox News" in html
    assert "Right" in html
    assert "Reporting differences" in html


def test_render_marks_single_source_story():
    story = _story()
    story.source_links = [SourceLink("Local Source", "https://local.test/story", "center", "medium")]
    story.source_count = 1
    story.is_single_source = True
    story.geo_tier = "local"
    run = DigestRun(run_id=1, period="evening", started_at=datetime(2026, 5, 21, tzinfo=timezone.utc))

    html = EmailRenderer().render([story], run)

    assert "Single source" in html
    assert "Local Source" in html


def test_render_empty_digest():
    run = DigestRun(run_id=1, period="morning", started_at=datetime(2026, 5, 21, tzinfo=timezone.utc))
    html = EmailRenderer().render([], run)
    assert "0 stories" in html
    assert "No reportable stories" in html
