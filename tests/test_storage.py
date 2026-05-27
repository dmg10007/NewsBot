"""Tests for SQLite persistence."""

from datetime import datetime, timezone

from domain.models import Article, DigestRun, DigestStory, ReportingComparison, SourceLink
from storage import SQLiteStore


def test_store_migrates_and_records_digest_story(tmp_path):
    store = SQLiteStore(tmp_path / "newsbot.sqlite")
    store.migrate()
    article = Article(
        article_id=None,
        source_name="AP News",
        source_url="https://ap.test/feed",
        headline="Budget passes",
        url="https://ap.test/story",
        canonical_url="https://ap.test/story",
        url_hash="hash-1",
        published_at=None,
        bias_lean="center",
    )
    saved = store.upsert_articles([article])[0]
    run = store.create_digest_run(DigestRun(run_id=None, period="morning", started_at=datetime.now(timezone.utc)))
    story = DigestStory(
        story_id=None,
        headline="Budget passes",
        summary="A budget passed.",
        geo_tier="national",
        topic="politics",
        importance_score=0.8,
        source_links=[SourceLink("AP News", "https://ap.test/story", "center", "high")],
        comparison=ReportingComparison(cluster_id=1, shared_facts=["Budget passes"]),
        source_count=1,
        is_single_source=True,
        article_ids=[saved.article_id],
    )
    store.save_digest_stories(run, [story])
    store.finish_digest_run(run, story_count=1)

    assert story.story_id is not None
    assert "hash-1" in store.recently_delivered_hashes(24)
    store.close()
