"""Tests for monitoring.health."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from monitoring.health import SourceHealthMonitor, FAILURE_ALERT_THRESHOLD


def _monitor_with_tmp_log(tmp_path: Path) -> SourceHealthMonitor:
    log_path = tmp_path / "health.json"
    with patch("monitoring.health.HEALTH_LOG_PATH", log_path):
        return SourceHealthMonitor()


def test_record_success_resets_consecutive_failures(tmp_path):
    log_path = tmp_path / "health.json"
    with patch("monitoring.health.HEALTH_LOG_PATH", log_path):
        monitor = SourceHealthMonitor()
        monitor.record_failure("Test Source", "timeout")
        monitor.record_failure("Test Source", "timeout")
        monitor.record_success("Test Source", article_count=5)
        assert monitor._state["Test Source"].consecutive_failures == 0


def test_degraded_after_threshold_failures(tmp_path):
    log_path = tmp_path / "health.json"
    with patch("monitoring.health.HEALTH_LOG_PATH", log_path):
        monitor = SourceHealthMonitor()
        for _ in range(FAILURE_ALERT_THRESHOLD):
            monitor.record_failure("Failing Source", "connection error")
        assert monitor._state["Failing Source"].is_degraded is True
        assert "Failing Source" in monitor.degraded_sources()


def test_success_rate_calculation(tmp_path):
    log_path = tmp_path / "health.json"
    with patch("monitoring.health.HEALTH_LOG_PATH", log_path):
        monitor = SourceHealthMonitor()
        monitor.record_success("Source", 10)
        monitor.record_success("Source", 8)
        monitor.record_failure("Source", "error")
        monitor.record_success("Source", 12)
        rate = monitor._state["Source"].success_rate
        assert abs(rate - 0.75) < 0.001


def test_health_state_persists_to_json(tmp_path):
    log_path = tmp_path / "health.json"
    with patch("monitoring.health.HEALTH_LOG_PATH", log_path):
        monitor = SourceHealthMonitor()
        monitor.record_success("AP News", article_count=15)
        assert log_path.exists()
        with open(log_path) as f:
            data = json.load(f)
        assert "AP News" in data
        assert data["AP News"]["last_article_count"] == 15


def test_zero_articles_logs_warning(tmp_path, caplog):
    import logging
    log_path = tmp_path / "health.json"
    with patch("monitoring.health.HEALTH_LOG_PATH", log_path):
        monitor = SourceHealthMonitor()
        with caplog.at_level(logging.WARNING, logger="monitoring.health"):
            monitor.record_success("Broken Scraper", article_count=0)
        assert "0 articles" in caplog.text
