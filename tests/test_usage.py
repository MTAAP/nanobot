"""Tests for cost and token usage tracking."""

import pytest

from nanobot.agent.usage import UsageRecord, UsageTracker


@pytest.fixture
def tracker(tmp_path):
    db = tmp_path / "test_usage.db"
    t = UsageTracker(db_path=db)
    yield t
    t.close()


def test_record_and_session_total(tracker):
    tracker.record(
        UsageRecord(
            model="gpt-4o",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            session_key="test:1",
        )
    )
    tracker.record(
        UsageRecord(
            model="gpt-4o",
            prompt_tokens=200,
            completion_tokens=100,
            total_tokens=300,
            session_key="test:1",
        )
    )
    totals = tracker.get_session_total("test:1")
    assert totals["prompt_tokens"] == 300
    assert totals["completion_tokens"] == 150
    assert totals["total_tokens"] == 450


def test_daily_total(tracker):
    tracker.record(
        UsageRecord(
            model="gpt-4o",
            prompt_tokens=500,
            completion_tokens=200,
            total_tokens=700,
            session_key="test:2",
        )
    )
    totals = tracker.get_daily_total(days=1)
    assert totals["total_tokens"] == 700


def test_model_breakdown(tracker):
    tracker.record(
        UsageRecord(
            model="gpt-4o",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            session_key="s1",
        )
    )
    tracker.record(
        UsageRecord(
            model="claude-3-5-sonnet",
            prompt_tokens=200,
            completion_tokens=100,
            total_tokens=300,
            session_key="s1",
        )
    )
    breakdown = tracker.get_model_breakdown(days=7)
    assert len(breakdown) == 2
    models = {r["model"] for r in breakdown}
    assert "gpt-4o" in models
    assert "claude-3-5-sonnet" in models


def test_empty_session(tracker):
    totals = tracker.get_session_total("nonexistent")
    assert totals["total_tokens"] == 0
