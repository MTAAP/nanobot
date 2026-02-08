"""Tests for progress event types."""

from nanobot.bus.progress import ProgressEvent, ProgressKind


def test_progress_kind_values():
    assert ProgressKind.THINKING.value == "thinking"
    assert ProgressKind.TOOL_START.value == "tool_start"
    assert ProgressKind.TOOL_COMPLETE.value == "tool_complete"
    assert ProgressKind.STREAMING.value == "streaming"
    assert ProgressKind.CLARIFICATION.value == "clarification"
    assert ProgressKind.ERROR.value == "error"


def test_progress_event_defaults():
    event = ProgressEvent(
        channel="discord", chat_id="123", kind=ProgressKind.THINKING
    )
    assert event.channel == "discord"
    assert event.chat_id == "123"
    assert event.detail == ""
    assert event.tool_name is None
    assert event.iteration == 0
    assert event.timestamp > 0


def test_progress_event_full():
    event = ProgressEvent(
        channel="telegram",
        chat_id="456",
        kind=ProgressKind.TOOL_START,
        detail="Running exec",
        tool_name="exec",
        iteration=3,
        total_iterations=20,
        metadata={"key": "val"},
    )
    assert event.tool_name == "exec"
    assert event.iteration == 3
    assert event.metadata == {"key": "val"}
