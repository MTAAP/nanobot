"""Tests for interactive features: clarification detection, components, follow-ups."""

from nanobot.bus.events import OutboundMessage


def test_outbound_message_components_default():
    msg = OutboundMessage(channel="test", chat_id="1", content="hello")
    assert msg.components == []


def test_outbound_message_with_components():
    components = [
        {"type": "button", "label": "Option A", "callback_data": "a"},
        {"type": "button", "label": "Option B", "callback_data": "b"},
    ]
    msg = OutboundMessage(
        channel="test", chat_id="1", content="Pick one", components=components
    )
    assert len(msg.components) == 2
    assert msg.components[0]["label"] == "Option A"


def _make_loop():
    """Create a minimal AgentLoop-like object for testing standalone methods."""
    from nanobot.agent.loop import AgentLoop

    # We only need the methods, so use __new__ to skip __init__
    loop = object.__new__(AgentLoop)
    return loop


def test_detect_clarification_with_question():
    loop = _make_loop()
    assert loop._detect_clarification("What do you mean?") is True


def test_detect_clarification_long_response():
    loop = _make_loop()
    long_text = "A" * 501 + "?"
    assert loop._detect_clarification(long_text) is False


def test_parse_clarification_options():
    loop = _make_loop()
    content = "Which option?\n1. Create a new file\n2. Edit the existing file\n3. Delete it"
    options = loop._parse_clarification_options(content)
    assert len(options) == 3
    assert options[0]["type"] == "button"
    assert "Create a new file" in options[0]["label"]
    assert options[0]["callback_data"] == "Create a new file"


def test_parse_clarification_no_options():
    loop = _make_loop()
    content = "Can you explain more about what you want?"
    options = loop._parse_clarification_options(content)
    assert options == []


def test_check_follow_up_cron():
    loop = _make_loop()
    hint = loop._check_follow_up("cron", {"action": "add"}, "ok")
    assert hint is not None
    assert "cron" in hint.lower()


def test_check_follow_up_safe():
    loop = _make_loop()
    hint = loop._check_follow_up("read_file", {"path": "/tmp/test.txt"}, "contents")
    assert hint is None
