"""Tests for tool execution guardrails."""

import asyncio

import pytest

from nanobot.agent.guardrails import DEFAULT_RULES, GuardrailEngine, GuardrailRule


def test_guardrail_rule_creation():
    rule = GuardrailRule(
        tool_name="exec",
        condition="pattern",
        pattern=r"rm\s+-rf",
        description="Dangerous command",
    )
    assert rule.tool_name == "exec"
    assert rule.condition == "pattern"
    assert rule.pattern == r"rm\s+-rf"
    assert rule.description == "Dangerous command"


def test_default_rules_exist():
    assert len(DEFAULT_RULES) == 3
    names = [r.tool_name for r in DEFAULT_RULES]
    assert "exec" in names
    assert "write_file" in names
    assert "edit_file" in names


def test_check_disabled_returns_none():
    engine = GuardrailEngine(enabled=False)
    result = engine.check("exec", {"command": "rm -rf /"})
    assert result is None


def test_check_matches_dangerous_exec():
    engine = GuardrailEngine(enabled=True)
    result = engine.check("exec", {"command": "rm -rf /"})
    assert result is not None
    assert result.tool_name == "exec"
    assert result.description == "Dangerous shell command"


def test_check_matches_sensitive_file():
    engine = GuardrailEngine(enabled=True)
    result = engine.check("write_file", {"path": "/home/user/.env"})
    assert result is not None
    assert result.tool_name == "write_file"
    assert result.description == "Write to sensitive file"


def test_check_no_match_safe_command():
    engine = GuardrailEngine(enabled=True)
    result = engine.check("exec", {"command": "ls -la"})
    assert result is None


def test_approval_flow():
    engine = GuardrailEngine(enabled=True)
    rule = DEFAULT_RULES[0]
    approval_id = engine.request_approval(rule, "exec", {"command": "rm -rf /"})
    assert approval_id in engine._pending
    engine.approve(approval_id)
    assert engine._approvals.get(approval_id) is True


def test_deny_flow():
    engine = GuardrailEngine(enabled=True)
    rule = DEFAULT_RULES[0]
    approval_id = engine.request_approval(rule, "exec", {"command": "rm -rf /"})
    engine.deny(approval_id)
    assert engine._approvals.get(approval_id) is False


@pytest.mark.asyncio
async def test_approval_timeout():
    engine = GuardrailEngine(enabled=True, timeout_s=60)
    rule = DEFAULT_RULES[0]
    approval_id = engine.request_approval(rule, "exec", {"command": "rm -rf /"})
    approved = await engine.wait_for_approval(approval_id, timeout=0.05)
    assert approved is False
